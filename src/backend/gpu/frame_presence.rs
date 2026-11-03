use super::{
    ConvParams, GpuContext, GpuKernel, Model, Result, RtError, create_pipeline, f32_bytes, hal,
    packed_dense, scalar_parameter, sequential_bind_group, softplus, spatial_dense_conv_shader,
    storage_buffer,
};
use crate::backend::frame_presence::{ENCODER_LAYERS, fold_context_projection, frame_input_shape};

pub(super) struct FramePresenceGpu {
    input: hal::Buffer,
    staging: Option<hal::Buffer>,
    output: hal::Buffer,
    layers: Vec<GpuKernel>,
}

impl FramePresenceGpu {
    pub(super) fn compile(model: &Model, context: &GpuContext, max_batch: usize) -> Result<Self> {
        Self::compile_with_upload(
            model,
            context,
            max_batch,
            context.device.stages_input_upload(),
        )
    }

    fn compile_with_upload(
        model: &Model,
        context: &GpuContext,
        max_batch: usize,
        staged: bool,
    ) -> Result<Self> {
        let (height, width) = frame_input_shape(model)?;
        let device = &context.device;
        let allocate =
            |label, elements, usage| storage_buffer(device, label, max_batch, elements, usage);
        let stem_elements = height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| {
                RtError::Inference("frame presence GPU activation size overflow".into())
            })?;
        // Shader indices are u32. The stem uses the largest buffer per sample.
        if stem_elements
            .checked_mul(max_batch)
            .is_none_or(|v| u32::try_from(v).is_err())
        {
            return Err(RtError::Inference(
                "frame presence GPU index range overflow".into(),
            ));
        }
        let input = allocate(
            "frame presence input",
            height * width * 3,
            if staged {
                hal::BufferUses::COPY_DST
            } else {
                hal::BufferUses::MAP_WRITE
            },
        )?;
        let staging = staged
            .then(|| {
                device.create_buffer(
                    "frame presence upload staging",
                    input.size(),
                    hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
                )
            })
            .transpose()?;
        let stage_a = allocate(
            "frame presence scratch A",
            stem_elements,
            hal::BufferUses::empty(),
        )?;
        let stage_b = allocate(
            "frame presence scratch B",
            (height / 4) * (width / 4) * 24,
            hal::BufferUses::empty(),
        )?;
        let (region_h, region_w) = (height / 8, width / 8);
        let region = allocate(
            "frame presence region features",
            region_h * region_w * 32,
            hal::BufferUses::empty(),
        )?;
        let output = allocate("frame presence scores", 1, hal::BufferUses::MAP_READ)?;
        // The region features stay live until the fused head.
        let buffers = [&input, &stage_a, &stage_b, &region, &stage_a, &stage_b];
        let mut layers = Vec::with_capacity(7);
        let (mut input_h, mut input_w, mut input_c) = (height, width, 3);
        for (index, (name, channels, kernel, stride, dilation)) in
            ENCODER_LAYERS.into_iter().take(5).enumerate()
        {
            let (h, w) = (input_h.div_ceil(stride), input_w.div_ceil(stride));
            let (weights, bias) = packed_dense(model, name, input_c, channels, [kernel; 2])?;
            let mut params = ConvParams::dense(
                input_h,
                input_w,
                input_c,
                h,
                w,
                channels,
                [kernel; 2],
                [stride; 2],
                [kernel / 2 * dilation; 2],
                [dilation; 2],
                false,
            );
            params.source_nchw = index == 0;
            let groups = channels / 4;
            let block_groups = if groups.is_multiple_of(4) { 4 } else { 3 };
            let blocks = groups / block_groups;
            let workgroup = if index < 2 { 64 } else { 32 };
            let spatial_tile = if index != 0 && w.is_multiple_of(2) {
                2
            } else {
                1
            };
            let weights = device.create_buffer_init(
                "frame presence weights",
                f32_bytes(&weights),
                hal::BufferUses::STORAGE_READ_ONLY,
            )?;
            let bias = device.create_buffer_init(
                "frame presence biases",
                f32_bytes(&bias),
                hal::BufferUses::STORAGE_READ_ONLY,
            )?;
            let bindings = [buffers[index], &weights, &bias, buffers[index + 1]];
            let shader = spatial_dense_conv_shader(
                params,
                false,
                1,
                workgroup,
                true,
                spatial_tile,
                block_groups,
            );
            layers.push(kernel_dispatch(
                device,
                "frame presence convolution",
                &shader,
                (h * w * blocks / spatial_tile).div_ceil(workgroup),
                &bindings,
            )?);
            (input_h, input_w, input_c) = (h, w, channels);
        }

        let (mut head, bias) = packed_dense(model, "region_head", 32, 1, [1, 1])?;
        let (projection, projection_bias) =
            packed_dense(model, "context_projection", 48, 32, [1, 1])?;
        let (context_head, context_bias) =
            fold_context_projection(&projection, &projection_bias, &head);
        head.extend(context_head);
        head.extend([bias[0] + context_bias, 0.0, 0.0, 0.0]);
        let head = device.create_buffer_init(
            "frame presence head weights",
            f32_bytes(&head),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let head_shader = include_str!("frame_presence_head.wgsl")
            .replace("__REGION_H__", &region_h.to_string())
            .replace("__REGION_W__", &region_w.to_string())
            .replace("__CONTEXT_H__", &input_h.to_string())
            .replace("__CONTEXT_W__", &input_w.to_string());
        // Convolutions have finished with A, which can now store the logit map.
        layers.push(kernel_dispatch(
            device,
            "frame presence region head",
            &head_shader,
            (region_h * region_w).div_ceil(64),
            &[&region, &stage_b, &head, &stage_a],
        )?);

        let scale = softplus(scalar_parameter(
            model,
            "local_evidence_aggregator.log_scale",
        )?) + 0.5;
        let bias = scalar_parameter(model, "local_evidence_aggregator.bias")?;
        let evidence = device.create_buffer_init(
            "frame presence evidence weights",
            f32_bytes(&[scale, bias]),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let pool_shader = include_str!("frame_presence_pool.wgsl")
            .replace("__REGION_H__", &region_h.to_string())
            .replace("__REGION_W__", &region_w.to_string());
        layers.push(kernel_dispatch(
            device,
            "frame presence evidence pooling",
            &pool_shader,
            1,
            &[&stage_a, &evidence, &output],
        )?);

        Ok(Self {
            input,
            staging,
            output,
            layers,
        })
    }

    pub(super) fn run(
        &self,
        context: &mut GpuContext,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<()> {
        context.runner.begin_with_input(
            "frame presence inference",
            &self.input,
            self.staging.as_ref(),
            input,
        )?;
        for layer in &self.layers {
            context.runner.dispatch(
                &layer.pipeline,
                &layer.bind_group,
                [layer.workgroups_per_sample, output.len() as u32, 1],
            );
        }
        context.runner.finish()?;
        self.output.read_f32(output)
    }
}

fn kernel_dispatch(
    device: &hal::Device,
    label: &'static str,
    shader: &str,
    workgroups: usize,
    buffers: &[&hal::Buffer],
) -> Result<GpuKernel> {
    let limit = device.limits().max_compute_workgroups_per_dimension as usize;
    if workgroups > limit {
        return Err(RtError::Inference(format!(
            "{label} dispatch requires {workgroups} workgroups, device limit is {limit}"
        )));
    }
    let pipeline = create_pipeline(device, label, shader)?;
    let bind_group = sequential_bind_group(device, &pipeline, label, buffers)?;
    Ok(GpuKernel {
        pipeline,
        bind_group,
        workgroups_per_sample: workgroups as u32,
        _resources: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::frame_presence::FramePresenceCpu;

    #[test]
    fn wgsl_matches_cpu_with_both_upload_paths() {
        let Some(path) = std::env::var_os("SUBFAST_RT_FRAME_PRESENCE_DIR") else {
            return;
        };
        if std::env::var_os("SUBFAST_RT_TEST_GPU").is_none() {
            return;
        }
        let model = Model::load(path).unwrap();
        let elements = model.input_info()[0].shape.iter().product::<usize>();
        let input = (0..elements)
            .flat_map(|index| (((index % 251) as f32) / 250.0).to_le_bytes())
            .collect::<Vec<_>>();
        let expected = FramePresenceCpu::compile(&model)
            .unwrap()
            .run_nchw_bytes(&input)
            .unwrap();
        for staged in [false, true] {
            let mut context = GpuContext::new(false).unwrap();
            let executable =
                FramePresenceGpu::compile_with_upload(&model, &context, 2, staged).unwrap();
            let batch = input.repeat(2);
            for bytes in [batch.as_slice(), input.as_slice(), batch.as_slice()] {
                let mut output = [0.0; 2];
                let count = bytes.len() / input.len();
                executable
                    .run(&mut context, bytes, &mut output[..count])
                    .unwrap();
                for actual in &output[..count] {
                    assert!((actual - expected).abs() <= 2e-4 * expected.abs().max(1.0));
                }
            }
        }
    }
}
