//! Pure-Rust CPU execution for the Frame Presence (architecture version 8) model.
//!
//! The network is a stride-8 encoder with a context-fused residual region
//! branch: `stem -> detail -> region -> context -> dilation-3 refine`, a 1x1
//! projection of the refined context added back to the region features, a 1x1
//! region head and a coherent local-average/max evidence reduction. Every
//! activation is ReLU except the two 1x1 projections.
//!
//! Feature maps live in channel-last scratch buffers with a fixed zero margin,
//! so the convolution kernels never need per-tap bounds checks inside their hot
//! loops. Weights are packed once at compile time in kernel-major
//! `[ky][kx][ic][oc]` order.

use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use crate::error::{Result, RtError};
use crate::ir::tensor::{DType, TensorKind};
use crate::model::{Model, ModelType};

// (parameter prefix, output channels, kernel, stride, dilation)
pub(super) const ENCODER_LAYERS: [(&str, usize, usize, usize, usize); 6] = [
    ("stem.0", 16, 5, 2, 1),
    ("detail.0", 24, 3, 2, 1),
    ("region_encoder.0", 32, 3, 2, 1),
    ("context_encoder.0", 48, 3, 2, 1),
    ("context_refine.0", 48, 3, 1, 3),
    ("context_projection", 32, 1, 1, 1),
];

pub(super) fn frame_input_shape(model: &Model) -> Result<(usize, usize)> {
    if model.model_type != ModelType::FramePresence || model.inputs.len() != 1 {
        return Err(invalid_model(
            "expected a frame_presence model with one input",
        ));
    }
    let info = model.input_info()[0];
    if info.dtype != DType::Float32 || !matches!(info.shape.as_slice(), [1, 3, _, _]) {
        return Err(RtError::InvalidTensor {
            tensor: info.name.clone(),
            message: format!(
                "expected float32 [1, 3, H, W], got {} {:?}",
                info.dtype, info.shape
            ),
        });
    }
    let (height, width) = (info.shape[2], info.shape[3]);
    if height < 8 || width < 8 || height % 8 != 0 || width % 8 != 0 {
        return Err(invalid_model(
            "input height and width must be positive multiples of 8",
        ));
    }
    Ok((height, width))
}

/// Zero margin kept around every feature map. It covers the widest halo in the
/// model, which is the dilation-3 refine stage reading three samples outside
/// the map.
const MARGIN: usize = 3;

/// Widest row segment processed together by the blocked convolution kernel.
#[cfg(target_arch = "aarch64")]
const WIDE_TILE: usize = 10;

/// Output channels processed together by the blocked convolution kernel.
const CHANNEL_BLOCK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AvxFma,
    Portable,
}

impl Kernel {
    fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma")
        {
            return Self::AvxFma;
        }
        #[allow(unreachable_code)]
        Self::Portable
    }

    fn name(self) -> &'static str {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "neon",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::AvxFma => "avx+fma",
            Self::Portable => "portable",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Relu,
    Linear,
}

impl Activation {
    #[inline(always)]
    fn apply(self, value: f32) -> f32 {
        match self {
            Self::Relu => value.max(0.0),
            Self::Linear => value,
        }
    }
}

/// The presence decision of one frame plus its stride-8 region-logit map.
#[derive(Debug, Clone)]
pub struct FramePresenceFrame {
    pub presence_logit: f32,
    /// Row-major `[height, width]` region logits in the stride-8 padded-image
    /// coordinate space.
    pub region_logits: Vec<f32>,
    pub region_shape: [usize; 2],
}

/// Compiled Frame Presence V8 forward pass with reusable scratch arenas.
pub struct FramePresenceCpu {
    kernel: Kernel,
    parallel: bool,
    stem: DenseConv,
    detail: DenseConv,
    region_encoder: DenseConv,
    context_encoder: DenseConv,
    context_refine: DenseConv,
    context_projection: DenseConv,
    region_head: DenseConv,
    evidence_scale: f32,
    evidence_bias: f32,
    region_height: usize,
    region_width: usize,
    input: Map,
    stem_map: Map,
    detail_map: Map,
    region_map: Map,
    context_map: Map,
    refine_map: Map,
    projected_map: Map,
    logit_map: Map,
    support_rows: Vec<f32>,
}

impl FramePresenceCpu {
    pub fn compile(model: &Model) -> Result<Self> {
        let (input_h, input_w) = frame_input_shape(model)?;
        let compile_layer = |index: usize, in_channels| {
            let (name, channels, kernel, stride, dilation) = ENCODER_LAYERS[index];
            DenseConv::compile(
                model,
                name,
                in_channels,
                channels,
                [kernel; 2],
                [stride; 2],
                [kernel / 2 * dilation; 2],
                [dilation; 2],
            )
        };
        let stem = compile_layer(0, 3)?;
        let detail = compile_layer(1, stem.out_channels)?;
        let region_encoder = compile_layer(2, detail.out_channels)?;
        let context_encoder = compile_layer(3, region_encoder.out_channels)?;
        let context_refine = compile_layer(4, context_encoder.out_channels)?;
        let mut context_projection = compile_layer(5, context_refine.out_channels)?;
        let region_head =
            DenseConv::compile(model, "region_head", 32, 1, [1, 1], [1, 1], [0, 0], [1, 1])?;

        let (weights, bias) = fold_context_projection(
            &context_projection.weights,
            &context_projection.bias,
            &region_head.weights,
        );
        context_projection.out_channels = 1;
        context_projection.weights = weights.into();
        context_projection.bias = Arc::new([bias]);

        let (stem_h, stem_w) = stem.output_shape(input_h, input_w)?;
        let (detail_h, detail_w) = detail.output_shape(stem_h, stem_w)?;
        let (region_h, region_w) = region_encoder.output_shape(detail_h, detail_w)?;
        let (context_h, context_w) = context_encoder.output_shape(region_h, region_w)?;
        if context_refine.output_shape(context_h, context_w)? != (context_h, context_w) {
            return Err(invalid_model(
                "context refine must preserve the context grid",
            ));
        }
        if context_projection.output_shape(context_h, context_w)? != (context_h, context_w) {
            return Err(invalid_model(
                "context projection must preserve the context grid",
            ));
        }
        if region_head.output_shape(region_h, region_w)? != (region_h, region_w) {
            return Err(invalid_model("region head must preserve the region grid"));
        }
        if context_h == 0 || context_w == 0 {
            return Err(invalid_model("input is too small for the feature pyramid"));
        }
        let log_scale = scalar_parameter(model, "local_evidence_aggregator.log_scale")?;
        let evidence_bias = scalar_parameter(model, "local_evidence_aggregator.bias")?;

        Ok(Self {
            kernel: Kernel::detect(),
            parallel: true,
            stem,
            detail,
            region_encoder,
            context_encoder,
            context_refine,
            context_projection,
            region_head,
            evidence_scale: softplus(log_scale) + 0.5,
            evidence_bias,
            region_height: region_h,
            region_width: region_w,
            input: Map::new(3, input_h, input_w),
            stem_map: Map::new(16, stem_h, stem_w),
            detail_map: Map::new(24, detail_h, detail_w),
            region_map: Map::new(32, region_h, region_w),
            context_map: Map::new(48, context_h, context_w),
            refine_map: Map::new(48, context_h, context_w),
            projected_map: Map::new(1, context_h, context_w),
            logit_map: Map::new(1, region_h, region_w),
            support_rows: vec![0.0; region_h * region_w],
        })
    }

    /// Clone the compiled weights into an independent scratch arena, which is
    /// how the packed-batch workers get their own state.
    pub fn worker_clone(&self) -> Self {
        Self {
            kernel: self.kernel,
            parallel: false,
            stem: self.stem.clone(),
            detail: self.detail.clone(),
            region_encoder: self.region_encoder.clone(),
            context_encoder: self.context_encoder.clone(),
            context_refine: self.context_refine.clone(),
            context_projection: self.context_projection.clone(),
            region_head: self.region_head.clone(),
            evidence_scale: self.evidence_scale,
            evidence_bias: self.evidence_bias,
            region_height: self.region_height,
            region_width: self.region_width,
            input: self.input.empty_like(),
            stem_map: self.stem_map.empty_like(),
            detail_map: self.detail_map.empty_like(),
            region_map: self.region_map.empty_like(),
            context_map: self.context_map.empty_like(),
            refine_map: self.refine_map.empty_like(),
            projected_map: self.projected_map.empty_like(),
            logit_map: self.logit_map.empty_like(),
            support_rows: vec![0.0; self.support_rows.len()],
        }
    }

    pub fn kernel_name(&self) -> &'static str {
        self.kernel.name()
    }

    pub fn region_shape(&self) -> [usize; 2] {
        [self.region_height, self.region_width]
    }

    /// Run one preprocessed `[1, 3, H, W]` float32 NCHW sample.
    pub fn run_nchw_bytes(&mut self, input: &[u8]) -> Result<f32> {
        self.load_input(input)?;
        self.forward()
    }

    /// Run one sample and also return the stride-8 region-logit map. Intended
    /// for parity checks against the training-side reference implementation.
    pub fn run_nchw_bytes_with_frame(&mut self, input: &[u8]) -> Result<FramePresenceFrame> {
        self.load_input(input)?;
        let presence_logit = self.forward()?;
        let region_logits = (0..self.region_height)
            .flat_map(|y| (0..self.region_width).map(move |x| (y, x)))
            .map(|(y, x)| self.logit_map.interior(y, x)[0])
            .collect();
        Ok(FramePresenceFrame {
            presence_logit,
            region_logits,
            region_shape: self.region_shape(),
        })
    }

    fn load_input(&mut self, input: &[u8]) -> Result<()> {
        let plane = self.input.height * self.input.width;
        if input.len() != plane * 3 * std::mem::size_of::<f32>() {
            return Err(RtError::InvalidTensor {
                tensor: "<frame-presence-input>".into(),
                message: format!(
                    "expected {} float32 NCHW values, got {} bytes",
                    plane * 3,
                    input.len()
                ),
            });
        }
        let row = self.input.row;
        let width = self.input.width;
        let (red, rest) = input.split_at(plane * 4);
        let (green, blue) = rest.split_at(plane * 4);
        for y in 0..self.input.height {
            let start = y * width;
            let range = start * 4..(start + width) * 4;
            let target = ((y + MARGIN) * row + MARGIN) * 3;
            let target = &mut self.input.data[target..][..width * 3];
            for (((rgb, red), green), blue) in target
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(red[range.clone()].as_chunks::<4>().0)
                .zip(green[range.clone()].as_chunks::<4>().0)
                .zip(blue[range].as_chunks::<4>().0)
            {
                *rgb = [
                    f32::from_le_bytes(*red),
                    f32::from_le_bytes(*green),
                    f32::from_le_bytes(*blue),
                ];
            }
        }
        Ok(())
    }

    /// Run the network on the mapped input; leaves the region logits in
    /// `logit_map` and returns the presence logit.
    fn forward(&mut self) -> Result<f32> {
        conv_into(
            self.kernel,
            self.parallel,
            &self.stem,
            &self.input,
            &mut self.stem_map,
            Activation::Relu,
        );
        conv_into(
            self.kernel,
            self.parallel,
            &self.detail,
            &self.stem_map,
            &mut self.detail_map,
            Activation::Relu,
        );
        conv_into(
            self.kernel,
            self.parallel,
            &self.region_encoder,
            &self.detail_map,
            &mut self.region_map,
            Activation::Relu,
        );
        conv_into(
            self.kernel,
            self.parallel,
            &self.context_encoder,
            &self.region_map,
            &mut self.context_map,
            Activation::Relu,
        );
        conv_into(
            self.kernel,
            self.parallel,
            &self.context_refine,
            &self.context_map,
            &mut self.refine_map,
            Activation::Relu,
        );
        conv_into(
            self.kernel,
            self.parallel,
            &self.context_projection,
            &self.refine_map,
            &mut self.projected_map,
            Activation::Linear,
        );
        conv_into(
            self.kernel,
            self.parallel,
            &self.region_head,
            &self.region_map,
            &mut self.logit_map,
            Activation::Linear,
        );
        upsample_bilinear_add(&self.projected_map, &mut self.logit_map);
        let evidence = coherent_evidence(&self.logit_map, &mut self.support_rows);
        Ok(self.evidence_scale.mul_add(evidence, self.evidence_bias))
    }
}

/// Zero-padded channel-last feature map.
struct Map {
    channels: usize,
    height: usize,
    width: usize,
    /// Row stride in elements: `width + 2 * MARGIN`.
    row: usize,
    data: Vec<f32>,
}

impl Map {
    fn new(channels: usize, height: usize, width: usize) -> Self {
        let row = width + 2 * MARGIN;
        Self {
            channels,
            height,
            width,
            row,
            data: vec![0.0; (height + 2 * MARGIN) * row * channels],
        }
    }

    fn empty_like(&self) -> Self {
        Self {
            channels: self.channels,
            height: self.height,
            width: self.width,
            row: self.row,
            data: vec![0.0; self.data.len()],
        }
    }

    /// Values of one interior pixel; the slice length is the channel count.
    #[inline(always)]
    fn interior(&self, y: usize, x: usize) -> &[f32] {
        let base = ((y + MARGIN) * self.row + x + MARGIN) * self.channels;
        &self.data[base..base + self.channels]
    }

    /// Linear offset of the first channel of one interior row.
    #[inline(always)]
    fn interior_ptr(&self, y: usize) -> usize {
        (y + MARGIN) * self.row * self.channels
    }
}

#[derive(Clone)]
struct DenseConv {
    in_channels: usize,
    out_channels: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    /// Kernel-major weights: `[ky][kx][ic][oc]`.
    weights: Arc<[f32]>,
    bias: Arc<[f32]>,
}

impl DenseConv {
    #[allow(clippy::too_many_arguments)]
    fn compile(
        model: &Model,
        target: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let source = parameter(
            model,
            &format!("{target}.weight"),
            &[out_channels, in_channels, kernel[0], kernel[1]],
        )?;
        let bias = parameter(model, &format!("{target}.bias"), &[out_channels])?;
        let mut weights = vec![0.0; source.len()];
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for ky in 0..kernel[0] {
                    let source_row = ((oc * in_channels + ic) * kernel[0] + ky) * kernel[1];
                    let target_row = ((ky * kernel[1]) * in_channels + ic) * out_channels + oc;
                    for kx in 0..kernel[1] {
                        weights[target_row + kx * in_channels * out_channels] =
                            source[source_row + kx];
                    }
                }
            }
        }
        Ok(Self {
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            weights: weights.into(),
            bias: bias.into(),
        })
    }

    fn output_shape(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        let effective_h = self.dilation[0]
            .checked_mul(self.kernel[0].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_model("convolution height overflow"))?;
        let effective_w = self.dilation[1]
            .checked_mul(self.kernel[1].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_model("convolution width overflow"))?;
        let padded_h = input_h + self.padding[0] * 2;
        let padded_w = input_w + self.padding[1] * 2;
        if padded_h < effective_h || padded_w < effective_w {
            return Err(invalid_model("convolution kernel exceeds its input"));
        }
        Ok((
            (padded_h - effective_h) / self.stride[0] + 1,
            (padded_w - effective_w) / self.stride[1] + 1,
        ))
    }
}

fn conv_into(
    kernel: Kernel,
    parallel: bool,
    conv: &DenseConv,
    input: &Map,
    output: &mut Map,
    activation: Activation,
) {
    let (out_h, out_w) = conv
        .output_shape(input.height, input.width)
        .expect("validated convolution shape");
    debug_assert_eq!(output.channels, conv.out_channels);
    debug_assert_eq!((out_h, out_w), (output.height, output.width));
    if conv.out_channels >= CHANNEL_BLOCK && conv.out_channels.is_multiple_of(CHANNEL_BLOCK) {
        let row_len = output.row * output.channels;
        let left = MARGIN * output.channels;
        let right = left + out_w * output.channels;
        let rows = &mut output.data[MARGIN * row_len..][..out_h * row_len];
        let run_row = |y: usize, row: &mut [f32]| {
            let input_y = y * conv.stride[0] + MARGIN - conv.padding[0];
            let row = &mut row[left..right];
            match kernel {
                #[cfg(target_arch = "aarch64")]
                Kernel::Neon => for_tiles::<WIDE_TILE>(out_w, |ox, width| {
                    // SAFETY: the interior output row and the padded input
                    // include the entire tile. NEON is baseline on AArch64.
                    unsafe {
                        match width {
                            WIDE_TILE => conv_tile_neon::<WIDE_TILE>(
                                conv, input, row, input_y, ox, activation,
                            ),
                            4 => conv_tile_neon::<4>(conv, input, row, input_y, ox, activation),
                            2 => conv_tile_neon::<2>(conv, input, row, input_y, ox, activation),
                            _ => conv_tile_neon::<1>(conv, input, row, input_y, ox, activation),
                        }
                    }
                }),
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                Kernel::AvxFma => for_tiles::<4>(out_w, |ox, width| {
                    // SAFETY: Kernel::detect checked AVX and FMA, and the
                    // interior output row and padded input include the tile.
                    unsafe {
                        match width {
                            4 => conv_tile_avx::<4>(conv, input, row, input_y, ox, activation),
                            2 => conv_tile_avx::<2>(conv, input, row, input_y, ox, activation),
                            _ => conv_tile_avx::<1>(conv, input, row, input_y, ox, activation),
                        }
                    }
                }),
                Kernel::Portable => for_tiles::<4>(out_w, |ox, width| match width {
                    4 => conv_tile_portable::<4>(conv, input, row, input_y, ox, activation),
                    2 => conv_tile_portable::<2>(conv, input, row, input_y, ox, activation),
                    _ => conv_tile_portable::<1>(conv, input, row, input_y, ox, activation),
                }),
            }
        };
        // Parallelize large single-frame layers. Packed-batch workers already
        // process independent images, so they use the serial row path.
        let work =
            out_h * out_w * conv.in_channels * conv.out_channels * conv.kernel[0] * conv.kernel[1];
        if parallel
            && work >= 2_000_000
            && let Some(pool) = row_pool()
        {
            let rows_per_task = out_h.div_ceil(pool.current_num_threads());
            pool.install(|| {
                rows.par_chunks_mut(row_len * rows_per_task)
                    .enumerate()
                    .for_each(|(task, band)| {
                        for (row, target) in band.chunks_exact_mut(row_len).enumerate() {
                            run_row(task * rows_per_task + row, target);
                        }
                    });
            });
        } else {
            for (y, row) in rows.chunks_exact_mut(row_len).enumerate() {
                run_row(y, row);
            }
        }
    } else {
        debug_assert_eq!(conv.kernel, [1, 1]);
        debug_assert_eq!(conv.out_channels, 1);
        debug_assert_eq!(activation, Activation::Linear);
        project_pointwise(conv, input, output);
    }
}

fn project_pointwise(conv: &DenseConv, input: &Map, output: &mut Map) {
    let channels = input.channels;
    let weights = conv.weights.as_chunks::<4>().0;
    let bias = conv.bias[0];
    for y in 0..output.height {
        let source_start = input.interior_ptr(y) + MARGIN * channels;
        let source = &input.data[source_start..][..input.width * channels];
        let target_start = output.interior_ptr(y) + MARGIN;
        let target = &mut output.data[target_start..][..output.width];
        for (pixel, logit) in source.chunks_exact(channels).zip(target) {
            let mut sum = [bias, 0.0, 0.0, 0.0];
            for (values, weights) in pixel.as_chunks::<4>().0.iter().zip(weights) {
                for lane in 0..4 {
                    sum[lane] += values[lane] * weights[lane];
                }
            }
            *logit = sum.into_iter().sum();
        }
    }
}

fn row_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().ok()?.get().min(4);
        if threads <= 1 {
            return None;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("frame-presence-row-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

#[inline(always)]
fn for_tiles<const WIDTH: usize>(end: usize, mut tile: impl FnMut(usize, usize)) {
    let mut x = 0;
    for width in [WIDTH, 4, 2, 1] {
        while end - x >= width {
            tile(x, width);
            x += width;
        }
    }
}

/// One row segment of `PIXELS` output pixels across a block of eight output
/// channels.
fn conv_tile_portable<const PIXELS: usize>(
    conv: &DenseConv,
    input: &Map,
    output: &mut [f32],
    iy0: usize,
    ox: usize,
    activation: Activation,
) {
    let in_c = conv.in_channels;
    let out_c = conv.out_channels;
    let stride_x = conv.stride[1];
    let ox0 = ox * stride_x + MARGIN - conv.padding[1];
    let mut block = 0;
    while block < out_c {
        let mut accumulator = [[0.0f32; CHANNEL_BLOCK]; PIXELS];
        for values in &mut accumulator {
            values.copy_from_slice(&conv.bias[block..block + CHANNEL_BLOCK]);
        }
        for ky in 0..conv.kernel[0] {
            let row = iy0 + ky * conv.dilation[0];
            for kx in 0..conv.kernel[1] {
                let pixel = row * input.row + ox0 + kx * conv.dilation[1];
                let weight = ((ky * conv.kernel[1] + kx) * in_c) * out_c + block;
                for ic in 0..in_c {
                    let weights = &conv.weights[weight + ic * out_c..][..CHANNEL_BLOCK];
                    for (p, target) in accumulator.iter_mut().enumerate() {
                        let value = input.data[(pixel + p * stride_x) * in_c + ic];
                        for lane in 0..CHANNEL_BLOCK {
                            #[cfg(target_arch = "aarch64")]
                            {
                                target[lane] = value.mul_add(weights[lane], target[lane]);
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                target[lane] += value * weights[lane];
                            }
                        }
                    }
                }
            }
        }
        let out_base = ox * out_c + block;
        for (p, values) in accumulator.iter().enumerate() {
            let target = &mut output[out_base + p * out_c..][..CHANNEL_BLOCK];
            for (lane, value) in values.iter().enumerate() {
                target[lane] = activation.apply(*value);
            }
        }
        block += CHANNEL_BLOCK;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn conv_tile_neon<const PIXELS: usize>(
    conv: &DenseConv,
    input: &Map,
    output: &mut [f32],
    iy0: usize,
    ox: usize,
    activation: Activation,
) {
    use std::arch::aarch64::*;

    let in_c = conv.in_channels;
    let out_c = conv.out_channels;
    let stride_x = conv.stride[1];
    let input_row = input.row;
    let input_data = input.data.as_ptr();
    let output_data = output.as_mut_ptr();
    let weights = conv.weights.as_ptr();
    let bias = conv.bias.as_ptr();
    let ox0 = ox * stride_x + MARGIN - conv.padding[1];
    let mut block = 0;
    while block < out_c {
        // SAFETY: the caller validated the map shapes, so every computed pixel
        // index is inside the padded arenas and `block + CHANNEL_BLOCK <= out_c`.
        unsafe {
            let mut accumulator = [vdupq_n_f32(0.0); WIDE_TILE * 2];
            for p in 0..PIXELS {
                accumulator[p * 2] = vld1q_f32(bias.add(block));
                accumulator[p * 2 + 1] = vld1q_f32(bias.add(block + 4));
            }
            for ky in 0..conv.kernel[0] {
                let row = iy0 + ky * conv.dilation[0];
                for kx in 0..conv.kernel[1] {
                    let pixel = row * input_row + ox0 + kx * conv.dilation[1];
                    let weight = ((ky * conv.kernel[1] + kx) * in_c) * out_c + block;
                    for ic in 0..in_c {
                        let weight_lo = vld1q_f32(weights.add(weight + ic * out_c));
                        let weight_hi = vld1q_f32(weights.add(weight + ic * out_c + 4));
                        for p in 0..PIXELS {
                            let value =
                                vdupq_n_f32(*input_data.add((pixel + p * stride_x) * in_c + ic));
                            accumulator[p * 2] = vfmaq_f32(accumulator[p * 2], value, weight_lo);
                            accumulator[p * 2 + 1] =
                                vfmaq_f32(accumulator[p * 2 + 1], value, weight_hi);
                        }
                    }
                }
            }
            let zero = vdupq_n_f32(0.0);
            let out_base = ox * out_c + block;
            for p in 0..PIXELS {
                let mut lo = accumulator[p * 2];
                let mut hi = accumulator[p * 2 + 1];
                if activation == Activation::Relu {
                    lo = vmaxq_f32(lo, zero);
                    hi = vmaxq_f32(hi, zero);
                }
                let target = output_data.add(out_base + p * out_c);
                vst1q_f32(target, lo);
                vst1q_f32(target.add(4), hi);
            }
            block += CHANNEL_BLOCK;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
unsafe fn conv_tile_avx<const PIXELS: usize>(
    conv: &DenseConv,
    input: &Map,
    output: &mut [f32],
    iy0: usize,
    ox: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let in_c = conv.in_channels;
    let out_c = conv.out_channels;
    let stride_x = conv.stride[1];
    let ox0 = ox * stride_x + MARGIN - conv.padding[1];
    // SAFETY: the caller supplies a complete tile inside the output row.
    // All tap indices stay inside the input halo and every vector fits its
    // eight-channel block in the packed weights and output.
    unsafe {
        for block in (0..out_c).step_by(CHANNEL_BLOCK) {
            let initial = _mm256_loadu_ps(conv.bias.as_ptr().add(block));
            let mut sum = [initial; PIXELS];
            for ky in 0..conv.kernel[0] {
                let row = iy0 + ky * conv.dilation[0];
                for kx in 0..conv.kernel[1] {
                    let pixel = row * input.row + ox0 + kx * conv.dilation[1];
                    let weight = ((ky * conv.kernel[1] + kx) * in_c) * out_c + block;
                    for ic in 0..in_c {
                        let values =
                            _mm256_loadu_ps(conv.weights.as_ptr().add(weight + ic * out_c));
                        for (p, target) in sum.iter_mut().enumerate() {
                            let value =
                                *input.data.as_ptr().add((pixel + p * stride_x) * in_c + ic);
                            *target = _mm256_fmadd_ps(_mm256_set1_ps(value), values, *target);
                        }
                    }
                }
            }
            let zero = _mm256_setzero_ps();
            for (p, value) in sum.into_iter().enumerate() {
                let value = if activation == Activation::Relu {
                    _mm256_max_ps(value, zero)
                } else {
                    value
                };
                _mm256_storeu_ps(output.as_mut_ptr().add((ox + p) * out_c + block), value);
            }
        }
    }
}

/// Move the linear region head through the linear context interpolation, so
/// only one channel is projected and interpolated on the context branch.
pub(super) fn fold_context_projection(
    weights: &[f32],
    bias: &[f32],
    head: &[f32],
) -> (Vec<f32>, f32) {
    let dot = |row: &[f32]| {
        row.iter()
            .zip(head)
            .map(|(&value, &weight)| f64::from(value) * f64::from(weight))
            .sum::<f64>() as f32
    };
    (
        weights.chunks_exact(head.len()).map(dot).collect(),
        dot(bias),
    )
}

/// Bilinear `align_corners=False` upsample of the single-channel context.
fn upsample_bilinear_add(context: &Map, region: &mut Map) {
    let scale_y = context.height as f32 / region.height as f32;
    let scale_x = context.width as f32 / region.width as f32;
    for oy in 0..region.height {
        let (y0, y1, weight_y) = source_tap(scale_y, oy, context.height);
        let top = context.interior_ptr(y0) + MARGIN;
        let bottom = context.interior_ptr(y1) + MARGIN;
        let output_start = region.interior_ptr(oy) + MARGIN;
        let row = &mut region.data[output_start..][..region.width];
        for (ox, target) in row.iter_mut().enumerate() {
            let (x0, x1, weight_x) = source_tap(scale_x, ox, context.width);
            let upper =
                (1.0 - weight_x).mul_add(context.data[top + x0], weight_x * context.data[top + x1]);
            let lower = (1.0 - weight_x).mul_add(
                context.data[bottom + x0],
                weight_x * context.data[bottom + x1],
            );
            *target += (1.0 - weight_y).mul_add(upper, weight_y * lower);
        }
    }
}

/// PyTorch's `area_pixel_compute_source_index` plus the two neighbours used by
/// bilinear interpolation with `align_corners=False`.
fn source_tap(scale: f32, destination: usize, source_size: usize) -> (usize, usize, f32) {
    let source = scale.mul_add(destination as f32 + 0.5, -0.5).max(0.0);
    let low = (source as usize).min(source_size - 1);
    let high = (low + 1).min(source_size - 1);
    (low, high, source - low as f32)
}

/// `CoherentEvidencePooling` without a valid mask: a 5x5 local mean with
/// in-bounds divisors, reduced by a spatial maximum and clamped from below.
fn coherent_evidence(logits: &Map, horizontal: &mut [f32]) -> f32 {
    debug_assert_eq!(logits.channels, 1);
    let (height, width) = (logits.height, logits.width);
    for (y, row) in horizontal.chunks_exact_mut(width).enumerate() {
        let start = logits.interior_ptr(y) + MARGIN;
        let source = &logits.data[start..][..width];
        for (x, sum) in row.iter_mut().enumerate() {
            *sum = source[x.saturating_sub(2)..(x + 3).min(width)].iter().sum();
        }
    }
    let mut evidence = -10_000.0f32;
    for y in 0..height {
        let y0 = y.saturating_sub(2);
        let y1 = (y + 3).min(height);
        for x in 0..width {
            let mut sum = 0.0;
            for row in horizontal[y0 * width..y1 * width].chunks_exact(width) {
                sum += row[x];
            }
            let columns = (x + 3).min(width) - x.saturating_sub(2);
            evidence = evidence.max(sum / ((y1 - y0) * columns) as f32);
        }
    }
    evidence
}

fn parameter(model: &Model, target: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = model
        .tensors
        .iter()
        .find(|info| {
            matches!(info.kind, TensorKind::Parameter | TensorKind::Buffer)
                && info.target.as_deref() == Some(target)
        })
        .ok_or_else(|| invalid_model(format!("missing parameter {target}")))?;
    if info.dtype != DType::Float32 || info.shape.as_slice() != shape {
        return Err(RtError::InvalidTensor {
            tensor: target.into(),
            message: format!(
                "expected float32 shape {shape:?}, got {} {:?}",
                info.dtype, info.shape
            ),
        });
    }
    let weight = info
        .weight
        .ok_or_else(|| invalid_model(format!("parameter {target} has no stored data")))?;
    let bytes = model.weights.slice(weight.offset, weight.byte_length)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn scalar_parameter(model: &Model, target: &str) -> Result<f32> {
    let info = model
        .tensors
        .iter()
        .find(|info| {
            matches!(info.kind, TensorKind::Parameter | TensorKind::Buffer)
                && info.target.as_deref() == Some(target)
        })
        .ok_or_else(|| invalid_model(format!("missing parameter {target}")))?;
    if !info.shape.is_empty() && info.shape != [1] {
        return Err(RtError::InvalidTensor {
            tensor: target.into(),
            message: format!("expected scalar shape [] or [1], got {:?}", info.shape),
        });
    }
    parameter(model, target, &info.shape)?
        .first()
        .copied()
        .ok_or_else(|| invalid_model(format!("scalar parameter {target} is empty")))
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn invalid_model(message: impl Into<String>) -> RtError {
    RtError::InvalidModel(format!("frame-presence backend: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::{FramePresenceCpu, Kernel};
    use crate::model::Model;

    /// Compare the portable and selected SIMD kernels with the actual checkpoint.
    #[test]
    fn portable_kernel_matches_selected_kernel() {
        let Ok(path) = std::env::var("SUBFAST_RT_FRAME_PRESENCE_DIR") else {
            eprintln!("skipping frame-presence kernel parity; set SUBFAST_RT_FRAME_PRESENCE_DIR");
            return;
        };
        let model = Model::load(path).unwrap();
        let shape = model.input_info()[0].shape.clone();
        let elements = shape[1..].iter().product::<usize>();
        let mut state = 0x5eed_1234u32;
        let values = (0..elements)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
            })
            .collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();

        let mut selected = FramePresenceCpu::compile(&model).unwrap();
        let expected = selected.run_nchw_bytes(&bytes).unwrap();
        let mut portable = FramePresenceCpu::compile(&model).unwrap();
        portable.kernel = Kernel::Portable;
        let actual = portable.run_nchw_bytes(&bytes).unwrap();
        assert!((actual - expected).abs() <= 2e-4 * expected.abs().max(1.0));
    }
}
