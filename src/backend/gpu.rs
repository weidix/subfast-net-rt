//! GPU backend for the ROI Presence and Pair Matcher models.
//!
//! WGSL compute kernels are compiled through `wgpu-hal` and Naga to Metal on
//! macOS and Vulkan on the other supported native targets. Input upload,
//! command submission, and scalar readback are host operations; every model
//! operator executes on the GPU. Weights, pipelines, bindings, and buffers are
//! reused.

use crate::backend::cpu::validate_roi_model;
use crate::backend::gpu_hal as hal;
use crate::backend::{Backend, Executable};
use crate::error::{Result, RtError};
use crate::ir::tensor::{DType, TensorInfo, TensorKind};
use crate::model::Model;
use crate::tensor::Tensor;

const WORKGROUP_SIZE: usize = 64;

const CONV_SHADER: &str = r#"
struct Params {
    input_h: u32,
    input_w: u32,
    in_channels: u32,
    output_h: u32,
    output_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    depthwise: u32,
    activation: u32,
}

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> destination: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

fn fast_exp(value: f32) -> f32 {
    let x = clamp(value, -80.0, 80.0);
    let scaled = x * 1.4426950408889634;
    let n = sign(scaled) * floor(abs(scaled) + 0.5);
    let r = fma(-n, 0.6931471805599453, x);
    var polynomial = 1.0 / 720.0;
    polynomial = fma(polynomial, r, 1.0 / 120.0);
    polynomial = fma(polynomial, r, 1.0 / 24.0);
    polynomial = fma(polynomial, r, 1.0 / 6.0);
    polynomial = fma(polynomial, r, 0.5);
    polynomial = fma(polynomial, r, 1.0);
    polynomial = fma(polynomial, r, 1.0);
    let exponent = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return polynomial * exponent;
}

fn silu(value: f32) -> f32 {
    return value / (1.0 + fast_exp(-value));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let output_elements = p.output_h * p.output_w * p.out_channels;
    let sample = invocation.y;
    let local = invocation.x;
    if (local >= output_elements) {
        return;
    }
    let index = sample * output_elements + local;
    let output_channel = local % p.out_channels;
    let spatial = local / p.out_channels;
    let output_x = spatial % p.output_w;
    let output_y = spatial / p.output_w;
    let input_elements = p.input_h * p.input_w * p.in_channels;
    var sum = bias[output_channel];

    for (var kernel_y = 0u; kernel_y < p.kernel_h; kernel_y += 1u) {
        let input_y = i32(output_y * p.stride_h + kernel_y * p.dilation_h)
            - i32(p.padding_h);
        if (input_y < 0 || input_y >= i32(p.input_h)) {
            continue;
        }
        for (var kernel_x = 0u; kernel_x < p.kernel_w; kernel_x += 1u) {
            let input_x = i32(output_x * p.stride_w + kernel_x * p.dilation_w)
                - i32(p.padding_w);
            if (input_x < 0 || input_x >= i32(p.input_w)) {
                continue;
            }
            let input_base = sample * input_elements
                + (u32(input_y) * p.input_w + u32(input_x)) * p.in_channels;
            let kernel_base = (kernel_y * p.kernel_w + kernel_x);
            if (p.depthwise == 1u) {
                let weight_index = kernel_base * p.out_channels + output_channel;
                sum = fma(source[input_base + output_channel], weights[weight_index], sum);
            } else {
                let weight_base = kernel_base * p.in_channels * p.out_channels;
                for (var input_channel = 0u; input_channel < p.in_channels; input_channel += 1u) {
                    let weight_index = weight_base
                        + input_channel * p.out_channels
                        + output_channel;
                    var source_index = input_base + input_channel;
                    if (p.depthwise == 2u) {
                        source_index = sample * input_elements
                            + input_channel * p.input_h * p.input_w
                            + u32(input_y) * p.input_w
                            + u32(input_x);
                    }
                    sum = fma(source[source_index], weights[weight_index], sum);
                }
            }
        }
    }
    destination[index] = select(sum, silu(sum), p.activation != 0u);
}
"#;

const PRESENCE_HEAD_FINISH_SHADER: &str = r#"
const INPUT_H: u32 = __INPUT_H__u;
const INPUT_W: u32 = __INPUT_W__u;
const REGION_H: u32 = __REGION_H__u;
const REGION_W: u32 = __REGION_W__u;
const INPUT_PLANE: u32 = INPUT_H * INPUT_W;
const INPUT_ELEMENTS: u32 = INPUT_PLANE * 3u;
const REGION_ELEMENTS: u32 = REGION_H * REGION_W;

struct Params {
    head_bias: f32,
    scale: f32,
    output_bias: f32,
    _padding: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> head_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

var<workgroup> region: array<f32, REGION_ELEMENTS>;
var<workgroup> valid_region: array<u32, REGION_ELEMENTS>;
var<workgroup> horizontal_sum: array<f32, REGION_ELEMENTS>;
var<workgroup> horizontal_count: array<u32, REGION_ELEMENTS>;
var<workgroup> maxima: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let sample = workgroup.x;
    let input_base = sample * INPUT_ELEMENTS;
    let feature_base = sample * REGION_ELEMENTS * 32u;

    for (var index = local; index < REGION_ELEMENTS; index += 256u) {
        let output_y = index / REGION_W;
        let output_x = index % REGION_W;
        var sum = p.head_bias;
        for (var kernel_y = 0u; kernel_y < 3u; kernel_y += 1u) {
            let input_y = i32(output_y + kernel_y * 2u) - 2;
            if (input_y < 0 || input_y >= i32(REGION_H)) {
                continue;
            }
            for (var kernel_x = 0u; kernel_x < 3u; kernel_x += 1u) {
                let input_x = i32(output_x + kernel_x * 2u) - 2;
                if (input_x < 0 || input_x >= i32(REGION_W)) {
                    continue;
                }
                let source_base = feature_base
                    + (u32(input_y) * REGION_W + u32(input_x)) * 32u;
                let weight_base = (kernel_y * 3u + kernel_x) * 32u;
                for (var channel = 0u; channel < 32u; channel += 1u) {
                    sum = fma(
                        features[source_base + channel],
                        head_weights[weight_base + channel],
                        sum,
                    );
                }
            }
        }
        region[index] = sum;

        let oy = index / REGION_W;
        let ox = index % REGION_W;
        var count = 0u;
        for (var local_y = 0u; local_y < 4u; local_y += 1u) {
            let y = oy * 4u + local_y;
            for (var local_x = 0u; local_x < 4u; local_x += 1u) {
                let x = ox * 4u + local_x;
                let pixel = y * INPUT_W + x;
                var magnitude = 0.0;
                for (var channel = 0u; channel < 3u; channel += 1u) {
                    magnitude = max(
                        magnitude,
                        abs(input[input_base + channel * INPUT_PLANE + pixel]),
                    );
                }
                count += select(0u, 1u, magnitude > 1e-8);
            }
        }
        valid_region[index] = select(0u, 1u, count > 8u);
    }
    workgroupBarrier();

    for (var index = local; index < REGION_ELEMENTS; index += 256u) {
        let y = index / REGION_W;
        let x = index % REGION_W;
        let x0 = select(x - 2u, 0u, x < 2u);
        let x1 = min(x + 3u, REGION_W);
        var sum = 0.0;
        var count = 0u;
        for (var xx = x0; xx < x1; xx += 1u) {
            let neighbor = y * REGION_W + xx;
            if (valid_region[neighbor] != 0u) {
                sum += region[neighbor];
                count += 1u;
            }
        }
        horizontal_sum[index] = sum;
        horizontal_count[index] = count;
    }
    workgroupBarrier();

    var best = -3.402823466e+38;
    for (var index = local; index < REGION_ELEMENTS; index += 256u) {
        if (valid_region[index] != 0u) {
            let y = index / REGION_W;
            let x = index % REGION_W;
            let y0 = select(y - 2u, 0u, y < 2u);
            let y1 = min(y + 3u, REGION_H);
            var sum = 0.0;
            var count = 0u;
            for (var yy = y0; yy < y1; yy += 1u) {
                let neighbor = yy * REGION_W + x;
                sum += horizontal_sum[neighbor];
                count += horizontal_count[neighbor];
            }
            best = max(best, sum / f32(max(count, 1u)));
        }
    }
    maxima[local] = best;
    workgroupBarrier();

    for (var stride = 128u; stride != 0u; stride >>= 1u) {
        if (local < stride) {
            maxima[local] = max(maxima[local], maxima[local + stride]);
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        output[sample] = p.scale * max(maxima[0], -10000.0) + p.output_bias;
    }
}
"#;

const PAIR_FEATURE_TILED_SHADER: &str = r#"
const INPUT_H: u32 = __INPUT_H__u;
const INPUT_W: u32 = __INPUT_W__u;
const INPUT_PLANE: u32 = INPUT_H * INPUT_W;
const INPUT_ELEMENTS: u32 = INPUT_PLANE * 3u;

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn value_channel(red_input: f32, green_input: f32, blue_input: f32) -> f32 {
    let red = red_input * 0.229 + 0.485;
    let green = green_input * 0.224 + 0.456;
    let blue = blue_input * 0.225 + 0.406;
    return (max(max(red, green), blue) - 0.449) / 0.226;
}

@compute @workgroup_size(128)
fn main(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let sample = workgroup.y;
    let pixel = workgroup.x * 128u + local;
    if (pixel >= INPUT_PLANE) {
        return;
    }
    let y = pixel / INPUT_W;
    let x = pixel % INPUT_W;
    let input_base = sample * INPUT_ELEMENTS;
    let output_base = sample * INPUT_ELEMENTS + pixel * 3u;
    let left_value = value_channel(
        left[input_base + pixel],
        left[input_base + INPUT_PLANE + pixel],
        left[input_base + INPUT_PLANE * 2u + pixel],
    );
    let right_value = value_channel(
        right[input_base + pixel],
        right[input_base + INPUT_PLANE + pixel],
        right[input_base + INPUT_PLANE * 2u + pixel],
    );
    let delta = left_value - right_value;
    let y0 = select(y - 2u, 0u, y < 2u);
    let y1 = min(y + 3u, INPUT_H);
    let x0 = select(x - 2u, 0u, x < 2u);
    let x1 = min(x + 3u, INPUT_W);
    let count = f32((y1 - y0) * (x1 - x0));
    var background = 0.0;
    for (var yy = y0; yy < y1; yy += 1u) {
        for (var xx = x0; xx < x1; xx += 1u) {
            let neighbor = yy * INPUT_W + xx;
            background += value_channel(
                left[input_base + neighbor],
                left[input_base + INPUT_PLANE + neighbor],
                left[input_base + INPUT_PLANE * 2u + neighbor],
            ) - value_channel(
                right[input_base + neighbor],
                right[input_base + INPUT_PLANE + neighbor],
                right[input_base + INPUT_PLANE * 2u + neighbor],
            );
        }
    }
    background /= count;
    output[output_base] = 0.5 * (left_value + right_value);
    output[output_base + 1u] = abs(delta);
    output[output_base + 2u] = abs(delta - background);
}
"#;

const PAIR_TAIL_SHADER: &str = r#"
const INPUT_H: u32 = __INPUT_H__u;
const INPUT_W: u32 = __INPUT_W__u;
const CHANNELS: u32 = 48u;
const TILE_H: u32 = 4u;
const TILE_W: u32 = 16u;
const TILE_PIXELS: u32 = TILE_H * TILE_W;
const TILE_COLUMNS: u32 = (INPUT_W + TILE_W - 1u) / TILE_W;
const TILE_ROWS: u32 = (INPUT_H + TILE_H - 1u) / TILE_H;
const TILE_COUNT: u32 = TILE_ROWS * TILE_COLUMNS;
const INPUT_ELEMENTS: u32 = INPUT_H * INPUT_W * CHANNELS;

struct Params {
    mask_bias: f32,
    classifier2_bias: f32,
    _padding0: f32,
    _padding1: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> mask_weight: array<f32>;
@group(0) @binding(2) var<storage, read> classifier1_weight: array<f32>;
@group(0) @binding(3) var<storage, read> classifier1_bias: array<f32>;
@group(0) @binding(4) var<storage, read> classifier2_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

var<workgroup> gated_tile: array<f32, TILE_PIXELS * CHANNELS>;
var<workgroup> pooled: array<f32, 144>;
var<workgroup> hidden: array<f32, 32>;

fn fast_exp(value: f32) -> f32 {
    let x = clamp(value, -80.0, 80.0);
    let scaled = x * 1.4426950408889634;
    let n = sign(scaled) * floor(abs(scaled) + 0.5);
    let r = fma(-n, 0.6931471805599453, x);
    var polynomial = 1.0 / 720.0;
    polynomial = fma(polynomial, r, 1.0 / 120.0);
    polynomial = fma(polynomial, r, 1.0 / 24.0);
    polynomial = fma(polynomial, r, 1.0 / 6.0);
    polynomial = fma(polynomial, r, 0.5);
    polynomial = fma(polynomial, r, 1.0);
    polynomial = fma(polynomial, r, 1.0);
    let exponent = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return polynomial * exponent;
}

fn silu(value: f32) -> f32 {
    return value / (1.0 + fast_exp(-value));
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let sample = workgroup.y;
    let input_base = sample * INPUT_ELEMENTS;
    var sum = 0.0;
    var peak = -3.402823466e+38;
    var width_peak = -3.402823466e+38;
    var width_sums: array<f32, INPUT_W>;

    for (var tile = 0u; tile < TILE_COUNT; tile += 1u) {
        let tile_x = tile % TILE_COLUMNS;
        let tile_y = tile / TILE_COLUMNS;
        let x = tile_x * TILE_W + local % TILE_W;
        let y = tile_y * TILE_H + local / TILE_W;
        let pixel_base = input_base + (y * INPUT_W + x) * CHANNELS;
        var mask = p.mask_bias;
        let is_active = x < INPUT_W && y < INPUT_H;
        if (is_active) {
            for (var channel = 0u; channel < CHANNELS; channel += 1u) {
                mask = fma(input[pixel_base + channel], mask_weight[channel], mask);
            }
        }
        let gate = 0.25 + 0.75 / (1.0 + fast_exp(-mask));
        for (var channel = 0u; channel < CHANNELS; channel += 1u) {
            var value = 0.0;
            if (is_active) {
                value = input[pixel_base + channel] * gate;
            }
            gated_tile[local * CHANNELS + channel] = value;
        }
        workgroupBarrier();

        if (local < CHANNELS) {
            for (var column = 0u; column < TILE_W; column += 1u) {
                var width_sum = 0.0;
                for (var tile_y = 0u; tile_y < TILE_H; tile_y += 1u) {
                    let value = gated_tile[(tile_y * TILE_W + column) * CHANNELS + local];
                    let actual_x = tile_x * TILE_W + column;
                    let actual_y = (tile / TILE_COLUMNS) * TILE_H + tile_y;
                    if (actual_x < INPUT_W && actual_y < INPUT_H) {
                        sum += value;
                        width_sum += value;
                        peak = max(peak, value);
                    }
                }
                let actual_x = tile_x * TILE_W + column;
                if (actual_x < INPUT_W) {
                    width_sums[actual_x] += width_sum;
                }
            }
        }
        workgroupBarrier();
    }

    if (local < CHANNELS) {
        for (var x = 0u; x < INPUT_W; x += 1u) {
            width_peak = max(width_peak, width_sums[x] / f32(INPUT_H));
        }
        pooled[local] = sum / f32(INPUT_H * INPUT_W);
        pooled[CHANNELS + local] = peak;
        pooled[CHANNELS * 2u + local] = width_peak;
    }
    workgroupBarrier();

    if (local < 32u) {
        var classifier_sum = classifier1_bias[local];
        for (var column = 0u; column < 144u; column += 1u) {
            classifier_sum = fma(
                pooled[column],
                classifier1_weight[local * 144u + column],
                classifier_sum,
            );
        }
        hidden[local] = silu(classifier_sum);
    }
    workgroupBarrier();

    if (local == 0u) {
        var classifier_sum = p.classifier2_bias;
        for (var column = 0u; column < 32u; column += 1u) {
            classifier_sum = fma(
                hidden[column],
                classifier2_weight[column],
                classifier_sum,
            );
        }
        output[sample] = classifier_sum;
    }
}
"#;

#[derive(Debug, Clone, Copy)]
pub struct GpuBackend {
    max_batch_size: usize,
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self { max_batch_size: 1 }
    }
}

impl GpuBackend {
    pub fn with_max_batch_size(max_batch_size: usize) -> Result<Self> {
        if max_batch_size == 0 {
            return Err(RtError::Inference(
                "GPU max batch size must be greater than zero".into(),
            ));
        }
        Ok(Self { max_batch_size })
    }
}

pub struct GpuExecutable {
    input_shapes: Vec<Vec<usize>>,
    max_batch_size: usize,
    context: GpuContext,
    model: CompiledGpuModel,
}

enum CompiledGpuModel {
    Presence(Box<PresenceGpu>),
    Pair(Box<PairGpu>),
}

impl Backend for GpuBackend {
    type Executable = GpuExecutable;

    fn compile(&self, model: &Model) -> Result<Self::Executable> {
        // Keep GPU model acceptance exactly aligned with the CPU backend.
        validate_roi_model(model)?;
        let input_shapes = model
            .input_info()
            .into_iter()
            .map(|info| info.shape.clone())
            .collect();
        let context = GpuContext::new()?;
        context.validate_max_batch_size(self.max_batch_size)?;
        let compiled = match model.inputs.len() {
            1 => CompiledGpuModel::Presence(Box::new(PresenceGpu::compile(
                model,
                &context,
                self.max_batch_size,
            )?)),
            2 => CompiledGpuModel::Pair(Box::new(PairGpu::compile(
                model,
                &context,
                self.max_batch_size,
            )?)),
            count => {
                return Err(RtError::Inference(format!(
                    "GPU ROI backend expects one or two inputs, got {count}"
                )));
            }
        };
        Ok(GpuExecutable {
            input_shapes,
            max_batch_size: self.max_batch_size,
            context,
            model: compiled,
        })
    }
}

impl GpuExecutable {
    pub fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }

    pub fn backend_name(&self) -> &'static str {
        self.context.backend_name
    }

    pub fn adapter_name(&self) -> &str {
        &self.context.adapter_name
    }

    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    pub fn run_scalar(&mut self, inputs: &[Tensor]) -> Result<f32> {
        validate_runtime_inputs(inputs, &self.input_shapes)?;
        let mut output = [0.0];
        self.run_validated_batch(inputs, &mut output)?;
        Ok(output[0])
    }

    pub fn run_batch_scalar(&mut self, inputs: &[Tensor]) -> Result<Vec<f32>> {
        let batch_size = validate_packed_inputs(inputs, &self.input_shapes, self.max_batch_size)?;
        let mut output = vec![0.0; batch_size];
        self.run_validated_batch(inputs, &mut output)?;
        Ok(output)
    }

    pub fn run_batch_scalar_into(&mut self, inputs: &[Tensor], output: &mut [f32]) -> Result<()> {
        let batch_size = validate_packed_inputs(inputs, &self.input_shapes, self.max_batch_size)?;
        if output.len() != batch_size {
            return Err(RtError::InvalidTensor {
                tensor: "<gpu-batch-output>".into(),
                message: format!(
                    "expected {batch_size} scalar output slots, got {}",
                    output.len()
                ),
            });
        }
        self.run_validated_batch(inputs, output)
    }

    fn run_validated_batch(&mut self, inputs: &[Tensor], output: &mut [f32]) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }
        let batch_size = output.len();
        match &mut self.model {
            CompiledGpuModel::Presence(model) => {
                model.run(&mut self.context, inputs, batch_size, output)?
            }
            CompiledGpuModel::Pair(model) => {
                model.run(&mut self.context, inputs, batch_size, output)?
            }
        }
        Ok(())
    }

    pub fn run_batch(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        let values = self.run_batch_scalar(inputs)?;
        Tensor::from_f32(vec![values.len()], values)
    }
}

impl Executable for GpuExecutable {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let value = self.run_scalar(inputs)?;
        Ok(vec![Tensor::from_f32(vec![1], vec![value])?])
    }
}

struct GpuContext {
    device: hal::Device,
    runner: hal::Runner,
    conv_pipeline: hal::Pipeline,
    backend_name: &'static str,
    adapter_name: String,
}

impl GpuContext {
    fn new() -> Result<Self> {
        let (device, info) = hal::Device::new()?;
        let runner = device.create_runner()?;
        let conv_pipeline = create_pipeline(&device, "subfast ROI convolution", CONV_SHADER)?;
        Ok(Self {
            device,
            runner,
            conv_pipeline,
            backend_name: hal::backend_name(),
            adapter_name: info.name,
        })
    }

    fn validate_max_batch_size(&self, max_batch_size: usize) -> Result<()> {
        let limit = self.device.limits().max_compute_workgroups_per_dimension as usize;
        if max_batch_size > limit {
            return Err(RtError::Inference(format!(
                "GPU max batch size {max_batch_size} exceeds device dispatch limit {limit}"
            )));
        }
        Ok(())
    }

    fn write_input(&self, buffer: &hal::Buffer, data: &[u8]) -> Result<()> {
        buffer.write(data)
    }

    fn execute_presence(
        &mut self,
        layers: &[GpuLayer],
        finish: &GpuKernel,
        output: &hal::Buffer,
        batch_size: usize,
        host_output: &mut [f32],
    ) -> Result<()> {
        self.runner.begin("subfast ROI inference")?;
        for layer in layers {
            let pipeline = layer.pipeline.as_ref().unwrap_or(&self.conv_pipeline);
            self.runner.dispatch(
                pipeline,
                &layer.bind_group,
                [layer.workgroups_per_sample, batch_size as u32, 1],
            );
        }
        self.runner.dispatch(
            &finish.pipeline,
            &finish.bind_group,
            [finish.workgroups_per_sample * batch_size as u32, 1, 1],
        );
        self.runner.finish()?;
        output.read_f32(&mut host_output[..batch_size])
    }

    fn execute_pair(
        &mut self,
        features: &GpuKernel,
        layers: &[GpuLayer],
        tail: &GpuKernel,
        output: &hal::Buffer,
        batch_size: usize,
        host_output: &mut [f32],
    ) -> Result<()> {
        self.runner.begin("subfast pair inference")?;
        self.runner.dispatch(
            &features.pipeline,
            &features.bind_group,
            [features.workgroups_per_sample, batch_size as u32, 1],
        );
        for layer in layers {
            let pipeline = layer.pipeline.as_ref().unwrap_or(&self.conv_pipeline);
            self.runner.dispatch(
                pipeline,
                &layer.bind_group,
                [layer.workgroups_per_sample, batch_size as u32, 1],
            );
        }
        self.runner.dispatch(
            &tail.pipeline,
            &tail.bind_group,
            [tail.workgroups_per_sample, batch_size as u32, 1],
        );
        self.runner.finish()?;
        output.read_f32(&mut host_output[..batch_size])
    }
}

fn create_pipeline(
    device: &hal::Device,
    label: &'static str,
    source: &str,
) -> Result<hal::Pipeline> {
    device.create_pipeline(label, source)
}

fn presence_finish_shader(
    input_h: usize,
    input_w: usize,
    region_h: usize,
    region_w: usize,
) -> String {
    PRESENCE_HEAD_FINISH_SHADER
        .replace("__INPUT_H__", &input_h.to_string())
        .replace("__INPUT_W__", &input_w.to_string())
        .replace("__REGION_H__", &region_h.to_string())
        .replace("__REGION_W__", &region_w.to_string())
}

fn pair_feature_shader(input_h: usize, input_w: usize) -> String {
    PAIR_FEATURE_TILED_SHADER
        .replace("__INPUT_H__", &input_h.to_string())
        .replace("__INPUT_W__", &input_w.to_string())
}

fn pair_tail_shader(input_h: usize, input_w: usize) -> String {
    PAIR_TAIL_SHADER
        .replace("__INPUT_H__", &input_h.to_string())
        .replace("__INPUT_W__", &input_w.to_string())
}

struct GpuLayer {
    pipeline: Option<hal::Pipeline>,
    bind_group: hal::BindGroup,
    workgroups_per_sample: u32,
    _resources: Vec<hal::Buffer>,
}

struct GpuKernel {
    pipeline: hal::Pipeline,
    bind_group: hal::BindGroup,
    workgroups_per_sample: u32,
    _resources: Vec<hal::Buffer>,
}

#[derive(Clone, Copy)]
struct ConvParams {
    input_h: usize,
    input_w: usize,
    in_channels: usize,
    output_h: usize,
    output_w: usize,
    out_channels: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    depthwise: bool,
    source_nchw: bool,
    activation: bool,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct SeparableParams {
    input_h: usize,
    input_w: usize,
    in_channels: usize,
    output_h: usize,
    output_w: usize,
    out_channels: usize,
    stride_h: usize,
}

#[allow(dead_code)]
fn fused_separable_spatial_shader(params: SeparableParams) -> String {
    let output_groups = params.out_channels / 4;
    let accumulators = (0..output_groups)
        .map(|group| format!("    var sum{group} = pointwise_bias[{group}u];"))
        .collect::<Vec<_>>()
        .join("\n");
    let updates = (0..4)
        .flat_map(|component| {
            (0..output_groups).map(move |group| {
                format!(
                    "        sum{group} = fma(\n            vec4<f32>(depthwise_value[{component}u]),\n            pointwise_weights[(input_group * 4u + {component}u)\n                * OUTPUT_GROUPS + {group}u],\n            sum{group},\n        );"
                )
            })
        })
        .collect::<Vec<_>>()
        .join("\n");
    let stores = (0..output_groups)
        .map(|group| {
            format!(
                "    destination[sample * OUTPUT_H * OUTPUT_W * OUTPUT_GROUPS\n        + local * OUTPUT_GROUPS + {group}u] = silu(sum{group});"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"
const INPUT_H: u32 = {input_h}u;
const INPUT_W: u32 = {input_w}u;
const INPUT_GROUPS: u32 = {input_groups}u;
const OUTPUT_H: u32 = {output_h}u;
const OUTPUT_W: u32 = {output_w}u;
const OUTPUT_GROUPS: u32 = {output_groups}u;
const STRIDE_H: u32 = {stride_h}u;

@group(0) @binding(0) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> depthwise_weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> depthwise_bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pointwise_weights: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pointwise_bias: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> destination: array<vec4<f32>>;

fn fast_exp(value: f32) -> f32 {{
    let x = clamp(value, -80.0, 80.0);
    let scaled = x * 1.4426950408889634;
    let n = sign(scaled) * floor(abs(scaled) + 0.5);
    let r = fma(-n, 0.6931471805599453, x);
    var polynomial = 1.0 / 720.0;
    polynomial = fma(polynomial, r, 1.0 / 120.0);
    polynomial = fma(polynomial, r, 1.0 / 24.0);
    polynomial = fma(polynomial, r, 1.0 / 6.0);
    polynomial = fma(polynomial, r, 0.5);
    polynomial = fma(polynomial, r, 1.0);
    polynomial = fma(polynomial, r, 1.0);
    let exponent = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return polynomial * exponent;
}}

fn silu(value: vec4<f32>) -> vec4<f32> {{
    return vec4<f32>(
        value.x / (1.0 + fast_exp(-value.x)),
        value.y / (1.0 + fast_exp(-value.y)),
        value.z / (1.0 + fast_exp(-value.z)),
        value.w / (1.0 + fast_exp(-value.w)),
    );
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {{
    let local = invocation.x;
    let sample = invocation.y;
    let output_x = local % OUTPUT_W;
    let output_y = local / OUTPUT_W;
{accumulators}

    for (var input_group = 0u; input_group < INPUT_GROUPS; input_group += 1u) {{
        var depthwise_value = depthwise_bias[input_group];
        for (var kernel_y = 0u; kernel_y < 3u; kernel_y += 1u) {{
            let input_y = i32(output_y * STRIDE_H + kernel_y) - 1;
            if (input_y < 0 || input_y >= i32(INPUT_H)) {{
                continue;
            }}
            for (var kernel_x = 0u; kernel_x < 3u; kernel_x += 1u) {{
                let input_x = i32(output_x + kernel_x) - 1;
                if (input_x < 0 || input_x >= i32(INPUT_W)) {{
                    continue;
                }}
                let source_index = sample * INPUT_H * INPUT_W * INPUT_GROUPS
                    + (u32(input_y) * INPUT_W + u32(input_x)) * INPUT_GROUPS
                    + input_group;
                let weight_index = (kernel_y * 3u + kernel_x) * INPUT_GROUPS
                    + input_group;
                depthwise_value = fma(
                    source[source_index],
                    depthwise_weights[weight_index],
                    depthwise_value,
                );
            }}
        }}
        depthwise_value = silu(depthwise_value);
{updates}
    }}
{stores}
}}
"#,
        input_h = params.input_h,
        input_w = params.input_w,
        input_groups = params.in_channels / 4,
        output_h = params.output_h,
        output_w = params.output_w,
        output_groups = output_groups,
        stride_h = params.stride_h,
    )
}

#[allow(dead_code)]
fn spatial_dense_conv_shader(params: ConvParams) -> String {
    let output_groups = params.out_channels / 4;
    let groups_per_invocation = output_groups;
    let blocks = 1;
    let accumulators = (0..groups_per_invocation)
        .map(|group| format!("    var sum{group} = bias[output_group_base + {group}u];"))
        .collect::<Vec<_>>()
        .join("\n");
    let updates = (0..groups_per_invocation)
        .map(|group| {
            format!(
                "                sum{group} = fma(\n                    vec4<f32>(source_value),\n                    weights[weight_base + output_group_base + {group}u],\n                    sum{group},\n                );"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let stores = (0..groups_per_invocation)
        .map(|group| {
            let value = if params.activation {
                format!("silu(sum{group})")
            } else {
                format!("sum{group}")
            };
            format!(
                "    destination[sample * OUTPUT_H * OUTPUT_W * OUT_GROUPS\n        + spatial * OUT_GROUPS + output_group_base + {group}u] = {value};"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let source_index = if params.source_nchw {
        r#"sample * INPUT_H * INPUT_W * IN_CHANNELS
                    + input_channel * INPUT_H * INPUT_W
                    + u32(input_y) * INPUT_W + u32(input_x)"#
    } else {
        r#"sample * INPUT_H * INPUT_W * IN_CHANNELS
                    + (u32(input_y) * INPUT_W + u32(input_x)) * IN_CHANNELS
                    + input_channel"#
    };
    format!(
        r#"
const INPUT_H: u32 = {input_h}u;
const INPUT_W: u32 = {input_w}u;
const IN_CHANNELS: u32 = {in_channels}u;
const OUTPUT_H: u32 = {output_h}u;
const OUTPUT_W: u32 = {output_w}u;
const OUT_GROUPS: u32 = {out_groups}u;
const BLOCK_GROUPS: u32 = {block_groups}u;
const BLOCKS: u32 = {blocks}u;
const KERNEL_H: u32 = {kernel_h}u;
const KERNEL_W: u32 = {kernel_w}u;
const STRIDE_H: u32 = {stride_h}u;
const STRIDE_W: u32 = {stride_w}u;
const PADDING_H: u32 = {padding_h}u;
const PADDING_W: u32 = {padding_w}u;
const DILATION_H: u32 = {dilation_h}u;
const DILATION_W: u32 = {dilation_w}u;

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> destination: array<vec4<f32>>;

fn fast_exp(value: f32) -> f32 {{
    let x = clamp(value, -80.0, 80.0);
    let scaled = x * 1.4426950408889634;
    let n = sign(scaled) * floor(abs(scaled) + 0.5);
    let r = fma(-n, 0.6931471805599453, x);
    var polynomial = 1.0 / 720.0;
    polynomial = fma(polynomial, r, 1.0 / 120.0);
    polynomial = fma(polynomial, r, 1.0 / 24.0);
    polynomial = fma(polynomial, r, 1.0 / 6.0);
    polynomial = fma(polynomial, r, 0.5);
    polynomial = fma(polynomial, r, 1.0);
    polynomial = fma(polynomial, r, 1.0);
    let exponent = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return polynomial * exponent;
}}

fn silu(value: vec4<f32>) -> vec4<f32> {{
    return vec4<f32>(
        value.x / (1.0 + fast_exp(-value.x)),
        value.y / (1.0 + fast_exp(-value.y)),
        value.z / (1.0 + fast_exp(-value.z)),
        value.w / (1.0 + fast_exp(-value.w)),
    );
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {{
    let local = invocation.x;
    let sample = invocation.y;
    let spatial = local / BLOCKS;
    let output_group_base = (local % BLOCKS) * BLOCK_GROUPS;
    let output_x = spatial % OUTPUT_W;
    let output_y = spatial / OUTPUT_W;
{accumulators}

    for (var kernel_y = 0u; kernel_y < KERNEL_H; kernel_y += 1u) {{
        let input_y = i32(output_y * STRIDE_H + kernel_y * DILATION_H)
            - i32(PADDING_H);
        if (input_y < 0 || input_y >= i32(INPUT_H)) {{
            continue;
        }}
        for (var kernel_x = 0u; kernel_x < KERNEL_W; kernel_x += 1u) {{
            let input_x = i32(output_x * STRIDE_W + kernel_x * DILATION_W)
                - i32(PADDING_W);
            if (input_x < 0 || input_x >= i32(INPUT_W)) {{
                continue;
            }}
            let kernel_base = kernel_y * KERNEL_W + kernel_x;
            for (var input_channel = 0u; input_channel < IN_CHANNELS; input_channel += 1u) {{
                let source_index = {source_index};
                let source_value = source[source_index];
                let weight_base = (kernel_base * IN_CHANNELS + input_channel) * OUT_GROUPS;
{updates}
            }}
        }}
    }}
{stores}
}}
"#,
        input_h = params.input_h,
        input_w = params.input_w,
        in_channels = params.in_channels,
        output_h = params.output_h,
        output_w = params.output_w,
        out_groups = output_groups,
        block_groups = groups_per_invocation,
        blocks = blocks,
        kernel_h = params.kernel[0],
        kernel_w = params.kernel[1],
        stride_h = params.stride[0],
        stride_w = params.stride[1],
        padding_h = params.padding[0],
        padding_w = params.padding[1],
        dilation_h = params.dilation[0],
        dilation_w = params.dilation[1],
    )
}

#[allow(dead_code)]
fn tiled_dense_conv_shader(params: ConvParams) -> String {
    let accumulators = (0..4)
        .map(|index| format!("    var sum{index} = bias[output_group];"))
        .collect::<Vec<_>>()
        .join("\n");
    let input_x = (0..4)
        .map(|index| {
            format!(
                "            let input_x{index} = i32((output_x + {index}u) * STRIDE_W\n                + kernel_x * DILATION_W) - i32(PADDING_W);"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let updates = (0..4)
        .map(|index| {
            let source_index = if params.source_nchw {
                format!(
                    "sample * INPUT_H * INPUT_W * IN_CHANNELS\n                        + input_channel * INPUT_H * INPUT_W\n                        + u32(input_y) * INPUT_W + u32(input_x{index})"
                )
            } else {
                format!(
                    "sample * INPUT_H * INPUT_W * IN_CHANNELS\n                        + (u32(input_y) * INPUT_W + u32(input_x{index})) * IN_CHANNELS\n                        + input_channel"
                )
            };
            format!(
                r#"                if (input_x{index} >= 0 && input_x{index} < i32(INPUT_W)) {{
                    let source_index{index} = {source_index};
                    sum{index} = fma(
                        vec4<f32>(source[source_index{index}]),
                        weight,
                        sum{index},
                    );
                }}"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let stores = (0..4)
        .map(|index| {
            let value = if params.activation {
                format!("silu(sum{index})")
            } else {
                format!("sum{index}")
            };
            format!(
                "    destination[sample * OUTPUT_H * OUTPUT_W * OUT_GROUPS\n        + (spatial_base + {index}u) * OUT_GROUPS + output_group] = {value};"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"
const INPUT_H: u32 = {input_h}u;
const INPUT_W: u32 = {input_w}u;
const IN_CHANNELS: u32 = {in_channels}u;
const OUTPUT_H: u32 = {output_h}u;
const OUTPUT_W: u32 = {output_w}u;
const OUT_GROUPS: u32 = {out_groups}u;
const KERNEL_H: u32 = {kernel_h}u;
const KERNEL_W: u32 = {kernel_w}u;
const STRIDE_H: u32 = {stride_h}u;
const STRIDE_W: u32 = {stride_w}u;
const PADDING_H: u32 = {padding_h}u;
const PADDING_W: u32 = {padding_w}u;
const DILATION_H: u32 = {dilation_h}u;
const DILATION_W: u32 = {dilation_w}u;

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> destination: array<vec4<f32>>;

fn fast_exp(value: f32) -> f32 {{
    let x = clamp(value, -80.0, 80.0);
    let scaled = x * 1.4426950408889634;
    let n = sign(scaled) * floor(abs(scaled) + 0.5);
    let r = fma(-n, 0.6931471805599453, x);
    var polynomial = 1.0 / 720.0;
    polynomial = fma(polynomial, r, 1.0 / 120.0);
    polynomial = fma(polynomial, r, 1.0 / 24.0);
    polynomial = fma(polynomial, r, 1.0 / 6.0);
    polynomial = fma(polynomial, r, 0.5);
    polynomial = fma(polynomial, r, 1.0);
    polynomial = fma(polynomial, r, 1.0);
    let exponent = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return polynomial * exponent;
}}

fn silu(value: vec4<f32>) -> vec4<f32> {{
    return vec4<f32>(
        value.x / (1.0 + fast_exp(-value.x)),
        value.y / (1.0 + fast_exp(-value.y)),
        value.z / (1.0 + fast_exp(-value.z)),
        value.w / (1.0 + fast_exp(-value.w)),
    );
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {{
    let sample = invocation.y;
    let local = invocation.x;
    let output_group = local % OUT_GROUPS;
    let spatial_base = (local / OUT_GROUPS) * 4u;
    let output_x = spatial_base % OUTPUT_W;
    let output_y = spatial_base / OUTPUT_W;
{accumulators}

    for (var kernel_y = 0u; kernel_y < KERNEL_H; kernel_y += 1u) {{
        let input_y = i32(output_y * STRIDE_H + kernel_y * DILATION_H)
            - i32(PADDING_H);
        if (input_y < 0 || input_y >= i32(INPUT_H)) {{
            continue;
        }}
        for (var kernel_x = 0u; kernel_x < KERNEL_W; kernel_x += 1u) {{
{input_x}
            let kernel_base = kernel_y * KERNEL_W + kernel_x;
            for (var input_channel = 0u; input_channel < IN_CHANNELS; input_channel += 1u) {{
                let weight_index = (kernel_base * IN_CHANNELS + input_channel)
                    * OUT_GROUPS + output_group;
                let weight = weights[weight_index];
{updates}
            }}
        }}
    }}
{stores}
}}
"#,
        input_h = params.input_h,
        input_w = params.input_w,
        in_channels = params.in_channels,
        output_h = params.output_h,
        output_w = params.output_w,
        out_groups = params.out_channels / 4,
        kernel_h = params.kernel[0],
        kernel_w = params.kernel[1],
        stride_h = params.stride[0],
        stride_w = params.stride[1],
        padding_h = params.padding[0],
        padding_w = params.padding[1],
        dilation_h = params.dilation[0],
        dilation_w = params.dilation[1],
    )
}

#[allow(dead_code)]
fn vector_conv_shader(params: ConvParams) -> String {
    if !params.depthwise {
        return tiled_dense_conv_shader(params);
    }
    let (source_type, accumulate) = if params.depthwise {
        (
            "array<vec4<f32>>",
            r#"
            let source_index = sample * INPUT_H * INPUT_W * OUT_GROUPS
                + (u32(input_y) * INPUT_W + u32(input_x)) * OUT_GROUPS
                + output_group;
            let weight_index = kernel_base * OUT_GROUPS + output_group;
            sum = fma(source[source_index], weights[weight_index], sum);
"#
            .to_owned(),
        )
    } else {
        let source_index = if params.source_nchw {
            r#"sample * INPUT_H * INPUT_W * IN_CHANNELS
                    + input_channel * INPUT_H * INPUT_W
                    + u32(input_y) * INPUT_W + u32(input_x)"#
        } else {
            r#"sample * INPUT_H * INPUT_W * IN_CHANNELS
                    + (u32(input_y) * INPUT_W + u32(input_x)) * IN_CHANNELS
                    + input_channel"#
        };
        (
            "array<f32>",
            format!(
                r#"
            for (var input_channel = 0u; input_channel < IN_CHANNELS; input_channel += 1u) {{
                let source_index = {source_index};
                let weight_index = (kernel_base * IN_CHANNELS + input_channel)
                    * OUT_GROUPS + output_group;
                sum = fma(
                    vec4<f32>(source[source_index]),
                    weights[weight_index],
                    sum,
                );
            }}
"#
            ),
        )
    };
    let activation = if params.activation {
        "silu(sum)"
    } else {
        "sum"
    };
    format!(
        r#"
const INPUT_H: u32 = {input_h}u;
const INPUT_W: u32 = {input_w}u;
const IN_CHANNELS: u32 = {in_channels}u;
const OUTPUT_H: u32 = {output_h}u;
const OUTPUT_W: u32 = {output_w}u;
const OUT_GROUPS: u32 = {out_groups}u;
const KERNEL_H: u32 = {kernel_h}u;
const KERNEL_W: u32 = {kernel_w}u;
const STRIDE_H: u32 = {stride_h}u;
const STRIDE_W: u32 = {stride_w}u;
const PADDING_H: u32 = {padding_h}u;
const PADDING_W: u32 = {padding_w}u;
const DILATION_H: u32 = {dilation_h}u;
const DILATION_W: u32 = {dilation_w}u;

@group(0) @binding(0) var<storage, read> source: {source_type};
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> destination: array<vec4<f32>>;

fn fast_exp(value: f32) -> f32 {{
    let x = clamp(value, -80.0, 80.0);
    let scaled = x * 1.4426950408889634;
    let n = sign(scaled) * floor(abs(scaled) + 0.5);
    let r = fma(-n, 0.6931471805599453, x);
    var polynomial = 1.0 / 720.0;
    polynomial = fma(polynomial, r, 1.0 / 120.0);
    polynomial = fma(polynomial, r, 1.0 / 24.0);
    polynomial = fma(polynomial, r, 1.0 / 6.0);
    polynomial = fma(polynomial, r, 0.5);
    polynomial = fma(polynomial, r, 1.0);
    polynomial = fma(polynomial, r, 1.0);
    let exponent = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return polynomial * exponent;
}}

fn silu(value: vec4<f32>) -> vec4<f32> {{
    return vec4<f32>(
        value.x / (1.0 + fast_exp(-value.x)),
        value.y / (1.0 + fast_exp(-value.y)),
        value.z / (1.0 + fast_exp(-value.z)),
        value.w / (1.0 + fast_exp(-value.w)),
    );
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {{
    let sample = invocation.y;
    let local = invocation.x;
    let output_group = local % OUT_GROUPS;
    let spatial = local / OUT_GROUPS;
    let output_x = spatial % OUTPUT_W;
    let output_y = spatial / OUTPUT_W;
    var sum = bias[output_group];

    for (var kernel_y = 0u; kernel_y < KERNEL_H; kernel_y += 1u) {{
        let input_y = i32(output_y * STRIDE_H + kernel_y * DILATION_H)
            - i32(PADDING_H);
        if (input_y < 0 || input_y >= i32(INPUT_H)) {{
            continue;
        }}
        for (var kernel_x = 0u; kernel_x < KERNEL_W; kernel_x += 1u) {{
            let input_x = i32(output_x * STRIDE_W + kernel_x * DILATION_W)
                - i32(PADDING_W);
            if (input_x < 0 || input_x >= i32(INPUT_W)) {{
                continue;
            }}
            let kernel_base = kernel_y * KERNEL_W + kernel_x;
{accumulate}
        }}
    }}
    destination[sample * OUTPUT_H * OUTPUT_W * OUT_GROUPS + local] = {activation};
}}
"#,
        input_h = params.input_h,
        input_w = params.input_w,
        in_channels = params.in_channels,
        output_h = params.output_h,
        output_w = params.output_w,
        out_groups = params.out_channels / 4,
        kernel_h = params.kernel[0],
        kernel_w = params.kernel[1],
        stride_h = params.stride[0],
        stride_w = params.stride[1],
        padding_h = params.padding[0],
        padding_w = params.padding[1],
        dilation_h = params.dilation[0],
        dilation_w = params.dilation[1],
    )
}

impl GpuLayer {
    fn new(
        context: &GpuContext,
        source: &hal::Buffer,
        target: &hal::Buffer,
        weights: &[f32],
        bias: &[f32],
        params: ConvParams,
    ) -> Result<Self> {
        let elements = params
            .output_h
            .checked_mul(params.output_w)
            .and_then(|value| value.checked_mul(params.out_channels))
            .ok_or_else(|| RtError::Inference("GPU layer output size overflow".into()))?;
        let weight_buffer = context.device.create_buffer_init(
            "subfast ROI weights",
            f32_bytes(weights),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let bias_buffer = context.device.create_buffer_init(
            "subfast ROI bias",
            f32_bytes(bias),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;

        let values = [
            params.input_h as u32,
            params.input_w as u32,
            params.in_channels as u32,
            params.output_h as u32,
            params.output_w as u32,
            params.out_channels as u32,
            params.kernel[0] as u32,
            params.kernel[1] as u32,
            params.stride[0] as u32,
            params.stride[1] as u32,
            params.padding[0] as u32,
            params.padding[1] as u32,
            params.dilation[0] as u32,
            params.dilation[1] as u32,
            if params.depthwise {
                1
            } else if params.source_nchw {
                2
            } else {
                0
            },
            u32::from(params.activation),
        ];
        let params_buffer = context.device.create_buffer_init(
            "subfast ROI convolution parameters",
            u32_bytes(&values),
            hal::BufferUses::UNIFORM,
        )?;
        let bind_group = sequential_bind_group(
            &context.device,
            &context.conv_pipeline,
            "subfast ROI convolution bindings",
            &[source, &weight_buffer, &bias_buffer, target, &params_buffer],
        )?;
        Ok(Self {
            pipeline: None,
            bind_group,
            workgroups_per_sample: elements.div_ceil(WORKGROUP_SIZE) as u32,
            _resources: vec![weight_buffer, bias_buffer, params_buffer],
        })
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    fn new_fused_separable(
        context: &GpuContext,
        source: &hal::Buffer,
        target: &hal::Buffer,
        depthwise_weights: &[f32],
        depthwise_bias: &[f32],
        pointwise_weights: &[f32],
        pointwise_bias: &[f32],
        params: SeparableParams,
    ) -> Result<Self> {
        if !params.in_channels.is_multiple_of(4)
            || !params.out_channels.is_multiple_of(4)
            || !(params.output_h * params.output_w).is_multiple_of(WORKGROUP_SIZE)
            || params.input_w != params.output_w
            || !matches!(params.stride_h, 1 | 2)
        {
            return Err(RtError::Inference(
                "unsupported fused separable convolution shape".into(),
            ));
        }
        let depthwise_weight_buffer = context.device.create_buffer_init(
            "subfast fused depthwise weights",
            f32_bytes(depthwise_weights),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let depthwise_bias_buffer = context.device.create_buffer_init(
            "subfast fused depthwise bias",
            f32_bytes(depthwise_bias),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let pointwise_weight_buffer = context.device.create_buffer_init(
            "subfast fused pointwise weights",
            f32_bytes(pointwise_weights),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let pointwise_bias_buffer = context.device.create_buffer_init(
            "subfast fused pointwise bias",
            f32_bytes(pointwise_bias),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let pipeline = create_pipeline(
            &context.device,
            "subfast fused separable convolution",
            &fused_separable_spatial_shader(params),
        )?;
        let bind_group = sequential_bind_group(
            &context.device,
            &pipeline,
            "subfast fused separable convolution bindings",
            &[
                source,
                &depthwise_weight_buffer,
                &depthwise_bias_buffer,
                &pointwise_weight_buffer,
                &pointwise_bias_buffer,
                target,
            ],
        )?;
        Ok(Self {
            pipeline: Some(pipeline),
            bind_group,
            workgroups_per_sample: (params.output_h * params.output_w / WORKGROUP_SIZE) as u32,
            _resources: vec![
                depthwise_weight_buffer,
                depthwise_bias_buffer,
                pointwise_weight_buffer,
                pointwise_bias_buffer,
            ],
        })
    }
}

struct PresenceGpu {
    input: hal::Buffer,
    _stage_a: hal::Buffer,
    _stage_b: hal::Buffer,
    output: hal::Buffer,
    layers: Vec<GpuLayer>,
    finish: GpuKernel,
}

impl PresenceGpu {
    fn compile(model: &Model, context: &GpuContext, max_batch: usize) -> Result<Self> {
        let shape = &model.input_info()[0].shape;
        let (input_h, input_w) = (shape[2], shape[3]);
        if input_h % 4 != 0 || input_w % 4 != 0 {
            return Err(invalid_gpu_model(
                "presence input height and width must be divisible by 4",
            ));
        }
        let (stage1_h, stage1_w) = (input_h / 2, input_w / 2);
        let (stage2_h, stage2_w) = (input_h / 4, input_w / 4);
        let input_elements = 3 * input_h * input_w;
        let stage_elements = (stage1_h * stage1_w * 16).max(stage2_h * stage2_w * 32);
        let input = storage_buffer(
            &context.device,
            "subfast presence input",
            max_batch,
            input_elements,
            hal::BufferUses::MAP_WRITE,
        )?;
        let stage_a = storage_buffer(
            &context.device,
            "subfast presence stage A",
            max_batch,
            stage_elements,
            hal::BufferUses::empty(),
        )?;
        let stage_b = storage_buffer(
            &context.device,
            "subfast presence stage B",
            max_batch,
            stage_elements,
            hal::BufferUses::empty(),
        )?;
        let output = storage_buffer(
            &context.device,
            "subfast presence output",
            max_batch,
            1,
            hal::BufferUses::MAP_READ,
        )?;

        let (conv1_weights, conv1_bias) = packed_dense(model, "backbone.0", 3, 16, [5, 5])?;
        let (conv2_weights, conv2_bias) = packed_dense(model, "backbone.2", 16, 32, [5, 5])?;
        let (head_weights, head_bias) = packed_dense(model, "presence_head.local", 32, 1, [3, 3])?;
        let layers = vec![
            GpuLayer::new(
                context,
                &input,
                &stage_a,
                &conv1_weights,
                &conv1_bias,
                ConvParams::dense(
                    input_h,
                    input_w,
                    3,
                    stage1_h,
                    stage1_w,
                    16,
                    [5, 5],
                    [2, 2],
                    [2, 2],
                    [1, 1],
                    true,
                )
                .with_nchw_source(),
            )?,
            GpuLayer::new(
                context,
                &stage_a,
                &stage_b,
                &conv2_weights,
                &conv2_bias,
                ConvParams::dense(
                    stage1_h,
                    stage1_w,
                    16,
                    stage2_h,
                    stage2_w,
                    32,
                    [5, 5],
                    [2, 2],
                    [2, 2],
                    [1, 1],
                    true,
                ),
            )?,
        ];
        let log_scale = scalar_parameter(model, "presence_head.evidence_pool.log_scale")?;
        let head_weight_buffer = context.device.create_buffer_init(
            "subfast presence head weights",
            f32_bytes(&head_weights),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let finish_params = context.device.create_buffer_init(
            "subfast presence finish parameters",
            f32_bytes(&[
                head_bias[0],
                softplus(log_scale) + 0.5,
                scalar_parameter(model, "presence_head.evidence_pool.bias")?,
                0.0,
            ]),
            hal::BufferUses::UNIFORM,
        )?;
        let finish_pipeline = create_pipeline(
            &context.device,
            "subfast presence finish",
            &presence_finish_shader(input_h, input_w, stage2_h, stage2_w),
        )?;
        let finish_bind_group = sequential_bind_group(
            &context.device,
            &finish_pipeline,
            "subfast presence head and finish bindings",
            &[
                &input,
                &stage_b,
                &head_weight_buffer,
                &output,
                &finish_params,
            ],
        )?;
        Ok(Self {
            input,
            _stage_a: stage_a,
            _stage_b: stage_b,
            output,
            layers,
            finish: GpuKernel {
                pipeline: finish_pipeline,
                bind_group: finish_bind_group,
                workgroups_per_sample: 1,
                _resources: vec![head_weight_buffer, finish_params],
            },
        })
    }

    fn run(
        &mut self,
        context: &mut GpuContext,
        inputs: &[Tensor],
        batch_size: usize,
        output: &mut [f32],
    ) -> Result<()> {
        context.write_input(&self.input, &inputs[0].data)?;
        context.execute_presence(&self.layers, &self.finish, &self.output, batch_size, output)
    }
}

impl ConvParams {
    #[allow(clippy::too_many_arguments)]
    const fn dense(
        input_h: usize,
        input_w: usize,
        in_channels: usize,
        output_h: usize,
        output_w: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        activation: bool,
    ) -> Self {
        Self {
            input_h,
            input_w,
            in_channels,
            output_h,
            output_w,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            depthwise: false,
            source_nchw: false,
            activation,
        }
    }

    const fn with_nchw_source(mut self) -> Self {
        self.source_nchw = true;
        self
    }

    const fn depthwise(
        input_h: usize,
        input_w: usize,
        channels: usize,
        output_h: usize,
        output_w: usize,
        stride: [usize; 2],
    ) -> Self {
        Self {
            input_h,
            input_w,
            in_channels: channels,
            output_h,
            output_w,
            out_channels: channels,
            kernel: [3, 3],
            stride,
            padding: [1, 1],
            dilation: [1, 1],
            depthwise: true,
            source_nchw: false,
            activation: true,
        }
    }
}

struct PairGpu {
    left_input: hal::Buffer,
    right_input: hal::Buffer,
    _features_buffer: hal::Buffer,
    _stage_a: hal::Buffer,
    _stage_b: hal::Buffer,
    output: hal::Buffer,
    features: GpuKernel,
    layers: Vec<GpuLayer>,
    tail: GpuKernel,
}

impl PairGpu {
    fn compile(model: &Model, context: &GpuContext, max_batch: usize) -> Result<Self> {
        let shape = &model.input_info()[0].shape;
        let (input_h, input_w) = (shape[2], shape[3]);
        let (stem_h, stem_w) = (input_h.div_ceil(2), input_w.div_ceil(2));
        let (down_h, down_w) = (stem_h.div_ceil(2), stem_w.div_ceil(2));
        let (final_h, final_w) = (down_h.div_ceil(2), down_w);
        let input_elements = input_h * input_w * 3;
        let stage_elements = (stem_h * stem_w * 16)
            .max(down_h * down_w * 32)
            .max(final_h * final_w * 48);
        let left_input = storage_buffer(
            &context.device,
            "subfast pair left input",
            max_batch,
            input_elements,
            hal::BufferUses::MAP_WRITE,
        )?;
        let right_input = storage_buffer(
            &context.device,
            "subfast pair right input",
            max_batch,
            input_elements,
            hal::BufferUses::MAP_WRITE,
        )?;
        let features_buffer = storage_buffer(
            &context.device,
            "subfast pair features",
            max_batch,
            input_elements,
            hal::BufferUses::empty(),
        )?;
        let stage_a = storage_buffer(
            &context.device,
            "subfast pair stage A",
            max_batch,
            stage_elements,
            hal::BufferUses::empty(),
        )?;
        let stage_b = storage_buffer(
            &context.device,
            "subfast pair stage B",
            max_batch,
            stage_elements,
            hal::BufferUses::empty(),
        )?;
        let output = storage_buffer(
            &context.device,
            "subfast pair output",
            max_batch,
            1,
            hal::BufferUses::MAP_READ,
        )?;

        let (stem_weights, stem_b) = packed_dense(model, "stem.0", 3, 16, [3, 3])?;
        let (down_weights, down_b) = packed_dense(model, "downsample.0", 16, 32, [3, 3])?;
        let (detail_dw_w, detail_dw_b) = packed_depthwise(model, "detail.0", 32)?;
        let (detail_pw_w, detail_pw_b) = packed_dense(model, "detail.3", 32, 32, [1, 1])?;
        let (vertical_dw_w, vertical_dw_b) = packed_depthwise(model, "vertical_downsample.0", 32)?;
        let (vertical_pw_w, vertical_pw_b) =
            packed_dense(model, "vertical_downsample.3", 32, 48, [1, 1])?;
        let (context_dw_w, context_dw_b) = packed_depthwise(model, "context.0", 48)?;
        let (context_pw_w, context_pw_b) = packed_dense(model, "context.3", 48, 48, [1, 1])?;

        let feature_pipeline = create_pipeline(
            &context.device,
            "subfast pair features",
            &pair_feature_shader(input_h, input_w),
        )?;
        let features = GpuKernel {
            pipeline: feature_pipeline.clone(),
            bind_group: sequential_bind_group(
                &context.device,
                &feature_pipeline,
                "subfast pair feature bindings",
                &[&left_input, &right_input, &features_buffer],
            )?,
            workgroups_per_sample: (input_h * input_w).div_ceil(128) as u32,
            _resources: Vec::new(),
        };

        let layers = vec![
            GpuLayer::new(
                context,
                &features_buffer,
                &stage_a,
                &stem_weights,
                &stem_b,
                ConvParams::dense(
                    input_h,
                    input_w,
                    3,
                    stem_h,
                    stem_w,
                    16,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    [1, 1],
                    true,
                ),
            )?,
            GpuLayer::new(
                context,
                &stage_a,
                &stage_b,
                &down_weights,
                &down_b,
                ConvParams::dense(
                    stem_h,
                    stem_w,
                    16,
                    down_h,
                    down_w,
                    32,
                    [3, 3],
                    [2, 2],
                    [1, 1],
                    [1, 1],
                    true,
                ),
            )?,
            GpuLayer::new(
                context,
                &stage_b,
                &stage_a,
                &detail_dw_w,
                &detail_dw_b,
                ConvParams::depthwise(down_h, down_w, 32, down_h, down_w, [1, 1]),
            )?,
            GpuLayer::new(
                context,
                &stage_a,
                &stage_b,
                &detail_pw_w,
                &detail_pw_b,
                ConvParams::dense(
                    down_h,
                    down_w,
                    32,
                    down_h,
                    down_w,
                    32,
                    [1, 1],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    true,
                ),
            )?,
            GpuLayer::new(
                context,
                &stage_b,
                &stage_a,
                &vertical_dw_w,
                &vertical_dw_b,
                ConvParams::depthwise(down_h, down_w, 32, final_h, final_w, [2, 1]),
            )?,
            GpuLayer::new(
                context,
                &stage_a,
                &stage_b,
                &vertical_pw_w,
                &vertical_pw_b,
                ConvParams::dense(
                    final_h,
                    final_w,
                    32,
                    final_h,
                    final_w,
                    48,
                    [1, 1],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    true,
                ),
            )?,
            GpuLayer::new(
                context,
                &stage_b,
                &stage_a,
                &context_dw_w,
                &context_dw_b,
                ConvParams::depthwise(final_h, final_w, 48, final_h, final_w, [1, 1]),
            )?,
            GpuLayer::new(
                context,
                &stage_a,
                &stage_b,
                &context_pw_w,
                &context_pw_b,
                ConvParams::dense(
                    final_h,
                    final_w,
                    48,
                    final_h,
                    final_w,
                    48,
                    [1, 1],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    true,
                ),
            )?,
        ];

        let mask_weight = context.device.create_buffer_init(
            "subfast pair mask weight",
            f32_bytes(&parameter(model, "text_mask.weight", &[1, 48, 1, 1])?),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let classifier1_weight = context.device.create_buffer_init(
            "subfast pair classifier 1 weight",
            f32_bytes(&parameter(model, "classifier.0.weight", &[32, 144])?),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let classifier1_bias = context.device.create_buffer_init(
            "subfast pair classifier 1 bias",
            f32_bytes(&parameter(model, "classifier.0.bias", &[32])?),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let classifier2_weight = context.device.create_buffer_init(
            "subfast pair classifier 2 weight",
            f32_bytes(&parameter(model, "classifier.2.weight", &[1, 32])?),
            hal::BufferUses::STORAGE_READ_ONLY,
        )?;
        let tail_params = context.device.create_buffer_init(
            "subfast pair tail parameters",
            f32_bytes(&[
                scalar_parameter(model, "text_mask.bias")?,
                scalar_parameter(model, "classifier.2.bias")?,
                0.0,
                0.0,
            ]),
            hal::BufferUses::UNIFORM,
        )?;
        let tail_pipeline = create_pipeline(
            &context.device,
            "subfast pair tail",
            &pair_tail_shader(final_h, final_w),
        )?;
        let tail = GpuKernel {
            pipeline: tail_pipeline.clone(),
            bind_group: sequential_bind_group(
                &context.device,
                &tail_pipeline,
                "subfast pair tail bindings",
                &[
                    &stage_b,
                    &mask_weight,
                    &classifier1_weight,
                    &classifier1_bias,
                    &classifier2_weight,
                    &output,
                    &tail_params,
                ],
            )?,
            workgroups_per_sample: 1,
            _resources: vec![
                mask_weight,
                classifier1_weight,
                classifier1_bias,
                classifier2_weight,
                tail_params,
            ],
        };
        Ok(Self {
            left_input,
            right_input,
            _features_buffer: features_buffer,
            _stage_a: stage_a,
            _stage_b: stage_b,
            output,
            features,
            layers,
            tail,
        })
    }

    fn run(
        &mut self,
        context: &mut GpuContext,
        inputs: &[Tensor],
        batch_size: usize,
        output: &mut [f32],
    ) -> Result<()> {
        context.write_input(&self.left_input, &inputs[0].data)?;
        context.write_input(&self.right_input, &inputs[1].data)?;
        context.execute_pair(
            &self.features,
            &self.layers,
            &self.tail,
            &self.output,
            batch_size,
            output,
        )
    }
}

fn sequential_bind_group(
    device: &hal::Device,
    pipeline: &hal::Pipeline,
    label: &'static str,
    buffers: &[&hal::Buffer],
) -> Result<hal::BindGroup> {
    device.create_bind_group(pipeline, label, buffers)
}

fn validate_runtime_inputs(inputs: &[Tensor], shapes: &[Vec<usize>]) -> Result<()> {
    if inputs.len() != shapes.len() {
        return Err(RtError::InvalidTensor {
            tensor: "<gpu-inputs>".into(),
            message: format!("expected {} inputs, got {}", shapes.len(), inputs.len()),
        });
    }
    for (index, (input, shape)) in inputs.iter().zip(shapes).enumerate() {
        if input.dtype != DType::Float32
            || input.shape != *shape
            || DType::Float32.byte_len_for_shape(shape) != Some(input.data.len())
        {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: format!(
                    "expected float32 shape {shape:?}, got {} {:?}",
                    input.dtype, input.shape
                ),
            });
        }
    }
    Ok(())
}

fn validate_packed_inputs(
    inputs: &[Tensor],
    shapes: &[Vec<usize>],
    max_batch_size: usize,
) -> Result<usize> {
    if inputs.len() != shapes.len() {
        return Err(RtError::InvalidTensor {
            tensor: "<gpu-batch-inputs>".into(),
            message: format!("expected {} inputs, got {}", shapes.len(), inputs.len()),
        });
    }
    let mut batch_size = None;
    for (index, (input, model_shape)) in inputs.iter().zip(shapes).enumerate() {
        let Some(size) = input.shape.first().copied() else {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: "packed GPU input must have a batch dimension".into(),
            });
        };
        let shape_matches =
            input.shape.len() == model_shape.len() && input.shape[1..] == model_shape[1..];
        if input.dtype != DType::Float32
            || !shape_matches
            || DType::Float32.byte_len_for_shape(&input.shape) != Some(input.data.len())
        {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: format!(
                    "expected packed float32 shape [batch, 3, H, W] matching the model, got {} {:?}",
                    input.dtype, input.shape
                ),
            });
        }
        if size > max_batch_size {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: format!("batch size {size} exceeds compiled GPU maximum {max_batch_size}"),
            });
        }
        if batch_size
            .replace(size)
            .is_some_and(|previous| previous != size)
        {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: "all packed GPU inputs must use the same batch size".into(),
            });
        }
    }
    Ok(batch_size.unwrap_or(0))
}

fn packed_dense(
    model: &Model,
    target: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: [usize; 2],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let source = parameter(
        model,
        &format!("{target}.weight"),
        &[out_channels, in_channels, kernel[0], kernel[1]],
    )?;
    let k = in_channels * kernel[0] * kernel[1];
    let mut packed = vec![0.0; k * out_channels];
    for output_channel in 0..out_channels {
        for input_channel in 0..in_channels {
            for kernel_y in 0..kernel[0] {
                for kernel_x in 0..kernel[1] {
                    let source_index =
                        (((output_channel * in_channels + input_channel) * kernel[0] + kernel_y)
                            * kernel[1])
                            + kernel_x;
                    let target_index = (((kernel_y * kernel[1] + kernel_x) * in_channels
                        + input_channel)
                        * out_channels)
                        + output_channel;
                    packed[target_index] = source[source_index];
                }
            }
        }
    }
    let bias = parameter(model, &format!("{target}.bias"), &[out_channels])?;
    Ok((packed, bias))
}

fn packed_depthwise(model: &Model, target: &str, channels: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let source = parameter(model, &format!("{target}.weight"), &[channels, 1, 3, 3])?;
    let mut packed = vec![0.0; source.len()];
    for channel in 0..channels {
        for kernel_y in 0..3 {
            for kernel_x in 0..3 {
                packed[(kernel_y * 3 + kernel_x) * channels + channel] =
                    source[(channel * 3 + kernel_y) * 3 + kernel_x];
            }
        }
    }
    let bias = parameter(model, &format!("{target}.bias"), &[channels])?;
    Ok((packed, bias))
}

fn parameter(model: &Model, target: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = model
        .tensors
        .iter()
        .find(|info| {
            matches!(info.kind, TensorKind::Parameter | TensorKind::Buffer)
                && info.target.as_deref() == Some(target)
        })
        .ok_or_else(|| invalid_gpu_model(format!("missing parameter {target}")))?;
    validate_parameter(info, target, shape)?;
    let weight = info
        .weight
        .ok_or_else(|| invalid_gpu_model(format!("parameter {target} has no data")))?;
    let bytes = model.weights.slice(weight.offset, weight.byte_length)?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(invalid_gpu_model(format!(
            "parameter {target} byte length is not float32-aligned"
        )));
    }
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
        .ok_or_else(|| invalid_gpu_model(format!("missing parameter {target}")))?;
    if !info.shape.is_empty() && info.shape != [1] {
        return Err(RtError::InvalidTensor {
            tensor: target.into(),
            message: format!("expected scalar shape [] or [1], got {:?}", info.shape),
        });
    }
    parameter(model, target, &info.shape)?
        .first()
        .copied()
        .ok_or_else(|| invalid_gpu_model(format!("scalar parameter {target} is empty")))
}

fn validate_parameter(info: &TensorInfo, target: &str, shape: &[usize]) -> Result<()> {
    if info.dtype != DType::Float32 || info.shape != shape {
        return Err(RtError::InvalidTensor {
            tensor: target.into(),
            message: format!(
                "expected float32 shape {shape:?}, got {} {:?}",
                info.dtype, info.shape
            ),
        });
    }
    Ok(())
}

fn invalid_gpu_model(message: impl Into<String>) -> RtError {
    RtError::InvalidModel(format!("GPU ROI backend: {}", message.into()))
}

fn storage_buffer(
    device: &hal::Device,
    label: &'static str,
    batch_size: usize,
    elements_per_sample: usize,
    extra_usage: hal::BufferUses,
) -> Result<hal::Buffer> {
    let size = checked_bytes(batch_size, elements_per_sample)?;
    let limits = device.limits();
    if size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size {
        return Err(RtError::Inference(format!(
            "GPU storage buffer {label} requires {size} bytes, device limit is {}",
            limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size)
        )));
    }
    device.create_buffer(
        label,
        size,
        hal::BufferUses::STORAGE_READ_ONLY | hal::BufferUses::STORAGE_READ_WRITE | extra_usage,
    )
}

fn checked_bytes(batch_size: usize, elements_per_sample: usize) -> Result<u64> {
    batch_size
        .checked_mul(elements_per_sample)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| RtError::Inference("GPU buffer size overflow".into()))
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: every `f32` is plain data, the returned slice has the same
    // lifetime, and its byte length exactly covers the initialized input.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    // SAFETY: every `u32` is plain data and the byte view does not outlive it.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
