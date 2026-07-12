//! Pure-Rust, fixed-shape CPU backend for the exported ROI models.
//!
//! The graph is validated and fused at compile time. Runtime tensors stay in the
//! public NCHW byte representation, while weights and scratch buffers use typed
//! NHWC storage. This keeps the public format portable without ever assuming
//! that a `Vec<u8>` is aligned for `f32` or SIMD loads.

use crate::backend::{Backend, Executable};
use crate::error::{Result, RtError};
use crate::ir::op::Op;
use crate::ir::tensor::{DType, TensorInfo, TensorKind};
use crate::model::UnifiedModel;
use crate::tensor::Tensor;

const INPUT_SHAPE: [usize; 4] = [1, 3, 32, 256];
const INPUT_H: usize = 32;
const INPUT_W: usize = 256;

const PRESENCE_OPS: [&str; 28] = [
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.abs.default",
    "aten.amax.default",
    "aten.gt.Scalar",
    "aten.to.dtype",
    "aten.avg_pool2d.default",
    "aten.gt.Scalar",
    "aten.to.dtype",
    "aten.gt.Scalar",
    "aten.to.dtype",
    "aten.avg_pool2d.default",
    "aten.mul.Tensor",
    "aten.avg_pool2d.default",
    "aten.clamp_min.default",
    "aten.div.Tensor",
    "aten.gt.Scalar",
    "aten.bitwise_not.default",
    "aten.masked_fill.Scalar",
    "aten.amax.default",
    "aten.clamp_min.default",
    "aten.softplus.default",
    "aten.add.Tensor",
    "aten.mul.Tensor",
    "aten.add.Tensor",
];

const PAIR_OPS: [&str; 38] = [
    "aten.sub.Tensor",
    "aten.avg_pool2d.default",
    "aten.add.Tensor",
    "aten.mul.Tensor",
    "aten.abs.default",
    "aten.sub.Tensor",
    "aten.abs.default",
    "aten.cat.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.silu_.default",
    "aten.conv2d.default",
    "aten.sigmoid.default",
    "aten.mul.Tensor",
    "aten.add.Tensor",
    "aten.mul.Tensor",
    "aten.mean.dim",
    "aten.amax.default",
    "aten.mean.dim",
    "aten.amax.default",
    "aten.cat.default",
    "aten.linear.default",
    "aten.silu_.default",
    "aten.linear.default",
    "aten.squeeze.dim",
];

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

pub struct CpuExecutable {
    input_shapes: Vec<Vec<usize>>,
    model: CompiledRoiModel,
}

enum CompiledRoiModel {
    Presence(Box<PresenceModel>),
    Pair(Box<PairModel>),
}

impl Backend for CpuBackend {
    type Executable = CpuExecutable;

    fn compile(&self, model: &UnifiedModel) -> Result<Self::Executable> {
        validate_model_io(model)?;
        let compiled = match model.inputs.len() {
            1 => CompiledRoiModel::Presence(Box::new(PresenceModel::compile(model)?)),
            2 => CompiledRoiModel::Pair(Box::new(PairModel::compile(model)?)),
            count => {
                return Err(RtError::Inference(format!(
                    "CPU ROI backend expects one or two inputs, got {count}"
                )));
            }
        };
        Ok(CpuExecutable {
            input_shapes: model
                .input_info()
                .into_iter()
                .map(|info| info.shape.clone())
                .collect(),
            model: compiled,
        })
    }
}

impl CpuExecutable {
    pub fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }

    /// Execute the scalar-output ROI graph without allocating an output tensor.
    ///
    /// All activation arenas are owned by the executable and reused between
    /// calls. The executable is intentionally `&mut self`, which also prevents
    /// concurrent access to the same scratch storage.
    pub fn run_scalar(&mut self, inputs: &[Tensor]) -> Result<f32> {
        validate_runtime_inputs(inputs, &self.input_shapes)?;
        match &mut self.model {
            CompiledRoiModel::Presence(model) => model.run(&inputs[0]),
            CompiledRoiModel::Pair(model) => model.run(&inputs[0], &inputs[1]),
        }
    }

    pub fn kernel_name(&self) -> &'static str {
        match &self.model {
            CompiledRoiModel::Presence(model) => model.kernel.name(),
            CompiledRoiModel::Pair(model) => model.kernel.name(),
        }
    }
}

impl Executable for CpuExecutable {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let value = self.run_scalar(inputs)?;
        Ok(vec![Tensor::from_f32(vec![1], vec![value])?])
    }
}

fn validate_model_io(model: &UnifiedModel) -> Result<()> {
    if model.outputs.len() != 1 {
        return Err(RtError::Inference(format!(
            "CPU ROI backend expects one output, got {}",
            model.outputs.len()
        )));
    }
    for info in model.input_info() {
        if info.dtype != DType::Float32 || info.shape != INPUT_SHAPE {
            return Err(RtError::InvalidTensor {
                tensor: info.name.clone(),
                message: format!(
                    "CPU ROI backend requires float32 shape {INPUT_SHAPE:?}, got {} {:?}",
                    info.dtype, info.shape
                ),
            });
        }
    }
    let output = model.output_info()[0];
    if output.dtype != DType::Float32 || output.shape != [1] {
        return Err(RtError::InvalidTensor {
            tensor: output.name.clone(),
            message: format!(
                "CPU ROI backend requires one float32 scalar output, got {} {:?}",
                output.dtype, output.shape
            ),
        });
    }
    Ok(())
}

fn validate_runtime_inputs(inputs: &[Tensor], shapes: &[Vec<usize>]) -> Result<()> {
    if inputs.len() != shapes.len() {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-inputs>".into(),
            message: format!("expected {} inputs, got {}", shapes.len(), inputs.len()),
        });
    }
    for (index, (input, shape)) in inputs.iter().zip(shapes).enumerate() {
        if input.dtype != DType::Float32 || input.shape != *shape {
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

#[derive(Clone, Copy)]
enum KernelKind {
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Scalar,
}

impl KernelKind {
    fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Self::Avx2;
            }
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }

    fn name(self) -> &'static str {
        match self {
            #[cfg(target_arch = "aarch64")]
            Self::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => "avx2+fma",
            Self::Scalar => "scalar",
        }
    }
}

struct DenseConv {
    in_channels: usize,
    out_channels: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    /// K-major weights: `[ky][kx][ic][oc]`.
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl DenseConv {
    #[allow(clippy::too_many_arguments)]
    fn compile(
        model: &UnifiedModel,
        target: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let weight_target = format!("{target}.weight");
        let bias_target = format!("{target}.bias");
        let source = parameter(
            model,
            &weight_target,
            &[out_channels, in_channels, kernel[0], kernel[1]],
        )?;
        let bias = parameter(model, &bias_target, &[out_channels])?;
        let mut weights = vec![0.0; source.len()];
        for oc in 0..out_channels {
            for ic in 0..in_channels {
                for ky in 0..kernel[0] {
                    for kx in 0..kernel[1] {
                        let src = (((oc * in_channels + ic) * kernel[0] + ky) * kernel[1]) + kx;
                        let dst = (((ky * kernel[1] + kx) * in_channels + ic) * out_channels) + oc;
                        weights[dst] = source[src];
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
            weights,
            bias,
        })
    }

    fn output_shape(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        let effective_h = self.dilation[0]
            .checked_mul(self.kernel[0].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_cpu_graph("convolution height overflow"))?;
        let effective_w = self.dilation[1]
            .checked_mul(self.kernel[1].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_cpu_graph("convolution width overflow"))?;
        let padded_h = input_h
            .checked_add(self.padding[0] * 2)
            .ok_or_else(|| invalid_cpu_graph("padded height overflow"))?;
        let padded_w = input_w
            .checked_add(self.padding[1] * 2)
            .ok_or_else(|| invalid_cpu_graph("padded width overflow"))?;
        if padded_h < effective_h || padded_w < effective_w {
            return Err(invalid_cpu_graph("convolution kernel exceeds input"));
        }
        Ok((
            (padded_h - effective_h) / self.stride[0] + 1,
            (padded_w - effective_w) / self.stride[1] + 1,
        ))
    }
}

struct DepthwiseConv {
    channels: usize,
    stride: [usize; 2],
    /// K-major weights: `[ky][kx][channel]`, fixed 3x3.
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl DepthwiseConv {
    fn compile(
        model: &UnifiedModel,
        target: &str,
        channels: usize,
        stride: [usize; 2],
    ) -> Result<Self> {
        let source = parameter(model, &format!("{target}.weight"), &[channels, 1, 3, 3])?;
        let bias = parameter(model, &format!("{target}.bias"), &[channels])?;
        let mut weights = vec![0.0; source.len()];
        for channel in 0..channels {
            for ky in 0..3 {
                for kx in 0..3 {
                    weights[(ky * 3 + kx) * channels + channel] =
                        source[(channel * 3 + ky) * 3 + kx];
                }
            }
        }
        Ok(Self {
            channels,
            stride,
            weights,
            bias,
        })
    }
}

struct Linear {
    in_features: usize,
    out_features: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn compile(
        model: &UnifiedModel,
        target: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        Ok(Self {
            in_features,
            out_features,
            weights: parameter(
                model,
                &format!("{target}.weight"),
                &[out_features, in_features],
            )?,
            bias: parameter(model, &format!("{target}.bias"), &[out_features])?,
        })
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct DenseJob {
    conv: usize,
    input: usize,
    input_len: usize,
    input_h: usize,
    input_w: usize,
    output: usize,
    output_len: usize,
    output_h: usize,
    output_w: usize,
}

#[cfg(target_arch = "aarch64")]
struct PoolShared {
    job: std::cell::UnsafeCell<DenseJob>,
    generation: std::sync::atomic::AtomicU64,
    completed: std::sync::atomic::AtomicUsize,
    failed: std::sync::atomic::AtomicBool,
    shutdown: std::sync::atomic::AtomicBool,
    sleep_lock: std::sync::Mutex<()>,
    wake: std::sync::Condvar,
}

// SAFETY: the sole dispatcher publishes `job` with a release increment of
// `generation`, and it never overwrites the slot before all readers complete.
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for PoolShared {}

/// Three persistent helpers plus the caller. No thread is created on the hot path.
#[cfg(target_arch = "aarch64")]
struct PresencePool {
    shared: std::sync::Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    total_threads: usize,
}

#[cfg(target_arch = "aarch64")]
impl PresencePool {
    fn new() -> Result<Self> {
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let total_threads = available.clamp(1, 4);
        let shared = std::sync::Arc::new(PoolShared {
            job: std::cell::UnsafeCell::new(DenseJob {
                conv: 0,
                input: 0,
                input_len: 0,
                input_h: 0,
                input_w: 0,
                output: 0,
                output_len: 0,
                output_h: 0,
                output_w: 0,
            }),
            generation: std::sync::atomic::AtomicU64::new(0),
            completed: std::sync::atomic::AtomicUsize::new(0),
            failed: std::sync::atomic::AtomicBool::new(false),
            shutdown: std::sync::atomic::AtomicBool::new(false),
            sleep_lock: std::sync::Mutex::new(()),
            wake: std::sync::Condvar::new(),
        });
        let mut threads: Vec<std::thread::JoinHandle<()>> =
            Vec::with_capacity(total_threads.saturating_sub(1));
        for worker_index in 1..total_threads {
            let worker_shared = std::sync::Arc::clone(&shared);
            let handle = match std::thread::Builder::new()
                .name(format!("subfast-cpu-{worker_index}"))
                .spawn(move || presence_worker(worker_shared, worker_index, total_threads))
            {
                Ok(handle) => handle,
                Err(error) => {
                    let guard = shared
                        .sleep_lock
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    shared
                        .shutdown
                        .store(true, std::sync::atomic::Ordering::Release);
                    shared
                        .generation
                        .fetch_add(1, std::sync::atomic::Ordering::Release);
                    shared.wake.notify_all();
                    drop(guard);
                    for thread in threads {
                        let _ = thread.join();
                    }
                    return Err(RtError::Inference(format!(
                        "failed to create CPU inference worker: {error}"
                    )));
                }
            };
            threads.push(handle);
        }
        Ok(Self {
            shared,
            threads,
            total_threads,
        })
    }

    fn dispatch(
        &self,
        conv: &DenseConv,
        input: &[f32],
        input_h: usize,
        input_w: usize,
        output: &mut [f32],
    ) -> Result<()> {
        if self.total_threads == 1 {
            dense_silu(KernelKind::Neon, conv, input, input_h, input_w, output);
            return Ok(());
        }
        let (output_h, output_w) = conv.output_shape(input_h, input_w)?;
        let job = DenseJob {
            conv: conv as *const DenseConv as usize,
            input: input.as_ptr() as usize,
            input_len: input.len(),
            input_h,
            input_w,
            output: output.as_mut_ptr() as usize,
            output_len: output.len(),
            output_h,
            output_w,
        };
        self.shared
            .failed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.shared
            .completed
            .store(0, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: `&mut CpuExecutable` serializes dispatches and every previous
        // generation completed before its dispatch returned.
        unsafe {
            *self.shared.job.get() = job;
        }
        let wake_guard = self.shared.sleep_lock.lock().map_err(|_| {
            RtError::Inference("CPU inference worker sleep lock is poisoned".into())
        })?;
        self.shared
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.shared.wake.notify_all();
        drop(wake_guard);

        // SAFETY: `job` points to live buffers borrowed for the whole dispatch.
        // Each participant receives a non-overlapping output element range.
        unsafe {
            execute_dense_job(job, 0, self.total_threads);
        }

        while self
            .shared
            .completed
            .load(std::sync::atomic::Ordering::Acquire)
            != self.threads.len()
        {
            std::hint::spin_loop();
        }
        if self
            .shared
            .failed
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(RtError::Inference(
                "a CPU inference worker failed while executing a SIMD kernel".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
impl Drop for PresencePool {
    fn drop(&mut self) {
        let guard = self
            .shared
            .sleep_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.shared
            .shutdown
            .store(true, std::sync::atomic::Ordering::Release);
        self.shared
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.shared.wake.notify_all();
        drop(guard);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn presence_worker(shared: std::sync::Arc<PoolShared>, worker_index: usize, total_threads: usize) {
    let mut observed_generation = 0u64;
    loop {
        let mut spins = 0usize;
        let generation = loop {
            let generation = shared.generation.load(std::sync::atomic::Ordering::Acquire);
            if generation != observed_generation {
                break generation;
            }
            if spins < 262_144 {
                spins += 1;
                std::hint::spin_loop();
                continue;
            }
            let guard = match shared.sleep_lock.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            if shared.generation.load(std::sync::atomic::Ordering::Acquire) == observed_generation
                && !shared.shutdown.load(std::sync::atomic::Ordering::Acquire)
                && shared.wake.wait(guard).is_err()
            {
                return;
            }
        };
        if shared.shutdown.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        // SAFETY: the acquire generation load observes the published job, which
        // remains unchanged until every worker increments `completed`.
        let job = unsafe { *shared.job.get() };
        let completed = std::panic::catch_unwind(|| {
            // SAFETY: the dispatcher keeps all pointed-to storage alive and gives
            // this worker a disjoint output range before waiting for completion.
            unsafe {
                execute_dense_job(job, worker_index, total_threads);
            }
        })
        .is_ok();
        if !completed {
            shared
                .failed
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        shared
            .completed
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        observed_generation = generation;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn execute_dense_job(job: DenseJob, worker_index: usize, total_threads: usize) {
    let x_start = job.output_w * worker_index / total_threads;
    let x_end = job.output_w * (worker_index + 1) / total_threads;
    if x_start == x_end {
        return;
    }
    unsafe {
        let conv = &*(job.conv as *const DenseConv);
        let input = std::slice::from_raw_parts(job.input as *const f32, job.input_len);
        debug_assert_eq!(
            job.output_len,
            job.output_h * job.output_w * conv.out_channels
        );
        simd::aarch64::dense_columns(
            conv,
            input,
            job.input_h,
            job.input_w,
            job.output_h,
            job.output_w,
            x_start,
            x_end,
            job.output as *mut f32,
        );
    }
}

struct PresenceModel {
    kernel: KernelKind,
    #[cfg(target_arch = "aarch64")]
    pool: PresencePool,
    conv1: DenseConv,
    conv2: DenseConv,
    head: DenseConv,
    output_scale: f32,
    output_bias: f32,
    input: Vec<f32>,
    valid_input: Vec<u8>,
    stage1: Vec<f32>,
    stage2: Vec<f32>,
    #[cfg(target_arch = "aarch64")]
    head_padded: Vec<f32>,
    region: Vec<f32>,
    valid_region: Vec<u8>,
    horizontal_sum: Vec<f32>,
    horizontal_count: Vec<u8>,
}

impl PresenceModel {
    fn compile(model: &UnifiedModel) -> Result<Self> {
        require_architecture(model, 28, 3, "presence")?;
        let conv1 = DenseConv::compile(model, "backbone.0", 3, 16, [5, 5], [2, 2], [2, 2], [1, 1])?;
        let conv2 =
            DenseConv::compile(model, "backbone.2", 16, 32, [5, 5], [2, 2], [2, 2], [1, 1])?;
        let head = DenseConv::compile(
            model,
            "presence_head.local",
            32,
            1,
            [3, 3],
            [1, 1],
            [2, 2],
            [2, 2],
        )?;
        if conv1.output_shape(INPUT_H, INPUT_W)? != (16, 128)
            || conv2.output_shape(16, 128)? != (8, 64)
            || head.output_shape(8, 64)? != (8, 64)
        {
            return Err(invalid_cpu_graph(
                "presence convolution shapes do not match export",
            ));
        }
        let log_scale = scalar_parameter(model, "presence_head.evidence_pool.log_scale")?;
        let output_bias = scalar_parameter(model, "presence_head.evidence_pool.bias")?;
        Ok(Self {
            kernel: KernelKind::detect(),
            #[cfg(target_arch = "aarch64")]
            pool: PresencePool::new()?,
            conv1,
            conv2,
            head,
            output_scale: softplus(log_scale) + 0.5,
            output_bias,
            input: vec![0.0; INPUT_H * INPUT_W * 3],
            valid_input: vec![0; INPUT_H * INPUT_W],
            stage1: vec![0.0; 16 * 128 * 16],
            stage2: vec![0.0; 8 * 64 * 32],
            #[cfg(target_arch = "aarch64")]
            head_padded: vec![0.0; 12 * 68 * 32],
            region: vec![0.0; 8 * 64],
            valid_region: vec![0; 8 * 64],
            horizontal_sum: vec![0.0; 8 * 64],
            horizontal_count: vec![0; 8 * 64],
        })
    }

    fn run(&mut self, input: &Tensor) -> Result<f32> {
        nchw_bytes_to_nhwc(&input.data, 3, INPUT_H, INPUT_W, &mut self.input)?;
        for (index, pixel) in self.input.chunks_exact(3).enumerate() {
            self.valid_input[index] =
                u8::from(pixel[0].abs().max(pixel[1].abs()).max(pixel[2].abs()) > 1e-8);
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.pool
                .dispatch(&self.conv1, &self.input, INPUT_H, INPUT_W, &mut self.stage1)?;
            self.pool
                .dispatch(&self.conv2, &self.stage1, 16, 128, &mut self.stage2)?;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            dense_silu(
                self.kernel,
                &self.conv1,
                &self.input,
                INPUT_H,
                INPUT_W,
                &mut self.stage1,
            );
            dense_silu(
                self.kernel,
                &self.conv2,
                &self.stage1,
                16,
                128,
                &mut self.stage2,
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the fixed ROI shape and buffers were checked at compile
            // time; the specialized head only touches those validated ranges.
            unsafe {
                simd::aarch64::presence_head(
                    &self.head,
                    &self.stage2,
                    &mut self.head_padded,
                    &mut self.region,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        dense_linear(
            self.kernel,
            &self.head,
            &self.stage2,
            8,
            64,
            &mut self.region,
        );

        // Exported mask path: any channel -> 4x4 average -> greater than 0.5.
        for oy in 0..8 {
            for ox in 0..64 {
                let mut count = 0usize;
                for y in oy * 4..oy * 4 + 4 {
                    let row = y * INPUT_W + ox * 4;
                    count += self.valid_input[row..row + 4]
                        .iter()
                        .map(|value| usize::from(*value))
                        .sum::<usize>();
                }
                self.valid_region[oy * 64 + ox] = u8::from(count > 8);
            }
        }

        // The numerator and denominator average pools have identical windows and
        // in-bounds divisors, so their quotient is exactly a valid-only mean apart
        // from the final floating-point rounding.
        for y in 0..8 {
            let row = y * 64;
            let mut sum = 0.0f32;
            let mut count = 0u8;
            for x in 0..64 {
                if x == 0 {
                    for xx in 0..3 {
                        if self.valid_region[row + xx] != 0 {
                            sum += self.region[row + xx];
                            count += 1;
                        }
                    }
                } else {
                    if x + 2 < 64 && self.valid_region[row + x + 2] != 0 {
                        sum += self.region[row + x + 2];
                        count += 1;
                    }
                    if x >= 3 && self.valid_region[row + x - 3] != 0 {
                        sum -= self.region[row + x - 3];
                        count -= 1;
                    }
                }
                self.horizontal_sum[row + x] = sum;
                self.horizontal_count[row + x] = count;
            }
        }
        let mut evidence = f32::NEG_INFINITY;
        for x in 0..64 {
            let mut sum = 0.0f32;
            let mut count = 0u16;
            for y in 0..8 {
                if y == 0 {
                    for yy in 0..3 {
                        sum += self.horizontal_sum[yy * 64 + x];
                        count += u16::from(self.horizontal_count[yy * 64 + x]);
                    }
                } else {
                    if y + 2 < 8 {
                        sum += self.horizontal_sum[(y + 2) * 64 + x];
                        count += u16::from(self.horizontal_count[(y + 2) * 64 + x]);
                    }
                    if y >= 3 {
                        sum -= self.horizontal_sum[(y - 3) * 64 + x];
                        count -= u16::from(self.horizontal_count[(y - 3) * 64 + x]);
                    }
                }
                if self.valid_region[y * 64 + x] != 0 {
                    evidence = evidence.max(sum / count.max(1) as f32);
                }
            }
        }
        let evidence = evidence.max(-10_000.0);
        Ok(self.output_scale * evidence + self.output_bias)
    }
}

struct PairModel {
    kernel: KernelKind,
    #[cfg(target_arch = "aarch64")]
    pool: PresencePool,
    stem: DenseConv,
    downsample: DenseConv,
    detail_dw: DepthwiseConv,
    detail_pw: DenseConv,
    vertical_dw: DepthwiseConv,
    vertical_pw: DenseConv,
    context_dw: DepthwiseConv,
    context_pw: DenseConv,
    mask_weight: Vec<f32>,
    mask_bias: f32,
    classifier1: Linear,
    classifier2: Linear,
    left: Vec<f32>,
    right: Vec<f32>,
    delta: Vec<f32>,
    integral: Vec<f32>,
    features: Vec<f32>,
    stage_a: Vec<f32>,
    stage_b: Vec<f32>,
    pooled: Vec<f32>,
    hidden: Vec<f32>,
    width_sum: Vec<f32>,
}

impl PairModel {
    fn compile(model: &UnifiedModel) -> Result<Self> {
        require_architecture(model, 38, 9, "pair matcher")?;
        let stem =
            DenseConv::compile(model, "model.stem.0", 9, 16, [3, 3], [2, 2], [1, 1], [1, 1])?;
        let downsample = DenseConv::compile(
            model,
            "model.downsample.0",
            16,
            32,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
        )?;
        let detail_dw = DepthwiseConv::compile(model, "model.detail.0", 32, [1, 1])?;
        let detail_pw = DenseConv::compile(
            model,
            "model.detail.3",
            32,
            32,
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
        )?;
        let vertical_dw = DepthwiseConv::compile(model, "model.vertical_downsample.0", 32, [2, 1])?;
        let vertical_pw = DenseConv::compile(
            model,
            "model.vertical_downsample.3",
            32,
            48,
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
        )?;
        let context_dw = DepthwiseConv::compile(model, "model.context.0", 48, [1, 1])?;
        let context_pw = DenseConv::compile(
            model,
            "model.context.3",
            48,
            48,
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
        )?;
        let mask_weight = parameter(model, "model.text_mask.weight", &[1, 48, 1, 1])?;
        let mask_bias = scalar_parameter(model, "model.text_mask.bias")?;
        let classifier1 = Linear::compile(model, "model.classifier.0", 144, 32)?;
        let classifier2 = Linear::compile(model, "model.classifier.2", 32, 1)?;
        if stem.output_shape(INPUT_H, INPUT_W)? != (16, 128)
            || downsample.output_shape(16, 128)? != (8, 64)
            || detail_pw.output_shape(8, 64)? != (8, 64)
            || vertical_pw.output_shape(4, 64)? != (4, 64)
            || context_pw.output_shape(4, 64)? != (4, 64)
        {
            return Err(invalid_cpu_graph(
                "pair convolution shapes do not match export",
            ));
        }
        Ok(Self {
            kernel: KernelKind::detect(),
            #[cfg(target_arch = "aarch64")]
            pool: PresencePool::new()?,
            stem,
            downsample,
            detail_dw,
            detail_pw,
            vertical_dw,
            vertical_pw,
            context_dw,
            context_pw,
            mask_weight,
            mask_bias,
            classifier1,
            classifier2,
            left: vec![0.0; INPUT_H * INPUT_W * 3],
            right: vec![0.0; INPUT_H * INPUT_W * 3],
            delta: vec![0.0; INPUT_H * INPUT_W * 3],
            integral: vec![0.0; (INPUT_H + 1) * (INPUT_W + 1) * 3],
            features: vec![0.0; INPUT_H * INPUT_W * 9],
            stage_a: vec![0.0; 16 * 128 * 16],
            stage_b: vec![0.0; 16 * 128 * 16],
            pooled: vec![0.0; 144],
            hidden: vec![0.0; 32],
            width_sum: vec![0.0; 64 * 48],
        })
    }

    fn run(&mut self, left: &Tensor, right: &Tensor) -> Result<f32> {
        nchw_bytes_to_nhwc(&left.data, 3, INPUT_H, INPUT_W, &mut self.left)?;
        nchw_bytes_to_nhwc(&right.data, 3, INPUT_H, INPUT_W, &mut self.right)?;
        pair_features(
            &self.left,
            &self.right,
            &mut self.delta,
            &mut self.integral,
            &mut self.features,
        );
        #[cfg(target_arch = "aarch64")]
        {
            self.pool.dispatch(
                &self.stem,
                &self.features,
                INPUT_H,
                INPUT_W,
                &mut self.stage_a,
            )?;
            self.pool.dispatch(
                &self.downsample,
                &self.stage_a,
                16,
                128,
                &mut self.stage_b[..8 * 64 * 32],
            )?;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            dense_silu(
                self.kernel,
                &self.stem,
                &self.features,
                INPUT_H,
                INPUT_W,
                &mut self.stage_a,
            );
            dense_silu(
                self.kernel,
                &self.downsample,
                &self.stage_a,
                16,
                128,
                &mut self.stage_b[..8 * 64 * 32],
            );
        }
        depthwise_silu(
            self.kernel,
            &self.detail_dw,
            &self.stage_b[..8 * 64 * 32],
            8,
            64,
            &mut self.stage_a[..8 * 64 * 32],
        );
        dense_silu(
            self.kernel,
            &self.detail_pw,
            &self.stage_a[..8 * 64 * 32],
            8,
            64,
            &mut self.stage_b[..8 * 64 * 32],
        );
        depthwise_silu(
            self.kernel,
            &self.vertical_dw,
            &self.stage_b[..8 * 64 * 32],
            8,
            64,
            &mut self.stage_a[..4 * 64 * 32],
        );
        dense_silu(
            self.kernel,
            &self.vertical_pw,
            &self.stage_a[..4 * 64 * 32],
            4,
            64,
            &mut self.stage_b[..4 * 64 * 48],
        );
        depthwise_silu(
            self.kernel,
            &self.context_dw,
            &self.stage_b[..4 * 64 * 48],
            4,
            64,
            &mut self.stage_a[..4 * 64 * 48],
        );
        dense_silu(
            self.kernel,
            &self.context_pw,
            &self.stage_a[..4 * 64 * 48],
            4,
            64,
            &mut self.stage_b[..4 * 64 * 48],
        );
        gated_pool(
            &self.stage_b[..4 * 64 * 48],
            &self.mask_weight,
            self.mask_bias,
            &mut self.width_sum,
            &mut self.pooled,
        );
        linear_silu(
            self.kernel,
            &self.classifier1,
            &self.pooled,
            &mut self.hidden,
        );
        Ok(linear_scalar(&self.classifier2, &self.hidden))
    }
}

fn require_architecture(
    model: &UnifiedModel,
    expected_nodes: usize,
    expected_convs: usize,
    name: &str,
) -> Result<()> {
    let expected_ops: &[&str] = match name {
        "presence" => &PRESENCE_OPS,
        "pair matcher" => &PAIR_OPS,
        _ => return Err(invalid_cpu_graph(format!("unknown architecture {name}"))),
    };
    if model
        .graph
        .nodes
        .iter()
        .map(|node| node.original_op.as_str())
        .ne(expected_ops.iter().copied())
    {
        return Err(invalid_cpu_graph(format!(
            "unsupported {name} operator sequence"
        )));
    }
    let conv_nodes = model
        .graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, Op::Conv2d { .. }))
        .collect::<Vec<_>>();
    if model.graph.nodes.len() != expected_nodes || conv_nodes.len() != expected_convs {
        return Err(invalid_cpu_graph(format!(
            "unsupported {name} graph: expected {expected_nodes} nodes/{expected_convs} convolutions, got {}/{}",
            model.graph.nodes.len(),
            conv_nodes.len()
        )));
    }
    type ConvSpec<'a> = (&'a str, [usize; 2], [usize; 2], [usize; 2], usize);
    let presence_specs: [ConvSpec<'_>; 3] = [
        ("backbone.0", [2, 2], [2, 2], [1, 1], 1),
        ("backbone.2", [2, 2], [2, 2], [1, 1], 1),
        ("presence_head.local", [1, 1], [2, 2], [2, 2], 1),
    ];
    let pair_specs: [ConvSpec<'_>; 9] = [
        ("model.stem.0", [2, 2], [1, 1], [1, 1], 1),
        ("model.downsample.0", [2, 2], [1, 1], [1, 1], 1),
        ("model.detail.0", [1, 1], [1, 1], [1, 1], 32),
        ("model.detail.3", [1, 1], [0, 0], [1, 1], 1),
        ("model.vertical_downsample.0", [2, 1], [1, 1], [1, 1], 32),
        ("model.vertical_downsample.3", [1, 1], [0, 0], [1, 1], 1),
        ("model.context.0", [1, 1], [1, 1], [1, 1], 48),
        ("model.context.3", [1, 1], [0, 0], [1, 1], 1),
        ("model.text_mask", [1, 1], [0, 0], [1, 1], 1),
    ];
    let specs: &[ConvSpec<'_>] = if name == "presence" {
        &presence_specs
    } else {
        &pair_specs
    };
    for (node, (target, expected_stride, expected_padding, expected_dilation, expected_groups)) in
        conv_nodes.into_iter().zip(specs)
    {
        let Op::Conv2d {
            stride,
            padding,
            dilation,
            groups,
        } = node.op
        else {
            unreachable!();
        };
        let weight_target = node
            .inputs
            .get(1)
            .and_then(|id| model.tensors.get(id.0))
            .and_then(|info| info.target.as_deref());
        let bias_target = node
            .inputs
            .get(2)
            .and_then(|id| model.tensors.get(id.0))
            .and_then(|info| info.target.as_deref());
        if stride != *expected_stride
            || padding != *expected_padding
            || dilation != *expected_dilation
            || groups != *expected_groups
            || weight_target.and_then(|value| value.strip_suffix(".weight")) != Some(*target)
            || bias_target.and_then(|value| value.strip_suffix(".bias")) != Some(*target)
        {
            return Err(invalid_cpu_graph(format!(
                "convolution {} does not match the {name} export",
                node.name
            )));
        }
    }
    Ok(())
}

fn parameter(model: &UnifiedModel, target: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = model
        .tensors
        .iter()
        .find(|info| {
            matches!(info.kind, TensorKind::Parameter | TensorKind::Buffer)
                && info.target.as_deref() == Some(target)
        })
        .ok_or_else(|| invalid_cpu_graph(format!("missing parameter {target}")))?;
    validate_parameter(info, target, shape)?;
    let weight = info
        .weight
        .ok_or_else(|| invalid_cpu_graph(format!("parameter {target} has no data")))?;
    let bytes = model.weights.slice(weight.offset, weight.byte_length)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid_cpu_graph(format!(
            "parameter {target} byte length is not float32-aligned"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn scalar_parameter(model: &UnifiedModel, target: &str) -> Result<f32> {
    let info = model
        .tensors
        .iter()
        .find(|info| {
            matches!(info.kind, TensorKind::Parameter | TensorKind::Buffer)
                && info.target.as_deref() == Some(target)
        })
        .ok_or_else(|| invalid_cpu_graph(format!("missing parameter {target}")))?;
    if !info.shape.is_empty() && info.shape != [1] {
        return Err(RtError::InvalidTensor {
            tensor: target.into(),
            message: format!("expected scalar shape [] or [1], got {:?}", info.shape),
        });
    }
    let values = parameter(model, target, &info.shape)?;
    values
        .first()
        .copied()
        .ok_or_else(|| invalid_cpu_graph(format!("scalar parameter {target} is empty")))
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

fn invalid_cpu_graph(message: impl Into<String>) -> RtError {
    RtError::InvalidGraph(format!("CPU ROI backend: {}", message.into()))
}

fn nchw_bytes_to_nhwc(
    bytes: &[u8],
    channels: usize,
    height: usize,
    width: usize,
    output: &mut [f32],
) -> Result<()> {
    let elements = channels * height * width;
    if bytes.len() != elements * 4 || output.len() != elements {
        return Err(RtError::InvalidTensor {
            tensor: "<cpu-input>".into(),
            message: "input byte length or CPU arena length mismatch".into(),
        });
    }
    let plane = height * width;
    for channel in 0..channels {
        for index in 0..plane {
            let offset = (channel * plane + index) * 4;
            output[index * channels + channel] = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
        }
    }
    Ok(())
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn pair_features(
    left: &[f32],
    right: &[f32],
    delta: &mut [f32],
    integral: &mut [f32],
    output: &mut [f32],
) {
    debug_assert_eq!(left.len(), INPUT_H * INPUT_W * 3);
    debug_assert_eq!(right.len(), left.len());
    debug_assert_eq!(delta.len(), left.len());
    debug_assert_eq!(integral.len(), (INPUT_H + 1) * (INPUT_W + 1) * 3);
    debug_assert_eq!(output.len(), INPUT_H * INPUT_W * 9);

    for index in 0..delta.len() {
        delta[index] = left[index] - right[index];
    }
    integral.fill(0.0);
    let integral_w = INPUT_W + 1;
    for y in 0..INPUT_H {
        let mut row_sum = [0.0f32; 3];
        for x in 0..INPUT_W {
            let pixel = (y * INPUT_W + x) * 3;
            let above = (y * integral_w + x + 1) * 3;
            let target = ((y + 1) * integral_w + x + 1) * 3;
            for channel in 0..3 {
                row_sum[channel] += delta[pixel + channel];
                integral[target + channel] = integral[above + channel] + row_sum[channel];
            }
        }
    }
    for y in 0..INPUT_H {
        let y0 = y.saturating_sub(2);
        let y1 = (y + 3).min(INPUT_H);
        for x in 0..INPUT_W {
            let x0 = x.saturating_sub(2);
            let x1 = (x + 3).min(INPUT_W);
            let count = ((y1 - y0) * (x1 - x0)) as f32;
            let pixel = (y * INPUT_W + x) * 3;
            let destination = (y * INPUT_W + x) * 9;
            for channel in 0..3 {
                let a = ((y1 * integral_w + x1) * 3) + channel;
                let b = ((y0 * integral_w + x1) * 3) + channel;
                let c = ((y1 * integral_w + x0) * 3) + channel;
                let d = ((y0 * integral_w + x0) * 3) + channel;
                let background = (integral[a] - integral[b] - integral[c] + integral[d]) / count;
                output[destination + channel] =
                    0.5 * (left[pixel + channel] + right[pixel + channel]);
                output[destination + 3 + channel] = delta[pixel + channel].abs();
                output[destination + 6 + channel] = (delta[pixel + channel] - background).abs();
            }
        }
    }
}

fn dense_silu(
    kind: KernelKind,
    conv: &DenseConv,
    input: &[f32],
    input_h: usize,
    input_w: usize,
    output: &mut [f32],
) {
    let (output_h, output_w) = conv
        .output_shape(input_h, input_w)
        .expect("validated convolution shape");
    debug_assert_eq!(input.len(), input_h * input_w * conv.in_channels);
    debug_assert_eq!(output.len(), output_h * output_w * conv.out_channels);
    match kind {
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => {
            // SAFETY: compile-time shape checks and the assertions above prove all
            // pointer ranges used by the private NEON kernel are in bounds.
            unsafe {
                simd::aarch64::dense::<true>(conv, input, input_h, input_w, output);
            }
        }
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => {
            // SAFETY: AVX2/FMA was detected once during compile; buffer ranges are
            // validated before this private kernel is entered.
            unsafe {
                simd::x86_64::dense::<true>(conv, input, input_h, input_w, output);
            }
        }
        KernelKind::Scalar => dense_scalar::<true>(conv, input, input_h, input_w, output),
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn dense_linear(
    kind: KernelKind,
    conv: &DenseConv,
    input: &[f32],
    input_h: usize,
    input_w: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(conv.out_channels, 1);
    match kind {
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => unsafe {
            simd::x86_64::dense_single(conv, input, input_h, input_w, output);
        },
        KernelKind::Scalar => dense_scalar::<false>(conv, input, input_h, input_w, output),
    }
}

fn dense_scalar<const SILU: bool>(
    conv: &DenseConv,
    input: &[f32],
    input_h: usize,
    input_w: usize,
    output: &mut [f32],
) {
    let (output_h, output_w) = conv.output_shape(input_h, input_w).unwrap();
    for oy in 0..output_h {
        let input_origin_y = oy as isize * conv.stride[0] as isize - conv.padding[0] as isize;
        for ox in 0..output_w {
            let input_origin_x = ox as isize * conv.stride[1] as isize - conv.padding[1] as isize;
            let output_base = (oy * output_w + ox) * conv.out_channels;
            for oc in 0..conv.out_channels {
                let mut sum = conv.bias[oc];
                for ky in 0..conv.kernel[0] {
                    let iy = input_origin_y + (ky * conv.dilation[0]) as isize;
                    if !(0..input_h as isize).contains(&iy) {
                        continue;
                    }
                    for kx in 0..conv.kernel[1] {
                        let ix = input_origin_x + (kx * conv.dilation[1]) as isize;
                        if !(0..input_w as isize).contains(&ix) {
                            continue;
                        }
                        let input_base = (iy as usize * input_w + ix as usize) * conv.in_channels;
                        let weight_base =
                            (ky * conv.kernel[1] + kx) * conv.in_channels * conv.out_channels;
                        for ic in 0..conv.in_channels {
                            sum = input[input_base + ic].mul_add(
                                conv.weights[weight_base + ic * conv.out_channels + oc],
                                sum,
                            );
                        }
                    }
                }
                output[output_base + oc] = if SILU { silu_scalar(sum) } else { sum };
            }
        }
    }
}

fn depthwise_silu(
    kind: KernelKind,
    conv: &DepthwiseConv,
    input: &[f32],
    input_h: usize,
    input_w: usize,
    output: &mut [f32],
) {
    let output_h = (input_h + 2 - 3) / conv.stride[0] + 1;
    let output_w = (input_w + 2 - 3) / conv.stride[1] + 1;
    debug_assert_eq!(input.len(), input_h * input_w * conv.channels);
    debug_assert_eq!(output.len(), output_h * output_w * conv.channels);
    match kind {
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => unsafe {
            simd::aarch64::depthwise(conv, input, input_h, input_w, output);
        },
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => unsafe {
            simd::x86_64::depthwise(conv, input, input_h, input_w, output);
        },
        KernelKind::Scalar => {
            for oy in 0..output_h {
                for ox in 0..output_w {
                    let out_base = (oy * output_w + ox) * conv.channels;
                    for channel in 0..conv.channels {
                        let mut sum = conv.bias[channel];
                        for ky in 0..3 {
                            let iy = oy as isize * conv.stride[0] as isize + ky as isize - 1;
                            if !(0..input_h as isize).contains(&iy) {
                                continue;
                            }
                            for kx in 0..3 {
                                let ix = ox as isize * conv.stride[1] as isize + kx as isize - 1;
                                if !(0..input_w as isize).contains(&ix) {
                                    continue;
                                }
                                sum = input[(iy as usize * input_w + ix as usize) * conv.channels
                                    + channel]
                                    .mul_add(
                                        conv.weights[(ky * 3 + kx) * conv.channels + channel],
                                        sum,
                                    );
                            }
                        }
                        output[out_base + channel] = silu_scalar(sum);
                    }
                }
            }
        }
    }
}

fn gated_pool(
    input: &[f32],
    mask_weight: &[f32],
    mask_bias: f32,
    width_sum: &mut [f32],
    output: &mut [f32],
) {
    const CHANNELS: usize = 48;
    const HEIGHT: usize = 4;
    const WIDTH: usize = 64;
    debug_assert_eq!(input.len(), HEIGHT * WIDTH * CHANNELS);
    debug_assert_eq!(mask_weight.len(), CHANNELS);
    debug_assert_eq!(width_sum.len(), WIDTH * CHANNELS);
    debug_assert_eq!(output.len(), CHANNELS * 3);
    output[..CHANNELS].fill(0.0);
    output[CHANNELS..CHANNELS * 2].fill(f32::NEG_INFINITY);
    width_sum.fill(0.0);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let base = (y * WIDTH + x) * CHANNELS;
            let mut mask = mask_bias;
            for channel in 0..CHANNELS {
                mask = input[base + channel].mul_add(mask_weight[channel], mask);
            }
            let gate = 0.25 + 0.75 * sigmoid_scalar(mask);
            for channel in 0..CHANNELS {
                let value = input[base + channel] * gate;
                output[channel] += value;
                output[CHANNELS + channel] = output[CHANNELS + channel].max(value);
                width_sum[x * CHANNELS + channel] += value;
            }
        }
    }
    for channel in 0..CHANNELS {
        output[channel] /= (HEIGHT * WIDTH) as f32;
        let mut width_peak = f32::NEG_INFINITY;
        for x in 0..WIDTH {
            width_peak = width_peak.max(width_sum[x * CHANNELS + channel] / HEIGHT as f32);
        }
        output[CHANNELS * 2 + channel] = width_peak;
    }
}

fn linear_silu(kind: KernelKind, linear: &Linear, input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), linear.in_features);
    debug_assert_eq!(output.len(), linear.out_features);
    match kind {
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => unsafe {
            simd::aarch64::linear_silu(linear, input, output);
        },
        #[cfg(target_arch = "x86_64")]
        KernelKind::Avx2 => unsafe {
            simd::x86_64::linear_silu(linear, input, output);
        },
        KernelKind::Scalar => {
            for (row, (weights, target)) in linear
                .weights
                .chunks_exact(linear.in_features)
                .zip(output)
                .enumerate()
            {
                let mut sum = linear.bias[row];
                for (value, weight) in input.iter().zip(weights) {
                    sum = value.mul_add(*weight, sum);
                }
                *target = silu_scalar(sum);
            }
        }
    }
}

fn linear_scalar(linear: &Linear, input: &[f32]) -> f32 {
    debug_assert_eq!(linear.out_features, 1);
    let mut sum = linear.bias[0];
    for (value, weight) in input.iter().zip(&linear.weights) {
        sum = value.mul_add(*weight, sum);
    }
    sum
}

#[inline(always)]
fn silu_scalar(value: f32) -> f32 {
    value * sigmoid_scalar(value)
}

#[inline(always)]
fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + fast_exp_scalar(-value))
}

/// Degree-six range-reduced exponential used by scalar fallback and SIMD lanes.
/// Maximum relative error over the sigmoid's useful range is below 2e-6.
#[inline(always)]
fn fast_exp_scalar(value: f32) -> f32 {
    let x = value.clamp(-80.0, 80.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = (-n).mul_add(std::f32::consts::LN_2, x);
    let mut polynomial: f32 = 1.0 / 720.0;
    polynomial = polynomial.mul_add(r, 1.0 / 120.0);
    polynomial = polynomial.mul_add(r, 1.0 / 24.0);
    polynomial = polynomial.mul_add(r, 1.0 / 6.0);
    polynomial = polynomial.mul_add(r, 0.5);
    polynomial = polynomial.mul_add(r, 1.0);
    polynomial = polynomial.mul_add(r, 1.0);
    let exponent = ((n as i32 + 127) as u32) << 23;
    polynomial * f32::from_bits(exponent)
}

mod simd {
    #[cfg(target_arch = "aarch64")]
    pub(super) mod aarch64 {
        use std::arch::aarch64::*;

        use super::super::{DenseConv, DepthwiseConv, Linear};

        #[inline(always)]
        unsafe fn exp_ps(value: float32x4_t) -> float32x4_t {
            unsafe {
                let lower = vdupq_n_f32(-80.0);
                let upper = vdupq_n_f32(80.0);
                let x = vmaxq_f32(lower, vminq_f32(upper, value));
                let n = vcvtnq_s32_f32(vmulq_n_f32(x, std::f32::consts::LOG2_E));
                let nf = vcvtq_f32_s32(n);
                let r = vfmsq_n_f32(x, nf, std::f32::consts::LN_2);
                let mut p = vdupq_n_f32(1.0 / 720.0);
                p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), p, r);
                p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
                p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
                p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
                p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
                p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
                let exponent = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
                vmulq_f32(p, vreinterpretq_f32_s32(exponent))
            }
        }

        #[inline(always)]
        unsafe fn silu_ps(value: float32x4_t) -> float32x4_t {
            unsafe {
                let denominator = vaddq_f32(vdupq_n_f32(1.0), exp_ps(vnegq_f32(value)));
                vmulq_f32(value, vdivq_f32(vdupq_n_f32(1.0), denominator))
            }
        }

        macro_rules! neon_dense_kernel {
            ($name:ident, $range_name:ident, $columns_name:ident, $channels:expr, $(($sum:ident, $offset:expr)),+ $(,)?) => {
                #[inline(always)]
                unsafe fn $name(
                    conv: &DenseConv,
                    input: &[f32],
                    input_h: usize,
                    input_w: usize,
                    output: &mut [f32],
                ) {
                    let (output_h, output_w) = conv.output_shape(input_h, input_w).unwrap();
                    unsafe {
                        $range_name(
                            conv,
                            input,
                            input_h,
                            input_w,
                            output_w,
                            0,
                            output_h * output_w,
                            output,
                        );
                    }
                }

                #[inline(always)]
                #[allow(clippy::too_many_arguments)]
                unsafe fn $range_name(
                    conv: &DenseConv,
                    input: &[f32],
                    input_h: usize,
                    input_w: usize,
                    output_w: usize,
                    start: usize,
                    end: usize,
                    output: &mut [f32],
                ) {
                    unsafe {
                        for position in start..end {
                            let oy = position / output_w;
                            let ox = position % output_w;
                            let origin_y = oy as isize * conv.stride[0] as isize
                                - conv.padding[0] as isize;
                            let origin_x = ox as isize * conv.stride[1] as isize
                                - conv.padding[1] as isize;
                            let output_base = (position - start) * $channels;
                            $(let mut $sum = vld1q_f32(conv.bias.as_ptr().add($offset));)+
                            for ky in 0..conv.kernel[0] {
                                let iy = origin_y + (ky * conv.dilation[0]) as isize;
                                if !(0..input_h as isize).contains(&iy) {
                                    continue;
                                }
                                for kx in 0..conv.kernel[1] {
                                    let ix = origin_x + (kx * conv.dilation[1]) as isize;
                                    if !(0..input_w as isize).contains(&ix) {
                                        continue;
                                    }
                                    let input_base = (iy as usize * input_w + ix as usize)
                                        * conv.in_channels;
                                    let weight_base = (ky * conv.kernel[1] + kx)
                                        * conv.in_channels
                                        * $channels;
                                    for ic in 0..conv.in_channels {
                                        let value = input[input_base + ic];
                                        let weights = conv.weights.as_ptr().add(
                                            weight_base + ic * $channels,
                                        );
                                        $($sum = vfmaq_n_f32(
                                            $sum,
                                            vld1q_f32(weights.add($offset)),
                                            value,
                                        );)+
                                    }
                                }
                            }
                            $(vst1q_f32(
                                output.as_mut_ptr().add(output_base + $offset),
                                silu_ps($sum),
                            );)+
                        }
                    }
                }

                #[inline(always)]
                #[allow(clippy::too_many_arguments)]
                unsafe fn $columns_name(
                    conv: &DenseConv,
                    input: &[f32],
                    input_h: usize,
                    input_w: usize,
                    output_h: usize,
                    output_w: usize,
                    x_start: usize,
                    x_end: usize,
                    output: *mut f32,
                ) {
                    unsafe {
                        for oy in 0..output_h {
                            let origin_y = oy as isize * conv.stride[0] as isize
                                - conv.padding[0] as isize;
                            for ox in x_start..x_end {
                                let origin_x = ox as isize * conv.stride[1] as isize
                                    - conv.padding[1] as isize;
                                let output_base = (oy * output_w + ox) * $channels;
                                $(let mut $sum = vld1q_f32(conv.bias.as_ptr().add($offset));)+
                                for ky in 0..conv.kernel[0] {
                                    let iy = origin_y + (ky * conv.dilation[0]) as isize;
                                    if !(0..input_h as isize).contains(&iy) {
                                        continue;
                                    }
                                    for kx in 0..conv.kernel[1] {
                                        let ix = origin_x + (kx * conv.dilation[1]) as isize;
                                        if !(0..input_w as isize).contains(&ix) {
                                            continue;
                                        }
                                        let input_base = (iy as usize * input_w + ix as usize)
                                            * conv.in_channels;
                                        let weight_base = (ky * conv.kernel[1] + kx)
                                            * conv.in_channels
                                            * $channels;
                                        for ic in 0..conv.in_channels {
                                            let value = input[input_base + ic];
                                            let weights = conv.weights.as_ptr().add(
                                                weight_base + ic * $channels,
                                            );
                                            $($sum = vfmaq_n_f32(
                                                $sum,
                                                vld1q_f32(weights.add($offset)),
                                                value,
                                            );)+
                                        }
                                    }
                                }
                                $(vst1q_f32(
                                    output.add(output_base + $offset),
                                    silu_ps($sum),
                                );)+
                            }
                        }
                    }
                }
            };
        }

        neon_dense_kernel!(
            dense16,
            dense16_range,
            dense16_columns,
            16,
            (sum0, 0),
            (sum1, 4),
            (sum2, 8),
            (sum3, 12),
        );
        neon_dense_kernel!(
            dense32,
            dense32_range,
            dense32_columns,
            32,
            (sum0, 0),
            (sum1, 4),
            (sum2, 8),
            (sum3, 12),
            (sum4, 16),
            (sum5, 20),
            (sum6, 24),
            (sum7, 28),
        );
        neon_dense_kernel!(
            dense48,
            dense48_range,
            dense48_columns,
            48,
            (sum0, 0),
            (sum1, 4),
            (sum2, 8),
            (sum3, 12),
            (sum4, 16),
            (sum5, 20),
            (sum6, 24),
            (sum7, 28),
            (sum8, 32),
            (sum9, 36),
            (sum10, 40),
            (sum11, 44),
        );

        pub(crate) unsafe fn dense<const SILU: bool>(
            conv: &DenseConv,
            input: &[f32],
            input_h: usize,
            input_w: usize,
            output: &mut [f32],
        ) {
            debug_assert!(SILU);
            unsafe {
                match conv.out_channels {
                    16 => dense16(conv, input, input_h, input_w, output),
                    32 => dense32(conv, input, input_h, input_w, output),
                    48 => dense48(conv, input, input_h, input_w, output),
                    _ => unreachable!("validated ROI dense channel count"),
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn dense_columns(
            conv: &DenseConv,
            input: &[f32],
            input_h: usize,
            input_w: usize,
            output_h: usize,
            output_w: usize,
            x_start: usize,
            x_end: usize,
            output: *mut f32,
        ) {
            unsafe {
                match conv.out_channels {
                    16 => dense16_columns(
                        conv, input, input_h, input_w, output_h, output_w, x_start, x_end, output,
                    ),
                    32 => dense32_columns(
                        conv, input, input_h, input_w, output_h, output_w, x_start, x_end, output,
                    ),
                    48 => dense48_columns(
                        conv, input, input_h, input_w, output_h, output_w, x_start, x_end, output,
                    ),
                    _ => unreachable!("validated ROI dense channel count"),
                }
            }
        }

        pub(crate) unsafe fn presence_head(
            conv: &DenseConv,
            input: &[f32],
            padded: &mut [f32],
            output: &mut [f32],
        ) {
            const CHANNELS: usize = 32;
            const INPUT_H: usize = 8;
            const INPUT_W: usize = 64;
            const PADDED_W: usize = 68;
            debug_assert_eq!(conv.in_channels, CHANNELS);
            debug_assert_eq!(conv.out_channels, 1);
            debug_assert_eq!(input.len(), INPUT_H * INPUT_W * CHANNELS);
            debug_assert_eq!(padded.len(), 12 * PADDED_W * CHANNELS);
            debug_assert_eq!(output.len(), INPUT_H * INPUT_W);
            for y in 0..INPUT_H {
                let source = &input[y * INPUT_W * CHANNELS..(y + 1) * INPUT_W * CHANNELS];
                let destination_start = ((y + 2) * PADDED_W + 2) * CHANNELS;
                padded[destination_start..destination_start + source.len()].copy_from_slice(source);
            }
            unsafe {
                for y in 0..INPUT_H {
                    for x in 0..INPUT_W {
                        let mut sum0 = vdupq_n_f32(0.0);
                        let mut sum1 = vdupq_n_f32(0.0);
                        let mut sum2 = vdupq_n_f32(0.0);
                        let mut sum3 = vdupq_n_f32(0.0);
                        let mut sum4 = vdupq_n_f32(0.0);
                        let mut sum5 = vdupq_n_f32(0.0);
                        let mut sum6 = vdupq_n_f32(0.0);
                        let mut sum7 = vdupq_n_f32(0.0);
                        let mut sum8 = vdupq_n_f32(0.0);
                        let input0 = padded.as_ptr().add((y * PADDED_W + x) * CHANNELS);
                        let input1 = input0.add(2 * CHANNELS);
                        let input2 = input0.add(4 * CHANNELS);
                        let input3 = input0.add(2 * PADDED_W * CHANNELS);
                        let input4 = input3.add(2 * CHANNELS);
                        let input5 = input3.add(4 * CHANNELS);
                        let input6 = input0.add(4 * PADDED_W * CHANNELS);
                        let input7 = input6.add(2 * CHANNELS);
                        let input8 = input6.add(4 * CHANNELS);
                        for channel in (0..CHANNELS).step_by(4) {
                            sum0 = vfmaq_f32(
                                sum0,
                                vld1q_f32(input0.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(channel)),
                            );
                            sum1 = vfmaq_f32(
                                sum1,
                                vld1q_f32(input1.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(CHANNELS + channel)),
                            );
                            sum2 = vfmaq_f32(
                                sum2,
                                vld1q_f32(input2.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(2 * CHANNELS + channel)),
                            );
                            sum3 = vfmaq_f32(
                                sum3,
                                vld1q_f32(input3.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(3 * CHANNELS + channel)),
                            );
                            sum4 = vfmaq_f32(
                                sum4,
                                vld1q_f32(input4.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(4 * CHANNELS + channel)),
                            );
                            sum5 = vfmaq_f32(
                                sum5,
                                vld1q_f32(input5.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(5 * CHANNELS + channel)),
                            );
                            sum6 = vfmaq_f32(
                                sum6,
                                vld1q_f32(input6.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(6 * CHANNELS + channel)),
                            );
                            sum7 = vfmaq_f32(
                                sum7,
                                vld1q_f32(input7.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(7 * CHANNELS + channel)),
                            );
                            sum8 = vfmaq_f32(
                                sum8,
                                vld1q_f32(input8.add(channel)),
                                vld1q_f32(conv.weights.as_ptr().add(8 * CHANNELS + channel)),
                            );
                        }
                        let sum01 = vaddq_f32(sum0, sum1);
                        let sum23 = vaddq_f32(sum2, sum3);
                        let sum45 = vaddq_f32(sum4, sum5);
                        let sum67 = vaddq_f32(sum6, sum7);
                        let sum = vaddq_f32(
                            vaddq_f32(sum01, sum23),
                            vaddq_f32(vaddq_f32(sum45, sum67), sum8),
                        );
                        output[y * INPUT_W + x] = conv.bias[0] + vaddvq_f32(sum);
                    }
                }
            }
        }

        pub(crate) unsafe fn depthwise(
            conv: &DepthwiseConv,
            input: &[f32],
            input_h: usize,
            input_w: usize,
            output: &mut [f32],
        ) {
            debug_assert_eq!(conv.channels % 4, 0);
            let output_h = (input_h - 1) / conv.stride[0] + 1;
            let output_w = (input_w - 1) / conv.stride[1] + 1;
            unsafe {
                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let output_base = (oy * output_w + ox) * conv.channels;
                        for channel in (0..conv.channels).step_by(4) {
                            let mut sum = vld1q_f32(conv.bias.as_ptr().add(channel));
                            for ky in 0..3 {
                                let iy = oy as isize * conv.stride[0] as isize + ky as isize - 1;
                                if !(0..input_h as isize).contains(&iy) {
                                    continue;
                                }
                                for kx in 0..3 {
                                    let ix =
                                        ox as isize * conv.stride[1] as isize + kx as isize - 1;
                                    if !(0..input_w as isize).contains(&ix) {
                                        continue;
                                    }
                                    let values = vld1q_f32(input.as_ptr().add(
                                        (iy as usize * input_w + ix as usize) * conv.channels
                                            + channel,
                                    ));
                                    let weights = vld1q_f32(
                                        conv.weights
                                            .as_ptr()
                                            .add((ky * 3 + kx) * conv.channels + channel),
                                    );
                                    sum = vfmaq_f32(sum, values, weights);
                                }
                            }
                            vst1q_f32(output.as_mut_ptr().add(output_base + channel), silu_ps(sum));
                        }
                    }
                }
            }
        }

        pub(crate) unsafe fn linear_silu(linear: &Linear, input: &[f32], output: &mut [f32]) {
            debug_assert_eq!(linear.in_features % 4, 0);
            unsafe {
                for (row, target) in output.iter_mut().enumerate() {
                    let mut sum = vdupq_n_f32(0.0);
                    let weights = linear.weights.as_ptr().add(row * linear.in_features);
                    for index in (0..linear.in_features).step_by(4) {
                        sum = vfmaq_f32(
                            sum,
                            vld1q_f32(input.as_ptr().add(index)),
                            vld1q_f32(weights.add(index)),
                        );
                    }
                    let value = linear.bias[row] + vaddvq_f32(sum);
                    *target = value * (1.0 / (1.0 + super::super::fast_exp_scalar(-value)));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(super) mod x86_64 {
        use std::arch::x86_64::*;

        use super::super::{DenseConv, DepthwiseConv, Linear, silu_scalar};

        #[target_feature(enable = "avx2,fma")]
        unsafe fn exp_ps(value: __m256) -> __m256 {
            let lower = _mm256_set1_ps(-80.0);
            let upper = _mm256_set1_ps(80.0);
            let x = _mm256_max_ps(lower, _mm256_min_ps(upper, value));
            let n = _mm256_cvtps_epi32(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)));
            let nf = _mm256_cvtepi32_ps(n);
            let r = _mm256_fnmadd_ps(nf, _mm256_set1_ps(std::f32::consts::LN_2), x);
            let mut p = _mm256_set1_ps(1.0 / 720.0);
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 120.0));
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 24.0));
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0 / 6.0));
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5));
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
            p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
            let exponent = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
            _mm256_mul_ps(p, _mm256_castsi256_ps(exponent))
        }

        #[target_feature(enable = "avx2,fma")]
        unsafe fn silu_ps(value: __m256) -> __m256 {
            let exponential = unsafe { exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), value)) };
            _mm256_mul_ps(
                value,
                _mm256_div_ps(
                    _mm256_set1_ps(1.0),
                    _mm256_add_ps(_mm256_set1_ps(1.0), exponential),
                ),
            )
        }

        macro_rules! avx_dense_kernel {
            ($name:ident, $channels:expr, $(($sum:ident, $offset:expr)),+ $(,)?) => {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn $name(
                    conv: &DenseConv,
                    input: &[f32],
                    input_h: usize,
                    input_w: usize,
                    output: &mut [f32],
                ) {
                    let (output_h, output_w) = conv.output_shape(input_h, input_w).unwrap();
                    unsafe {
                        for oy in 0..output_h {
                            let origin_y = oy as isize * conv.stride[0] as isize
                                - conv.padding[0] as isize;
                            for ox in 0..output_w {
                                let origin_x = ox as isize * conv.stride[1] as isize
                                    - conv.padding[1] as isize;
                                let output_base = (oy * output_w + ox) * $channels;
                                $(let mut $sum = _mm256_loadu_ps(
                                    conv.bias.as_ptr().add($offset),
                                );)+
                                for ky in 0..conv.kernel[0] {
                                    let iy = origin_y + (ky * conv.dilation[0]) as isize;
                                    if !(0..input_h as isize).contains(&iy) {
                                        continue;
                                    }
                                    for kx in 0..conv.kernel[1] {
                                        let ix = origin_x + (kx * conv.dilation[1]) as isize;
                                        if !(0..input_w as isize).contains(&ix) {
                                            continue;
                                        }
                                        let input_base = (iy as usize * input_w + ix as usize)
                                            * conv.in_channels;
                                        let weight_base = (ky * conv.kernel[1] + kx)
                                            * conv.in_channels
                                            * $channels;
                                        for ic in 0..conv.in_channels {
                                            let value = _mm256_set1_ps(input[input_base + ic]);
                                            let weights = conv.weights.as_ptr().add(
                                                weight_base + ic * $channels,
                                            );
                                            $($sum = _mm256_fmadd_ps(
                                                _mm256_loadu_ps(weights.add($offset)),
                                                value,
                                                $sum,
                                            );)+
                                        }
                                    }
                                }
                                $($sum = silu_ps($sum);)+
                                $(_mm256_storeu_ps(
                                    output.as_mut_ptr().add(output_base + $offset),
                                    $sum,
                                );)+
                            }
                        }
                    }
                }
            };
        }

        avx_dense_kernel!(dense16, 16, (sum0, 0), (sum1, 8),);
        avx_dense_kernel!(dense32, 32, (sum0, 0), (sum1, 8), (sum2, 16), (sum3, 24),);
        avx_dense_kernel!(
            dense48,
            48,
            (sum0, 0),
            (sum1, 8),
            (sum2, 16),
            (sum3, 24),
            (sum4, 32),
            (sum5, 40),
        );

        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn dense<const SILU: bool>(
            conv: &DenseConv,
            input: &[f32],
            input_h: usize,
            input_w: usize,
            output: &mut [f32],
        ) {
            debug_assert!(SILU);
            unsafe {
                match conv.out_channels {
                    16 => dense16(conv, input, input_h, input_w, output),
                    32 => dense32(conv, input, input_h, input_w, output),
                    48 => dense48(conv, input, input_h, input_w, output),
                    _ => unreachable!("validated ROI dense channel count"),
                }
            }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn dense_single(
            conv: &DenseConv,
            input: &[f32],
            input_h: usize,
            input_w: usize,
            output: &mut [f32],
        ) {
            let (output_h, output_w) = conv.output_shape(input_h, input_w).unwrap();
            unsafe {
                for oy in 0..output_h {
                    let origin_y = oy as isize * conv.stride[0] as isize - conv.padding[0] as isize;
                    for ox in 0..output_w {
                        let origin_x =
                            ox as isize * conv.stride[1] as isize - conv.padding[1] as isize;
                        let mut vector_sum = _mm256_setzero_ps();
                        for ky in 0..conv.kernel[0] {
                            let iy = origin_y + (ky * conv.dilation[0]) as isize;
                            if !(0..input_h as isize).contains(&iy) {
                                continue;
                            }
                            for kx in 0..conv.kernel[1] {
                                let ix = origin_x + (kx * conv.dilation[1]) as isize;
                                if !(0..input_w as isize).contains(&ix) {
                                    continue;
                                }
                                let input_base =
                                    (iy as usize * input_w + ix as usize) * conv.in_channels;
                                let weight_base = (ky * conv.kernel[1] + kx) * conv.in_channels;
                                for ic in (0..conv.in_channels).step_by(8) {
                                    vector_sum = _mm256_fmadd_ps(
                                        _mm256_loadu_ps(input.as_ptr().add(input_base + ic)),
                                        _mm256_loadu_ps(
                                            conv.weights.as_ptr().add(weight_base + ic),
                                        ),
                                        vector_sum,
                                    );
                                }
                            }
                        }
                        let mut lanes = [0.0f32; 8];
                        _mm256_storeu_ps(lanes.as_mut_ptr(), vector_sum);
                        output[oy * output_w + ox] = conv.bias[0] + lanes.iter().sum::<f32>();
                    }
                }
            }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn depthwise(
            conv: &DepthwiseConv,
            input: &[f32],
            input_h: usize,
            input_w: usize,
            output: &mut [f32],
        ) {
            debug_assert_eq!(conv.channels % 8, 0);
            let output_h = (input_h - 1) / conv.stride[0] + 1;
            let output_w = (input_w - 1) / conv.stride[1] + 1;
            unsafe {
                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let output_base = (oy * output_w + ox) * conv.channels;
                        for channel in (0..conv.channels).step_by(8) {
                            let mut sum = _mm256_loadu_ps(conv.bias.as_ptr().add(channel));
                            for ky in 0..3 {
                                let iy = oy as isize * conv.stride[0] as isize + ky as isize - 1;
                                if !(0..input_h as isize).contains(&iy) {
                                    continue;
                                }
                                for kx in 0..3 {
                                    let ix =
                                        ox as isize * conv.stride[1] as isize + kx as isize - 1;
                                    if !(0..input_w as isize).contains(&ix) {
                                        continue;
                                    }
                                    sum = _mm256_fmadd_ps(
                                        _mm256_loadu_ps(input.as_ptr().add(
                                            (iy as usize * input_w + ix as usize) * conv.channels
                                                + channel,
                                        )),
                                        _mm256_loadu_ps(
                                            conv.weights
                                                .as_ptr()
                                                .add((ky * 3 + kx) * conv.channels + channel),
                                        ),
                                        sum,
                                    );
                                }
                            }
                            _mm256_storeu_ps(
                                output.as_mut_ptr().add(output_base + channel),
                                silu_ps(sum),
                            );
                        }
                    }
                }
            }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn linear_silu(linear: &Linear, input: &[f32], output: &mut [f32]) {
            unsafe {
                for (row, target) in output.iter_mut().enumerate() {
                    let weights = linear.weights.as_ptr().add(row * linear.in_features);
                    let mut sum = _mm256_setzero_ps();
                    let mut index = 0;
                    while index + 8 <= linear.in_features {
                        sum = _mm256_fmadd_ps(
                            _mm256_loadu_ps(input.as_ptr().add(index)),
                            _mm256_loadu_ps(weights.add(index)),
                            sum,
                        );
                        index += 8;
                    }
                    let mut lanes = [0.0f32; 8];
                    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
                    let mut value = linear.bias[row] + lanes.iter().sum::<f32>();
                    while index < linear.in_features {
                        value = input[index].mul_add(*weights.add(index), value);
                        index += 1;
                    }
                    *target = silu_scalar(value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fast_exp_scalar;

    #[test]
    fn fast_exp_tracks_standard_exp() {
        for step in -800..=800 {
            let value = step as f32 * 0.01;
            let expected = value.exp();
            let relative = (fast_exp_scalar(value) - expected).abs() / expected;
            assert!(relative < 2e-6, "x={value}, relative={relative}");
        }
    }
}
