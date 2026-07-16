//! Pure-Rust CPU backend for the exported ROI models.
//!
//! The model contract is validated and weights are packed at compile time. Runtime tensors stay in the
//! public NCHW byte representation, while weights and scratch buffers use typed
//! NHWC storage. This keeps the public format portable without ever assuming
//! that a `Vec<u8>` is aligned for `f32` or SIMD loads.

use crate::backend::{Backend, Executable};
use crate::error::{Result, RtError};
use crate::ir::tensor::{DType, TensorInfo, TensorKind};
use crate::model::{Model, ModelType};
use crate::tensor::Tensor;

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

pub struct CpuExecutable {
    input_shapes: Vec<Vec<usize>>,
    model: CompiledRoiModel,
    batch_pool: Option<BatchPool>,
}

enum CompiledRoiModel {
    Presence(Box<PresenceModel>),
    Pair(Box<PairModel>),
}

impl CompiledRoiModel {
    fn compile(model: &Model, inner_parallelism: bool) -> Result<Self> {
        match model.inputs.len() {
            1 => Ok(Self::Presence(Box::new(PresenceModel::compile(
                model,
                inner_parallelism,
            )?))),
            2 => Ok(Self::Pair(Box::new(PairModel::compile(
                model,
                inner_parallelism,
            )?))),
            count => Err(RtError::Inference(format!(
                "CPU ROI backend expects one or two inputs, got {count}"
            ))),
        }
    }

    fn run_packed_sample(
        &mut self,
        inputs: &[Tensor],
        sample: usize,
        sample_bytes: usize,
    ) -> Result<f32> {
        let start = sample * sample_bytes;
        let end = start + sample_bytes;
        match self {
            Self::Presence(model) => model.run_bytes(&inputs[0].data[start..end]),
            Self::Pair(model) => {
                model.run_bytes(&inputs[0].data[start..end], &inputs[1].data[start..end])
            }
        }
    }

    fn worker_clone(&self) -> Self {
        match self {
            Self::Presence(model) => Self::Presence(Box::new(model.worker_clone())),
            Self::Pair(model) => Self::Pair(Box::new(model.worker_clone())),
        }
    }
}

impl Backend for CpuBackend {
    type Executable = CpuExecutable;

    fn compile(&self, model: &Model) -> Result<Self::Executable> {
        validate_model_io(model)?;
        let compiled = CompiledRoiModel::compile(model, true)?;
        Ok(CpuExecutable {
            input_shapes: model
                .input_info()
                .into_iter()
                .map(|info| info.shape.clone())
                .collect(),
            model: compiled,
            batch_pool: None,
        })
    }
}

impl CpuExecutable {
    pub fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }

    /// Execute the scalar-output ROI model without allocating an output tensor.
    ///
    /// All activation arenas are owned by the executable and reused between
    /// calls. The executable is intentionally `&mut self`, which also prevents
    /// concurrent access to the same scratch storage.
    pub fn run_scalar(&mut self, inputs: &[Tensor]) -> Result<f32> {
        validate_runtime_inputs(inputs, &self.input_shapes)?;
        match &mut self.model {
            CompiledRoiModel::Presence(model) => model.run_bytes(&inputs[0].data),
            CompiledRoiModel::Pair(model) => model.run_bytes(&inputs[0].data, &inputs[1].data),
        }
    }

    /// Run a packed batch and return one scalar per sample without hot-path
    /// allocation inside the inference workers.
    ///
    /// Each model input is a packed float32 NCHW tensor matching the model shape.
    /// Samples are scheduled dynamically across persistent workers, while small
    /// batches retain the optimized batch-1 SIMD path.
    pub fn run_batch_scalar(&mut self, inputs: &[Tensor]) -> Result<Vec<f32>> {
        let batch_size = validate_packed_batch_inputs(inputs, &self.input_shapes)?;
        let mut output = vec![0.0; batch_size];
        self.run_validated_batch(inputs, &mut output)?;
        Ok(output)
    }

    /// Allocation-free variant of [`Self::run_batch_scalar`].
    pub fn run_batch_scalar_into(&mut self, inputs: &[Tensor], output: &mut [f32]) -> Result<()> {
        let batch_size = validate_packed_batch_inputs(inputs, &self.input_shapes)?;
        if output.len() != batch_size {
            return Err(RtError::InvalidTensor {
                tensor: "<runtime-batch-output>".into(),
                message: format!(
                    "expected {batch_size} scalar output slots, got {}",
                    output.len()
                ),
            });
        }
        self.run_validated_batch(inputs, output)
    }

    fn run_validated_batch(&mut self, inputs: &[Tensor], output: &mut [f32]) -> Result<()> {
        let batch_size = output.len();
        if batch_size == 0 {
            return Ok(());
        }
        let sample_bytes =
            self.input_shapes[0][1..].iter().product::<usize>() * std::mem::size_of::<f32>();

        // Presence already occupies four SIMD lanes per sample; a larger gate
        // avoids losing throughput to heterogeneous-core scheduling on small batches.
        let parallel_threshold = match &self.model {
            #[cfg(target_arch = "aarch64")]
            CompiledRoiModel::Presence(_) => scalar_parallelism() * 4,
            _ => scalar_parallelism() * 2,
        };
        if batch_size < parallel_threshold {
            for (sample, target) in output.iter_mut().enumerate() {
                *target = self.model.run_packed_sample(inputs, sample, sample_bytes)?;
            }
        } else {
            if self.batch_pool.is_none() {
                self.batch_pool = Some(BatchPool::new(&self.model)?);
            }
            self.batch_pool.as_mut().expect("initialized above").run(
                inputs,
                batch_size,
                sample_bytes,
                output,
            )?;
        }
        Ok(())
    }

    /// Run a packed batch and return the conventional `[batch]` output tensor.
    pub fn run_batch(&mut self, inputs: &[Tensor]) -> Result<Tensor> {
        let values = self.run_batch_scalar(inputs)?;
        Tensor::from_f32(vec![values.len()], values)
    }

    /// Number of CPU lanes available to the batch scheduler.
    pub fn batch_parallelism(&self) -> usize {
        available_batch_parallelism()
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

fn validate_model_io(model: &Model) -> Result<()> {
    if model.outputs.len() != 1 {
        return Err(RtError::Inference(format!(
            "CPU ROI backend expects one output, got {}",
            model.outputs.len()
        )));
    }
    for info in model.input_info() {
        if info.dtype != DType::Float32
            || info.shape.len() != 4
            || info.shape[0] != 1
            || info.shape[1] != 3
            || info.shape[2] == 0
            || info.shape[3] == 0
        {
            return Err(RtError::InvalidTensor {
                tensor: info.name.clone(),
                message: format!(
                    "CPU ROI backend requires float32 shape [1, 3, H, W], got {} {:?}",
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

#[cfg(feature = "gpu")]
pub(crate) fn validate_roi_model(model: &Model) -> Result<()> {
    validate_model_io(model)?;
    match model.inputs.len() {
        1 => require_architecture(model, 28, 3, "presence"),
        2 => require_architecture(model, 38, 9, "pair matcher"),
        count => Err(RtError::Inference(format!(
            "ROI backend expects one or two inputs, got {count}"
        ))),
    }
}

fn validate_runtime_inputs(inputs: &[Tensor], shapes: &[Vec<usize>]) -> Result<()> {
    if inputs.len() != shapes.len() {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-inputs>".into(),
            message: format!("expected {} inputs, got {}", shapes.len(), inputs.len()),
        });
    }
    for (index, (input, shape)) in inputs.iter().zip(shapes).enumerate() {
        let expected_bytes = DType::Float32.byte_len_for_shape(shape);
        if input.dtype != DType::Float32
            || input.shape != *shape
            || expected_bytes != Some(input.data.len())
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

fn validate_packed_batch_inputs(inputs: &[Tensor], shapes: &[Vec<usize>]) -> Result<usize> {
    if inputs.len() != shapes.len() {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-batch-inputs>".into(),
            message: format!("expected {} inputs, got {}", shapes.len(), inputs.len()),
        });
    }
    let mut batch_size = None;
    for (index, (input, model_shape)) in inputs.iter().zip(shapes).enumerate() {
        let Some(first_dim) = input.shape.first().copied() else {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: "packed input must have a batch dimension".into(),
            });
        };
        let shape_matches =
            input.shape.len() == model_shape.len() && input.shape[1..] == model_shape[1..];
        let expected_bytes = DType::Float32.byte_len_for_shape(&input.shape);
        if input.dtype != DType::Float32
            || !shape_matches
            || expected_bytes != Some(input.data.len())
        {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: format!(
                    "expected packed float32 shape [batch, 3, H, W] matching the model, got {} {:?}",
                    input.dtype, input.shape
                ),
            });
        }
        if batch_size
            .replace(first_dim)
            .is_some_and(|size| size != first_dim)
        {
            return Err(RtError::InvalidTensor {
                tensor: format!("input[{index}]"),
                message: "all packed inputs must use the same batch size".into(),
            });
        }
    }
    Ok(batch_size.unwrap_or(0))
}

fn available_batch_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .clamp(1, 16)
}

fn scalar_parallelism() -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        return available_batch_parallelism().min(4);
    }
    #[allow(unreachable_code)]
    1
}

#[derive(Clone, Copy)]
struct BatchJob {
    inputs: usize,
    input_count: usize,
    batch_size: usize,
    sample_bytes: usize,
    output: usize,
}

struct BatchShared {
    job: std::cell::UnsafeCell<BatchJob>,
    generation: std::sync::atomic::AtomicU64,
    next_sample: std::sync::atomic::AtomicUsize,
    completed: std::sync::atomic::AtomicUsize,
    failed: std::sync::atomic::AtomicBool,
    error: std::sync::Mutex<Option<String>>,
    shutdown: std::sync::atomic::AtomicBool,
    sleep_lock: std::sync::Mutex<()>,
    wake: std::sync::Condvar,
}

// SAFETY: one `&mut CpuExecutable` dispatcher owns the job slot. It publishes
// each job with a release generation increment and waits for every helper before
// the borrowed input/output pointers can expire or the slot can be overwritten.
unsafe impl Sync for BatchShared {}

struct BatchPool {
    shared: std::sync::Arc<BatchShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    caller_model: CompiledRoiModel,
}

impl BatchPool {
    fn new(model: &CompiledRoiModel) -> Result<Self> {
        let helper_count = available_batch_parallelism().saturating_sub(1);
        let caller_model = model.worker_clone();
        let workers = (0..helper_count)
            .map(|_| model.worker_clone())
            .collect::<Vec<_>>();
        let shared = std::sync::Arc::new(BatchShared {
            job: std::cell::UnsafeCell::new(BatchJob {
                inputs: 0,
                input_count: 0,
                batch_size: 0,
                sample_bytes: 0,
                output: 0,
            }),
            generation: std::sync::atomic::AtomicU64::new(0),
            next_sample: std::sync::atomic::AtomicUsize::new(0),
            completed: std::sync::atomic::AtomicUsize::new(0),
            failed: std::sync::atomic::AtomicBool::new(false),
            error: std::sync::Mutex::new(None),
            shutdown: std::sync::atomic::AtomicBool::new(false),
            sleep_lock: std::sync::Mutex::new(()),
            wake: std::sync::Condvar::new(),
        });
        let mut threads = Vec::with_capacity(helper_count);
        for (index, worker) in workers.into_iter().enumerate() {
            let worker_shared = std::sync::Arc::clone(&shared);
            let handle = match std::thread::Builder::new()
                .name(format!("subfast-batch-{}", index + 1))
                .spawn(move || batch_worker(worker_shared, worker))
            {
                Ok(handle) => handle,
                Err(error) => {
                    stop_batch_workers(&shared, &mut threads);
                    return Err(RtError::Inference(format!(
                        "failed to create CPU batch worker: {error}"
                    )));
                }
            };
            threads.push(handle);
        }
        Ok(Self {
            shared,
            threads,
            caller_model,
        })
    }

    fn run(
        &mut self,
        inputs: &[Tensor],
        batch_size: usize,
        sample_bytes: usize,
        output: &mut [f32],
    ) -> Result<()> {
        debug_assert_eq!(batch_size, output.len());
        if self.threads.is_empty() {
            for (sample, target) in output.iter_mut().enumerate() {
                *target = self
                    .caller_model
                    .run_packed_sample(inputs, sample, sample_bytes)?;
            }
            return Ok(());
        }

        let job = BatchJob {
            inputs: inputs.as_ptr() as usize,
            input_count: inputs.len(),
            batch_size,
            sample_bytes,
            output: output.as_mut_ptr() as usize,
        };
        let wake_guard =
            self.shared.sleep_lock.lock().map_err(|_| {
                RtError::Inference("CPU batch worker sleep lock is poisoned".into())
            })?;
        // Sample zero is reserved for the caller's single-threaded SIMD worker.
        self.shared
            .next_sample
            .store(1, std::sync::atomic::Ordering::Relaxed);
        self.shared
            .completed
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.shared
            .failed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        *self
            .shared
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        // SAFETY: dispatches are serialized by `&mut BatchPool`; all readers
        // of the previous generation completed before its call returned.
        unsafe {
            *self.shared.job.get() = job;
        }
        self.shared
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        self.shared.wake.notify_all();
        drop(wake_guard);

        let caller_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let value = self
                .caller_model
                .run_packed_sample(inputs, 0, sample_bytes)?;
            // SAFETY: helpers start at sample one, so slot zero belongs to caller.
            unsafe {
                (job.output as *mut f32).write(value);
            }
            // SAFETY: the input and output borrows remain live until all helpers
            // complete below. Atomic indices give every participant disjoint output.
            unsafe { execute_batch_job(&self.shared, job, &mut self.caller_model) }
        }));
        match caller_result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => record_batch_failure(&self.shared, error.to_string()),
            Err(_) => record_batch_failure(&self.shared, "CPU batch caller panicked".into()),
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
            .load(std::sync::atomic::Ordering::Acquire)
        {
            let message = self
                .shared
                .error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
                .unwrap_or_else(|| "CPU batch worker failed".into());
            return Err(RtError::Inference(message));
        }
        Ok(())
    }
}

impl Drop for BatchPool {
    fn drop(&mut self) {
        stop_batch_workers(&self.shared, &mut self.threads);
    }
}

fn stop_batch_workers(
    shared: &std::sync::Arc<BatchShared>,
    threads: &mut Vec<std::thread::JoinHandle<()>>,
) {
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
    for thread in threads.drain(..) {
        let _ = thread.join();
    }
}

fn batch_worker(shared: std::sync::Arc<BatchShared>, mut model: CompiledRoiModel) {
    let mut observed_generation = 0u64;
    loop {
        let generation = loop {
            let generation = shared.generation.load(std::sync::atomic::Ordering::Acquire);
            if generation != observed_generation {
                break generation;
            }
            let guard = shared
                .sleep_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if shared.generation.load(std::sync::atomic::Ordering::Acquire) == observed_generation
                && !shared.shutdown.load(std::sync::atomic::Ordering::Acquire)
            {
                let guard = shared
                    .wake
                    .wait(guard)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                drop(guard);
            }
        };
        if shared.shutdown.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        // SAFETY: the acquire generation load observes an immutable job that the
        // dispatcher retains until this worker increments `completed`.
        let job = unsafe { *shared.job.get() };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the dispatcher validates and retains every pointed-to buffer;
            // the atomic scheduler assigns each output slot once.
            unsafe { execute_batch_job(&shared, job, &mut model) }
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => record_batch_failure(&shared, error.to_string()),
            Err(_) => record_batch_failure(&shared, "CPU batch worker panicked".into()),
        }
        shared
            .completed
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        observed_generation = generation;
    }
}

unsafe fn execute_batch_job(
    shared: &BatchShared,
    job: BatchJob,
    model: &mut CompiledRoiModel,
) -> Result<()> {
    // SAFETY: `BatchPool::run` keeps this immutable slice alive for the entire job.
    let inputs =
        unsafe { std::slice::from_raw_parts(job.inputs as *const Tensor, job.input_count) };
    loop {
        if shared.failed.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        let sample = shared
            .next_sample
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if sample >= job.batch_size {
            return Ok(());
        }
        let value = model.run_packed_sample(inputs, sample, job.sample_bytes)?;
        // SAFETY: `sample` is unique and within the initialized output allocation.
        unsafe {
            (job.output as *mut f32).add(sample).write(value);
        }
    }
}

fn record_batch_failure(shared: &BatchShared, message: String) {
    if !shared
        .failed
        .swap(true, std::sync::atomic::Ordering::AcqRel)
    {
        *shared
            .error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(message);
    }
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

#[derive(Clone)]
struct DenseConv {
    in_channels: usize,
    out_channels: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    /// K-major weights: `[ky][kx][ic][oc]`.
    weights: std::sync::Arc<[f32]>,
    bias: std::sync::Arc<[f32]>,
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
            weights: weights.into(),
            bias: bias.into(),
        })
    }

    fn output_shape(&self, input_h: usize, input_w: usize) -> Result<(usize, usize)> {
        let effective_h = self.dilation[0]
            .checked_mul(self.kernel[0].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_cpu_model("convolution height overflow"))?;
        let effective_w = self.dilation[1]
            .checked_mul(self.kernel[1].saturating_sub(1))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_cpu_model("convolution width overflow"))?;
        let padded_h = input_h
            .checked_add(self.padding[0] * 2)
            .ok_or_else(|| invalid_cpu_model("padded height overflow"))?;
        let padded_w = input_w
            .checked_add(self.padding[1] * 2)
            .ok_or_else(|| invalid_cpu_model("padded width overflow"))?;
        if padded_h < effective_h || padded_w < effective_w {
            return Err(invalid_cpu_model("convolution kernel exceeds input"));
        }
        Ok((
            (padded_h - effective_h) / self.stride[0] + 1,
            (padded_w - effective_w) / self.stride[1] + 1,
        ))
    }
}

#[derive(Clone)]
struct DepthwiseConv {
    channels: usize,
    stride: [usize; 2],
    /// K-major weights: `[ky][kx][channel]`, fixed 3x3.
    weights: std::sync::Arc<[f32]>,
    bias: std::sync::Arc<[f32]>,
}

impl DepthwiseConv {
    fn compile(model: &Model, target: &str, channels: usize, stride: [usize; 2]) -> Result<Self> {
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
            weights: weights.into(),
            bias: bias.into(),
        })
    }
}

#[derive(Clone)]
struct Linear {
    in_features: usize,
    out_features: usize,
    weights: std::sync::Arc<[f32]>,
    bias: std::sync::Arc<[f32]>,
}

impl Linear {
    fn compile(
        model: &Model,
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
            )?
            .into(),
            bias: parameter(model, &format!("{target}.bias"), &[out_features])?.into(),
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

/// Persistent helpers plus the caller. No thread is created on the hot path.
#[cfg(target_arch = "aarch64")]
struct PresencePool {
    shared: std::sync::Arc<PoolShared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    total_threads: usize,
}

#[cfg(target_arch = "aarch64")]
impl PresencePool {
    fn new() -> Result<Self> {
        Self::with_threads(scalar_parallelism(), "subfast-cpu")
    }

    fn with_threads(total_threads: usize, thread_name: &str) -> Result<Self> {
        let total_threads = total_threads.max(1);
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
                .name(format!("{thread_name}-{worker_index}"))
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
        let (output_h, output_w) = conv.output_shape(input_h, input_w)?;
        if self.total_threads == 1 {
            dense_silu(KernelKind::Neon, conv, input, input_h, input_w, output);
            return Ok(());
        }
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
        let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the dispatcher keeps all pointed-to storage alive and gives
            // this worker a disjoint output range before waiting for completion.
            unsafe {
                execute_dense_job(job, worker_index, total_threads);
            }
        }))
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
    pool: Option<PresencePool>,
    conv1: DenseConv,
    conv2: DenseConv,
    head: DenseConv,
    output_scale: f32,
    output_bias: f32,
    input_h: usize,
    input_w: usize,
    stage1_h: usize,
    stage1_w: usize,
    stage2_h: usize,
    stage2_w: usize,
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
    fn compile(model: &Model, inner_parallelism: bool) -> Result<Self> {
        #[cfg(not(target_arch = "aarch64"))]
        let _ = inner_parallelism;
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
        let (input_h, input_w) = model_input_hw(model)?;
        if input_h % 4 != 0 || input_w % 4 != 0 {
            return Err(invalid_cpu_model(
                "presence input height and width must be divisible by 4",
            ));
        }
        let (stage1_h, stage1_w) = conv1.output_shape(input_h, input_w)?;
        let (stage2_h, stage2_w) = conv2.output_shape(stage1_h, stage1_w)?;
        if (stage2_h, stage2_w) != (input_h / 4, input_w / 4)
            || head.output_shape(stage2_h, stage2_w)? != (stage2_h, stage2_w)
        {
            return Err(invalid_cpu_model(
                "presence convolution shapes do not match input",
            ));
        }
        let log_scale = scalar_parameter(model, "presence_head.evidence_pool.log_scale")?;
        let output_bias = scalar_parameter(model, "presence_head.evidence_pool.bias")?;
        Ok(Self {
            kernel: KernelKind::detect(),
            #[cfg(target_arch = "aarch64")]
            pool: inner_parallelism.then(PresencePool::new).transpose()?,
            conv1,
            conv2,
            head,
            output_scale: softplus(log_scale) + 0.5,
            output_bias,
            input_h,
            input_w,
            stage1_h,
            stage1_w,
            stage2_h,
            stage2_w,
            input: vec![0.0; input_h * input_w * 3],
            valid_input: vec![0; input_h * input_w],
            stage1: vec![0.0; stage1_h * stage1_w * 16],
            stage2: vec![0.0; stage2_h * stage2_w * 32],
            #[cfg(target_arch = "aarch64")]
            head_padded: vec![0.0; (stage2_h + 4) * (stage2_w + 4) * 32],
            region: vec![0.0; stage2_h * stage2_w],
            valid_region: vec![0; stage2_h * stage2_w],
            horizontal_sum: vec![0.0; stage2_h * stage2_w],
            horizontal_count: vec![0; stage2_h * stage2_w],
        })
    }

    fn worker_clone(&self) -> Self {
        Self {
            kernel: self.kernel,
            #[cfg(target_arch = "aarch64")]
            pool: None,
            conv1: self.conv1.clone(),
            conv2: self.conv2.clone(),
            head: self.head.clone(),
            output_scale: self.output_scale,
            output_bias: self.output_bias,
            input_h: self.input_h,
            input_w: self.input_w,
            stage1_h: self.stage1_h,
            stage1_w: self.stage1_w,
            stage2_h: self.stage2_h,
            stage2_w: self.stage2_w,
            input: vec![0.0; self.input.len()],
            valid_input: vec![0; self.valid_input.len()],
            stage1: vec![0.0; self.stage1.len()],
            stage2: vec![0.0; self.stage2.len()],
            #[cfg(target_arch = "aarch64")]
            head_padded: vec![0.0; self.head_padded.len()],
            region: vec![0.0; self.region.len()],
            valid_region: vec![0; self.valid_region.len()],
            horizontal_sum: vec![0.0; self.horizontal_sum.len()],
            horizontal_count: vec![0; self.horizontal_count.len()],
        }
    }

    fn run_bytes(&mut self, input: &[u8]) -> Result<f32> {
        nchw_bytes_to_nhwc(input, 3, self.input_h, self.input_w, &mut self.input)?;
        for (index, pixel) in self.input.chunks_exact(3).enumerate() {
            self.valid_input[index] =
                u8::from(pixel[0].abs().max(pixel[1].abs()).max(pixel[2].abs()) > 1e-8);
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(pool) = &self.pool {
                pool.dispatch(
                    &self.conv1,
                    &self.input,
                    self.input_h,
                    self.input_w,
                    &mut self.stage1,
                )?;
                pool.dispatch(
                    &self.conv2,
                    &self.stage1,
                    self.stage1_h,
                    self.stage1_w,
                    &mut self.stage2,
                )?;
            } else {
                dense_silu(
                    self.kernel,
                    &self.conv1,
                    &self.input,
                    self.input_h,
                    self.input_w,
                    &mut self.stage1,
                );
                dense_silu(
                    self.kernel,
                    &self.conv2,
                    &self.stage1,
                    self.stage1_h,
                    self.stage1_w,
                    &mut self.stage2,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            dense_silu(
                self.kernel,
                &self.conv1,
                &self.input,
                self.input_h,
                self.input_w,
                &mut self.stage1,
            );
            dense_silu(
                self.kernel,
                &self.conv2,
                &self.stage1,
                self.stage1_h,
                self.stage1_w,
                &mut self.stage2,
            );
        }
        Ok(presence_finish(
            self.kernel,
            &self.head,
            &self.stage2,
            &self.valid_input,
            self.input_w,
            self.stage2_h,
            self.stage2_w,
            self.output_scale,
            self.output_bias,
            #[cfg(target_arch = "aarch64")]
            &mut self.head_padded,
            &mut self.region,
            &mut self.valid_region,
            &mut self.horizontal_sum,
            &mut self.horizontal_count,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn presence_finish(
    _kernel: KernelKind,
    head: &DenseConv,
    stage2: &[f32],
    valid_input: &[u8],
    input_w: usize,
    region_h: usize,
    region_w: usize,
    output_scale: f32,
    output_bias: f32,
    #[cfg(target_arch = "aarch64")] head_padded: &mut [f32],
    region: &mut [f32],
    valid_region: &mut [u8],
    horizontal_sum: &mut [f32],
    horizontal_count: &mut [u8],
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        simd::aarch64::presence_head(head, stage2, region_h, region_w, head_padded, region);
    }
    #[cfg(not(target_arch = "aarch64"))]
    dense_linear(_kernel, head, stage2, region_h, region_w, region);

    // Exported mask path: any channel -> 4x4 average -> greater than 0.5.
    for oy in 0..region_h {
        for ox in 0..region_w {
            let mut count = 0usize;
            for y in oy * 4..oy * 4 + 4 {
                let row = y * input_w + ox * 4;
                count += valid_input[row..row + 4]
                    .iter()
                    .map(|value| usize::from(*value))
                    .sum::<usize>();
            }
            valid_region[oy * region_w + ox] = u8::from(count > 8);
        }
    }

    // The numerator and denominator average pools have identical windows and
    // in-bounds divisors, so their quotient is exactly a valid-only mean apart
    // from the final floating-point rounding.
    for y in 0..region_h {
        let row = y * region_w;
        let mut sum = 0.0f32;
        let mut count = 0u8;
        for x in 0..region_w {
            if x == 0 {
                for xx in 0..3 {
                    if valid_region[row + xx] != 0 {
                        sum += region[row + xx];
                        count += 1;
                    }
                }
            } else {
                if x + 2 < region_w && valid_region[row + x + 2] != 0 {
                    sum += region[row + x + 2];
                    count += 1;
                }
                if x >= 3 && valid_region[row + x - 3] != 0 {
                    sum -= region[row + x - 3];
                    count -= 1;
                }
            }
            horizontal_sum[row + x] = sum;
            horizontal_count[row + x] = count;
        }
    }
    let mut evidence = f32::NEG_INFINITY;
    for x in 0..region_w {
        let mut sum = 0.0f32;
        let mut count = 0u16;
        for y in 0..region_h {
            if y == 0 {
                for yy in 0..3 {
                    sum += horizontal_sum[yy * region_w + x];
                    count += u16::from(horizontal_count[yy * region_w + x]);
                }
            } else {
                if y + 2 < region_h {
                    sum += horizontal_sum[(y + 2) * region_w + x];
                    count += u16::from(horizontal_count[(y + 2) * region_w + x]);
                }
                if y >= 3 {
                    sum -= horizontal_sum[(y - 3) * region_w + x];
                    count -= u16::from(horizontal_count[(y - 3) * region_w + x]);
                }
            }
            if valid_region[y * region_w + x] != 0 {
                evidence = evidence.max(sum / count.max(1) as f32);
            }
        }
    }
    output_scale * evidence.max(-10_000.0) + output_bias
}

struct PairModel {
    kernel: KernelKind,
    #[cfg(target_arch = "aarch64")]
    pool: Option<PresencePool>,
    stem: DenseConv,
    downsample: DenseConv,
    detail_dw: DepthwiseConv,
    detail_pw: DenseConv,
    vertical_dw: DepthwiseConv,
    vertical_pw: DenseConv,
    context_dw: DepthwiseConv,
    context_pw: DenseConv,
    mask_weight: std::sync::Arc<[f32]>,
    mask_bias: f32,
    classifier1: Linear,
    classifier2: Linear,
    input_h: usize,
    input_w: usize,
    stem_h: usize,
    stem_w: usize,
    down_h: usize,
    down_w: usize,
    final_h: usize,
    final_w: usize,
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
    fn compile(model: &Model, inner_parallelism: bool) -> Result<Self> {
        #[cfg(not(target_arch = "aarch64"))]
        let _ = inner_parallelism;
        require_architecture(model, 38, 9, "pair matcher")?;
        let stem = DenseConv::compile(model, "stem.0", 3, 16, [3, 3], [2, 2], [1, 1], [1, 1])?;
        let downsample = DenseConv::compile(
            model,
            "downsample.0",
            16,
            32,
            [3, 3],
            [2, 2],
            [1, 1],
            [1, 1],
        )?;
        let detail_dw = DepthwiseConv::compile(model, "detail.0", 32, [1, 1])?;
        let detail_pw =
            DenseConv::compile(model, "detail.3", 32, 32, [1, 1], [1, 1], [0, 0], [1, 1])?;
        let vertical_dw = DepthwiseConv::compile(model, "vertical_downsample.0", 32, [2, 1])?;
        let vertical_pw = DenseConv::compile(
            model,
            "vertical_downsample.3",
            32,
            48,
            [1, 1],
            [1, 1],
            [0, 0],
            [1, 1],
        )?;
        let context_dw = DepthwiseConv::compile(model, "context.0", 48, [1, 1])?;
        let context_pw =
            DenseConv::compile(model, "context.3", 48, 48, [1, 1], [1, 1], [0, 0], [1, 1])?;
        let mask_weight = parameter(model, "text_mask.weight", &[1, 48, 1, 1])?;
        let mask_bias = scalar_parameter(model, "text_mask.bias")?;
        let classifier1 = Linear::compile(model, "classifier.0", 144, 32)?;
        let classifier2 = Linear::compile(model, "classifier.2", 32, 1)?;
        let (input_h, input_w) = model_input_hw(model)?;
        let (stem_h, stem_w) = stem.output_shape(input_h, input_w)?;
        let (down_h, down_w) = downsample.output_shape(stem_h, stem_w)?;
        let (detail_h, detail_w) = detail_pw.output_shape(down_h, down_w)?;
        let final_h = (detail_h + 1) / 2;
        let final_w = detail_w;
        if vertical_pw.output_shape(final_h, final_w)? != (final_h, final_w)
            || context_pw.output_shape(final_h, final_w)? != (final_h, final_w)
        {
            return Err(invalid_cpu_model(
                "pair convolution shapes do not match input",
            ));
        }
        let input_elements = input_h * input_w;
        let arena_elements = (stem_h * stem_w * 16)
            .max(down_h * down_w * 32)
            .max(final_h * final_w * 48);
        Ok(Self {
            kernel: KernelKind::detect(),
            #[cfg(target_arch = "aarch64")]
            pool: inner_parallelism.then(PresencePool::new).transpose()?,
            stem,
            downsample,
            detail_dw,
            detail_pw,
            vertical_dw,
            vertical_pw,
            context_dw,
            context_pw,
            mask_weight: mask_weight.into(),
            mask_bias,
            classifier1,
            classifier2,
            input_h,
            input_w,
            stem_h,
            stem_w,
            down_h,
            down_w,
            final_h,
            final_w,
            left: vec![0.0; input_elements * 3],
            right: vec![0.0; input_elements * 3],
            delta: vec![0.0; input_elements],
            integral: vec![0.0; (input_h + 1) * (input_w + 1)],
            features: vec![0.0; input_elements * 3],
            stage_a: vec![0.0; arena_elements],
            stage_b: vec![0.0; arena_elements],
            pooled: vec![0.0; 144],
            hidden: vec![0.0; 32],
            width_sum: vec![0.0; final_w * 48],
        })
    }

    fn worker_clone(&self) -> Self {
        Self {
            kernel: self.kernel,
            #[cfg(target_arch = "aarch64")]
            pool: None,
            stem: self.stem.clone(),
            downsample: self.downsample.clone(),
            detail_dw: self.detail_dw.clone(),
            detail_pw: self.detail_pw.clone(),
            vertical_dw: self.vertical_dw.clone(),
            vertical_pw: self.vertical_pw.clone(),
            context_dw: self.context_dw.clone(),
            context_pw: self.context_pw.clone(),
            mask_weight: std::sync::Arc::clone(&self.mask_weight),
            mask_bias: self.mask_bias,
            classifier1: self.classifier1.clone(),
            classifier2: self.classifier2.clone(),
            input_h: self.input_h,
            input_w: self.input_w,
            stem_h: self.stem_h,
            stem_w: self.stem_w,
            down_h: self.down_h,
            down_w: self.down_w,
            final_h: self.final_h,
            final_w: self.final_w,
            left: vec![0.0; self.left.len()],
            right: vec![0.0; self.right.len()],
            delta: vec![0.0; self.delta.len()],
            integral: vec![0.0; self.integral.len()],
            features: vec![0.0; self.features.len()],
            stage_a: vec![0.0; self.stage_a.len()],
            stage_b: vec![0.0; self.stage_b.len()],
            pooled: vec![0.0; 144],
            hidden: vec![0.0; 32],
            width_sum: vec![0.0; self.width_sum.len()],
        }
    }

    fn run_bytes(&mut self, left: &[u8], right: &[u8]) -> Result<f32> {
        nchw_bytes_to_nhwc(left, 3, self.input_h, self.input_w, &mut self.left)?;
        nchw_bytes_to_nhwc(right, 3, self.input_h, self.input_w, &mut self.right)?;
        pair_features(
            &self.left,
            &self.right,
            &mut self.delta,
            &mut self.integral,
            &mut self.features,
            self.input_h,
            self.input_w,
        );
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(pool) = &self.pool {
                pool.dispatch(
                    &self.stem,
                    &self.features,
                    self.input_h,
                    self.input_w,
                    &mut self.stage_a,
                )?;
                pool.dispatch(
                    &self.downsample,
                    &self.stage_a,
                    self.stem_h,
                    self.stem_w,
                    &mut self.stage_b[..self.down_h * self.down_w * 32],
                )?;
            } else {
                dense_silu(
                    self.kernel,
                    &self.stem,
                    &self.features,
                    self.input_h,
                    self.input_w,
                    &mut self.stage_a,
                );
                dense_silu(
                    self.kernel,
                    &self.downsample,
                    &self.stage_a,
                    self.stem_h,
                    self.stem_w,
                    &mut self.stage_b[..self.down_h * self.down_w * 32],
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            dense_silu(
                self.kernel,
                &self.stem,
                &self.features,
                self.input_h,
                self.input_w,
                &mut self.stage_a,
            );
            dense_silu(
                self.kernel,
                &self.downsample,
                &self.stage_a,
                self.stem_h,
                self.stem_w,
                &mut self.stage_b[..self.down_h * self.down_w * 32],
            );
        }
        depthwise_silu(
            self.kernel,
            &self.detail_dw,
            &self.stage_b[..self.down_h * self.down_w * 32],
            self.down_h,
            self.down_w,
            &mut self.stage_a[..self.down_h * self.down_w * 32],
        );
        dense_silu(
            self.kernel,
            &self.detail_pw,
            &self.stage_a[..self.down_h * self.down_w * 32],
            self.down_h,
            self.down_w,
            &mut self.stage_b[..self.down_h * self.down_w * 32],
        );
        depthwise_silu(
            self.kernel,
            &self.vertical_dw,
            &self.stage_b[..self.down_h * self.down_w * 32],
            self.down_h,
            self.down_w,
            &mut self.stage_a[..self.final_h * self.final_w * 32],
        );
        dense_silu(
            self.kernel,
            &self.vertical_pw,
            &self.stage_a[..self.final_h * self.final_w * 32],
            self.final_h,
            self.final_w,
            &mut self.stage_b[..self.final_h * self.final_w * 48],
        );
        depthwise_silu(
            self.kernel,
            &self.context_dw,
            &self.stage_b[..self.final_h * self.final_w * 48],
            self.final_h,
            self.final_w,
            &mut self.stage_a[..self.final_h * self.final_w * 48],
        );
        dense_silu(
            self.kernel,
            &self.context_pw,
            &self.stage_a[..self.final_h * self.final_w * 48],
            self.final_h,
            self.final_w,
            &mut self.stage_b[..self.final_h * self.final_w * 48],
        );
        gated_pool(
            &self.stage_b[..self.final_h * self.final_w * 48],
            &self.mask_weight,
            self.mask_bias,
            &mut self.width_sum,
            &mut self.pooled,
            self.final_h,
            self.final_w,
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
    model: &Model,
    _expected_nodes: usize,
    _expected_convs: usize,
    name: &str,
) -> Result<()> {
    let expected = match name {
        "presence" => ModelType::RoiPresence,
        "pair matcher" => ModelType::RoiPairMatcher,
        _ => return Err(invalid_cpu_model(format!("unknown architecture {name}"))),
    };
    if model.model_type != expected {
        return Err(invalid_cpu_model(format!(
            "model type does not match {name} architecture"
        )));
    }
    Ok(())
}

fn parameter(model: &Model, target: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = model
        .tensors
        .iter()
        .find(|info| {
            matches!(info.kind, TensorKind::Parameter | TensorKind::Buffer)
                && info.target.as_deref() == Some(target)
        })
        .ok_or_else(|| invalid_cpu_model(format!("missing parameter {target}")))?;
    validate_parameter(info, target, shape)?;
    let weight = info
        .weight
        .ok_or_else(|| invalid_cpu_model(format!("parameter {target} has no data")))?;
    let bytes = model.weights.slice(weight.offset, weight.byte_length)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid_cpu_model(format!(
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
        .ok_or_else(|| invalid_cpu_model(format!("missing parameter {target}")))?;
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
        .ok_or_else(|| invalid_cpu_model(format!("scalar parameter {target} is empty")))
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

fn invalid_cpu_model(message: impl Into<String>) -> RtError {
    RtError::InvalidModel(format!("CPU ROI backend: {}", message.into()))
}

fn model_input_hw(model: &Model) -> Result<(usize, usize)> {
    let info = model
        .input_info()
        .into_iter()
        .next()
        .ok_or_else(|| invalid_cpu_model("model has no input"))?;
    Ok((info.shape[2], info.shape[3]))
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
    height: usize,
    width: usize,
) {
    debug_assert_eq!(left.len(), height * width * 3);
    debug_assert_eq!(right.len(), left.len());
    debug_assert_eq!(delta.len(), height * width);
    debug_assert_eq!(integral.len(), (height + 1) * (width + 1));
    debug_assert_eq!(output.len(), height * width * 3);

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    const VALUE_MEAN: f32 = (0.485 + 0.456 + 0.406) / 3.0;
    const VALUE_STD: f32 = (0.229 + 0.224 + 0.225) / 3.0;
    for (index, (left, right)) in left.chunks_exact(3).zip(right.chunks_exact(3)).enumerate() {
        let left_value = (left[0] * STD[0] + MEAN[0])
            .max(left[1] * STD[1] + MEAN[1])
            .max(left[2] * STD[2] + MEAN[2]);
        let right_value = (right[0] * STD[0] + MEAN[0])
            .max(right[1] * STD[1] + MEAN[1])
            .max(right[2] * STD[2] + MEAN[2]);
        delta[index] = (left_value - right_value) / VALUE_STD;
        output[index * 3] = ((left_value + right_value) * 0.5 - VALUE_MEAN) / VALUE_STD;
        output[index * 3 + 1] = delta[index].abs();
    }
    integral.fill(0.0);
    let integral_w = width + 1;
    for y in 0..height {
        let mut row_sum = 0.0f32;
        for x in 0..width {
            let pixel = y * width + x;
            let above = y * integral_w + x + 1;
            let target = (y + 1) * integral_w + x + 1;
            row_sum += delta[pixel];
            integral[target] = integral[above] + row_sum;
        }
    }
    for y in 0..height {
        let y0 = y.saturating_sub(2);
        let y1 = (y + 3).min(height);
        for x in 0..width {
            let x0 = x.saturating_sub(2);
            let x1 = (x + 3).min(width);
            let count = ((y1 - y0) * (x1 - x0)) as f32;
            let pixel = y * width + x;
            let destination = pixel * 3;
            let a = y1 * integral_w + x1;
            let b = y0 * integral_w + x1;
            let c = y1 * integral_w + x0;
            let d = y0 * integral_w + x0;
            let background = (integral[a] - integral[b] - integral[c] + integral[d]) / count;
            output[destination + 2] = (delta[pixel] - background).abs();
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
    _kind: KernelKind,
    conv: &DenseConv,
    input: &[f32],
    input_h: usize,
    input_w: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(conv.out_channels, 1);
    dense_scalar::<false>(conv, input, input_h, input_w, output)
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
    height: usize,
    width: usize,
) {
    const CHANNELS: usize = 48;
    debug_assert_eq!(input.len(), height * width * CHANNELS);
    debug_assert_eq!(mask_weight.len(), CHANNELS);
    debug_assert_eq!(width_sum.len(), width * CHANNELS);
    debug_assert_eq!(output.len(), CHANNELS * 3);
    output[..CHANNELS].fill(0.0);
    output[CHANNELS..CHANNELS * 2].fill(f32::NEG_INFINITY);
    width_sum.fill(0.0);
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * CHANNELS;
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
        output[channel] /= (height * width) as f32;
        let mut width_peak = f32::NEG_INFINITY;
        for x in 0..width {
            width_peak = width_peak.max(width_sum[x * CHANNELS + channel] / height as f32);
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
    for (value, weight) in input.iter().zip(linear.weights.iter()) {
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
            input_h: usize,
            input_w: usize,
            padded: &mut [f32],
            output: &mut [f32],
        ) {
            const CHANNELS: usize = 32;
            let padded_w = input_w + 4;
            debug_assert_eq!(conv.in_channels, CHANNELS);
            debug_assert_eq!(conv.out_channels, 1);
            debug_assert_eq!(input.len(), input_h * input_w * CHANNELS);
            debug_assert_eq!(padded.len(), (input_h + 4) * padded_w * CHANNELS);
            debug_assert_eq!(output.len(), input_h * input_w);
            padded.fill(0.0);
            for y in 0..input_h {
                let source = &input[y * input_w * CHANNELS..(y + 1) * input_w * CHANNELS];
                let destination_start = ((y + 2) * padded_w + 2) * CHANNELS;
                padded[destination_start..destination_start + source.len()].copy_from_slice(source);
            }
            unsafe {
                for y in 0..input_h {
                    for x in 0..input_w {
                        let mut sum0 = vdupq_n_f32(0.0);
                        let mut sum1 = vdupq_n_f32(0.0);
                        let mut sum2 = vdupq_n_f32(0.0);
                        let mut sum3 = vdupq_n_f32(0.0);
                        let mut sum4 = vdupq_n_f32(0.0);
                        let mut sum5 = vdupq_n_f32(0.0);
                        let mut sum6 = vdupq_n_f32(0.0);
                        let mut sum7 = vdupq_n_f32(0.0);
                        let mut sum8 = vdupq_n_f32(0.0);
                        let input0 = padded.as_ptr().add((y * padded_w + x) * CHANNELS);
                        let input1 = input0.add(2 * CHANNELS);
                        let input2 = input0.add(4 * CHANNELS);
                        let input3 = input0.add(2 * padded_w * CHANNELS);
                        let input4 = input3.add(2 * CHANNELS);
                        let input5 = input3.add(4 * CHANNELS);
                        let input6 = input0.add(4 * padded_w * CHANNELS);
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
                        output[y * input_w + x] = conv.bias[0] + vaddvq_f32(sum);
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
