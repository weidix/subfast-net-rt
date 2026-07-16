use std::borrow::Cow;
use std::num::NonZeroU64;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};

use wgpu_hal as hal;
use wgpu_types as wgt;

use hal::{Adapter as _, CommandEncoder as _, Device as _, Instance as _, Queue as _};

use crate::error::{Result, RtError};

#[cfg(target_os = "macos")]
type Api = hal::api::Metal;
#[cfg(not(target_os = "macos"))]
type Api = hal::api::Vulkan;

type RawInstance = <Api as hal::Api>::Instance;
type RawAdapter = <Api as hal::Api>::Adapter;
type RawDevice = <Api as hal::Api>::Device;
type RawQueue = <Api as hal::Api>::Queue;
type RawBuffer = <Api as hal::Api>::Buffer;
type RawBindGroup = <Api as hal::Api>::BindGroup;
type RawBindGroupLayout = <Api as hal::Api>::BindGroupLayout;
type RawPipelineLayout = <Api as hal::Api>::PipelineLayout;
type RawShaderModule = <Api as hal::Api>::ShaderModule;
type RawComputePipeline = <Api as hal::Api>::ComputePipeline;
type RawCommandEncoder = <Api as hal::Api>::CommandEncoder;
type RawCommandBuffer = <Api as hal::Api>::CommandBuffer;
type RawFence = <Api as hal::Api>::Fence;

pub(crate) use wgt::BufferUses;

struct Shared {
    // Drop in dependency order: queue/device, then adapter/instance.
    queue: RawQueue,
    device: RawDevice,
    _adapter: RawAdapter,
    _instance: RawInstance,
    limits: wgt::Limits,
    features: wgt::Features,
}

#[derive(Clone)]
pub(crate) struct Device {
    shared: Arc<Shared>,
}

pub(crate) struct AdapterInfo {
    pub(crate) name: String,
}

impl Device {
    pub(crate) fn new() -> Result<(Self, AdapterInfo)> {
        let descriptor = hal::InstanceDescriptor {
            name: "subfast-net-rt",
            flags: wgt::InstanceFlags::from_build_config().with_env(),
            memory_budget_thresholds: wgt::MemoryBudgetThresholds::default(),
            backend_options: wgt::BackendOptions::default(),
            telemetry: None,
            display: None,
        };
        // SAFETY: the selected backend is fixed by the target and the descriptor
        // contains no borrowed display handle.
        let instance = unsafe { RawInstance::init(&descriptor) }.map_err(|error| {
            RtError::Inference(format!("failed to initialize {}: {error}", backend_name()))
        })?;
        // SAFETY: no surface compatibility is required for this compute-only device.
        let mut adapters = unsafe { instance.enumerate_adapters(None) };
        let selected = adapters
            .iter()
            .enumerate()
            .max_by_key(|(_, exposed)| adapter_score(exposed.info.device_type))
            .map(|(index, _)| index)
            .ok_or_else(|| {
                RtError::Inference(format!("{} GPU adapter is unavailable", backend_name()))
            })?;
        let exposed = adapters.swap_remove(selected);
        let name = exposed.info.name.clone();
        let limits = exposed.capabilities.limits.clone();
        let features = exposed.features & wgt::Features::SHADER_F16;
        // SAFETY: only an adapter-advertised optional feature is requested, and
        // default limits are supported by every exposed adapter from wgpu-hal.
        let hal::OpenDevice { device, queue } = unsafe {
            exposed.adapter.open(
                features,
                &wgt::Limits::default(),
                &wgt::MemoryHints::Performance,
            )
        }
        .map_err(|error| RtError::Inference(format!("failed to create GPU device: {error}")))?;
        Ok((
            Self {
                shared: Arc::new(Shared {
                    queue,
                    device,
                    _adapter: exposed.adapter,
                    _instance: instance,
                    limits,
                    features,
                }),
            },
            AdapterInfo { name },
        ))
    }

    pub(crate) fn limits(&self) -> &wgt::Limits {
        &self.shared.limits
    }

    pub(crate) fn supports_f16(&self) -> bool {
        self.shared.features.contains(wgt::Features::SHADER_F16)
    }

    pub(crate) fn create_buffer(
        &self,
        label: &'static str,
        size: u64,
        usage: BufferUses,
    ) -> Result<Buffer> {
        if size == 0 {
            return Err(RtError::Inference(format!(
                "GPU buffer {label} must not be empty"
            )));
        }
        let descriptor = hal::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            memory_flags: if usage.intersects(BufferUses::MAP_READ | BufferUses::MAP_WRITE) {
                hal::MemoryFlags::PREFER_COHERENT
            } else {
                hal::MemoryFlags::empty()
            },
        };
        // SAFETY: size is non-zero and every later use is included in `usage`.
        let raw = unsafe { self.shared.device.create_buffer(&descriptor) }.map_err(|error| {
            RtError::Inference(format!("failed to create GPU buffer {label}: {error}"))
        })?;
        let mapping = if usage.intersects(BufferUses::MAP_READ | BufferUses::MAP_WRITE) {
            // SAFETY: the descriptor includes a mapping use and the full range is valid.
            match unsafe { self.shared.device.map_buffer(&raw, 0..size) } {
                Ok(mapping) => Some(Mapping {
                    ptr: mapping.ptr.as_ptr() as usize,
                    is_coherent: mapping.is_coherent,
                }),
                Err(error) => {
                    // SAFETY: the newly created buffer has never been submitted.
                    unsafe { self.shared.device.destroy_buffer(raw) };
                    return Err(RtError::Inference(format!(
                        "failed to map GPU buffer {label}: {error}"
                    )));
                }
            }
        } else {
            None
        };
        Ok(Buffer(Arc::new(BufferInner {
            shared: Arc::clone(&self.shared),
            raw: Some(raw),
            size,
            usage,
            mapping,
            state: AtomicU16::new(BufferUses::empty().bits()),
        })))
    }

    pub(crate) fn create_buffer_init(
        &self,
        label: &'static str,
        contents: &[u8],
        usage: BufferUses,
    ) -> Result<Buffer> {
        let size = u64::try_from(contents.len())
            .map_err(|_| RtError::Inference("GPU buffer size overflow".into()))?;
        let target = self.create_buffer(label, size, usage | BufferUses::COPY_DST)?;
        let staging = self.create_buffer(
            "subfast GPU upload staging",
            size,
            BufferUses::MAP_WRITE | BufferUses::COPY_SRC,
        )?;
        staging.write(contents)?;
        let descriptor = hal::CommandEncoderDescriptor {
            label: Some("subfast GPU resource upload"),
            queue: &self.shared.queue,
        };
        // SAFETY: queue and device came from the same OpenDevice.
        let mut encoder = unsafe { self.shared.device.create_command_encoder(&descriptor) }
            .map_err(|error| {
                RtError::Inference(format!("failed to create GPU encoder: {error}"))
            })?;
        let mut fence = Fence::new(Arc::clone(&self.shared))?;
        // SAFETY: both buffers include the declared copy uses, are live, and the
        // complete copy range is valid and initialized.
        let command = unsafe {
            encoder
                .begin_encoding(Some("subfast GPU resource upload"))
                .map_err(|error| {
                    RtError::Inference(format!("failed to begin GPU upload: {error}"))
                })?;
            encoder.transition_buffers(
                [
                    hal::BufferBarrier {
                        buffer: staging.raw(),
                        usage: hal::StateTransition {
                            from: BufferUses::MAP_WRITE,
                            to: BufferUses::COPY_SRC,
                        },
                    },
                    hal::BufferBarrier {
                        buffer: target.raw(),
                        usage: hal::StateTransition {
                            from: BufferUses::empty(),
                            to: BufferUses::COPY_DST,
                        },
                    },
                ]
                .into_iter(),
            );
            encoder.copy_buffer_to_buffer(
                staging.raw(),
                target.raw(),
                std::iter::once(hal::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: wgt::BufferSize::new(size).expect("non-empty upload"),
                }),
            );
            encoder.transition_buffers(std::iter::once(hal::BufferBarrier {
                buffer: target.raw(),
                usage: hal::StateTransition {
                    from: BufferUses::COPY_DST,
                    to: usage,
                },
            }));
            encoder.end_encoding().map_err(|error| {
                RtError::Inference(format!("failed to finish GPU upload: {error}"))
            })?
        };
        // SAFETY: command, encoder, buffers, and fence stay alive through completion.
        if let Err(error) = unsafe {
            self.shared
                .queue
                .submit(&[&command], &[], (fence.raw_mut(), 1))
        } {
            // SAFETY: submission failed, so the command is not in flight.
            unsafe { encoder.reset_all(std::iter::once(command)) };
            return Err(RtError::Inference(format!("GPU upload failed: {error}")));
        }
        if let Err(error) = wait_for_fence(&self.shared, fence.raw(), 1, "GPU upload") {
            // Completion could not be proven. Leak this exceptional in-flight
            // resource set rather than violating wgpu-hal's lifetime contract.
            std::mem::forget(command);
            std::mem::forget(encoder);
            std::mem::forget(fence);
            std::mem::forget(staging);
            std::mem::forget(target);
            return Err(error);
        }
        // SAFETY: submission completed, so command storage can be reclaimed.
        unsafe { encoder.reset_all(std::iter::once(command)) };
        target.set_state(usage);
        Ok(target)
    }

    pub(crate) fn create_pipeline(&self, label: &'static str, source: &str) -> Result<Pipeline> {
        let module = naga::front::wgsl::Frontend::new()
            .parse(source)
            .map_err(|error| RtError::Inference(format!("invalid GPU shader {label}: {error}")))?;
        let capabilities = if self.supports_f16() {
            naga::valid::Capabilities::SHADER_FLOAT16
        } else {
            naga::valid::Capabilities::empty()
        };
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .map_err(|error| RtError::Inference(format!("invalid GPU shader {label}: {error}")))?;
        let binding_uses = shader_buffer_bindings(label, &module)?;
        let layout_entries = binding_uses
            .iter()
            .enumerate()
            .map(|(binding, usage)| wgt::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::Buffer {
                    ty: if *usage == BufferUses::UNIFORM {
                        wgt::BufferBindingType::Uniform
                    } else {
                        wgt::BufferBindingType::Storage {
                            read_only: *usage == BufferUses::STORAGE_READ_ONLY,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let bgl_descriptor = hal::BindGroupLayoutDescriptor {
            label: Some(label),
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &layout_entries,
        };
        // SAFETY: entries are sorted, unique, and derived from the validated shader.
        let bind_group_layout =
            unsafe { self.shared.device.create_bind_group_layout(&bgl_descriptor) }.map_err(
                |error| RtError::Inference(format!("failed to create GPU layout {label}: {error}")),
            )?;
        let pipeline_layout_descriptor = hal::PipelineLayoutDescriptor {
            label: Some(label),
            flags: hal::PipelineLayoutFlags::empty(),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        };
        // SAFETY: the single bind group layout is alive for the call.
        let pipeline_layout = match unsafe {
            self.shared
                .device
                .create_pipeline_layout(&pipeline_layout_descriptor)
        } {
            Ok(layout) => layout,
            Err(error) => {
                // SAFETY: the layout has not been used by submitted work.
                unsafe {
                    self.shared
                        .device
                        .destroy_bind_group_layout(bind_group_layout)
                };
                return Err(RtError::Inference(format!(
                    "failed to create GPU pipeline layout {label}: {error}"
                )));
            }
        };
        let shader_descriptor = hal::ShaderModuleDescriptor {
            label: Some(label),
            // SAFETY: dispatch sizes and buffer extents are validated by the fixed model compiler.
            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
        };
        let naga_shader = hal::NagaShader {
            module: Cow::Owned(module),
            info,
            debug_source: None,
        };
        // SAFETY: Naga validated the module and the descriptor matches its runtime policy.
        let shader = match unsafe {
            self.shared
                .device
                .create_shader_module(&shader_descriptor, hal::ShaderInput::Naga(naga_shader))
        } {
            Ok(shader) => shader,
            Err(error) => {
                // SAFETY: these resources have not been submitted.
                unsafe {
                    self.shared.device.destroy_pipeline_layout(pipeline_layout);
                    self.shared
                        .device
                        .destroy_bind_group_layout(bind_group_layout);
                }
                return Err(RtError::Inference(format!(
                    "failed to compile GPU shader {label}: {error}"
                )));
            }
        };
        let constants = naga::back::PipelineConstants::default();
        let pipeline_descriptor = hal::ComputePipelineDescriptor {
            label: Some(label),
            layout: &pipeline_layout,
            stage: hal::ProgrammableStage {
                module: &shader,
                entry_point: "main",
                constants: &constants,
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        };
        // SAFETY: shader, layout, and entry point were validated above.
        let pipeline = match unsafe {
            self.shared
                .device
                .create_compute_pipeline(&pipeline_descriptor)
        } {
            Ok(pipeline) => pipeline,
            Err(error) => {
                // SAFETY: these resources have not been submitted.
                unsafe {
                    self.shared.device.destroy_shader_module(shader);
                    self.shared.device.destroy_pipeline_layout(pipeline_layout);
                    self.shared
                        .device
                        .destroy_bind_group_layout(bind_group_layout);
                }
                return Err(RtError::Inference(format!(
                    "failed to create GPU pipeline {label}: {error}"
                )));
            }
        };
        Ok(Pipeline(Arc::new(PipelineInner {
            shared: Arc::clone(&self.shared),
            bind_group_layout: Some(bind_group_layout),
            pipeline_layout: Some(pipeline_layout),
            shader: Some(shader),
            pipeline: Some(pipeline),
            binding_uses,
        })))
    }

    pub(crate) fn create_bind_group(
        &self,
        pipeline: &Pipeline,
        label: &'static str,
        buffers: &[&Buffer],
    ) -> Result<BindGroup> {
        if buffers.len() != pipeline.0.binding_uses.len() {
            return Err(RtError::Inference(format!(
                "GPU bind group {label} expected {} buffers, got {}",
                pipeline.0.binding_uses.len(),
                buffers.len()
            )));
        }
        let bindings = buffers
            .iter()
            .map(|buffer| {
                hal::BufferBinding::new_unchecked(buffer.raw(), 0, NonZeroU64::new(buffer.size()))
            })
            .collect::<Vec<_>>();
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, _)| hal::BindGroupEntry {
                binding: binding as u32,
                resource_index: binding as u32,
                count: 1,
            })
            .collect::<Vec<_>>();
        let descriptor = hal::BindGroupDescriptor {
            label: Some(label),
            layout: pipeline.bind_group_layout(),
            buffers: &bindings,
            samplers: &[],
            textures: &[],
            entries: &entries,
            acceleration_structures: &[],
            external_textures: &[],
        };
        // SAFETY: entries exactly match the pipeline layout and all bound ranges
        // cover their complete, non-zero buffers.
        let raw =
            unsafe { self.shared.device.create_bind_group(&descriptor) }.map_err(|error| {
                RtError::Inference(format!("failed to create GPU bind group {label}: {error}"))
            })?;
        Ok(BindGroup(Arc::new(BindGroupInner {
            shared: Arc::clone(&self.shared),
            raw: Some(raw),
            pipeline: pipeline.clone(),
            buffers: buffers.iter().map(|buffer| (*buffer).clone()).collect(),
        })))
    }

    pub(crate) fn create_runner(&self) -> Result<Runner> {
        Runner::new(Arc::clone(&self.shared))
    }
}

fn adapter_score(device_type: wgt::DeviceType) -> u8 {
    match device_type {
        wgt::DeviceType::DiscreteGpu => 5,
        wgt::DeviceType::IntegratedGpu => 4,
        wgt::DeviceType::VirtualGpu => 3,
        wgt::DeviceType::Other => 2,
        wgt::DeviceType::Cpu => 1,
    }
}

#[cfg(target_os = "macos")]
pub(crate) const fn backend_name() -> &'static str {
    "metal"
}

#[cfg(not(target_os = "macos"))]
pub(crate) const fn backend_name() -> &'static str {
    "vulkan"
}

fn shader_buffer_bindings(label: &str, module: &naga::Module) -> Result<Vec<BufferUses>> {
    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| {
            let binding = variable.binding.as_ref()?;
            if binding.group != 0 {
                return Some(Err(RtError::Inference(format!(
                    "GPU shader {label} uses unsupported bind group {}",
                    binding.group
                ))));
            }
            let usage = match variable.space {
                naga::AddressSpace::Uniform => BufferUses::UNIFORM,
                naga::AddressSpace::Storage { access } => {
                    if access.contains(naga::StorageAccess::STORE) {
                        BufferUses::STORAGE_READ_WRITE
                    } else {
                        BufferUses::STORAGE_READ_ONLY
                    }
                }
                _ => {
                    return Some(Err(RtError::Inference(format!(
                        "GPU shader {label} binding {} is not a buffer",
                        binding.binding
                    ))));
                }
            };
            Some(Ok((binding.binding, usage)))
        })
        .collect::<Result<Vec<_>>>()?;
    bindings.sort_unstable_by_key(|(binding, _)| *binding);
    for (index, (binding, _)) in bindings.iter().enumerate() {
        if *binding != index as u32 {
            return Err(RtError::Inference(format!(
                "GPU shader {label} has non-sequential buffer bindings"
            )));
        }
    }
    Ok(bindings.into_iter().map(|(_, usage)| usage).collect())
}

#[derive(Clone)]
pub(crate) struct Buffer(Arc<BufferInner>);

struct BufferInner {
    shared: Arc<Shared>,
    raw: Option<RawBuffer>,
    size: u64,
    usage: BufferUses,
    mapping: Option<Mapping>,
    state: AtomicU16,
}

struct Mapping {
    ptr: usize,
    is_coherent: bool,
}

impl Buffer {
    fn raw(&self) -> &RawBuffer {
        self.0.raw.as_ref().expect("live HAL buffer")
    }

    pub(crate) fn size(&self) -> u64 {
        self.0.size
    }

    fn state(&self) -> BufferUses {
        BufferUses::from_bits_retain(self.0.state.load(Ordering::Relaxed))
    }

    fn set_state(&self, state: BufferUses) {
        self.0.state.store(state.bits(), Ordering::Relaxed);
    }

    pub(crate) fn write(&self, data: &[u8]) -> Result<()> {
        if !self.0.usage.contains(BufferUses::MAP_WRITE) || data.len() as u64 > self.0.size {
            return Err(RtError::Inference("invalid mapped GPU upload".into()));
        }
        let mapping = self
            .0
            .mapping
            .as_ref()
            .ok_or_else(|| RtError::Inference("GPU upload buffer is not mapped".into()))?;
        // SAFETY: mapping covers the whole buffer, `data` fits, and synchronous
        // inference ensures the GPU is no longer reading this range.
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr as *mut u8, data.len());
            if !mapping.is_coherent {
                self.0
                    .shared
                    .device
                    .flush_mapped_ranges(self.raw(), std::iter::once(0..data.len() as u64));
            }
        }
        self.set_state(BufferUses::MAP_WRITE);
        Ok(())
    }

    pub(crate) fn read_f32(&self, output: &mut [f32]) -> Result<()> {
        let byte_len = std::mem::size_of_val(output);
        if !self.0.usage.contains(BufferUses::MAP_READ) || byte_len as u64 > self.0.size {
            return Err(RtError::Inference("invalid mapped GPU readback".into()));
        }
        let mapping = self
            .0
            .mapping
            .as_ref()
            .ok_or_else(|| RtError::Inference("GPU output buffer is not mapped".into()))?;
        // SAFETY: the submission fence has completed, the mapping covers the
        // requested bytes, and `f32` output accepts their native representation.
        unsafe {
            if !mapping.is_coherent {
                self.0
                    .shared
                    .device
                    .invalidate_mapped_ranges(self.raw(), std::iter::once(0..byte_len as u64));
            }
            ptr::copy_nonoverlapping(
                mapping.ptr as *const u8,
                output.as_mut_ptr().cast::<u8>(),
                byte_len,
            );
        }
        self.set_state(BufferUses::MAP_READ);
        Ok(())
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        if self.mapping.take().is_some() {
            // SAFETY: all executions are synchronous and no submission is pending.
            unsafe { self.shared.device.unmap_buffer(self.raw.as_ref().unwrap()) };
        }
        if let Some(raw) = self.raw.take() {
            // SAFETY: the last Buffer owner is gone and no submission is pending.
            unsafe { self.shared.device.destroy_buffer(raw) };
        }
    }
}

#[derive(Clone)]
pub(crate) struct Pipeline(Arc<PipelineInner>);

struct PipelineInner {
    shared: Arc<Shared>,
    bind_group_layout: Option<RawBindGroupLayout>,
    pipeline_layout: Option<RawPipelineLayout>,
    shader: Option<RawShaderModule>,
    pipeline: Option<RawComputePipeline>,
    binding_uses: Vec<BufferUses>,
}

impl Pipeline {
    fn raw(&self) -> &RawComputePipeline {
        self.0.pipeline.as_ref().expect("live HAL pipeline")
    }

    fn layout(&self) -> &RawPipelineLayout {
        self.0.pipeline_layout.as_ref().expect("live HAL layout")
    }

    fn bind_group_layout(&self) -> &RawBindGroupLayout {
        self.0
            .bind_group_layout
            .as_ref()
            .expect("live HAL bind group layout")
    }
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        // SAFETY: pipelines are released only after synchronous submissions finish.
        unsafe {
            if let Some(pipeline) = self.pipeline.take() {
                self.shared.device.destroy_compute_pipeline(pipeline);
            }
            if let Some(shader) = self.shader.take() {
                self.shared.device.destroy_shader_module(shader);
            }
            if let Some(layout) = self.pipeline_layout.take() {
                self.shared.device.destroy_pipeline_layout(layout);
            }
            if let Some(layout) = self.bind_group_layout.take() {
                self.shared.device.destroy_bind_group_layout(layout);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct BindGroup(Arc<BindGroupInner>);

struct BindGroupInner {
    shared: Arc<Shared>,
    raw: Option<RawBindGroup>,
    pipeline: Pipeline,
    buffers: Vec<Buffer>,
}

impl BindGroup {
    fn raw(&self) -> &RawBindGroup {
        self.0.raw.as_ref().expect("live HAL bind group")
    }
}

impl Drop for BindGroupInner {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: the last group owner is gone and submissions are synchronous.
            unsafe { self.shared.device.destroy_bind_group(raw) };
        }
    }
}

struct Fence {
    shared: Arc<Shared>,
    raw: Option<RawFence>,
}

impl Fence {
    fn new(shared: Arc<Shared>) -> Result<Self> {
        // SAFETY: the fence is created and later destroyed on the same device.
        let raw = unsafe { shared.device.create_fence() }
            .map_err(|error| RtError::Inference(format!("failed to create GPU fence: {error}")))?;
        Ok(Self {
            shared,
            raw: Some(raw),
        })
    }

    fn raw(&self) -> &RawFence {
        self.raw.as_ref().expect("live HAL fence")
    }

    fn raw_mut(&mut self) -> &mut RawFence {
        self.raw.as_mut().expect("live HAL fence")
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: callers wait for every successful submission before the
            // fence owner is dropped. Failed submissions never put it in flight.
            unsafe { self.shared.device.destroy_fence(raw) };
        }
    }
}

fn wait_for_fence(
    shared: &Shared,
    fence: &RawFence,
    value: hal::FenceValue,
    operation: &str,
) -> Result<()> {
    let mut polls = 0usize;
    loop {
        // SAFETY: `value` was submitted to this device and fence before this call.
        match unsafe { shared.device.get_fence_value(fence) } {
            Ok(completed) if completed >= value => return Ok(()),
            Ok(_) => {}
            Err(poll_error) => {
                // A successful blocking fallback is also a completion proof. It
                // runs only on an exceptional path, so Metal's polling delay does
                // not affect normal inference latency.
                // SAFETY: `value` has already been submitted to this fence.
                return match unsafe { shared.device.wait(fence, value, None) } {
                    Ok(true) => Ok(()),
                    Ok(false) => Err(RtError::Inference(format!(
                        "{operation} wait unexpectedly timed out after polling failed: {poll_error}"
                    ))),
                    Err(wait_error) => Err(RtError::Inference(format!(
                        "{operation} wait failed: {poll_error}; blocking fallback failed: {wait_error}"
                    ))),
                };
            }
        }
        if polls < 32_768 {
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
        polls += 1;
    }
}

pub(crate) struct Runner {
    shared: Arc<Shared>,
    fence: Option<Fence>,
    encoder: Option<RawCommandEncoder>,
    pending_command: Option<RawCommandBuffer>,
    in_flight_groups: Vec<BindGroup>,
    next_fence_value: hal::FenceValue,
    recording: bool,
    poisoned: bool,
    abandon_in_flight: bool,
}

impl Runner {
    fn new(shared: Arc<Shared>) -> Result<Self> {
        let descriptor = hal::CommandEncoderDescriptor {
            label: Some("subfast-net-rt GPU command encoder"),
            queue: &shared.queue,
        };
        // SAFETY: queue and device came from the same OpenDevice.
        let encoder =
            unsafe { shared.device.create_command_encoder(&descriptor) }.map_err(|error| {
                RtError::Inference(format!("failed to create GPU encoder: {error}"))
            })?;
        let fence = Fence::new(Arc::clone(&shared))?;
        Ok(Self {
            shared,
            fence: Some(fence),
            encoder: Some(encoder),
            pending_command: None,
            in_flight_groups: Vec::new(),
            next_fence_value: 1,
            recording: false,
            poisoned: false,
            abandon_in_flight: false,
        })
    }

    pub(crate) fn begin(&mut self, label: &'static str) -> Result<()> {
        if self.poisoned {
            return Err(RtError::Inference(
                "GPU runner is unavailable after an earlier HAL failure".into(),
            ));
        }
        if self.next_fence_value.checked_add(1).is_none() {
            self.poisoned = true;
            return Err(RtError::Inference("GPU fence value overflow".into()));
        }
        // SAFETY: finish closes and resets the encoder before every subsequent call.
        let encoder = self.encoder.as_mut().expect("live HAL encoder");
        if let Err(error) = unsafe { encoder.begin_encoding(Some(label)) } {
            self.poisoned = true;
            return Err(RtError::Inference(format!(
                "failed to begin GPU encoding: {error}"
            )));
        }
        let descriptor = hal::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        };
        // SAFETY: the encoder is recording and no other pass is open.
        unsafe { encoder.begin_compute_pass(&descriptor) };
        self.recording = true;
        Ok(())
    }

    pub(crate) fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        group: &BindGroup,
        workgroups: [u32; 3],
    ) {
        debug_assert!(self.recording && !self.poisoned);
        debug_assert!(Arc::ptr_eq(&pipeline.0, &group.0.pipeline.0));
        self.in_flight_groups.push(group.clone());
        let mut barriers = Vec::with_capacity(group.0.buffers.len());
        for (buffer, &usage) in group.0.buffers.iter().zip(&pipeline.0.binding_uses) {
            let from = buffer.state();
            if from != usage || usage.intersects(BufferUses::EXCLUSIVE) {
                barriers.push(hal::BufferBarrier {
                    buffer: buffer.raw(),
                    usage: hal::StateTransition { from, to: usage },
                });
                buffer.set_state(usage);
            }
        }
        // SAFETY: barriers describe the tracked state of live buffers.
        let encoder = self.encoder.as_mut().expect("live HAL encoder");
        unsafe { encoder.transition_buffers(barriers.into_iter()) };
        // SAFETY: pipeline, layout, group, and dispatch extents were validated at compile time.
        unsafe {
            encoder.set_compute_pipeline(pipeline.raw());
            encoder.set_bind_group(pipeline.layout(), 0, group.raw(), &[]);
            encoder.dispatch(workgroups);
        }
    }

    pub(crate) fn finish(&mut self) -> Result<()> {
        if !self.recording {
            self.poisoned = true;
            return Err(RtError::Inference(
                "GPU runner finished without an active command recording".into(),
            ));
        }
        let encoder = self.encoder.as_mut().expect("live HAL encoder");
        // SAFETY: begin opened exactly one compute pass and the encoder is recording.
        unsafe { encoder.end_compute_pass() };
        self.recording = false;
        // SAFETY: the encoder is recording with no pass open.
        let command = match unsafe { encoder.end_encoding() } {
            Ok(command) => command,
            Err(error) => {
                self.poisoned = true;
                self.in_flight_groups.clear();
                return Err(RtError::Inference(format!(
                    "failed to finish GPU encoding: {error}"
                )));
            }
        };
        let fence_value = self.next_fence_value;
        self.next_fence_value += 1;
        // SAFETY: command, encoder, resources, and fence remain alive through wait.
        let fence = self.fence.as_mut().expect("live HAL fence");
        if let Err(error) = unsafe {
            self.shared
                .queue
                .submit(&[&command], &[], (fence.raw_mut(), fence_value))
        } {
            // SAFETY: submission failed, so the command is not in flight.
            unsafe { encoder.reset_all(std::iter::once(command)) };
            self.poisoned = true;
            self.in_flight_groups.clear();
            return Err(RtError::Inference(format!(
                "GPU submission failed: {error}"
            )));
        }
        self.pending_command = Some(command);
        if let Err(error) = wait_for_fence(&self.shared, fence.raw(), fence_value, "GPU submission")
        {
            self.poisoned = true;
            self.abandon_in_flight = true;
            return Err(error);
        }
        // SAFETY: the command completed and can be returned to its encoder pool.
        let command = self.pending_command.take().expect("submitted HAL command");
        unsafe { encoder.reset_all(std::iter::once(command)) };
        self.in_flight_groups.clear();
        Ok(())
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        if self.abandon_in_flight {
            // A driver error prevented us from proving completion. Retain every
            // object reachable by the submitted command for the process lifetime.
            if let Some(command) = self.pending_command.take() {
                std::mem::forget(command);
            }
            std::mem::forget(std::mem::take(&mut self.in_flight_groups));
            if let Some(fence) = self.fence.take() {
                std::mem::forget(fence);
            }
            if let Some(encoder) = self.encoder.take() {
                std::mem::forget(encoder);
            }
            std::mem::forget(Arc::clone(&self.shared));
            return;
        }
        if self.recording {
            // SAFETY: dropping after an internal early return is the only path here.
            let encoder = self.encoder.as_mut().expect("live HAL encoder");
            unsafe {
                encoder.end_compute_pass();
                encoder.discard_encoding();
            }
        }
    }
}
