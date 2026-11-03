use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;

use super::{Model, Result, RtError, checked_bytes, packed_dense, scalar_parameter, softplus};
use crate::backend::frame_presence::{ENCODER_LAYERS, fold_context_projection, frame_input_shape};

#[repr(C)]
struct Layer {
    weights: *const f32,
    bias: *const f32,
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
}

unsafe extern "C" {
    fn subfast_frame_graph_create(
        layers: *const Layer,
        height: usize,
        width: usize,
        max_batch: usize,
        input_bytes: usize,
        scale: f32,
        bias: f32,
        name: *mut c_char,
        name_bytes: usize,
    ) -> *mut c_void;
    fn subfast_frame_graph_run(
        handle: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut f32,
        batch: usize,
        error: *mut c_char,
        error_bytes: usize,
    ) -> bool;
    fn subfast_frame_graph_destroy(handle: *mut c_void);
}

pub(super) struct MetalFramePresence {
    handle: NonNull<c_void>,
    pub(super) adapter_name: String,
    failed: bool,
}

// SAFETY: the native graph has exclusive mutable access via run, completes
// GPU work synchronously, and uses a fresh autorelease pool on the calling thread.
unsafe impl Send for MetalFramePresence {}
// SAFETY: shared references expose only the immutable adapter name.
unsafe impl Sync for MetalFramePresence {}

impl MetalFramePresence {
    pub(super) fn compile(model: &Model, max_batch: usize) -> Result<Option<Self>> {
        let (height, width) = frame_input_shape(model)?;
        let elements = height
            .checked_mul(width)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| RtError::Inference("frame presence input size overflow".into()))?;
        let input_bytes = usize::try_from(checked_bytes(max_batch, elements)?)
            .map_err(|_| RtError::Inference("frame presence input size overflow".into()))?;
        let mut stored = Vec::with_capacity(7);
        let mut input_channels = 3;
        for (name, output_channels, kernel, stride, dilation) in ENCODER_LAYERS.into_iter().take(5)
        {
            let (weights, bias) =
                packed_dense(model, name, input_channels, output_channels, [kernel; 2])?;
            stored.push((weights, bias, input_channels, kernel, stride, dilation));
            input_channels = output_channels;
        }
        let (head, head_bias) = packed_dense(model, "region_head", 32, 1, [1, 1])?;
        let (projection, projection_bias) =
            packed_dense(model, "context_projection", 48, 32, [1, 1])?;
        let (context, context_bias) = fold_context_projection(&projection, &projection_bias, &head);
        stored.push((context, vec![context_bias], 48, 1, 1, 1));
        stored.push((head, head_bias, 32, 1, 1, 1));
        let layers: Vec<_> = stored
            .iter()
            .map(
                |(weights, bias, channels, kernel, stride, dilation)| Layer {
                    weights: weights.as_ptr(),
                    bias: bias.as_ptr(),
                    input_channels: *channels,
                    output_channels: bias.len(),
                    kernel: *kernel,
                    stride: *stride,
                    dilation: *dilation,
                },
            )
            .collect();
        let scale = softplus(scalar_parameter(
            model,
            "local_evidence_aggregator.log_scale",
        )?) + 0.5;
        let bias = scalar_parameter(model, "local_evidence_aggregator.bias")?;
        let mut name = [0; 256];
        // SAFETY: layers describe seven validated, live parameter buffers.
        // The native constructor copies the weights before returning and owns
        // its result until subfast_frame_graph_destroy.
        let handle = unsafe {
            subfast_frame_graph_create(
                layers.as_ptr(),
                height,
                width,
                max_batch,
                input_bytes,
                scale,
                bias,
                name.as_mut_ptr(),
                name.len(),
            )
        };
        Ok(NonNull::new(handle).map(|handle| Self {
            handle,
            // SAFETY: the zero-initialized output is written with bounded snprintf.
            adapter_name: unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            failed: false,
        }))
    }

    pub(super) fn run(&mut self, input: &[u8], output: &mut [f32]) -> Result<()> {
        if self.failed {
            return Err(RtError::Inference(
                "Metal graph failed on a previous call".into(),
            ));
        }
        let mut error = [0; 512];
        // SAFETY: the caller validated the tensor shape and output length, and
        // native execution is synchronous with exclusive access to the graph.
        let ok = unsafe {
            subfast_frame_graph_run(
                self.handle.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if !ok {
            self.failed = true;
            // SAFETY: the error buffer is zero initialized and bounded by snprintf.
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            return Err(RtError::Inference(format!(
                "Metal frame presence execution failed {message}"
            )));
        }
        Ok(())
    }
}

impl Drop for MetalFramePresence {
    fn drop(&mut self) {
        // SAFETY: this handle is owned by self and released exactly once.
        unsafe { subfast_frame_graph_destroy(self.handle.as_ptr()) };
    }
}
