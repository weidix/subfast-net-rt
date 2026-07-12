# subfast-net-rt

Small Rust runtime for the custom subfast-net.unified-model export format.

Current phase:

- Model loading, schema validation, typed IR normalization, and weight bounds checks are implemented.
- Presence and Pair Matcher have a dependency-free, pure Rust CPU execution path.
- AArch64 uses NEON kernels and persistent workers; x86_64 selects AVX2+FMA at runtime and retains a scalar fallback.
- CPU weights are packed into shared `Arc` storage once. Scalar and per-worker activation arenas are reused without hot-path allocation.
- Packed CPU batches use persistent whole-sample workers only when they improve throughput; small batches retain the scalar SIMD path.
- The optional pure-Rust GPU feature selects Metal on macOS and Vulkan on every other native target.

Supported ATen operators:

- aten.conv2d.default
- aten.batch_norm.default
- aten.silu.default
- aten.silu_.default
- aten.add.Tensor
- aten.upsample_bilinear2d.vec
- aten.abs.default
- aten.amax.default
- aten.mean.dim
- aten.gt.Scalar
- aten.to.dtype
- aten.avg_pool2d.default
- aten.mul.Tensor
- aten.sub.Tensor
- aten.div.Tensor
- aten.bitwise_not.default
- aten.masked_fill.Scalar
- aten.clamp_min.default
- aten.softplus.default
- aten.cat.default
- aten.sigmoid.default
- aten.linear.default
- aten.squeeze.dim

## ROI models

`roi-infer` supports the two scalar-output models exported by the sibling
Python project. It selects presence or pair-matcher inference from the model's
input count and applies the preprocessing used during training. The CLI uses
the CPU backend by default; pass `--backend gpu` with the `gpu` feature enabled
to use Metal on macOS or Vulkan on other native platforms.

~~~bash
# Unified presence probability for one ROI. The model path is the directory
# containing model.json and weights.bin.
cargo run --release --bin roi-infer -- \
  --model-path <path-to-presence-model> \
  --image <roi-image>

# Unified match probability for two ROIs.
cargo run --release --bin roi-infer -- \
  --model-path <path-to-pair-matcher-model> \
  --image <left-roi> \
  --right-image <right-roi>

# GPU inference (Metal on macOS, Vulkan on other native platforms).
cargo run --release --features gpu --bin roi-infer -- \
  --backend gpu \
  --model-path <path-to-presence-model> \
  --image <roi-image>
~~~

## Pure Rust CPU batch inference

Each input to `run_batch_scalar` is one packed float32 tensor shaped
`[batch, 3, 32, 256]`. Presence receives one tensor and Pair Matcher receives
two tensors. Outputs preserve sample order.

~~~rust
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;

let mut executable = CpuBackend.compile(&model)?;
let logits = executable.run_batch_scalar(&packed_inputs)?;
~~~

Use `run_batch_scalar_into` with a preallocated output slice to remove the last
per-call allocation.

## CPU benchmark

Benchmark scalar and packed-batch throughput on identical pre-generated inputs:

~~~bash
cargo run --release --bin cpu-bench -- \
  --model-path <path-to-model> \
  --warmup 2000 \
  --iterations 10000 \
  --batch-sizes 1,2,4,8,16,32
~~~

The benchmark verifies every batch against scalar results, alternates the two
paths across measurement rounds, and reports samples/second plus speedup. Model
loading, input generation, and preprocessing are excluded.

On a 10-core Apple M4, the current scheduler measured 1.86x/2.03x Presence
throughput at batch 16/32 and 2.68x/3.22x/3.33x Pair Matcher throughput at batch
8/16/32. Small batches stay on the scalar path and remain at scalar throughput.

## GPU inference

Enable the optional GPU backend:

~~~bash
cargo build --release --features gpu
~~~

~~~rust
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::gpu::GpuBackend;

let backend = GpuBackend::with_max_batch_size(8)?;
let mut executable = backend.compile(&model)?;
let logits = executable.run_batch_scalar(&packed_inputs)?;
let selected_backend = executable.backend_name(); // "metal" or "vulkan"
~~~

The backend compiles WGSL kernels for feature construction, convolution,
depthwise convolution, masking, pooling, and linear classification. It uploads
weights once and reuses pipelines, bind groups, activation buffers, and
readback buffers. macOS requests only Metal; every other native target requests
only Vulkan. Apart from input upload, command submission, and the final
scalar readback, every model operator executes on the GPU.

Build the CPU runtime with:

~~~bash
cargo build --release
~~~

The JSON manifest names weights.bin. The loader validates the supplied binary
content against all JSON offsets and lengths instead of relying on the filename.

Current limits:

- The pure Rust CPU compiler currently recognizes the fixed Presence and Pair Matcher export graphs with `[1, 3, 32, 256]` inputs.
- Packed batches use `[batch, 3, 32, 256]`; GPU batches cannot exceed the maximum selected at compile time.
- One executable owns mutable scratch storage; use one executable per concurrent request worker.
- Inference-mode BatchNorm only.
- FP16/FP32 float tensors for execution; int64 buffers are accepted for unused BatchNorm counters.
- Finite operator set.
- No training.
- Unknown dynamic shape is unsupported.
- Tests cover packed-batch parity, repeatability, Pair symmetry, backend
  selection, and GPU-to-CPU parity; configured real fixtures are also exercised
  when available.

Reference-output format reserved for future validation:

- test_input.bin
- expected_output.bin
- test_case.json

Default comparison tolerances:

- FP32: atol=1e-5, rtol=1e-4
- FP16: atol=1e-3, rtol=1e-2

Adding a new operator requires:

1. Parse and validate its ATen node attributes in src/model/loader.rs.
2. Add a typed variant to src/ir/op.rs.
3. Add a CPU fusion/pattern implementation.
4. Add loader and backend tests.
