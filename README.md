# subfast-net-rt

Small Rust runtime for the custom subfast-net.unified-model export format.

Current phase:

- Model loading, schema validation, typed IR normalization, and weight bounds checks are implemented.
- Presence and Pair Matcher have a dependency-free, pure Rust CPU execution path.
- AArch64 uses NEON kernels and a persistent four-way worker pool; x86_64 selects AVX2+FMA at runtime and retains a scalar fallback.
- Weights are packed once, while fixed activation arenas are reused by `run_scalar` without hot-path allocation.

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
input count and applies the preprocessing used during training.

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
~~~

## Pure Rust CPU benchmark

Benchmark warm, forward-only CPU inference with fixed pre-generated inputs:

~~~bash
cargo run --release --bin cpu-bench -- \
  --model-path <path-to-model> \
  --warmup 2000 \
  --iterations 10000
~~~

The reported average, median, P90, and throughput exclude model loading, input
generation, preprocessing, and output postprocessing.

Build the CPU runtime with:

~~~bash
cargo build --release
~~~

The JSON manifest names weights.bin. The loader validates the supplied binary
content against all JSON offsets and lengths instead of relying on the filename.

Current limits:

- The pure Rust CPU compiler currently recognizes the fixed Presence and Pair Matcher export graphs with `[1, 3, 32, 256]` inputs.
- One executable owns mutable scratch storage; use one executable per concurrent request worker.
- Inference-mode BatchNorm only.
- FP16/FP32 float tensors for execution; int64 buffers are accepted for unused BatchNorm counters.
- Finite operator set.
- No training.
- Unknown dynamic shape is unsupported.
- CPU tests cover repeatability and Pair symmetry on configured real fixtures.

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
