# subfast-net-rt

Pure Rust inference runtime for the `subfast-net.safetensors` ROI models exported
by the sibling Python project.

The loader accepts one model directory containing exactly:

- `config.json`
- `model.safetensors`

It validates the safetensors bundle and model contract, folds Pair Matcher
Conv+BatchNorm weights for inference, and rejects unsupported formats or model
types. There is no legacy model-format fallback.

Supported models:

- ROI Presence: one `[batch, 3, H, W]` NCHW input
- ROI Pair Matcher: two identically shaped `[batch, 3, H, W]` NCHW inputs

`H` and `W` are read from `config.json` and drive preprocessing, CPU/GPU
activation storage, GPU buffers, dispatch geometry, and WGSL specialization.
Presence dimensions must be divisible by four to match its exported mask path.

Both models have optimized CPU execution. AArch64 uses NEON kernels and
persistent workers; x86_64 selects AVX2+FMA at runtime. The optional GPU feature
uses Metal on macOS and Vulkan on other native targets through `wgpu-hal`.

## Image inference

~~~bash
cargo run --release --bin roi-infer -- \
  --model-path <path-to-safetensors-directory> \
  --image <roi-image>

cargo run --release --bin roi-infer -- \
  --model-path <path-to-pair-safetensors-directory> \
  --image <left-roi> \
  --right-image <right-roi>

cargo run --release --features gpu --bin roi-infer -- \
  --backend gpu \
  --model-path <path-to-safetensors-directory> \
  --image <roi-image>
~~~

The CLI reads preprocessing dimensions from the loaded model contract. Presence
uses centered letterboxing; Pair Matcher uses stretched resizing.

## Library use

~~~rust
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::Model;

let model = Model::load("outputs/roi_presence_v3/safetensors")?;
let mut executable = CpuBackend.compile(&model)?;
let logits = executable.run_batch_scalar(&packed_inputs)?;
~~~

`run_batch_scalar_into` accepts a preallocated output slice. One executable owns
mutable scratch storage, so concurrent request workers should each own an
executable.

## Benchmarks

~~~bash
cargo run --release --bin cpu-bench -- \
  --model-path <path-to-safetensors-directory> \
  --warmup 2000 \
  --iterations 10000 \
  --batch-sizes 1,2,4,8,16,32

cargo run --release --features gpu --bin gpu-bench -- \
  --model-path <path-to-safetensors-directory> \
  --warmup 5000 \
  --iterations 50000 \
  --batch-sizes 32 \
  --rounds 10
~~~

Model loading and preprocessing are excluded from inference benchmark timings.

## Build and test

~~~bash
cargo build --release
cargo build --release --features gpu
cargo test
~~~

Set `SUBFAST_RT_PRESENCE_DIR` and `SUBFAST_RT_PAIR_DIR` to safetensors model
directories to run the real-model loader and inference tests.
