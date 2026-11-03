# subfast-net-rt

Pure Rust inference runtime for the `subfast-net.safetensors` models exported by
the sibling Python project.

The loader accepts one model directory containing exactly:

- `config.json`
- `model.safetensors`

It validates the safetensors bundle and model contract, folds Pair Matcher
Conv+BatchNorm weights for inference, and rejects unsupported formats or model
types. There is no legacy model-format fallback.

Supported models:

- ROI Presence: one `[batch, 3, H, W]` NCHW input
- ROI Pair Matcher: two identically shaped `[batch, 3, H, W]` NCHW inputs
- Frame Presence: one `[batch, 3, H, W]` NCHW input (architecture version 8)

`H` and `W` are read from `config.json` and drive preprocessing, CPU/GPU
activation storage, GPU buffers, dispatch geometry, and WGSL specialization.
Presence dimensions must be divisible by four to match its exported mask path;
frame-presence dimensions must be multiples of 16.

The CPU uses NEON on AArch64 and runtime-dispatched AVX/FMA for Frame Presence
on x86 and x86-64. Frame Presence parallelizes large convolutions across rows
for single frames and uses persistent workers across frames in packed batches.

Build with `--features gpu` for Metal on macOS and Vulkan on Windows and Linux.
Frame Presence uses FP32 on both GPU paths. Apple Silicon on macOS 12.3 or later
uses cached Metal Performance Shaders Graph plans when available. For
mid-sized batches, the upload also converts planar input to the graph's NHWC
layout. For 16 or 32 frames at 480×272, Metal can overlap upload and inference
across two parts of the batch. The WGSL path uses a reusable staging upload on
discrete Vulkan adapters so the stem reads device-local input. Windows GPU
execution requires a Vulkan driver for the selected adapter.

## Image inference

~~~bash
cargo run --release --bin roi-infer -- \
  --model-path <path-to-safetensors-directory> \
  --image <roi-image>

cargo run --release --bin roi-infer -- \
  --model-path <path-to-pair-safetensors-directory> \
  --image <left-roi> \
  --right-image <right-roi>

cargo run --release --bin roi-infer -- \
  --model-path <path-to-frame-presence-safetensors-directory> \
  --image <frame>

cargo run --release --features gpu --bin roi-infer -- \
  --backend gpu \
  --model-path <path-to-frame-presence-safetensors-directory> \
  --image <frame>
~~~

The CLI reads the preprocessing contract from the loaded model bundle. ROI
Presence uses centered letterboxing, Pair Matcher uses stretched resizing, and
Frame Presence uses proportional resizing with centered zero padding and
[0, 1] value scaling. Results report the raw logit, its sigmoid probability, the
exported decision threshold, and the resulting prediction.

## Library use

~~~rust
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::image_infer::preprocessor_for_model;
use subfast_net_rt::Model;

let model = Model::load("outputs/frame_presence_v8/safetensors")?;
let mut preprocessor = preprocessor_for_model(&model)?;
let mut executable = CpuBackend.compile(&model)?;
let logits = executable.run_batch_scalar(&packed_inputs)?;
~~~

`preprocessor_for_model` builds the RGB8 pipeline from the exported resize mode,
normalization, and canvas size. `run_batch_scalar_into` accepts a preallocated
output slice. One executable owns mutable scratch storage, so concurrent request
workers should each own an executable.

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

For a Frame Presence V8 comparison, use `gpu-bench --separate` to measure both
backends, or `gpu-bench --gpu-only` to avoid timing CPU batches before GPU batches.
Match the batch sizes, iteration counts, and warmup counts in
`scripts/bench_frame_presence_pytorch.py`. Set `PYTHONPATH` to the Python project's
`src` directory and run the script with that project's environment. Pass the
original checkpoint directory to `--checkpoint` and run once with
`--device cpu` and once with `--device mps` or `--device cuda`, depending on
the GPU. Both benchmarks use identical generated tensors. The timing includes
GPU input upload and synchronous score readback. Model loading, image decoding,
and preprocessing are outside the timed section.

### Frame Presence V8 results

Apple M4, macOS 27.2, PyTorch 2.12.1, FP32 input at 480×272. Each run
used 768 warmup frames and 4096 measured frames per batch size. PyTorch used
four CPU threads. The CPU comparison uses Rust's separate CPU phase. The
Metal ranges cover two paired GPU-only runs against PyTorch MPS.

| Batch | Rust CPU / PyTorch CPU | Rust Metal (ms/batch) | Rust Metal / PyTorch MPS |
| ---: | ---: | ---: | ---: |
| 1 | 0.406× | 0.462–0.464 | 0.428× |
| 4 | 0.259× | 0.929–0.957 | 0.458–0.472× |
| 8 | 0.204× | 1.467–1.517 | 0.465–0.471× |
| 16 | 0.128× | 2.595–2.627 | 0.469–0.493× |
| 32 | 0.108× | 5.055–5.096 | 0.489–0.516× |

Batch 32 crossed the 0.5× threshold between runs. Windows i686 and x86_64
passed cross-target compilation checks with Vulkan enabled. Windows inference
latency still requires an on-device benchmark.

## Build and test

~~~bash
cargo build --release
cargo build --release --features gpu
cargo test
~~~

Set `SUBFAST_RT_PRESENCE_DIR` and `SUBFAST_RT_PAIR_DIR` to safetensors model
directories to run the real-model loader and inference tests. Frame Presence
tests additionally read `SUBFAST_RT_FRAME_PRESENCE_DIR` and a directory of
PyTorch reference tensors through `SUBFAST_RT_FRAME_PRESENCE_CASES`. With
`--features gpu`, set `SUBFAST_RT_TEST_GPU=1` to run the GPU parity test on an
adapter.
