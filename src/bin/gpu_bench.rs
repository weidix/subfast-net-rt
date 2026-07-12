use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::backend::gpu::GpuBackend;
use subfast_net_rt::ir::tensor::DType;
use subfast_net_rt::model::UnifiedModel;
use subfast_net_rt::tensor::Tensor;

#[derive(Debug, Parser)]
#[command(about = "Benchmark Metal/Vulkan inference against the CPU batch backend")]
struct Args {
    #[arg(long, value_name = "MODEL_DIR")]
    model_path: PathBuf,
    #[arg(long, default_value_t = 200)]
    warmup: usize,
    #[arg(long, default_value_t = 2_000)]
    iterations: usize,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32")]
    batch_sizes: Vec<usize>,
    #[arg(long, default_value_t = 6)]
    rounds: usize,
}

#[derive(Default)]
struct Totals {
    cpu: Duration,
    gpu: Duration,
    calls: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> subfast_net_rt::Result<()> {
    let args = Args::parse();
    if args.iterations == 0
        || args.rounds == 0
        || args.batch_sizes.is_empty()
        || args.batch_sizes.contains(&0)
    {
        return Err(subfast_net_rt::RtError::Inference(
            "iterations, rounds, and batch sizes must be positive".into(),
        ));
    }

    let model = UnifiedModel::load(
        args.model_path.join("model.json"),
        args.model_path.join("weights.bin"),
    )?;
    let max_batch = *args.batch_sizes.iter().max().expect("validated above");
    let mut cpu = CpuBackend.compile(&model)?;
    let mut gpu = GpuBackend::with_max_batch_size(max_batch)?.compile(&model)?;

    println!("backend: {}", gpu.backend_name());
    println!("adapter: {}", gpu.adapter_name());
    println!(
        "model: {}",
        if model.inputs.len() == 1 {
            "presence"
        } else {
            "pair_matcher"
        }
    );
    println!("cpu_kernel: {}", cpu.kernel_name());

    for batch_size in args.batch_sizes {
        let inputs = make_inputs(&model, batch_size)?;
        let mut cpu_output = vec![0.0; batch_size];
        let mut gpu_output = vec![0.0; batch_size];
        cpu.run_batch_scalar_into(&inputs, &mut cpu_output)?;
        gpu.run_batch_scalar_into(&inputs, &mut gpu_output)?;
        verify(&cpu_output, &gpu_output)?;

        let warmup_calls = args.warmup.div_ceil(batch_size);
        for _ in 0..warmup_calls {
            cpu.run_batch_scalar_into(black_box(&inputs), &mut cpu_output)?;
            gpu.run_batch_scalar_into(black_box(&inputs), &mut gpu_output)?;
        }

        let totals = measure(
            &mut cpu,
            &mut gpu,
            &inputs,
            &mut cpu_output,
            &mut gpu_output,
            args.iterations,
            args.rounds,
        )?;
        let samples = totals.calls * batch_size;
        let cpu_rate = samples as f64 / totals.cpu.as_secs_f64();
        let gpu_rate = samples as f64 / totals.gpu.as_secs_f64();
        let cpu_call_ms = totals.cpu.as_secs_f64() * 1_000.0 / totals.calls as f64;
        let gpu_call_ms = totals.gpu.as_secs_f64() * 1_000.0 / totals.calls as f64;
        println!(
            "batch={batch_size} samples={samples} cpu_ms={cpu_call_ms:.4} gpu_ms={gpu_call_ms:.4} cpu_samples_per_second={cpu_rate:.0} gpu_samples_per_second={gpu_rate:.0} gpu_vs_cpu={:.3}x",
            gpu_rate / cpu_rate
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn measure(
    cpu: &mut subfast_net_rt::backend::cpu::CpuExecutable,
    gpu: &mut subfast_net_rt::backend::gpu::GpuExecutable,
    inputs: &[Tensor],
    cpu_output: &mut [f32],
    gpu_output: &mut [f32],
    target_samples: usize,
    rounds: usize,
) -> subfast_net_rt::Result<Totals> {
    let batch_size = cpu_output.len();
    let calls = target_samples.div_ceil(batch_size);
    let mut totals = Totals {
        calls,
        ..Totals::default()
    };
    for round in 0..rounds {
        let round_calls = calls / rounds + usize::from(round < calls % rounds);
        if round.is_multiple_of(2) {
            totals.cpu += time_cpu(cpu, inputs, cpu_output, round_calls)?;
            totals.gpu += time_gpu(gpu, inputs, gpu_output, round_calls)?;
        } else {
            totals.gpu += time_gpu(gpu, inputs, gpu_output, round_calls)?;
            totals.cpu += time_cpu(cpu, inputs, cpu_output, round_calls)?;
        }
    }
    Ok(totals)
}

fn time_cpu(
    executable: &mut subfast_net_rt::backend::cpu::CpuExecutable,
    inputs: &[Tensor],
    output: &mut [f32],
    calls: usize,
) -> subfast_net_rt::Result<Duration> {
    let start = Instant::now();
    for _ in 0..calls {
        executable.run_batch_scalar_into(black_box(inputs), output)?;
        black_box(&output);
    }
    Ok(start.elapsed())
}

fn time_gpu(
    executable: &mut subfast_net_rt::backend::gpu::GpuExecutable,
    inputs: &[Tensor],
    output: &mut [f32],
    calls: usize,
) -> subfast_net_rt::Result<Duration> {
    let start = Instant::now();
    for _ in 0..calls {
        executable.run_batch_scalar_into(black_box(inputs), output)?;
        black_box(&output);
    }
    Ok(start.elapsed())
}

fn make_inputs(model: &UnifiedModel, batch_size: usize) -> subfast_net_rt::Result<Vec<Tensor>> {
    model
        .input_info()
        .into_iter()
        .enumerate()
        .map(|(input_index, info)| {
            if info.dtype != DType::Float32 || info.shape != [1, 3, 32, 256] {
                return Err(subfast_net_rt::RtError::InvalidTensor {
                    tensor: info.name.clone(),
                    message: format!("expected float32 [1, 3, 32, 256], got {:?}", info.shape),
                });
            }
            let elements = batch_size * info.shape[1..].iter().product::<usize>();
            let mut state = 0x9e37_79b9u32 ^ (input_index as u32).wrapping_mul(0x85eb_ca6b);
            let values = (0..elements)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
                })
                .collect();
            Tensor::from_f32(vec![batch_size, 3, 32, 256], values)
        })
        .collect()
}

fn verify(cpu: &[f32], gpu: &[f32]) -> subfast_net_rt::Result<()> {
    for (index, (&cpu, &gpu)) in cpu.iter().zip(gpu).enumerate() {
        let tolerance = 5e-4f32.max(cpu.abs() * 2e-4);
        if !gpu.is_finite() || (cpu - gpu).abs() > tolerance {
            return Err(subfast_net_rt::RtError::Inference(format!(
                "GPU output[{index}] {gpu} differs from CPU {cpu}"
            )));
        }
    }
    Ok(())
}
