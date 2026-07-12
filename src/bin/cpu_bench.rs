use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::ir::tensor::DType;
use subfast_net_rt::model::UnifiedModel;
use subfast_net_rt::tensor::Tensor;

#[derive(Debug, Parser)]
#[command(about = "Benchmark warm, batch-1 pure Rust CPU inference")]
struct Args {
    #[arg(long, value_name = "MODEL_DIR")]
    model_path: PathBuf,
    #[arg(long, default_value_t = 2_000)]
    warmup: usize,
    #[arg(long, default_value_t = 10_000)]
    iterations: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> subfast_net_rt::Result<()> {
    let args = Args::parse();
    if args.iterations == 0 {
        return Err(subfast_net_rt::RtError::Inference(
            "--iterations must be greater than zero".into(),
        ));
    }

    let model = UnifiedModel::load(
        args.model_path.join("model.json"),
        args.model_path.join("weights.bin"),
    )?;
    let inputs = make_inputs(&model)?;
    let mut executable = CpuBackend.compile(&model)?;

    for _ in 0..args.warmup {
        let output = executable.run_scalar(black_box(&inputs))?;
        black_box(output);
    }

    let mut timings = Vec::with_capacity(args.iterations);
    let mut last_output = 0.0f32;
    for _ in 0..args.iterations {
        let start = Instant::now();
        last_output = executable.run_scalar(black_box(&inputs))?;
        let elapsed = start.elapsed();
        black_box(last_output);
        timings.push(elapsed);
    }

    let total: Duration = timings.iter().copied().sum();
    timings.sort_unstable();
    let average_ms = total.as_secs_f64() * 1_000.0 / args.iterations as f64;
    let median_ms = median(&timings).as_secs_f64() * 1_000.0;
    let p90_ms = percentile_nearest_rank(&timings, 90).as_secs_f64() * 1_000.0;
    let throughput = args.iterations as f64 / total.as_secs_f64();

    println!("backend: pure-rust-cpu");
    println!("kernel: {}", executable.kernel_name());
    println!(
        "model: {}",
        if inputs.len() == 1 {
            "presence"
        } else {
            "pair_matcher"
        }
    );
    println!("model_path: {}", args.model_path.display());
    println!("inputs: {}", inputs.len());
    println!("warmup: {}", args.warmup);
    println!("iterations: {}", args.iterations);
    println!("last_output: {last_output:.9}");
    println!("average_ms: {average_ms:.4}");
    println!("median_ms: {median_ms:.4}");
    println!("p90_ms: {p90_ms:.4}");
    println!("throughput_per_second: {throughput:.0}");
    Ok(())
}

fn make_inputs(model: &UnifiedModel) -> subfast_net_rt::Result<Vec<Tensor>> {
    model
        .input_info()
        .into_iter()
        .enumerate()
        .map(|(input_index, info)| {
            if info.dtype != DType::Float32 {
                return Err(subfast_net_rt::RtError::InvalidTensor {
                    tensor: info.name.clone(),
                    message: format!("CPU benchmark requires float32 input, got {}", info.dtype),
                });
            }
            let element_count = info.shape.iter().copied().product();
            let mut state = 0x9e37_79b9u32 ^ (input_index as u32).wrapping_mul(0x85eb_ca6b);
            let values = (0..element_count)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let unit = (state >> 8) as f32 * (1.0 / 16_777_216.0);
                    unit * 4.0 - 2.0
                })
                .collect();
            Tensor::from_f32(info.shape.clone(), values)
        })
        .collect()
}

fn median(sorted: &[Duration]) -> Duration {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2
    } else {
        sorted[middle]
    }
}

fn percentile_nearest_rank(sorted: &[Duration], percentile: usize) -> Duration {
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}
