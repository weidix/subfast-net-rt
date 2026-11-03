use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::{CpuBackend, CpuExecutable};
use subfast_net_rt::ir::tensor::DType;
use subfast_net_rt::model::Model;
use subfast_net_rt::tensor::Tensor;

#[derive(Debug, Parser)]
#[command(about = "Benchmark scalar and packed-batch pure Rust CPU inference")]
struct Args {
    #[arg(long, value_name = "MODEL_DIR")]
    model_path: PathBuf,
    /// Warmup samples per path and batch size.
    #[arg(long, default_value_t = 2_000)]
    warmup: usize,
    /// Target measured samples per path and batch size.
    #[arg(long, default_value_t = 10_000)]
    iterations: usize,
    /// Packed batch sizes, as a comma-separated list.
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32")]
    batch_sizes: Vec<usize>,
    /// Alternating sequential/packed measurement rounds.
    #[arg(long, default_value_t = 6)]
    rounds: usize,
}

struct BatchCase {
    batch_size: usize,
    samples: Vec<Vec<Tensor>>,
    packed: Vec<Tensor>,
}

#[derive(Default)]
struct ThroughputTotals {
    sequential: Duration,
    packed: Duration,
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
    validate_args(&args)?;

    let model = Model::load(&args.model_path)?;
    let mut executable = CpuBackend.compile(&model)?;
    let batch_one = make_batch_case(&model, 1)?;
    let (last_output, mut timings) = measure_batch_one_latency(
        &mut executable,
        &batch_one.samples[0],
        args.warmup,
        args.iterations,
    )?;
    let latency_total: Duration = timings.iter().copied().sum();
    timings.sort_unstable();
    let average_ms = latency_total.as_secs_f64() * 1_000.0 / args.iterations as f64;
    let median_ms = median(&timings).as_secs_f64() * 1_000.0;
    let p90_ms = percentile_nearest_rank(&timings, 90).as_secs_f64() * 1_000.0;
    let latency_throughput = args.iterations as f64 / latency_total.as_secs_f64();

    println!("backend: pure-rust-cpu");
    println!("kernel: {}", executable.kernel_name());
    println!("model: {}", model_label(&model));
    println!("model_path: {}", args.model_path.display());
    println!("inputs: {}", batch_one.samples[0].len());
    println!("batch_parallelism: {}", executable.batch_parallelism());
    println!("warmup_samples_per_path: {}", args.warmup);
    println!("measured_samples_per_path: {}", args.iterations);
    println!("rounds: {}", args.rounds);
    println!("batch_sizes: {}", format_batch_sizes(&args.batch_sizes));
    println!("batch1_last_output: {last_output:.9}");
    println!("batch1_average_ms: {average_ms:.4}");
    println!("batch1_median_ms: {median_ms:.4}");
    println!("batch1_p90_ms: {p90_ms:.4}");
    println!("batch1_throughput_per_second: {latency_throughput:.0}");

    for &batch_size in &args.batch_sizes {
        let case = if batch_size == 1 {
            make_batch_case(&model, 1)?
        } else {
            make_batch_case(&model, batch_size)?
        };
        verify_batch(&mut executable, &case)?;
        warm_paths(&mut executable, &case, args.warmup)?;
        let totals = measure_throughput(&mut executable, &case, args.iterations, args.rounds)?;
        let measured_samples = totals.calls * batch_size;
        let sequential_rate = measured_samples as f64 / totals.sequential.as_secs_f64();
        let packed_rate = measured_samples as f64 / totals.packed.as_secs_f64();
        let speedup = packed_rate / sequential_rate;
        println!(
            "batch={batch_size} samples={measured_samples} sequential_samples_per_second={sequential_rate:.0} packed_samples_per_second={packed_rate:.0} speedup={speedup:.3}x"
        );
    }
    Ok(())
}

fn validate_args(args: &Args) -> subfast_net_rt::Result<()> {
    if args.iterations == 0 {
        return Err(subfast_net_rt::RtError::Inference(
            "--iterations must be greater than zero".into(),
        ));
    }
    if args.rounds == 0 {
        return Err(subfast_net_rt::RtError::Inference(
            "--rounds must be greater than zero".into(),
        ));
    }
    if args.batch_sizes.is_empty() || args.batch_sizes.contains(&0) {
        return Err(subfast_net_rt::RtError::Inference(
            "--batch-sizes must contain only positive sizes".into(),
        ));
    }
    Ok(())
}

fn measure_batch_one_latency(
    executable: &mut CpuExecutable,
    inputs: &[Tensor],
    warmup: usize,
    iterations: usize,
) -> subfast_net_rt::Result<(f32, Vec<Duration>)> {
    for _ in 0..warmup {
        black_box(executable.run_scalar(black_box(inputs))?);
    }
    let mut timings = Vec::with_capacity(iterations);
    let mut last_output = 0.0f32;
    for _ in 0..iterations {
        let start = Instant::now();
        last_output = executable.run_scalar(black_box(inputs))?;
        timings.push(start.elapsed());
        black_box(last_output);
    }
    Ok((last_output, timings))
}

fn verify_batch(executable: &mut CpuExecutable, case: &BatchCase) -> subfast_net_rt::Result<()> {
    let expected = case
        .samples
        .iter()
        .map(|sample| executable.run_scalar(sample))
        .collect::<subfast_net_rt::Result<Vec<_>>>()?;
    let actual = executable.run_batch_scalar(&case.packed)?;
    for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
        let tolerance = 2e-5f32.max(expected.abs() * 1e-5);
        if (expected - actual).abs() > tolerance {
            return Err(subfast_net_rt::RtError::Inference(format!(
                "packed batch output[{index}] differs from scalar: {actual} vs {expected}"
            )));
        }
    }
    Ok(())
}

fn warm_paths(
    executable: &mut CpuExecutable,
    case: &BatchCase,
    warmup_samples: usize,
) -> subfast_net_rt::Result<()> {
    let calls = warmup_samples.div_ceil(case.batch_size);
    let mut packed_output = vec![0.0; case.batch_size];
    for _ in 0..calls {
        run_sequential(executable, &case.samples)?;
        executable.run_batch_scalar_into(black_box(&case.packed), &mut packed_output)?;
        black_box(&packed_output);
    }
    Ok(())
}

fn measure_throughput(
    executable: &mut CpuExecutable,
    case: &BatchCase,
    target_samples: usize,
    rounds: usize,
) -> subfast_net_rt::Result<ThroughputTotals> {
    let calls = target_samples.div_ceil(case.batch_size);
    let mut totals = ThroughputTotals {
        calls,
        ..ThroughputTotals::default()
    };
    for round in 0..rounds {
        let round_calls = calls / rounds + usize::from(round < calls % rounds);
        if round_calls == 0 {
            continue;
        }
        if round.is_multiple_of(2) {
            totals.sequential += measure_sequential(executable, &case.samples, round_calls)?;
            totals.packed += measure_packed(executable, &case.packed, round_calls)?;
        } else {
            totals.packed += measure_packed(executable, &case.packed, round_calls)?;
            totals.sequential += measure_sequential(executable, &case.samples, round_calls)?;
        }
    }
    Ok(totals)
}

fn measure_sequential(
    executable: &mut CpuExecutable,
    samples: &[Vec<Tensor>],
    calls: usize,
) -> subfast_net_rt::Result<Duration> {
    let start = Instant::now();
    for _ in 0..calls {
        run_sequential(executable, black_box(samples))?;
    }
    Ok(start.elapsed())
}

fn run_sequential(
    executable: &mut CpuExecutable,
    samples: &[Vec<Tensor>],
) -> subfast_net_rt::Result<()> {
    for sample in samples {
        black_box(executable.run_scalar(black_box(sample))?);
    }
    Ok(())
}

fn measure_packed(
    executable: &mut CpuExecutable,
    packed: &[Tensor],
    calls: usize,
) -> subfast_net_rt::Result<Duration> {
    let batch_size = packed.first().map_or(0, |input| input.shape[0]);
    let mut output = vec![0.0; batch_size];
    let start = Instant::now();
    for _ in 0..calls {
        executable.run_batch_scalar_into(black_box(packed), &mut output)?;
        black_box(&output);
    }
    Ok(start.elapsed())
}

fn make_batch_case(model: &Model, batch_size: usize) -> subfast_net_rt::Result<BatchCase> {
    let samples = (0..batch_size)
        .map(|sample_index| make_inputs(model, sample_index))
        .collect::<subfast_net_rt::Result<Vec<_>>>()?;
    let packed = pack_inputs(model, &samples)?;
    Ok(BatchCase {
        batch_size,
        samples,
        packed,
    })
}

fn make_inputs(model: &Model, sample_index: usize) -> subfast_net_rt::Result<Vec<Tensor>> {
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
            let mut state = 0x9e37_79b9u32
                ^ (input_index as u32).wrapping_mul(0x85eb_ca6b)
                ^ (sample_index as u32).wrapping_mul(0xc2b2_ae35);
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

fn pack_inputs(model: &Model, samples: &[Vec<Tensor>]) -> subfast_net_rt::Result<Vec<Tensor>> {
    model
        .input_info()
        .into_iter()
        .enumerate()
        .map(|(input_index, info)| {
            let sample_bytes = info.dtype.byte_len_for_shape(&info.shape).ok_or_else(|| {
                subfast_net_rt::RtError::InvalidTensor {
                    tensor: info.name.clone(),
                    message: "input byte length overflow".into(),
                }
            })?;
            let mut data = Vec::with_capacity(sample_bytes * samples.len());
            for sample in samples {
                data.extend_from_slice(&sample[input_index].data);
            }
            let mut shape = info.shape.clone();
            shape[0] = samples.len();
            Tensor::new(info.dtype, shape, data)
        })
        .collect()
}

fn model_label(model: &Model) -> &'static str {
    match model.model_type {
        subfast_net_rt::model::ModelType::RoiPresence => "presence",
        subfast_net_rt::model::ModelType::RoiPairMatcher => "pair_matcher",
        subfast_net_rt::model::ModelType::FramePresence => "frame_presence",
    }
}

fn format_batch_sizes(batch_sizes: &[usize]) -> String {
    batch_sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
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
