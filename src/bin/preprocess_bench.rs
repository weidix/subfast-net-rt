use std::hint::black_box;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use subfast_net_rt::image_infer::{ImagePreprocessor, ResizeMode};

const INPUT_SHAPE: [usize; 4] = [1, 3, 32, 256];

#[derive(Debug, Parser)]
#[command(about = "Benchmark allocation-free RGB8 image preprocessing")]
struct Args {
    #[arg(long, default_value_t = 1920)]
    source_width: u32,
    #[arg(long, default_value_t = 1080)]
    source_height: u32,
    #[arg(long, value_enum, default_value = "centered")]
    mode: Mode,
    #[arg(long, default_value_t = 1_000)]
    warmup: usize,
    #[arg(long, default_value_t = 10_000)]
    iterations: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Stretch,
    TopLeft,
    Centered,
}

impl From<Mode> for ResizeMode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::Stretch => Self::Stretch,
            Mode::TopLeft => Self::LetterboxTopLeft,
            Mode::Centered => Self::LetterboxCentered,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> subfast_net_rt::Result<()> {
    let args = Args::parse();
    if args.source_width == 0 || args.source_height == 0 || args.iterations == 0 {
        return Err(subfast_net_rt::RtError::Inference(
            "source dimensions and --iterations must be greater than zero".into(),
        ));
    }

    let rgb = deterministic_rgb(args.source_width, args.source_height);
    let mut preprocessor = ImagePreprocessor::new(&INPUT_SHAPE, args.mode.into())?;
    let mut output = preprocessor.new_output_tensor();
    for _ in 0..args.warmup {
        black_box(preprocessor.preprocess_rgb_into(
            black_box(&rgb),
            args.source_width,
            args.source_height,
            black_box(&mut output),
        )?);
    }

    let mut timings = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let start = Instant::now();
        black_box(preprocessor.preprocess_rgb_into(
            black_box(&rgb),
            args.source_width,
            args.source_height,
            black_box(&mut output),
        )?);
        timings.push(start.elapsed());
    }
    timings.sort_unstable();
    let total: Duration = timings.iter().sum();
    let average_us = total.as_secs_f64() * 1_000_000.0 / args.iterations as f64;
    let p50_us = percentile(&timings, 50).as_secs_f64() * 1_000_000.0;
    let p90_us = percentile(&timings, 90).as_secs_f64() * 1_000_000.0;
    let p99_us = percentile(&timings, 99).as_secs_f64() * 1_000_000.0;

    println!("pipeline: decoded-rgb8 -> SIMD-bilinear -> normalized-f32-nchw");
    println!("source: {}x{}", args.source_width, args.source_height);
    println!("target: 256x32");
    println!("mode: {:?}", args.mode);
    println!("iterations: {}", args.iterations);
    println!("average_us: {average_us:.3}");
    println!("p50_us: {p50_us:.3}");
    println!("p90_us: {p90_us:.3}");
    println!("p99_us: {p99_us:.3}");
    println!(
        "target_0_1ms_p90: {}",
        if p90_us <= 100.0 { "met" } else { "missed" }
    );
    black_box(output);
    Ok(())
}

fn deterministic_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut state = 0x1234_5678u32;
    (0..width as usize * height as usize * 3)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

fn percentile(sorted: &[Duration], percentage: usize) -> Duration {
    sorted[(sorted.len() * percentage).div_ceil(100).saturating_sub(1)]
}
