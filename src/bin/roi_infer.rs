use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::Serialize;
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::Executable;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::image_infer::{ImagePreprocessor, preprocessor_for_model};
use subfast_net_rt::model::ModelType;

const DEFAULT_DECISION_THRESHOLD: f32 = 0.5;

#[derive(Debug, Parser)]
#[command(about = "Run ROI presence, pair-matcher, or frame-presence inference")]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    image: PathBuf,
    #[arg(long, help = "Second ROI image; required by pair-matcher models")]
    right_image: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    backend: BackendKind,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendKind {
    Cpu,
    Gpu,
}

struct LoadedModel {
    executable: Box<dyn Executable>,
    input_shapes: Vec<Vec<usize>>,
    preprocessor: ImagePreprocessor,
    label: &'static str,
    decision_threshold: f32,
}

impl LoadedModel {
    fn load(path: &std::path::Path, backend: BackendKind) -> subfast_net_rt::Result<Self> {
        ensure_backend_available(backend)?;
        let model = subfast_net_rt::model::Model::load(path)?;
        let input_shapes = model
            .input_info()
            .iter()
            .map(|info| info.shape.clone())
            .collect();
        let preprocessor = preprocessor_for_model(&model)?;
        let label = match model.model_type {
            ModelType::RoiPresence => "presence",
            ModelType::RoiPairMatcher => "pair_matcher",
            ModelType::FramePresence => "frame_presence",
        };
        let decision_threshold = model
            .decision_threshold
            .unwrap_or(DEFAULT_DECISION_THRESHOLD);
        let executable: Box<dyn Executable> = match backend {
            BackendKind::Cpu => Box::new(CpuBackend.compile(&model)?),
            BackendKind::Gpu => compile_gpu(&model)?,
        };
        Ok(Self {
            executable,
            input_shapes,
            preprocessor,
            label,
            decision_threshold,
        })
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.input_shapes.clone()
    }

    fn preprocess(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> subfast_net_rt::Result<subfast_net_rt::tensor::Tensor> {
        let mut tensor = self.preprocessor.new_output_tensor();
        self.preprocessor
            .preprocess_rgb_into(rgb, width, height, &mut tensor)?;
        Ok(tensor)
    }

    fn run(
        &mut self,
        inputs: &[subfast_net_rt::tensor::Tensor],
    ) -> subfast_net_rt::Result<subfast_net_rt::tensor::Tensor> {
        let mut outputs = self.executable.run(inputs)?;
        if outputs.len() != 1 {
            return Err(subfast_net_rt::RtError::Inference(format!(
                "ROI model must return one tensor, got {}",
                outputs.len()
            )));
        }
        Ok(outputs.remove(0))
    }
}

#[cfg(feature = "gpu")]
fn ensure_backend_available(_backend: BackendKind) -> subfast_net_rt::Result<()> {
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn ensure_backend_available(backend: BackendKind) -> subfast_net_rt::Result<()> {
    if matches!(backend, BackendKind::Gpu) {
        return Err(gpu_feature_error());
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn compile_gpu(
    model: &subfast_net_rt::model::Model,
) -> subfast_net_rt::Result<Box<dyn Executable>> {
    use subfast_net_rt::backend::gpu::GpuBackend;

    Ok(Box::new(GpuBackend::default().compile(model)?))
}

#[cfg(not(feature = "gpu"))]
fn compile_gpu(
    _model: &subfast_net_rt::model::Model,
) -> subfast_net_rt::Result<Box<dyn Executable>> {
    Err(gpu_feature_error())
}

#[cfg(not(feature = "gpu"))]
fn gpu_feature_error() -> subfast_net_rt::RtError {
    subfast_net_rt::RtError::Inference(
        "GPU backend requested, but roi-infer was built without the `gpu` feature; rebuild with `--features gpu`"
            .into(),
    )
}

#[derive(Serialize)]
struct ResultRecord<'a> {
    model: &'a str,
    inputs: Vec<String>,
    logit: f32,
    probability: f32,
    decision_threshold: f32,
    prediction: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> subfast_net_rt::Result<()> {
    let args = Args::parse();
    let mut model = LoadedModel::load(&args.model_path, args.backend)?;
    let input_shapes = model.input_shapes();
    let image_paths = match input_shapes.len() {
        1 if args.right_image.is_none() => vec![args.image],
        2 => vec![
            args.image,
            args.right_image.ok_or_else(|| {
                subfast_net_rt::RtError::Inference(
                    "pair-matcher model requires --right-image".into(),
                )
            })?,
        ],
        1 => {
            return Err(subfast_net_rt::RtError::Inference(
                "presence model accepts only --image".into(),
            ));
        }
        count => {
            return Err(subfast_net_rt::RtError::Inference(format!(
                "ROI inference supports one or two model inputs, got {count}"
            )));
        }
    };
    let inputs = image_paths
        .iter()
        .map(|path| {
            let image = image::open(path).map_err(|source| {
                subfast_net_rt::RtError::Inference(format!(
                    "failed to decode image {}: {source}",
                    path.display()
                ))
            })?;
            let rgb = image.to_rgb8();
            model.preprocess(rgb.as_raw(), rgb.width(), rgb.height())
        })
        .collect::<subfast_net_rt::Result<Vec<_>>>()?;
    let output = model.run(&inputs)?;
    let values = output.f32_values()?;
    if values.len() != 1 {
        return Err(subfast_net_rt::RtError::Inference(format!(
            "ROI model must return one logit, got shape {:?}",
            output.shape
        )));
    }
    let logit = values[0];
    let probability = 1.0 / (1.0 + (-logit).exp());
    let record = ResultRecord {
        model: model.label,
        inputs: image_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        logit,
        probability,
        decision_threshold: model.decision_threshold,
        prediction: probability >= model.decision_threshold,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&record)
            .map_err(|error| subfast_net_rt::RtError::Json(error.to_string()))?
    );
    Ok(())
}
