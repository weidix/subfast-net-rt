use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;
use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::Executable;
use subfast_net_rt::backend::cpu::{CpuBackend, CpuExecutable};
use subfast_net_rt::image_infer::{
    preprocess_centered_letterbox_image_path, preprocess_stretched_image_path,
};

#[derive(Debug, Parser)]
#[command(about = "Run ROI presence or pair-matcher inference")]
struct Args {
    #[arg(long)]
    model_path: PathBuf,
    #[arg(long)]
    image: PathBuf,
    #[arg(long, help = "Second ROI image; required by pair-matcher models")]
    right_image: Option<PathBuf>,
}

enum LoadedModel {
    Unified {
        executable: CpuExecutable,
        input_shapes: Vec<Vec<usize>>,
    },
}

impl LoadedModel {
    fn load(path: &std::path::Path) -> subfast_net_rt::Result<Self> {
        let model = subfast_net_rt::model::UnifiedModel::load(
            path.join("model.json"),
            path.join("weights.bin"),
        )?;
        let input_shapes = model
            .input_info()
            .iter()
            .map(|info| info.shape.clone())
            .collect();
        let executable = CpuBackend.compile(&model)?;
        Ok(Self::Unified {
            executable,
            input_shapes,
        })
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        match self {
            Self::Unified { input_shapes, .. } => input_shapes.clone(),
        }
    }

    fn run(
        &mut self,
        inputs: &[subfast_net_rt::tensor::Tensor],
    ) -> subfast_net_rt::Result<subfast_net_rt::tensor::Tensor> {
        match self {
            Self::Unified { executable, .. } => {
                let mut outputs = executable.run(inputs)?;
                if outputs.len() != 1 {
                    return Err(subfast_net_rt::RtError::Inference(format!(
                        "ROI model must return one tensor, got {}",
                        outputs.len()
                    )));
                }
                Ok(outputs.remove(0))
            }
        }
    }
}

#[derive(Serialize)]
struct ResultRecord<'a> {
    model: &'a str,
    inputs: Vec<String>,
    logit: f32,
    probability: f32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> subfast_net_rt::Result<()> {
    let args = Args::parse();
    let mut model = LoadedModel::load(&args.model_path)?;
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
        .zip(&input_shapes)
        .map(|(path, shape)| {
            if input_shapes.len() == 1 {
                preprocess_centered_letterbox_image_path(path, shape)
            } else {
                preprocess_stretched_image_path(path, shape)
            }
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
    let record = ResultRecord {
        model: if input_shapes.len() == 1 {
            "presence"
        } else {
            "pair_matcher"
        },
        inputs: image_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        logit,
        probability: 1.0 / (1.0 + (-logit).exp()),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&record)
            .map_err(|error| subfast_net_rt::RtError::Json(error.to_string()))?
    );
    Ok(())
}
