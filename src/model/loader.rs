use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype as SafeDtype, SafeTensors};

use crate::error::{Result, RtError};
use crate::ir::tensor::{DType, Layout, TensorId, TensorInfo, TensorKind, WeightRef};
use crate::model::schema::ModelConfig;
use crate::model::weights::WeightStore;

const FORMAT: &str = "subfast-net.safetensors";
const FORMAT_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    RoiPresence,
    RoiPairMatcher,
}

impl ModelType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "roi_presence" => Ok(Self::RoiPresence),
            "roi_pair_matcher" => Ok(Self::RoiPairMatcher),
            other => Err(RtError::Inference(format!(
                "unsupported safetensors model type {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub model_type: ModelType,
    pub tensors: Vec<TensorInfo>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub weights: WeightStore,
}

impl Model {
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| RtError::io(&config_path, e))?;
        let config: ModelConfig = serde_json::from_slice(&config_bytes)
            .map_err(|e| RtError::Json(format!("{}: {e}", config_path.display())))?;
        validate_config(&config)?;

        let model_type = ModelType::parse(&config.model_type)?;
        let weights_path = model_dir.join(&config.weights.file);
        let bytes = std::fs::read(&weights_path).map_err(|e| RtError::io(&weights_path, e))?;
        let (_, metadata) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| RtError::Safetensors(format!("{}: {e}", weights_path.display())))?;
        let metadata = metadata.metadata().as_ref().ok_or_else(|| {
            RtError::Safetensors("model.safetensors is missing required metadata".into())
        })?;
        for (key, expected) in [
            ("format", FORMAT),
            ("format_version", "1"),
            ("model_type", config.model_type.as_str()),
        ] {
            if metadata.get(key).map(String::as_str) != Some(expected) {
                return Err(RtError::Safetensors(format!(
                    "metadata {key} does not match config.json"
                )));
            }
        }
        let file = SafeTensors::deserialize(&bytes)
            .map_err(|e| RtError::Safetensors(format!("{}: {e}", weights_path.display())))?;
        let raw = collect_tensors(&file)?;

        let mut builder = ModelBuilder::default();
        let inputs = match model_type {
            ModelType::RoiPresence => validate_io(&config, &["images"], "presence_logit")?
                .into_iter()
                .map(|(name, shape)| builder.add_io(&name, TensorKind::UserInput, shape.to_vec()))
                .collect(),
            ModelType::RoiPairMatcher => {
                let inputs = validate_io(&config, &["left", "right"], "same_subtitle_logit")?;
                if inputs[0].1 != inputs[1].1 {
                    return Err(RtError::Json(
                        "pair matcher inputs must have identical shapes".into(),
                    ));
                }
                inputs
                    .into_iter()
                    .map(|(name, shape)| {
                        builder.add_io(&name, TensorKind::UserInput, shape.to_vec())
                    })
                    .collect()
            }
        };
        let output_name = match model_type {
            ModelType::RoiPresence => "presence_logit",
            ModelType::RoiPairMatcher => "same_subtitle_logit",
        };
        let output = builder.add_io(output_name, TensorKind::UserOutput, vec![1]);

        match model_type {
            ModelType::RoiPresence => load_presence_weights(&raw, &mut builder)?,
            ModelType::RoiPairMatcher => load_pair_weights(&raw, &mut builder)?,
        }
        Ok(Self {
            model_type,
            tensors: builder.tensors,
            inputs,
            outputs: vec![output],
            weights: WeightStore::from_bytes(builder.weights),
        })
    }

    pub fn input_info(&self) -> Vec<&TensorInfo> {
        self.inputs.iter().map(|id| &self.tensors[id.0]).collect()
    }

    pub fn output_info(&self) -> Vec<&TensorInfo> {
        self.outputs.iter().map(|id| &self.tensors[id.0]).collect()
    }
}

fn validate_config(config: &ModelConfig) -> Result<()> {
    if config.format != FORMAT || config.format_version != FORMAT_VERSION {
        return Err(RtError::FormatVersion {
            format: config.format.clone(),
            version: config.format_version,
        });
    }
    if config.weights.format != "safetensors" || config.weights.file != "model.safetensors" {
        return Err(RtError::Json(
            "weights must reference model.safetensors in safetensors format".into(),
        ));
    }
    if config.model.model_type != config.model_type {
        return Err(RtError::Json(
            "model.type does not match top-level model_type".into(),
        ));
    }
    match config.model_type.as_str() {
        "roi_presence" if config.model.architecture_version == 3 => {}
        "roi_pair_matcher"
            if config.model.architecture_version == 2
                && config.model.pooling_version == Some(2) => {}
        _ => {
            return Err(RtError::Json(format!(
                "unsupported {} architecture/pooling version",
                config.model_type
            )));
        }
    }
    Ok(())
}

fn validate_io(
    config: &ModelConfig,
    expected_inputs: &[&str],
    expected_output: &str,
) -> Result<Vec<(String, [usize; 4])>> {
    let required = config
        .input
        .tensors
        .iter()
        .filter(|tensor| !tensor.optional)
        .collect::<Vec<_>>();
    if required.len() != expected_inputs.len() {
        return Err(RtError::Json(format!(
            "expected {} required inputs, got {}",
            expected_inputs.len(),
            required.len()
        )));
    }
    let mut inputs = Vec::with_capacity(required.len());
    for (tensor, name) in required.iter().zip(expected_inputs) {
        let shape = tensor.shape.nchw().ok_or_else(|| {
            RtError::Json(format!("input {name} must have shape [batch, 3, H, W]"))
        })?;
        if tensor.name != *name || tensor.layout != "NCHW" || shape[1] != 3 {
            return Err(RtError::Json(format!(
                "input {name} must be NCHW with shape [batch, 3, H, W]"
            )));
        }
        if shape[2] == 0 || shape[3] == 0 {
            return Err(RtError::Json(format!(
                "input {name} dimensions must be non-zero"
            )));
        }
        inputs.push((tensor.name.clone(), shape));
    }
    if config
        .output
        .tensors
        .first()
        .is_none_or(|tensor| tensor.name != expected_output || !tensor.shape.is_batch_scalar())
    {
        return Err(RtError::Json(format!(
            "expected scalar output {expected_output}"
        )));
    }
    Ok(inputs)
}

struct RawTensor<'a> {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

fn collect_tensors<'a>(file: &'a SafeTensors<'a>) -> Result<HashMap<String, RawTensor<'a>>> {
    Ok(file
        .iter()
        .map(|(name, view)| {
            (
                name.to_owned(),
                RawTensor {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    data: view.data(),
                },
            )
        })
        .collect())
}

fn load_presence_weights(
    raw: &HashMap<String, RawTensor<'_>>,
    builder: &mut ModelBuilder,
) -> Result<()> {
    for (name, shape) in [
        ("backbone.0.weight", vec![16, 3, 5, 5]),
        ("backbone.0.bias", vec![16]),
        ("backbone.2.weight", vec![32, 16, 5, 5]),
        ("backbone.2.bias", vec![32]),
        ("presence_head.local.weight", vec![1, 32, 3, 3]),
        ("presence_head.local.bias", vec![1]),
        ("presence_head.evidence_pool.log_scale", vec![]),
        ("presence_head.evidence_pool.bias", vec![]),
    ] {
        builder.add_raw_f32(raw, name, &shape)?;
    }
    Ok(())
}

fn load_pair_weights(
    raw: &HashMap<String, RawTensor<'_>>,
    builder: &mut ModelBuilder,
) -> Result<()> {
    for (conv, bn, shape) in [
        ("stem.0", "stem.1", vec![16, 3, 3, 3]),
        ("downsample.0", "downsample.1", vec![32, 16, 3, 3]),
        ("detail.0", "detail.1", vec![32, 1, 3, 3]),
        ("detail.3", "detail.4", vec![32, 32, 1, 1]),
        (
            "vertical_downsample.0",
            "vertical_downsample.1",
            vec![32, 1, 3, 3],
        ),
        (
            "vertical_downsample.3",
            "vertical_downsample.4",
            vec![48, 32, 1, 1],
        ),
        ("context.0", "context.1", vec![48, 1, 3, 3]),
        ("context.3", "context.4", vec![48, 48, 1, 1]),
    ] {
        builder.add_folded_conv_bn(raw, conv, bn, &shape)?;
    }
    for (name, shape) in [
        ("text_mask.weight", vec![1, 48, 1, 1]),
        ("text_mask.bias", vec![1]),
        ("classifier.0.weight", vec![32, 144]),
        ("classifier.0.bias", vec![32]),
        ("classifier.2.weight", vec![1, 32]),
        ("classifier.2.bias", vec![1]),
    ] {
        builder.add_raw_f32(raw, name, &shape)?;
    }
    Ok(())
}

#[derive(Default)]
struct ModelBuilder {
    tensors: Vec<TensorInfo>,
    weights: Vec<u8>,
}

impl ModelBuilder {
    fn add_io(&mut self, name: &str, kind: TensorKind, shape: Vec<usize>) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(TensorInfo {
            id,
            name: name.into(),
            dtype: DType::Float32,
            shape,
            layout: Layout::DenseRowMajor,
            kind,
            target: None,
            weight: None,
        });
        id
    }

    fn add_raw_f32(
        &mut self,
        raw: &HashMap<String, RawTensor<'_>>,
        name: &str,
        shape: &[usize],
    ) -> Result<()> {
        let tensor = require_f32(raw, name, shape)?;
        self.add_parameter(name, shape.to_vec(), tensor.data);
        Ok(())
    }

    fn add_folded_conv_bn(
        &mut self,
        raw: &HashMap<String, RawTensor<'_>>,
        conv: &str,
        bn: &str,
        shape: &[usize],
    ) -> Result<()> {
        let weight = f32_values(require_f32(raw, &format!("{conv}.weight"), shape)?.data);
        let channels = shape[0];
        let gamma = f32_values(require_f32(raw, &format!("{bn}.weight"), &[channels])?.data);
        let beta = f32_values(require_f32(raw, &format!("{bn}.bias"), &[channels])?.data);
        let mean = f32_values(require_f32(raw, &format!("{bn}.running_mean"), &[channels])?.data);
        let variance =
            f32_values(require_f32(raw, &format!("{bn}.running_var"), &[channels])?.data);
        let per_channel = weight.len() / channels;
        let mut fused_weight = weight;
        let mut fused_bias = vec![0.0; channels];
        for channel in 0..channels {
            let scale = gamma[channel] / (variance[channel] + 1e-5).sqrt();
            for value in &mut fused_weight[channel * per_channel..(channel + 1) * per_channel] {
                *value *= scale;
            }
            fused_bias[channel] = beta[channel] - mean[channel] * scale;
        }
        self.add_parameter(
            &format!("{conv}.weight"),
            shape.to_vec(),
            &f32_bytes(&fused_weight),
        );
        self.add_parameter(
            &format!("{conv}.bias"),
            vec![channels],
            &f32_bytes(&fused_bias),
        );
        Ok(())
    }

    fn add_parameter(&mut self, name: &str, shape: Vec<usize>, data: &[u8]) {
        let offset = self.weights.len();
        self.weights.extend_from_slice(data);
        let id = TensorId(self.tensors.len());
        self.tensors.push(TensorInfo {
            id,
            name: name.into(),
            dtype: DType::Float32,
            shape,
            layout: Layout::DenseRowMajor,
            kind: TensorKind::Parameter,
            target: Some(name.into()),
            weight: Some(WeightRef {
                offset,
                byte_length: data.len(),
            }),
        });
    }
}

fn require_f32<'a>(
    raw: &'a HashMap<String, RawTensor<'a>>,
    name: &str,
    shape: &[usize],
) -> Result<&'a RawTensor<'a>> {
    let tensor = raw.get(name).ok_or_else(|| RtError::InvalidTensor {
        tensor: name.into(),
        message: "missing from model.safetensors".into(),
    })?;
    if tensor.dtype != SafeDtype::F32 || tensor.shape != shape {
        return Err(RtError::InvalidTensor {
            tensor: name.into(),
            message: format!(
                "expected F32 shape {shape:?}, got {} {:?}",
                tensor.dtype, tensor.shape
            ),
        });
    }
    Ok(tensor)
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
