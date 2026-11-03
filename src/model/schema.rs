use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub format: String,
    pub format_version: u64,
    pub model_type: String,
    pub model: ModelSection,
    pub input: TensorList,
    pub output: TensorList,
    pub weights: WeightsSection,
    #[serde(default)]
    pub preprocessing: Option<PreprocessingSection>,
    #[serde(default)]
    pub postprocessing: Option<PostprocessingSection>,
}

#[derive(Debug, Deserialize)]
pub struct ModelSection {
    #[serde(rename = "type")]
    pub model_type: String,
    pub architecture_version: u64,
    #[serde(default)]
    pub pooling_version: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct TensorList {
    pub tensors: Vec<TensorConfig>,
}

#[derive(Debug, Deserialize)]
pub struct TensorConfig {
    pub name: String,
    #[serde(default)]
    pub shape: ConfigShape,
    #[serde(default)]
    pub layout: String,
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct ConfigShape(Vec<serde_json::Value>);

impl ConfigShape {
    pub fn nchw(&self) -> Option<[usize; 4]> {
        if self.0.len() != 4 || self.0[0].as_str() != Some("batch") {
            return None;
        }
        let channel = usize::try_from(self.0[1].as_u64()?).ok()?;
        let height = usize::try_from(self.0[2].as_u64()?).ok()?;
        let width = usize::try_from(self.0[3].as_u64()?).ok()?;
        Some([1, channel, height, width])
    }

    pub fn is_batch_scalar(&self) -> bool {
        self.0.len() == 1 && self.0[0].as_str() == Some("batch")
    }
}

#[derive(Debug, Deserialize)]
pub struct WeightsSection {
    pub file: String,
    pub format: String,
}

#[derive(Debug, Deserialize)]
pub struct PreprocessingSection {
    #[serde(default)]
    pub normalization: Option<NormalizationSection>,
    #[serde(default)]
    pub resize: Option<ResizeSection>,
    #[serde(default)]
    pub value_scale: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct NormalizationSection {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ResizeSection {
    pub mode: String,
    #[serde(default)]
    pub target_size: Option<TargetSize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TargetSize {
    Named { width: usize, height: usize },
    Ordered(Vec<usize>),
}

impl TargetSize {
    /// Exporters write either `{width, height}` or the ordered `[width, height]`.
    pub fn width_height(&self) -> Option<(usize, usize)> {
        match self {
            Self::Named { width, height } => Some((*width, *height)),
            Self::Ordered(values) if values.len() == 2 => Some((values[0], values[1])),
            Self::Ordered(_) => None,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PostprocessingSection {
    #[serde(default)]
    pub decision_threshold: Option<f32>,
}
