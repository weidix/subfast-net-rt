use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, RtError>;

#[derive(Debug, thiserror::Error)]
pub enum RtError {
    #[error("IO error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("JSON/schema error: {0}")]
    Json(String),
    #[error("safetensors error: {0}")]
    Safetensors(String),
    #[error("unsupported format version: format={format}, version={version}")]
    FormatVersion { format: String, version: u64 },
    #[error(
        "weight tensor {tensor} is out of bounds: offset={offset}, length={length}, file length={file_len}"
    )]
    WeightsOutOfBounds {
        tensor: String,
        offset: u64,
        length: u64,
        file_len: u64,
    },
    #[error("invalid model: {0}")]
    InvalidModel(String),
    #[error("invalid tensor {tensor}: {message}")]
    InvalidTensor { tensor: String, message: String },
    #[error("unsupported dtype/layout for tensor {tensor}: dtype={dtype}, layout={layout}")]
    UnsupportedDTypeLayout {
        tensor: String,
        dtype: String,
        layout: String,
    },
    #[error("inference execution failed: {0}")]
    Inference(String),
}

impl RtError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
