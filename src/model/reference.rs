use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Result, RtError};
use crate::ir::tensor::DType;
use crate::tensor::Tensor;

#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceCase {
    pub input: ReferenceTensorFile,
    pub expected_output: ReferenceTensorFile,
    #[serde(default)]
    pub tolerance: ReferenceTolerance,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceTensorFile {
    pub file: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct ReferenceTolerance {
    #[serde(default = "default_atol")]
    pub atol: f32,
    #[serde(default = "default_rtol")]
    pub rtol: f32,
}

impl Default for ReferenceTolerance {
    fn default() -> Self {
        Self {
            atol: default_atol(),
            rtol: default_rtol(),
        }
    }
}

pub fn load_case(path: impl AsRef<Path>) -> Result<ReferenceCase> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| RtError::io(path, e))?;
    serde_json::from_slice(&bytes).map_err(|e| RtError::Json(format!("{}: {}", path.display(), e)))
}

pub fn load_tensor_file(base_dir: &Path, spec: &ReferenceTensorFile) -> Result<Tensor> {
    let dtype = DType::parse(&spec.dtype).ok_or_else(|| RtError::UnsupportedDTypeLayout {
        tensor: spec.file.display().to_string(),
        dtype: spec.dtype.clone(),
        layout: "dense_row_major".into(),
    })?;
    let path = if spec.file.is_absolute() {
        spec.file.clone()
    } else {
        base_dir.join(&spec.file)
    };
    let data = std::fs::read(&path).map_err(|e| RtError::io(&path, e))?;
    Tensor::new(dtype, spec.shape.clone(), data)
}

pub fn compare_tensors(
    actual: &Tensor,
    expected: &Tensor,
    tolerance: ReferenceTolerance,
) -> Result<()> {
    if actual.dtype != expected.dtype || actual.shape != expected.shape {
        return Err(RtError::InvalidTensor {
            tensor: "<reference-compare>".into(),
            message: format!(
                "actual dtype/shape {:?}/{:?} does not match expected {:?}/{:?}",
                actual.dtype, actual.shape, expected.dtype, expected.shape
            ),
        });
    }
    match actual.dtype {
        DType::Float32 => {
            let actual_values = actual.f32_values()?;
            let expected_values = expected.f32_values()?;
            for (idx, (actual, expected)) in
                actual_values.iter().zip(expected_values.iter()).enumerate()
            {
                let allowed = tolerance.atol + tolerance.rtol * expected.abs();
                if (actual - expected).abs() > allowed {
                    return Err(RtError::Inference(format!(
                        "reference mismatch at {idx}: actual={actual}, expected={expected}, allowed={allowed}"
                    )));
                }
            }
            Ok(())
        }
        DType::Float16 => compare_f16_bytes(actual, expected, tolerance),
        DType::Int64 | DType::Bool => Err(RtError::UnsupportedDTypeLayout {
            tensor: "<reference-compare>".into(),
            dtype: "int64".into(),
            layout: "dense_row_major".into(),
        }),
    }
}

fn compare_f16_bytes(
    actual: &Tensor,
    expected: &Tensor,
    tolerance: ReferenceTolerance,
) -> Result<()> {
    for (idx, (actual, expected)) in actual
        .data
        .chunks_exact(2)
        .zip(expected.data.chunks_exact(2))
        .enumerate()
    {
        let actual = half::f16::from_bits(u16::from_le_bytes([actual[0], actual[1]])).to_f32();
        let expected =
            half::f16::from_bits(u16::from_le_bytes([expected[0], expected[1]])).to_f32();
        let allowed = tolerance.atol + tolerance.rtol * expected.abs();
        if (actual - expected).abs() > allowed {
            return Err(RtError::Inference(format!(
                "reference mismatch at {idx}: actual={actual}, expected={expected}, allowed={allowed}"
            )));
        }
    }
    Ok(())
}

fn default_atol() -> f32 {
    1e-5
}

fn default_rtol() -> f32 {
    1e-4
}

pub fn default_tolerance_for(dtype: DType) -> ReferenceTolerance {
    match dtype {
        DType::Float32 => ReferenceTolerance {
            atol: 1e-5,
            rtol: 1e-4,
        },
        DType::Float16 => ReferenceTolerance {
            atol: 1e-3,
            rtol: 1e-2,
        },
        DType::Int64 => ReferenceTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        DType::Bool => ReferenceTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
    }
}
