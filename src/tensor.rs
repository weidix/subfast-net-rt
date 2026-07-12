use crate::error::{Result, RtError};
use crate::ir::tensor::DType;

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(dtype: DType, shape: Vec<usize>, data: Vec<u8>) -> Result<Self> {
        let expected = dtype
            .byte_len_for_shape(&shape)
            .ok_or_else(|| RtError::InvalidTensor {
                tensor: "<runtime>".into(),
                message: "shape byte length overflow".into(),
            })?;
        if expected != data.len() {
            return Err(RtError::InvalidTensor {
                tensor: "<runtime>".into(),
                message: format!(
                    "data length {} does not match dtype/shape byte length {}",
                    data.len(),
                    expected
                ),
            });
        }
        Ok(Self { dtype, shape, data })
    }

    pub fn from_f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self::new(DType::Float32, shape, data)
    }

    pub fn from_f16(shape: Vec<usize>, values: Vec<half::f16>) -> Result<Self> {
        let mut data = Vec::with_capacity(values.len() * 2);
        for value in values {
            data.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Self::new(DType::Float16, shape, data)
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().copied().product()
    }

    pub fn f32_values(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::Float32 {
            return Err(RtError::InvalidTensor {
                tensor: "<runtime>".into(),
                message: format!("expected float32, got {}", self.dtype),
            });
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
}
