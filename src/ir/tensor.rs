use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Float32,
    Float16,
    Int64,
    Bool,
}

impl DType {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "float32" | "fp32" => Some(Self::Float32),
            "float16" | "fp16" => Some(Self::Float16),
            "int64" => Some(Self::Int64),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }

    pub fn byte_size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
            Self::Int64 => 8,
            Self::Bool => 1,
        }
    }

    pub fn byte_len_for_shape(self, shape: &[usize]) -> Option<usize> {
        let elems = shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        elems.checked_mul(self.byte_size())
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::Float32 | Self::Float16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => f.write_str("float32"),
            Self::Float16 => f.write_str("float16"),
            Self::Int64 => f.write_str("int64"),
            Self::Bool => f.write_str("bool"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorKind {
    UserInput,
    Parameter,
    Buffer,
    Intermediate,
    UserOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    DenseRowMajor,
}

impl Layout {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "dense_row_major" => Some(Self::DenseRowMajor),
            _ => None,
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DenseRowMajor => f.write_str("dense_row_major"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub id: TensorId,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub layout: Layout,
    pub kind: TensorKind,
    pub target: Option<String>,
    pub weight: Option<WeightRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightRef {
    pub offset: usize,
    pub byte_length: usize,
}
