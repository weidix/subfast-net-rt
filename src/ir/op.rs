#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Conv2d {
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    },
    BatchNorm {
        epsilon: f32,
        training: bool,
    },
    Silu,
    Add {
        alpha: f32,
        value: Option<f32>,
    },
    UpsampleBilinear2d {
        output_size: [usize; 2],
        align_corners: bool,
        scale_factors: Option<[f32; 2]>,
    },
    Abs,
    ReduceMax {
        axes: Vec<isize>,
    },
    ReduceMean {
        axes: Vec<isize>,
    },
    GreaterThan {
        value: f32,
    },
    Cast {
        dtype: crate::ir::tensor::DType,
    },
    AvgPool2d {
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        ceil_mode: bool,
        count_include_pad: bool,
    },
    Mul {
        value: Option<f32>,
    },
    Sub,
    Div,
    BitwiseNot,
    MaskedFill {
        value: f32,
    },
    ClampMin {
        value: f32,
    },
    Softplus,
    Cat {
        dim: isize,
    },
    Sigmoid,
    Linear,
    Squeeze {
        dim: isize,
    },
}
