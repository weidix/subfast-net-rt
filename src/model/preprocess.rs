use crate::image_infer::ResizeMode;

/// Exported inference resolution is padded to a multiple of this value.
pub const INPUT_ALIGNMENT: usize = 16;

/// Per-channel input normalization applied after the source pixels are scaled
/// into the `[0, 1]` value range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Normalization {
    /// ImageNet statistics: `(value / 255 - mean[c]) / std[c]`.
    Imagenet,
    /// Plain `value * scale` scaling without mean subtraction.
    Scale(f32),
}

/// Preprocessing contract exported alongside the model weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessContract {
    pub resize_mode: ResizeMode,
    pub normalization: Normalization,
}
