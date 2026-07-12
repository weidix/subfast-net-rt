use std::collections::VecDeque;
use std::mem::size_of;
use std::path::Path;
use std::time::Duration;

use fast_image_resize::images::{Image as FirImage, ImageRef as FirImageRef};
use fast_image_resize::{
    FilterType as FirFilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::DynamicImage;
use serde::Serialize;

use crate::error::{Result, RtError};
use crate::ir::tensor::DType;
use crate::tensor::Tensor;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    Stretch,
    LetterboxTopLeft,
    LetterboxCentered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessGeometry {
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub padded_width: u32,
    pub padded_height: u32,
}

/// Reusable SIMD image preprocessor for decoded, tightly packed RGB8 frames.
///
/// File decoding is intentionally outside the hot path. Reuse both this value
/// and a tensor returned by [`Self::new_output_tensor`] between frames to avoid
/// all steady-state allocations.
pub struct ImagePreprocessor {
    shape: Vec<usize>,
    target_width: u32,
    target_height: u32,
    mode: ResizeMode,
    resizer: Resizer,
    resized: FirImage<'static>,
    normalize_lut: [[u32; 256]; 3],
}

impl std::fmt::Debug for ImagePreprocessor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImagePreprocessor")
            .field("shape", &self.shape)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl ImagePreprocessor {
    pub fn new(input_shape: &[usize], mode: ResizeMode) -> Result<Self> {
        let (target_width, target_height) = validate_input_shape(input_shape)?;
        let mut normalize_lut = [[0u32; 256]; 3];
        for (channel, channel_lut) in normalize_lut.iter_mut().enumerate() {
            for (value, normalized) in channel_lut.iter_mut().enumerate() {
                *normalized = (((value as f32 / 255.0) - IMAGENET_MEAN[channel])
                    / IMAGENET_STD[channel])
                    .to_bits()
                    .to_le();
            }
        }
        Ok(Self {
            shape: input_shape.to_vec(),
            target_width,
            target_height,
            mode,
            resizer: Resizer::new(),
            resized: FirImage::new(target_width, target_height, PixelType::U8x3),
            normalize_lut,
        })
    }

    pub fn new_output_tensor(&self) -> Tensor {
        Tensor {
            dtype: DType::Float32,
            shape: self.shape.clone(),
            data: vec![0; self.output_byte_len()],
        }
    }

    pub fn preprocess_rgb(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> Result<PreprocessedImage> {
        let mut tensor = self.new_output_tensor();
        let geometry = self.preprocess_rgb_into(rgb, width, height, &mut tensor)?;
        Ok(PreprocessedImage {
            tensor,
            original_width: geometry.original_width,
            original_height: geometry.original_height,
            resized_width: geometry.resized_width,
            resized_height: geometry.resized_height,
            padded_width: geometry.padded_width,
            padded_height: geometry.padded_height,
        })
    }

    pub fn preprocess_rgb_into(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        output: &mut Tensor,
    ) -> Result<PreprocessGeometry> {
        if width == 0 || height == 0 {
            return Err(invalid_preprocess("image dimensions must be non-zero"));
        }
        let expected_rgb_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| invalid_preprocess("RGB image byte length overflow"))?;
        if rgb.len() != expected_rgb_len {
            return Err(invalid_preprocess(format!(
                "expected {expected_rgb_len} tightly packed RGB bytes for {width}x{height}, got {}",
                rgb.len()
            )));
        }
        if output.dtype != DType::Float32
            || output.shape != self.shape
            || output.data.len() != self.output_byte_len()
        {
            return Err(RtError::InvalidTensor {
                tensor: "<runtime-input>".into(),
                message: format!("expected float32 shape {:?}", self.shape),
            });
        }

        let (resized_width, resized_height) = match self.mode {
            ResizeMode::Stretch => (self.target_width, self.target_height),
            ResizeMode::LetterboxTopLeft | ResizeMode::LetterboxCentered => {
                letterbox_dimensions(width, height, self.target_width, self.target_height)
            }
        };
        if self.resized.width() != resized_width || self.resized.height() != resized_height {
            self.resized = FirImage::new(resized_width, resized_height, PixelType::U8x3);
        }

        let source = FirImageRef::new(width, height, rgb, PixelType::U8x3)
            .map_err(|error| invalid_preprocess(format!("invalid RGB image: {error}")))?;
        let options =
            ResizeOptions::new().resize_alg(ResizeAlg::Interpolation(FirFilterType::Bilinear));
        self.resizer
            .resize(&source, &mut self.resized, &options)
            .map_err(|error| invalid_preprocess(format!("SIMD resize failed: {error}")))?;

        let (offset_x, offset_y) = match self.mode {
            ResizeMode::LetterboxCentered => (
                (self.target_width - resized_width) / 2,
                (self.target_height - resized_height) / 2,
            ),
            ResizeMode::Stretch | ResizeMode::LetterboxTopLeft => (0, 0),
        };
        if resized_width != self.target_width || resized_height != self.target_height {
            output.data.fill(0);
        }
        rgb_to_normalized_nchw(
            self.resized.buffer(),
            resized_width as usize,
            resized_height as usize,
            self.target_width as usize,
            offset_x as usize,
            offset_y as usize,
            &self.normalize_lut,
            &mut output.data,
        );

        Ok(PreprocessGeometry {
            original_width: width,
            original_height: height,
            resized_width,
            resized_height,
            padded_width: self.target_width,
            padded_height: self.target_height,
        })
    }

    fn output_byte_len(&self) -> usize {
        3 * self.target_width as usize * self.target_height as usize * size_of::<f32>()
    }
}

#[derive(Debug)]
pub struct PreprocessedImage {
    pub tensor: Tensor,
    pub original_width: u32,
    pub original_height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub padded_width: u32,
    pub padded_height: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct DetectionThresholds {
    pub region_threshold: f32,
    pub kernel_threshold: f32,
    pub min_size: f32,
}

impl Default for DetectionThresholds {
    fn default() -> Self {
        Self {
            region_threshold: 0.5,
            kernel_threshold: 0.5,
            min_size: 3.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub box_xyxy: [f32; 4],
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputAnalysis {
    pub max_region_probability: f32,
    pub max_kernel_probability: f32,
    pub has_subtitle: bool,
    pub detections: Vec<Detection>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct TimingStats {
    pub count: usize,
    pub total: Duration,
}

impl TimingStats {
    pub fn record(&mut self, duration: Duration) {
        self.record_many(duration, 1);
    }

    pub fn record_many(&mut self, duration: Duration, count: usize) {
        if count == 0 {
            return;
        }
        self.count += count;
        self.total += duration;
    }

    pub fn average_micros(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.total.as_secs_f64() * 1_000_000.0 / self.count as f64
    }

    pub fn images_per_second(&self) -> f64 {
        if self.total.is_zero() {
            return 0.0;
        }
        self.count as f64 / self.total.as_secs_f64()
    }
}

#[derive(Debug, Default)]
struct ComponentStats {
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    confidence_sum: f32,
    count: usize,
}

impl ComponentStats {
    fn new(width: usize, height: usize) -> Self {
        Self {
            min_x: width,
            min_y: height,
            ..Self::default()
        }
    }

    fn add(&mut self, x: usize, y: usize, region_prob: f32, kernel_prob: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.confidence_sum += region_prob * 0.8 + kernel_prob * 0.2;
        self.count += 1;
    }

    fn confidence(&self) -> f32 {
        self.confidence_sum / self.count.max(1) as f32
    }

    fn box_xyxy(&self) -> [f32; 4] {
        [
            self.min_x as f32,
            self.min_y as f32,
            (self.max_x + 1) as f32,
            (self.max_y + 1) as f32,
        ]
    }
}

pub fn preprocess_image_path(
    path: impl AsRef<Path>,
    input_shape: &[usize],
) -> Result<PreprocessedImage> {
    let path = path.as_ref();
    let image = image::open(path).map_err(|source| {
        RtError::Inference(format!(
            "failed to decode image {}: {source}",
            path.display()
        ))
    })?;
    preprocess_dynamic_image(&image, input_shape)
}

/// Resize an ROI directly to the model shape, matching the training pipeline's
/// `resize_mode=stretch` preprocessing.
pub fn preprocess_stretched_image_path(
    path: impl AsRef<Path>,
    input_shape: &[usize],
) -> Result<Tensor> {
    let path = path.as_ref();
    let image = image::open(path).map_err(|source| {
        RtError::Inference(format!(
            "failed to decode image {}: {source}",
            path.display()
        ))
    })?;
    Ok(preprocess_dynamic_image_mode(&image, input_shape, ResizeMode::Stretch)?.tensor)
}

pub fn preprocess_centered_letterbox_image_path(
    path: impl AsRef<Path>,
    input_shape: &[usize],
) -> Result<Tensor> {
    let path = path.as_ref();
    let image = image::open(path).map_err(|source| {
        RtError::Inference(format!(
            "failed to decode image {}: {source}",
            path.display()
        ))
    })?;
    Ok(preprocess_dynamic_image_mode(&image, input_shape, ResizeMode::LetterboxCentered)?.tensor)
}

pub fn preprocess_dynamic_image(
    image: &DynamicImage,
    input_shape: &[usize],
) -> Result<PreprocessedImage> {
    preprocess_dynamic_image_mode(image, input_shape, ResizeMode::LetterboxTopLeft)
}

fn preprocess_dynamic_image_mode(
    image: &DynamicImage,
    input_shape: &[usize],
    mode: ResizeMode,
) -> Result<PreprocessedImage> {
    let mut preprocessor = ImagePreprocessor::new(input_shape, mode)?;
    match image {
        DynamicImage::ImageRgb8(rgb) => {
            preprocessor.preprocess_rgb(rgb.as_raw(), rgb.width(), rgb.height())
        }
        _ => {
            let rgb = image.to_rgb8();
            preprocessor.preprocess_rgb(rgb.as_raw(), rgb.width(), rgb.height())
        }
    }
}

fn validate_input_shape(input_shape: &[usize]) -> Result<(u32, u32)> {
    if input_shape.len() != 4 || input_shape[0] != 1 || input_shape[1] != 3 {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-input>".into(),
            message: format!("expected NCHW input shape [1, 3, H, W], got {input_shape:?}"),
        });
    }
    let height = u32::try_from(input_shape[2]).map_err(|_| RtError::InvalidTensor {
        tensor: "<runtime-input>".into(),
        message: "input height does not fit u32".into(),
    })?;
    let width = u32::try_from(input_shape[3]).map_err(|_| RtError::InvalidTensor {
        tensor: "<runtime-input>".into(),
        message: "input width does not fit u32".into(),
    })?;
    if width == 0 || height == 0 {
        return Err(invalid_preprocess("input dimensions must be non-zero"));
    }
    Ok((width, height))
}

fn letterbox_dimensions(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> (u32, u32) {
    let scale = (target_width as f32 / source_width as f32)
        .min(target_height as f32 / source_height as f32);
    (
        ((source_width as f32 * scale).round() as u32)
            .max(1)
            .min(target_width),
        ((source_height as f32 * scale).round() as u32)
            .max(1)
            .min(target_height),
    )
}

#[allow(clippy::too_many_arguments)]
fn rgb_to_normalized_nchw(
    rgb: &[u8],
    width: usize,
    height: usize,
    output_width: usize,
    offset_x: usize,
    offset_y: usize,
    lut: &[[u32; 256]; 3],
    output: &mut [u8],
) {
    let plane = output.len() / (3 * size_of::<f32>());
    debug_assert_eq!(rgb.len(), width * height * 3);
    debug_assert!(offset_x + width <= output_width);
    debug_assert!((offset_y + height) * output_width <= plane);

    for y in 0..height {
        let source_row = &rgb[y * width * 3..(y + 1) * width * 3];
        let output_row = (y + offset_y) * output_width + offset_x;
        for (x, pixel) in source_row.chunks_exact(3).enumerate() {
            let index = output_row + x;
            // All offsets are proven in-bounds above. Unaligned writes are
            // required because Tensor intentionally stores portable LE bytes.
            unsafe {
                let base = output.as_mut_ptr();
                std::ptr::write_unaligned(
                    base.add(index * 4).cast::<u32>(),
                    lut[0][pixel[0] as usize],
                );
                std::ptr::write_unaligned(
                    base.add((plane + index) * 4).cast::<u32>(),
                    lut[1][pixel[1] as usize],
                );
                std::ptr::write_unaligned(
                    base.add((2 * plane + index) * 4).cast::<u32>(),
                    lut[2][pixel[2] as usize],
                );
            }
        }
    }
}

fn invalid_preprocess(message: impl Into<String>) -> RtError {
    RtError::Inference(format!("image preprocessing: {}", message.into()))
}

pub fn analyze_output(output: &Tensor, thresholds: DetectionThresholds) -> Result<OutputAnalysis> {
    let values = output.f32_values()?;
    if output.shape.len() != 4 || output.shape[0] != 1 || output.shape[1] != 2 {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-output>".into(),
            message: format!("expected output shape [1, 2, H, W], got {:?}", output.shape),
        });
    }
    let height = output.shape[2];
    let width = output.shape[3];
    let plane = height * width;
    if values.len() != 2 * plane {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-output>".into(),
            message: "output value count does not match shape".into(),
        });
    }

    let region_prob = values[..plane]
        .iter()
        .map(|value| sigmoid(*value))
        .collect::<Vec<_>>();
    let kernel_prob = values[plane..]
        .iter()
        .map(|value| sigmoid(*value))
        .collect::<Vec<_>>();
    let max_region_probability = region_prob
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let max_kernel_probability = kernel_prob
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let detections =
        detections_from_probabilities(&region_prob, &kernel_prob, width, height, thresholds);

    Ok(OutputAnalysis {
        max_region_probability,
        max_kernel_probability,
        has_subtitle: !detections.is_empty(),
        detections,
    })
}

fn detections_from_probabilities(
    region_prob: &[f32],
    kernel_prob: &[f32],
    width: usize,
    height: usize,
    thresholds: DetectionThresholds,
) -> Vec<Detection> {
    let mut owner = vec![-1i32; width * height];
    let mut queue = VecDeque::new();
    let mut seed_id = 0i32;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if kernel_prob[idx] < thresholds.kernel_threshold
                || region_prob[idx] < thresholds.region_threshold
                || owner[idx] != -1
            {
                continue;
            }
            owner[idx] = seed_id;
            queue.push_back((x, y, seed_id));
            while let Some((cx, cy, sid)) = queue.pop_front() {
                for (nx, ny) in neighbors(cx, cy, width, height) {
                    let nidx = ny * width + nx;
                    if kernel_prob[nidx] >= thresholds.kernel_threshold
                        && region_prob[nidx] >= thresholds.region_threshold
                        && owner[nidx] == -1
                    {
                        owner[nidx] = sid;
                        queue.push_back((nx, ny, sid));
                    }
                }
            }
            seed_id += 1;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if owner[idx] >= 0 {
                queue.push_back((x, y, owner[idx]));
            }
        }
    }
    while let Some((cx, cy, sid)) = queue.pop_front() {
        for (nx, ny) in neighbors(cx, cy, width, height) {
            let nidx = ny * width + nx;
            if region_prob[nidx] >= thresholds.region_threshold && owner[nidx] == -1 {
                owner[nidx] = sid;
                queue.push_back((nx, ny, sid));
            }
        }
    }

    let mut detections = Vec::new();
    for sid in 0..seed_id {
        let mut stats = ComponentStats::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if owner[idx] == sid {
                    stats.add(x, y, region_prob[idx], kernel_prob[idx]);
                }
            }
        }
        if stats.count == 0 {
            continue;
        }
        let box_xyxy = stats.box_xyxy();
        if box_xyxy[2] - box_xyxy[0] >= thresholds.min_size
            && box_xyxy[3] - box_xyxy[1] >= thresholds.min_size
        {
            detections.push(Detection {
                box_xyxy,
                score: stats.confidence(),
            });
        }
    }
    detections
}

fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut values = Vec::with_capacity(4);
    if x > 0 {
        values.push((x - 1, y));
    }
    if x + 1 < width {
        values.push((x + 1, y));
    }
    if y > 0 {
        values.push((x, y - 1));
    }
    if y + 1 < height {
        values.push((x, y + 1));
    }
    values.into_iter()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
