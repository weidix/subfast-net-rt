use std::collections::VecDeque;
use std::path::Path;
use std::time::Duration;

use image::imageops::{FilterType, resize};
use image::{DynamicImage, GenericImageView};
use serde::Serialize;

use crate::error::{Result, RtError};
use crate::tensor::Tensor;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

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
    if input_shape.len() != 4 || input_shape[0] != 1 || input_shape[1] != 3 {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-input>".into(),
            message: format!("expected NCHW input shape [1, 3, H, W], got {input_shape:?}"),
        });
    }
    let height = input_shape[2] as u32;
    let width = input_shape[3] as u32;
    let resized = resize(&image.to_rgb8(), width, height, FilterType::Triangle);
    let plane = (width * height) as usize;
    let mut values = vec![0.0f32; 3 * plane];
    for y in 0..height {
        for x in 0..width {
            let pixel = resized.get_pixel(x, y).0;
            let index = (y * width + x) as usize;
            for channel in 0..3 {
                let value = f32::from(pixel[channel]) / 255.0;
                values[channel * plane + index] =
                    (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }
    Tensor::from_f32(input_shape.to_vec(), values)
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
    if input_shape.len() != 4 || input_shape[0] != 1 || input_shape[1] != 3 {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-input>".into(),
            message: format!("expected NCHW input shape [1, 3, H, W], got {input_shape:?}"),
        });
    }
    let target_height = input_shape[2];
    let target_width = input_shape[3];
    let rgb = image.to_rgb8();
    let (source_width, source_height) = rgb.dimensions();
    let scale = (target_width as f32 / source_width as f32)
        .min(target_height as f32 / source_height as f32);
    let width = ((source_width as f32 * scale).round() as usize).max(1);
    let height = ((source_height as f32 * scale).round() as usize).max(1);
    let offset_x = (target_width - width) / 2;
    let offset_y = (target_height - height) / 2;
    let plane = target_width * target_height;
    let mut values = vec![0.0f32; 3 * plane];
    for y in 0..height {
        let source_y = ((y as f32 + 0.5) * source_height as f32 / height as f32 - 0.5)
            .clamp(0.0, source_height.saturating_sub(1) as f32);
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(source_height - 1);
        let wy = source_y - y0 as f32;
        for x in 0..width {
            let source_x = ((x as f32 + 0.5) * source_width as f32 / width as f32 - 0.5)
                .clamp(0.0, source_width.saturating_sub(1) as f32);
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(source_width - 1);
            let wx = source_x - x0 as f32;
            let index = (y + offset_y) * target_width + x + offset_x;
            for channel in 0..3 {
                let top = f32::from(rgb.get_pixel(x0, y0)[channel]) * (1.0 - wx)
                    + f32::from(rgb.get_pixel(x1, y0)[channel]) * wx;
                let bottom = f32::from(rgb.get_pixel(x0, y1)[channel]) * (1.0 - wx)
                    + f32::from(rgb.get_pixel(x1, y1)[channel]) * wx;
                let value = (top * (1.0 - wy) + bottom * wy) / 255.0;
                values[channel * plane + index] =
                    (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }
    Tensor::from_f32(input_shape.to_vec(), values)
}

pub fn preprocess_dynamic_image(
    image: &DynamicImage,
    input_shape: &[usize],
) -> Result<PreprocessedImage> {
    preprocess_dynamic_image_aligned(image, input_shape, false)
}

fn preprocess_dynamic_image_aligned(
    image: &DynamicImage,
    input_shape: &[usize],
    centered: bool,
) -> Result<PreprocessedImage> {
    if input_shape.len() != 4 || input_shape[0] != 1 || input_shape[1] != 3 {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-input>".into(),
            message: format!("expected NCHW input shape [1, 3, H, W], got {input_shape:?}"),
        });
    }

    let padded_height = u32::try_from(input_shape[2]).map_err(|_| RtError::InvalidTensor {
        tensor: "<runtime-input>".into(),
        message: "input height does not fit u32".into(),
    })?;
    let padded_width = u32::try_from(input_shape[3]).map_err(|_| RtError::InvalidTensor {
        tensor: "<runtime-input>".into(),
        message: "input width does not fit u32".into(),
    })?;
    let (original_width, original_height) = image.dimensions();
    if original_width == 0 || original_height == 0 || padded_width == 0 || padded_height == 0 {
        return Err(RtError::InvalidTensor {
            tensor: "<runtime-input>".into(),
            message: "image and input dimensions must be non-zero".into(),
        });
    }

    let scale = (padded_width as f32 / original_width as f32)
        .min(padded_height as f32 / original_height as f32);
    let resized_width = ((original_width as f32 * scale).round() as u32).max(1);
    let resized_height = ((original_height as f32 * scale).round() as u32).max(1);
    let rgb = image.to_rgb8();
    let resized = resize(&rgb, resized_width, resized_height, FilterType::Triangle);
    let offset_x = if centered {
        (padded_width - resized_width) / 2
    } else {
        0
    };
    let offset_y = if centered {
        (padded_height - resized_height) / 2
    } else {
        0
    };

    let plane = (padded_width * padded_height) as usize;
    let mut values = vec![0.0f32; 3 * plane];
    for y in 0..resized_height {
        for x in 0..resized_width {
            let pixel = resized.get_pixel(x, y).0;
            let idx = ((y + offset_y) * padded_width + x + offset_x) as usize;
            for channel in 0..3 {
                let value = f32::from(pixel[channel]) / 255.0;
                values[channel * plane + idx] =
                    (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }

    Ok(PreprocessedImage {
        tensor: Tensor::from_f32(input_shape.to_vec(), values)?,
        original_width,
        original_height,
        resized_width,
        resized_height,
        padded_width,
        padded_height,
    })
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
