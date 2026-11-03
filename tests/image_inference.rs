#[allow(dead_code)]
mod support;

use std::time::Duration;
use subfast_net_rt::image_infer::TimingStats;
use subfast_net_rt::image_infer::{
    DetectionThresholds, ImagePreprocessor, ResizeFilter, ResizeMode, analyze_output,
    preprocess_dynamic_image, preprocessor_for_model,
};
use subfast_net_rt::model::Normalization;
use subfast_net_rt::tensor::Tensor;

#[test]
fn preprocesses_rgb_image_as_nchw_imagenet_normalized_letterbox() {
    let mut image = image::RgbImage::new(2, 1);
    for pixel in image.pixels_mut() {
        *pixel = image::Rgb([255, 0, 0]);
    }

    let preprocessed =
        preprocess_dynamic_image(&image::DynamicImage::ImageRgb8(image), &[1, 3, 4, 4]).unwrap();
    let values = preprocessed.tensor.f32_values().unwrap();

    assert_eq!(preprocessed.resized_width, 4);
    assert_eq!(preprocessed.resized_height, 2);
    assert_eq!(preprocessed.padded_width, 4);
    assert_eq!(preprocessed.padded_height, 4);

    let red = (1.0 - 0.485) / 0.229;
    let green = (0.0 - 0.456) / 0.224;
    let blue = (0.0 - 0.406) / 0.225;
    assert!((values[0] - red).abs() < 1e-6);
    assert!((values[16] - green).abs() < 1e-6);
    assert!((values[32] - blue).abs() < 1e-6);
    assert_eq!(values[8], 0.0, "letterbox padding must remain zero");
    assert_eq!(values[24], 0.0, "letterbox padding must remain zero");
    assert_eq!(values[40], 0.0, "letterbox padding must remain zero");
}

#[test]
fn reusable_preprocessor_writes_into_the_same_tensor() {
    let rgb = [255, 0, 0, 0, 255, 0];
    let mut preprocessor = ImagePreprocessor::new(&[1, 3, 2, 4], ResizeMode::Stretch).unwrap();
    let mut output = preprocessor.new_output_tensor();
    let allocation = output.data.as_ptr();

    let geometry = preprocessor
        .preprocess_rgb_into(&rgb, 2, 1, &mut output)
        .unwrap();
    let values = output.f32_values().unwrap();

    assert_eq!(allocation, output.data.as_ptr());
    assert_eq!(geometry.resized_width, 4);
    assert_eq!(geometry.resized_height, 2);
    assert!((values[0] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
    assert!((values[3] - (0.0 - 0.485) / 0.229).abs() < 1e-6);
}

#[test]
fn reusable_preprocessor_rejects_invalid_rgb_length() {
    let mut preprocessor = ImagePreprocessor::new(&[1, 3, 32, 256], ResizeMode::Stretch).unwrap();
    let mut output = preprocessor.new_output_tensor();
    let error = preprocessor
        .preprocess_rgb_into(&[0; 5], 2, 1, &mut output)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected 6 tightly packed RGB bytes")
    );
}

#[test]
fn scale_normalization_writes_exact_pixel_values() {
    let rgb = [255u8, 128, 1, 0, 64, 255];
    let mut preprocessor = ImagePreprocessor::with_options(
        &[1, 3, 1, 2],
        ResizeMode::Stretch,
        Normalization::Scale(1.0 / 255.0),
        ResizeFilter::Interpolation,
    )
    .unwrap();
    let preprocessed = preprocessor.preprocess_rgb(&rgb, 2, 1).unwrap();
    let values = preprocessed.tensor.f32_values().unwrap();

    for (index, byte) in rgb.iter().enumerate() {
        let channel = index % 3;
        let column = index / 3;
        let value = values[channel * 2 + column];
        assert_eq!(value, *byte as f32 * (1.0 / 255.0));
        assert!((value - *byte as f32 / 255.0).abs() < 1e-6);
    }
    assert_eq!(values[0], 1.0, "a 255 pixel must normalize to exactly 1.0");
}

#[test]
fn scale_normalization_leaves_letterbox_padding_zero() {
    let rgb = [200u8, 10, 30, 40, 50, 60];
    let mut preprocessor = ImagePreprocessor::with_options(
        &[1, 3, 4, 4],
        ResizeMode::LetterboxCentered,
        Normalization::Scale(1.0 / 255.0),
        ResizeFilter::Interpolation,
    )
    .unwrap();
    let preprocessed = preprocessor.preprocess_rgb(&rgb, 2, 1).unwrap();

    assert_eq!(preprocessed.resized_width, 4);
    assert_eq!(preprocessed.resized_height, 2);
    let values = preprocessed.tensor.f32_values().unwrap();
    let plane = 16;
    for channel in 0..3 {
        for padding_row in [0, 3] {
            assert_eq!(
                values[channel * plane + padding_row * 4],
                0.0,
                "letterbox padding must remain zero"
            );
        }
    }
    assert!(values[4] > 0.0, "resized rows must hold image data");
}

#[test]
fn preprocessor_for_frame_presence_bundle_uses_exported_contract() {
    let Some(path) =
        std::env::var_os("SUBFAST_RT_FRAME_PRESENCE_DIR").map(std::path::PathBuf::from)
    else {
        eprintln!("skipping Frame Presence fixture; set SUBFAST_RT_FRAME_PRESENCE_DIR");
        return;
    };
    let model = subfast_net_rt::model::Model::load(path).unwrap();
    let mut preprocessor = preprocessor_for_model(&model).unwrap();
    assert_eq!(preprocessor.new_output_tensor().shape, [1, 3, 272, 480]);

    let mut image = image::RgbImage::from_pixel(480, 272, image::Rgb([51, 102, 204]));
    image.put_pixel(255, 0, image::Rgb([255, 0, 0]));
    let preprocessed = preprocessor
        .preprocess_rgb(image.as_raw(), 480, 272)
        .unwrap();
    assert_eq!(preprocessed.resized_width, 480);
    assert_eq!(preprocessed.resized_height, 272);
    let values = preprocessed.tensor.f32_values().unwrap();
    let plane = 480 * 272;
    assert!((values[0] - 51.0 / 255.0).abs() < 1e-6);
    assert!((values[plane] - 102.0 / 255.0).abs() < 1e-6);
    assert!((values[2 * plane] - 204.0 / 255.0).abs() < 1e-6);
    assert!((values[255] - 1.0).abs() < 1e-6);

    let rgb = vec![128u8; 960 * 272 * 3];
    let letterboxed = preprocessor.preprocess_rgb(&rgb, 960, 272).unwrap();
    assert_eq!(letterboxed.resized_width, 480);
    assert_eq!(letterboxed.resized_height, 136);
    assert_eq!(letterboxed.padded_width, 480);
    assert_eq!(letterboxed.padded_height, 272);
    let values = letterboxed.tensor.f32_values().unwrap();
    assert_eq!(values[0], 0.0, "top padding must remain zero");
    assert_eq!(values[271 * 480], 0.0, "bottom padding must remain zero");
    assert!((values[68 * 480] - 128.0 / 255.0).abs() < 1e-3);

    // One-pixel stripes downscaled four times must average to 128/255 exactly.
    // Plain interpolation would keep a single sampled stripe value instead.
    let mut stripes = image::RgbImage::new(1920, 272);
    for (x, _, pixel) in stripes.enumerate_pixels_mut() {
        let value = if x % 2 == 0 { 0 } else { 255 };
        *pixel = image::Rgb([value, value, value]);
    }
    let downscaled = preprocessor
        .preprocess_rgb(stripes.as_raw(), 1920, 272)
        .unwrap();
    assert_eq!(downscaled.resized_height, 68);
    let values = downscaled.tensor.f32_values().unwrap();
    for column in [100, 240, 400] {
        let value = values[102 * 480 + column];
        assert!(
            (value - 128.0 / 255.0).abs() < 1e-3,
            "column {column}: {value}"
        );
    }
}

#[test]
fn preprocessor_for_legacy_bundles_keeps_roi_geometry() {
    let mut presence = preprocessor_for_model(&support::synthetic_presence_model()).unwrap();
    assert_eq!(presence.new_output_tensor().shape, [1, 3, 32, 256]);

    let red = image::RgbImage::from_pixel(64, 32, image::Rgb([255, 0, 0]));
    let preprocessed = presence.preprocess_rgb(red.as_raw(), 64, 32).unwrap();
    assert_eq!(preprocessed.resized_width, 64);
    assert_eq!(preprocessed.resized_height, 32);
    let values = preprocessed.tensor.f32_values().unwrap();
    let plane = 32 * 256;
    let offset = (256 - 64) / 2;
    assert_eq!(values[0], 0.0, "centered letterbox padding must stay zero");
    assert!((values[offset] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
    assert!((values[plane + offset] - (0.0 - 0.456) / 0.224).abs() < 1e-6);

    let mut pair = preprocessor_for_model(&support::synthetic_pair_model()).unwrap();
    assert_eq!(pair.new_output_tensor().shape, [1, 3, 32, 256]);
    let stretched = pair.preprocess_rgb(red.as_raw(), 64, 32).unwrap();
    assert_eq!(stretched.resized_width, 256);
    assert_eq!(stretched.resized_height, 32);
}

#[test]
fn detects_subtitle_presence_from_region_and_kernel_logits() {
    let mut values = vec![-8.0f32; 2 * 4 * 4];
    for y in 1..3 {
        for x in 1..3 {
            values[y * 4 + x] = 8.0;
            values[16 + y * 4 + x] = 8.0;
        }
    }
    let output = Tensor::from_f32(vec![1, 2, 4, 4], values).unwrap();

    let analysis = analyze_output(
        &output,
        DetectionThresholds {
            min_size: 1.0,
            ..DetectionThresholds::default()
        },
    )
    .unwrap();

    assert!(analysis.has_subtitle);
    assert!(analysis.max_region_probability > 0.99);
    assert!(analysis.max_kernel_probability > 0.99);
    assert_eq!(analysis.detections.len(), 1);
    assert!(analysis.detections[0].score > 0.99);
    assert_eq!(analysis.detections[0].box_xyxy, [1.0, 1.0, 3.0, 3.0]);
}

#[test]
fn timing_stats_reports_total_average_and_images_per_second() {
    let mut stats = TimingStats::default();
    stats.record(Duration::from_millis(2));
    stats.record_many(Duration::from_millis(6), 3);

    assert_eq!(stats.count, 4);
    assert_eq!(stats.total.as_micros(), 8_000);
    assert_eq!(stats.average_micros(), 2_000.0);
    assert_eq!(stats.images_per_second(), 500.0);
}
