use std::time::Duration;
use subfast_net_rt::image_infer::TimingStats;
use subfast_net_rt::image_infer::{DetectionThresholds, analyze_output, preprocess_dynamic_image};
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
