//! End-to-end coverage for Frame Presence: exported preprocessing contract,
//! image resampling, and CPU inference compared against PyTorch references.

use std::path::{Path, PathBuf};

use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::image_infer::preprocessor_for_model;
use subfast_net_rt::model::Model;

/// The runtime resampler is an independent bilinear implementation; keeping the
/// decision equal on real frames is what matters.
const RESAMPLED_LOGIT_TOLERANCE: f32 = 0.1;
/// Images that already match the model canvas are copied without resampling.
const IDENTITY_LOGIT_TOLERANCE: f32 = 1e-3;

fn fixture(variable: &str) -> Option<PathBuf> {
    let path = std::env::var_os(variable).map(PathBuf::from)?;
    assert!(path.exists(), "{} does not exist", path.display());
    Some(path)
}

fn read_cases(cases_dir: &Path) -> Vec<serde_json::Value> {
    let bytes = std::fs::read(cases_dir.join("cases.json")).unwrap();
    serde_json::from_slice::<serde_json::Value>(&bytes)
        .unwrap()
        .as_array()
        .expect("cases.json must be an array")
        .clone()
}

#[test]
fn exported_preprocessing_contract_drives_frame_presence_decisions() {
    let (Some(model_dir), Some(cases_dir)) = (
        fixture("SUBFAST_RT_FRAME_PRESENCE_DIR"),
        fixture("SUBFAST_RT_FRAME_PRESENCE_CASES"),
    ) else {
        eprintln!(
            "skipping frame-presence end-to-end parity; set SUBFAST_RT_FRAME_PRESENCE_DIR and SUBFAST_RT_FRAME_PRESENCE_CASES"
        );
        return;
    };
    let model = Model::load(&model_dir).unwrap();
    let threshold = model.decision_threshold.unwrap_or(0.5);
    let mut preprocessor = preprocessor_for_model(&model).unwrap();
    let mut executable = CpuBackend.compile(&model).unwrap();
    let cases = read_cases(&cases_dir);
    assert!(!cases.is_empty());

    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let source = cases_dir.join(case["source_png"].as_str().unwrap());
        let rgb = image::open(&source).unwrap().to_rgb8();
        let mut tensor = preprocessor.new_output_tensor();
        preprocessor
            .preprocess_rgb_into(rgb.as_raw(), rgb.width(), rgb.height(), &mut tensor)
            .unwrap();
        let logit = executable
            .run_scalar(std::slice::from_ref(&tensor))
            .unwrap();

        let expected = case["presence_logit"].as_f64().unwrap() as f32;
        let expected_input =
            std::fs::read(cases_dir.join(case["input"]["file"].as_str().unwrap())).unwrap();
        let actual_values = tensor.f32_values().unwrap();
        let expected_values = expected_input
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        assert_eq!(actual_values.len(), expected_values.len());
        let max_input_diff = actual_values
            .iter()
            .zip(&expected_values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{name}: logit={logit:.7} expected={expected:.7} max_input_diff={max_input_diff:.3e}"
        );
        let source_size = case["source_size"].as_array().unwrap();
        let resized_exactly = source_size
            .iter()
            .zip(model.input_info()[0].shape[2..].iter().rev())
            .all(|(source, target)| source.as_u64().unwrap() as usize == *target);
        let tolerance = if resized_exactly {
            IDENTITY_LOGIT_TOLERANCE
        } else {
            RESAMPLED_LOGIT_TOLERANCE
        };
        assert!(
            (logit - expected).abs() <= tolerance,
            "{name}: runtime logit {logit} differs from PyTorch {expected} by more than {tolerance}"
        );

        let probability = 1.0 / (1.0 + (-logit).exp());
        let expected_probability = 1.0 / (1.0 + (-expected).exp());
        assert_eq!(
            probability >= threshold,
            expected_probability >= threshold,
            "{name}: runtime probability {probability} and PyTorch {expected_probability} disagree at threshold {threshold}"
        );
    }
}
