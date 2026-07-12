use std::path::PathBuf;

use subfast_net_rt::error::RtError;
use subfast_net_rt::ir::op::Op;
use subfast_net_rt::model::UnifiedModel;

fn fixture_paths() -> Option<(PathBuf, PathBuf)> {
    let json = std::env::var_os("SUBFAST_RT_MODEL_JSON").map(PathBuf::from)?;
    let model_path = std::env::var_os("SUBFAST_RT_MODEL_PATH").map(PathBuf::from)?;
    Some((json, model_path))
}

macro_rules! require_fixture_paths {
    () => {
        match fixture_paths() {
            Some(paths) => paths,
            None => {
                eprintln!(
                    "skipping real model fixture test; set SUBFAST_RT_MODEL_JSON and SUBFAST_RT_MODEL_PATH"
                );
                return;
            }
        }
    };
}

#[test]
fn loads_real_unified_model_and_normalizes_ir() {
    let (json, bin) = require_fixture_paths!();
    let model = UnifiedModel::load(&json, &bin).expect("real model should load");

    assert_eq!(model.format, "subfast-net.unified-model");
    assert_eq!(model.format_version, 1);
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.outputs.len(), 1);
    assert_eq!(model.graph.nodes.len(), 32);
    assert_eq!(model.tensors.len(), 89);
    assert_eq!(model.input_info()[0].name, "x");
    assert_eq!(model.input_info()[0].shape, [1, 3, 512, 512]);
    assert_eq!(model.output_info()[0].name, "upsample_bilinear2d");
    assert_eq!(model.output_info()[0].shape, [1, 2, 512, 512]);

    let mut conv = 0;
    let mut bn = 0;
    let mut silu = 0;
    let mut add = 0;
    let mut resize = 0;
    for node in &model.graph.nodes {
        match node.op {
            Op::Conv2d { .. } => conv += 1,
            Op::BatchNorm { .. } => bn += 1,
            Op::Silu => silu += 1,
            Op::Add { .. } => add += 1,
            Op::UpsampleBilinear2d { .. } => resize += 1,
            _ => {}
        }
    }
    assert_eq!((conv, bn, silu, add, resize), (10, 9, 9, 3, 1));
    assert_eq!(model.weights.total_bytes(), 201_040);
}

#[test]
fn rejects_truncated_weight_file_with_bounds_error() {
    let (json, bin) = require_fixture_paths!();
    let tmp = tempfile::tempdir().unwrap();
    let truncated = tmp.path().join("weights.bin");
    let bytes = std::fs::read(bin).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() - 1]).unwrap();

    let err = UnifiedModel::load(json, truncated).unwrap_err();
    assert!(matches!(err, RtError::WeightsOutOfBounds { .. }));
}

#[test]
fn unsupported_op_error_contains_node_op_and_input_info() {
    let (json, bin) = require_fixture_paths!();
    let tmp = tempfile::tempdir().unwrap();
    let mutated = tmp.path().join("model.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap();
    value["nodes"][0]["op"] = serde_json::Value::String("aten.relu.default".into());
    std::fs::write(&mutated, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    let err = UnifiedModel::load(mutated, bin).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("conv2d"));
    assert!(msg.contains("aten.relu.default"));
    assert!(msg.contains("x"));
    assert!(msg.contains("[1, 3, 512, 512]"));
}

#[test]
fn rejects_illegal_dtype_layout_shape_and_missing_input_reference() {
    let (json, bin) = require_fixture_paths!();

    type JsonMutation = fn(&mut serde_json::Value);
    let cases: [(&str, JsonMutation); 4] = [
        ("dtype", |value: &mut serde_json::Value| {
            value["tensors"]["p_stem_0_weight"]["dtype"] = "float64".into();
        }),
        ("layout", |value: &mut serde_json::Value| {
            value["tensors"]["p_stem_0_weight"]["layout"] = "channels_last".into();
        }),
        ("shape", |value: &mut serde_json::Value| {
            value["tensors"]["p_stem_0_weight"]["shape"] = serde_json::json!([31, 3, 3, 3]);
        }),
        ("missing_input", |value: &mut serde_json::Value| {
            value["nodes"][0]["inputs"][0] = "missing_tensor".into();
        }),
    ];

    for (name, mutate) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let mutated = tmp.path().join(format!("{name}.json"));
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
        mutate(&mut value);
        std::fs::write(&mutated, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(
            UnifiedModel::load(mutated, &bin).is_err(),
            "case {name} should fail"
        );
    }
}

#[test]
fn rejects_weight_offset_that_exceeds_binary_size() {
    let (json, bin) = require_fixture_paths!();
    let tmp = tempfile::tempdir().unwrap();
    let mutated = tmp.path().join("bad_offset.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap();
    value["tensors"]["p_stem_0_weight"]["data"]["offset"] = serde_json::json!(999_999_999u64);
    std::fs::write(&mutated, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    let err = UnifiedModel::load(mutated, bin).unwrap_err();
    assert!(matches!(err, RtError::WeightsOutOfBounds { .. }));
}

#[test]
fn loader_uses_actual_weights_bin_without_filename_binding() {
    let (json, bin) = require_fixture_paths!();
    assert_eq!(bin.file_name().unwrap(), "weights.bin");
    UnifiedModel::load(json, bin).unwrap();
}
