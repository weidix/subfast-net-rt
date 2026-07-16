use std::path::{Path, PathBuf};

use subfast_net_rt::model::{Model, ModelType};

fn fixture(variable: &str) -> Option<PathBuf> {
    let path = std::env::var_os(variable).map(PathBuf::from)?;
    assert!(path.join("config.json").is_file());
    assert!(path.join("model.safetensors").is_file());
    Some(path)
}

fn copy_bundle(source: &Path, destination: &Path) {
    std::fs::copy(source.join("config.json"), destination.join("config.json")).unwrap();
    std::fs::copy(
        source.join("model.safetensors"),
        destination.join("model.safetensors"),
    )
    .unwrap();
}

#[test]
fn loads_presence_safetensors_bundle() {
    let Some(path) = fixture("SUBFAST_RT_PRESENCE_DIR") else {
        eprintln!("skipping Presence fixture; set SUBFAST_RT_PRESENCE_DIR");
        return;
    };
    let model = Model::load(path).unwrap();
    assert_eq!(model.model_type, ModelType::RoiPresence);
    assert_eq!(model.input_info()[0].name, "images");
    assert_eq!(model.input_info()[0].shape[..2], [1, 3]);
    assert!(
        model.input_info()[0].shape[2..]
            .iter()
            .all(|size| *size > 0)
    );
    assert_eq!(model.output_info()[0].name, "presence_logit");
    assert!(model.tensors.iter().any(|tensor| {
        tensor.target.as_deref() == Some("presence_head.evidence_pool.log_scale")
    }));
}

#[test]
fn loads_pair_bundle_and_folds_batch_norm() {
    let Some(path) = fixture("SUBFAST_RT_PAIR_DIR") else {
        eprintln!("skipping Pair fixture; set SUBFAST_RT_PAIR_DIR");
        return;
    };
    let model = Model::load(path).unwrap();
    assert_eq!(model.model_type, ModelType::RoiPairMatcher);
    assert_eq!(model.input_info().len(), 2);
    assert_eq!(model.input_info()[0].shape[..2], [1, 3]);
    assert_eq!(model.input_info()[0].shape, model.input_info()[1].shape);
    assert!(
        model
            .tensors
            .iter()
            .any(|tensor| tensor.target.as_deref() == Some("stem.0.bias"))
    );
    assert!(
        !model
            .tensors
            .iter()
            .any(|tensor| tensor.target.as_deref() == Some("stem.1.running_mean"))
    );
}

#[test]
fn rejects_truncated_safetensors_file() {
    let Some(path) = fixture("SUBFAST_RT_PRESENCE_DIR") else {
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    copy_bundle(&path, temp.path());
    let weights = temp.path().join("model.safetensors");
    let mut bytes = std::fs::read(&weights).unwrap();
    bytes.pop();
    std::fs::write(weights, bytes).unwrap();
    assert!(Model::load(temp.path()).is_err());
}

#[test]
fn rejects_wrong_config_format_without_fallback() {
    let Some(path) = fixture("SUBFAST_RT_PRESENCE_DIR") else {
        return;
    };
    let temp = tempfile::tempdir().unwrap();
    copy_bundle(&path, temp.path());
    let config_path = temp.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["format"] = "subfast-net.unified-model".into();
    std::fs::write(config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
    assert!(Model::load(temp.path()).is_err());
}
