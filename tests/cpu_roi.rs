use std::path::{Path, PathBuf};

use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::model::Model;
use subfast_net_rt::tensor::Tensor;

fn fixture(variable: &str) -> Option<PathBuf> {
    let path = std::env::var_os(variable).map(PathBuf::from)?;
    if path.join("config.json").is_file() && path.join("model.safetensors").is_file() {
        Some(path)
    } else {
        panic!("{variable} does not point to a safetensors model directory");
    }
}

fn load(path: &Path) -> Model {
    Model::load(path).unwrap()
}

fn deterministic_input(shape: &[usize], seed: u32) -> Tensor {
    let mut state = seed;
    let values = (0..shape.iter().product())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 * (4.0 / 16_777_216.0) - 2.0
        })
        .collect();
    Tensor::from_f32(shape.to_vec(), values).unwrap()
}

#[test]
fn presence_cpu_is_finite_and_repeatable() {
    let Some(path) = fixture("SUBFAST_RT_PRESENCE_DIR") else {
        eprintln!("skipping Presence fixture; set SUBFAST_RT_PRESENCE_DIR");
        return;
    };
    let model = load(&path);
    let input = deterministic_input(&model.input_info()[0].shape, 0x1234_5678);
    let mut executable = CpuBackend.compile(&model).unwrap();
    let first = executable.run_scalar(std::slice::from_ref(&input)).unwrap();
    let second = executable.run_scalar(&[input]).unwrap();
    assert!(first.is_finite());
    assert_eq!(first, second);
    assert!((first - 4.158_833_5).abs() <= 1e-4, "{first}");
}

#[test]
fn pair_cpu_is_finite_repeatable_and_symmetric() {
    let Some(path) = fixture("SUBFAST_RT_PAIR_DIR") else {
        eprintln!("skipping Pair fixture; set SUBFAST_RT_PAIR_DIR");
        return;
    };
    let model = load(&path);
    let shape = &model.input_info()[0].shape;
    let left = deterministic_input(shape, 0x1357_9bdf);
    let right = deterministic_input(shape, 0x2468_ace0);
    let mut executable = CpuBackend.compile(&model).unwrap();
    let forward = executable
        .run_scalar(&[left.clone(), right.clone()])
        .unwrap();
    let repeated = executable
        .run_scalar(&[left.clone(), right.clone()])
        .unwrap();
    let swapped = executable.run_scalar(&[right, left]).unwrap();
    assert!(forward.is_finite());
    assert_eq!(forward, repeated);
    assert!((forward - swapped).abs() <= 2e-5, "{forward} vs {swapped}");
    assert!((forward - -31.207_836).abs() <= 2e-4, "{forward}");
}
