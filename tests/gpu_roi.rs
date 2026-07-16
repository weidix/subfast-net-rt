#![cfg(feature = "gpu")]

mod support;

use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::{CpuBackend, CpuExecutable};
use subfast_net_rt::backend::gpu::{GpuBackend, GpuExecutable};
use subfast_net_rt::model::Model;

use support::{
    deterministic_input, deterministic_input_with_shape, deterministic_pair_input, packed_input,
    packed_input_with_shape, packed_pair_input, synthetic_pair_model,
    synthetic_pair_model_with_shape, synthetic_presence_model, synthetic_presence_model_with_shape,
};

const MIXED_BATCH_SIZES: [usize; 5] = [32, 1, 17, 0, 32];

#[test]
fn dynamic_dimensions_match_cpu_on_gpu() {
    for (model, input_count) in [
        (synthetic_presence_model_with_shape([1, 3, 48, 320]), 1usize),
        (synthetic_pair_model_with_shape([1, 3, 96, 320]), 2usize),
    ] {
        let shape = model.input_info()[0].shape.clone();
        let inputs = (0..input_count)
            .map(|index| deterministic_input_with_shape(&shape, 0x3141_5926 ^ index as u32))
            .collect::<Vec<_>>();
        let mut cpu = CpuBackend.compile(&model).unwrap();
        let expected = cpu.run_scalar(&inputs).unwrap();
        let mut gpu = GpuBackend::with_max_batch_size(1)
            .unwrap()
            .compile(&model)
            .unwrap();
        assert_close("dynamic shape", gpu.run_scalar(&inputs).unwrap(), expected);
        assert_eq!(gpu.input_shapes()[0], shape);
    }
}

#[test]
fn metal_or_vulkan_matches_cpu_for_both_roi_models() {
    assert!(GpuBackend::with_max_batch_size(0).is_err());

    let presence = synthetic_presence_model();
    let presence_input = deterministic_input(0x1234_5678);
    let mut presence_cpu = CpuBackend.compile(&presence).unwrap();
    let expected = presence_cpu
        .run_scalar(std::slice::from_ref(&presence_input))
        .unwrap();
    let mut presence_gpu = GpuBackend::with_max_batch_size(32)
        .unwrap()
        .compile(&presence)
        .unwrap();
    assert_platform_backend(presence_gpu.backend_name());
    assert!(!presence_gpu.adapter_name().is_empty());
    assert_eq!(presence_gpu.max_batch_size(), 32);
    let actual = presence_gpu
        .run_scalar(std::slice::from_ref(&presence_input))
        .unwrap();
    assert_close("presence", actual, expected);
    assert_eq!(
        actual,
        presence_gpu
            .run_scalar(std::slice::from_ref(&presence_input))
            .unwrap()
    );

    let presence_seeds = [0x1020_3040, 0x5060_7080, 0x90a0_b0c0];
    let presence_expected = presence_seeds
        .iter()
        .map(|seed| {
            let input = deterministic_input(*seed);
            presence_cpu
                .run_scalar(std::slice::from_ref(&input))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let presence_actual = presence_gpu
        .run_batch_scalar(std::slice::from_ref(&packed_input(&presence_seeds)))
        .unwrap();
    assert_batch_close("presence", &presence_actual, &presence_expected);
    let mut presence_into = [0.0; 3];
    presence_gpu
        .run_batch_scalar_into(
            std::slice::from_ref(&packed_input(&presence_seeds)),
            &mut presence_into,
        )
        .unwrap();
    assert_eq!(presence_actual, presence_into);
    let empty = packed_input(&[]);
    assert!(
        presence_gpu
            .run_batch_scalar(std::slice::from_ref(&empty))
            .unwrap()
            .is_empty()
    );
    presence_gpu
        .run_batch_scalar_into(std::slice::from_ref(&empty), &mut [])
        .unwrap();
    assert_presence_mixed_batches(&mut presence_cpu, &mut presence_gpu);
    let oversized_presence = batch_seeds(33, 0, 0);
    assert!(
        presence_gpu
            .run_batch_scalar(std::slice::from_ref(&packed_input(&oversized_presence)))
            .is_err()
    );
    assert!(
        presence_gpu
            .run_batch_scalar_into(
                std::slice::from_ref(&packed_input(&presence_seeds)),
                &mut [0.0; 2],
            )
            .is_err()
    );

    let pair = synthetic_pair_model();
    let left = deterministic_pair_input(0x1357_9bdf);
    let right = deterministic_pair_input(0x2468_ace0);
    let mut pair_cpu = CpuBackend.compile(&pair).unwrap();
    let expected = pair_cpu.run_scalar(&[left.clone(), right.clone()]).unwrap();
    let mut pair_gpu = GpuBackend::with_max_batch_size(32)
        .unwrap()
        .compile(&pair)
        .unwrap();
    assert_platform_backend(pair_gpu.backend_name());
    let actual = pair_gpu.run_scalar(&[left.clone(), right.clone()]).unwrap();
    assert_close("pair", actual, expected);
    let swapped = pair_gpu.run_scalar(&[right, left]).unwrap();
    assert_close("pair swapped-input symmetry", swapped, actual);
    assert_eq!(
        actual,
        pair_gpu
            .run_scalar(&[
                deterministic_pair_input(0x1357_9bdf),
                deterministic_pair_input(0x2468_ace0),
            ])
            .unwrap()
    );

    let left_seeds = [0x0102_0304, 0x1112_1314, 0x2122_2324];
    let right_seeds = [0xa1a2_a3a4, 0xb1b2_b3b4, 0xc1c2_c3c4];
    let pair_expected = left_seeds
        .iter()
        .zip(right_seeds)
        .map(|(left_seed, right_seed)| {
            pair_cpu
                .run_scalar(&[
                    deterministic_pair_input(*left_seed),
                    deterministic_pair_input(right_seed),
                ])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let pair_actual = pair_gpu
        .run_batch_scalar(&[
            packed_pair_input(&left_seeds),
            packed_pair_input(&right_seeds),
        ])
        .unwrap();
    assert_batch_close("pair", &pair_actual, &pair_expected);
    let mut pair_into = [0.0; 3];
    pair_gpu
        .run_batch_scalar_into(
            &[
                packed_pair_input(&left_seeds),
                packed_pair_input(&right_seeds),
            ],
            &mut pair_into,
        )
        .unwrap();
    assert_eq!(pair_actual, pair_into);
    assert_pair_mixed_batches(&mut pair_cpu, &mut pair_gpu);
    assert!(
        pair_gpu
            .run_batch_scalar(std::slice::from_ref(&packed_pair_input(&[1])))
            .is_err()
    );
    assert!(
        pair_gpu
            .run_batch_scalar(&[packed_pair_input(&[1, 2]), packed_pair_input(&[3])])
            .is_err()
    );
}

fn assert_presence_mixed_batches(cpu: &mut CpuExecutable, gpu: &mut GpuExecutable) {
    for (run, batch_size) in MIXED_BATCH_SIZES.into_iter().enumerate() {
        let seeds = batch_seeds(batch_size, 0, run);
        let input = packed_input(&seeds);
        let inputs = std::slice::from_ref(&input);
        let expected = cpu.run_batch_scalar(inputs).unwrap();
        let actual = gpu.run_batch_scalar(inputs).unwrap();
        assert_batch_close(
            &format!("presence mixed run {run} batch {batch_size}"),
            &actual,
            &expected,
        );
    }
}

fn assert_pair_mixed_batches(cpu: &mut CpuExecutable, gpu: &mut GpuExecutable) {
    for (run, batch_size) in MIXED_BATCH_SIZES.into_iter().enumerate() {
        let left_seeds = batch_seeds(batch_size, 1, run);
        let right_seeds = batch_seeds(batch_size, 2, run);
        let inputs = [
            packed_pair_input(&left_seeds),
            packed_pair_input(&right_seeds),
        ];
        let expected = cpu.run_batch_scalar(&inputs).unwrap();
        let actual = gpu.run_batch_scalar(&inputs).unwrap();
        assert_batch_close(
            &format!("pair mixed run {run} batch {batch_size}"),
            &actual,
            &expected,
        );
    }
}

fn batch_seeds(batch_size: usize, stream: u32, run: usize) -> Vec<u32> {
    (0..batch_size)
        .map(|index| {
            0x1020_3040u32
                .wrapping_add(stream.wrapping_mul(0x1357_9bdf))
                .wrapping_add((run as u32).wrapping_mul(0x85eb_ca6b))
                .wrapping_add((index as u32).wrapping_mul(0x9e37_79b9))
        })
        .collect()
}

#[test]
fn configured_real_models_match_metal_or_vulkan() {
    for (variable, input_count) in [
        ("SUBFAST_RT_PRESENCE_DIR", 1usize),
        ("SUBFAST_RT_PAIR_DIR", 2usize),
    ] {
        let Some(path) = std::env::var_os(variable).map(std::path::PathBuf::from) else {
            eprintln!("skipping GPU fixture; set {variable}");
            continue;
        };
        let model = Model::load(&path).unwrap();
        let shape = model.input_info()[0].shape.clone();
        let inputs = (0..input_count)
            .map(|index| {
                let seed = 0x4242_1000 ^ (index as u32 * 0x1020_3040);
                deterministic_input_with_shape(&shape, seed)
            })
            .collect::<Vec<_>>();
        let mut cpu = CpuBackend.compile(&model).unwrap();
        let expected = cpu.run_scalar(&inputs).unwrap();
        let mut gpu = GpuBackend::with_max_batch_size(3)
            .unwrap()
            .compile(&model)
            .unwrap();
        assert_platform_backend(gpu.backend_name());
        assert_close(variable, gpu.run_scalar(&inputs).unwrap(), expected);

        let seed_sets = (0..input_count)
            .map(|input_index| {
                [0x1020_3040, 0x5060_7080, 0x90a0_b0c0]
                    .map(|seed| seed ^ (input_index as u32).wrapping_mul(0x1357_9bdf))
            })
            .collect::<Vec<_>>();
        let batch_expected = (0..3)
            .map(|sample| {
                let sample_inputs = seed_sets
                    .iter()
                    .map(|seeds| deterministic_input_with_shape(&shape, seeds[sample]))
                    .collect::<Vec<_>>();
                cpu.run_scalar(&sample_inputs).unwrap()
            })
            .collect::<Vec<_>>();
        let packed = seed_sets
            .iter()
            .map(|seeds| packed_input_with_shape(&shape, seeds))
            .collect::<Vec<_>>();
        let batch_actual = gpu.run_batch_scalar(&packed).unwrap();
        assert_batch_close(variable, &batch_actual, &batch_expected);
    }
}

#[test]
fn configured_real_models_use_dimensions_from_config() {
    for (variable, dimensions, input_count) in [
        ("SUBFAST_RT_PRESENCE_DIR", [48u64, 320], 1usize),
        ("SUBFAST_RT_PAIR_DIR", [96u64, 320], 2usize),
    ] {
        let Some(source) = std::env::var_os(variable).map(std::path::PathBuf::from) else {
            eprintln!("skipping dynamic GPU fixture; set {variable}");
            continue;
        };
        let temp = tempfile::tempdir().unwrap();
        std::fs::copy(
            source.join("model.safetensors"),
            temp.path().join("model.safetensors"),
        )
        .unwrap();
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap()).unwrap();
        for tensor in config["input"]["tensors"].as_array_mut().unwrap() {
            tensor["shape"][2] = dimensions[0].into();
            tensor["shape"][3] = dimensions[1].into();
        }
        std::fs::write(
            temp.path().join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let model = Model::load(temp.path()).unwrap();
        let shape = model.input_info()[0].shape.clone();
        assert_eq!(shape[2..], dimensions.map(|value| value as usize));
        let inputs = (0..input_count)
            .map(|index| deterministic_input_with_shape(&shape, 0x4242_1000 ^ index as u32))
            .collect::<Vec<_>>();
        let mut cpu = CpuBackend.compile(&model).unwrap();
        let expected = cpu.run_scalar(&inputs).unwrap();
        let mut gpu = GpuBackend::default().compile(&model).unwrap();
        assert_close(variable, gpu.run_scalar(&inputs).unwrap(), expected);
    }
}

fn assert_platform_backend(actual: &str) {
    #[cfg(target_os = "macos")]
    assert_eq!(actual, "metal");
    #[cfg(not(target_os = "macos"))]
    assert_eq!(actual, "vulkan");
}

fn assert_batch_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(&format!("{name}[{index}]"), actual, expected);
    }
}

fn assert_close(name: &str, actual: f32, expected: f32) {
    let tolerance = 5e-4f32.max(expected.abs() * 2e-4);
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance,
        "{name}: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}
