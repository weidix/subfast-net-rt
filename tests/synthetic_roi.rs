mod support;

use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;

use support::{
    deterministic_input, deterministic_input_with_shape, deterministic_pair_input, packed_input,
    packed_pair_input, synthetic_pair_model, synthetic_pair_model_with_shape,
    synthetic_presence_model, synthetic_presence_model_with_shape,
};

const BATCH_SIZES: [usize; 5] = [0, 1, 3, 8, 17];
const TOLERANCE: f32 = 2e-5;

#[test]
fn cpu_uses_dynamic_model_input_dimensions() {
    for (model, seed_count) in [
        (synthetic_presence_model_with_shape([1, 3, 48, 320]), 1usize),
        (synthetic_pair_model_with_shape([1, 3, 96, 320]), 2usize),
    ] {
        let shape = model.input_info()[0].shape.clone();
        let inputs = (0..seed_count)
            .map(|index| deterministic_input_with_shape(&shape, 0x1234_5678 ^ index as u32))
            .collect::<Vec<_>>();
        let mut executable = CpuBackend.compile(&model).unwrap();
        assert!(executable.run_scalar(&inputs).unwrap().is_finite());
        assert_eq!(executable.input_shapes()[0], shape);
    }
}

#[test]
fn backend_rejects_model_type_mismatch() {
    let mut presence = synthetic_presence_model();
    presence.model_type = subfast_net_rt::model::ModelType::RoiPairMatcher;
    assert!(CpuBackend.compile(&presence).is_err());

    let mut pair = synthetic_pair_model();
    pair.model_type = subfast_net_rt::model::ModelType::RoiPresence;
    assert!(CpuBackend.compile(&pair).is_err());
}

#[test]
fn synthetic_presence_compiles_and_is_finite_and_repeatable() {
    let model = synthetic_presence_model();
    let input = deterministic_input(0x1234_5678);
    let mut executable = CpuBackend.compile(&model).unwrap();

    let first = executable.run_scalar(std::slice::from_ref(&input)).unwrap();
    let repeated = executable.run_scalar(&[input]).unwrap();

    assert!(first.is_finite());
    assert_eq!(first, repeated);

    for batch_size in BATCH_SIZES {
        let seeds = seeds(batch_size, 0x3141_5926);
        let expected = seeds
            .iter()
            .map(|seed| {
                let input = deterministic_input(*seed);
                executable.run_scalar(std::slice::from_ref(&input)).unwrap()
            })
            .collect::<Vec<_>>();
        let packed = packed_input(&seeds);
        let actual = executable
            .run_batch_scalar(std::slice::from_ref(&packed))
            .unwrap();
        assert_batch_matches(batch_size, &actual, &expected);
        let mut repeated = vec![0.0; batch_size];
        executable
            .run_batch_scalar_into(std::slice::from_ref(&packed), &mut repeated)
            .unwrap();
        assert_eq!(
            actual, repeated,
            "presence batch {batch_size} was not repeatable"
        );
    }

    assert!(executable.run_batch_scalar(&[]).is_err());
    let wrong_shape =
        subfast_net_rt::Tensor::from_f32(vec![2, 3, 32, 255], vec![0.0; 2 * 3 * 32 * 255]).unwrap();
    assert!(
        executable
            .run_batch_scalar(std::slice::from_ref(&wrong_shape))
            .is_err()
    );
    let mut truncated = packed_input(&[1, 2]);
    truncated.data.pop();
    assert!(
        executable
            .run_batch_scalar(std::slice::from_ref(&truncated))
            .is_err()
    );
}

#[test]
fn synthetic_pair_compiles_and_is_finite_repeatable_and_symmetric() {
    let model = synthetic_pair_model();
    let left = deterministic_pair_input(0x1357_9bdf);
    let right = deterministic_pair_input(0x2468_ace0);
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

    for batch_size in BATCH_SIZES {
        let left_seeds = seeds(batch_size, 0x2718_2818);
        let right_seeds = seeds(batch_size, 0xdead_beef);
        let expected = left_seeds
            .iter()
            .zip(&right_seeds)
            .map(|(left_seed, right_seed)| {
                let left = deterministic_pair_input(*left_seed);
                let right = deterministic_pair_input(*right_seed);
                executable.run_scalar(&[left, right]).unwrap()
            })
            .collect::<Vec<_>>();
        let packed_left = packed_pair_input(&left_seeds);
        let packed_right = packed_pair_input(&right_seeds);
        let actual = executable
            .run_batch_scalar(&[packed_left.clone(), packed_right.clone()])
            .unwrap();
        assert_batch_matches(batch_size, &actual, &expected);
        let mut repeated = vec![0.0; batch_size];
        executable
            .run_batch_scalar_into(&[packed_left, packed_right], &mut repeated)
            .unwrap();
        assert_eq!(
            actual, repeated,
            "pair batch {batch_size} was not repeatable"
        );
    }

    let two = packed_input(&[1, 2]);
    let three = packed_input(&[3, 4, 5]);
    assert!(
        executable
            .run_batch_scalar(std::slice::from_ref(&two))
            .is_err()
    );
    assert!(executable.run_batch_scalar(&[two, three]).is_err());
    assert!(
        executable
            .run_batch_scalar_into(&[packed_input(&[1]), packed_input(&[2])], &mut [])
            .is_err()
    );
}

fn seeds(count: usize, base: u32) -> Vec<u32> {
    (0..count)
        .map(|index| base.wrapping_add((index as u32).wrapping_mul(0x9e37_79b9)))
        .collect()
}

fn assert_batch_matches(batch_size: usize, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), batch_size);
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= TOLERANCE,
            "batch {batch_size} output {index}: actual={actual}, expected={expected}"
        );
    }
}
