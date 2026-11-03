//! CPU coverage for the Frame Presence V8 forward pass.

use std::path::{Path, PathBuf};

use subfast_net_rt::backend::Backend;
use subfast_net_rt::backend::cpu::CpuBackend;
use subfast_net_rt::backend::frame_presence::FramePresenceCpu;
use subfast_net_rt::ir::tensor::{DType, Layout, TensorId, TensorInfo, TensorKind, WeightRef};
use subfast_net_rt::model::weights::WeightStore;
use subfast_net_rt::model::{Model, ModelType};
use subfast_net_rt::tensor::Tensor;

/// Exact exported parameter set of architecture version 8 at `width = 24`.
const PARAMETERS: &[(&str, &[usize])] = &[
    ("stem.0.weight", &[16, 3, 5, 5]),
    ("stem.0.bias", &[16]),
    ("detail.0.weight", &[24, 16, 3, 3]),
    ("detail.0.bias", &[24]),
    ("region_encoder.0.weight", &[32, 24, 3, 3]),
    ("region_encoder.0.bias", &[32]),
    ("context_encoder.0.weight", &[48, 32, 3, 3]),
    ("context_encoder.0.bias", &[48]),
    ("context_refine.0.weight", &[48, 48, 3, 3]),
    ("context_refine.0.bias", &[48]),
    ("context_projection.weight", &[32, 48, 1, 1]),
    ("context_projection.bias", &[32]),
    ("region_head.weight", &[1, 32, 1, 1]),
    ("region_head.bias", &[1]),
    ("local_evidence_aggregator.log_scale", &[]),
    ("local_evidence_aggregator.bias", &[]),
];

const SCALAR_TOLERANCE: f32 = 2e-5;

fn synthetic_model(shape: [usize; 4]) -> Model {
    let mut tensors = Vec::new();
    let mut weights = Vec::new();
    let input = push_io(
        &mut tensors,
        "images",
        TensorKind::UserInput,
        shape.to_vec(),
    );
    for (serial, (target, parameter_shape)) in PARAMETERS.iter().enumerate() {
        let values = match *target {
            "local_evidence_aggregator.log_scale" => vec![0.0],
            "local_evidence_aggregator.bias" => vec![0.025],
            _ => (0..parameter_shape.iter().product::<usize>())
                .map(|index| {
                    let bucket = (index.wrapping_mul(7) + serial.wrapping_mul(11)) % 19;
                    (bucket as f32 - 9.0) * 0.01
                })
                .collect(),
        };
        push_parameter(&mut tensors, &mut weights, target, parameter_shape, &values);
    }
    let output = push_io(
        &mut tensors,
        "presence_logit",
        TensorKind::UserOutput,
        vec![1],
    );
    finish(
        tensors,
        weights,
        vec![input],
        output,
        ModelType::FramePresence,
    )
}

fn push_io(
    tensors: &mut Vec<TensorInfo>,
    name: &str,
    kind: TensorKind,
    shape: Vec<usize>,
) -> TensorId {
    let id = TensorId(tensors.len());
    tensors.push(TensorInfo {
        id,
        name: name.into(),
        dtype: DType::Float32,
        shape,
        layout: Layout::DenseRowMajor,
        kind,
        target: None,
        weight: None,
    });
    id
}

fn push_parameter(
    tensors: &mut Vec<TensorInfo>,
    weights: &mut Vec<u8>,
    target: &str,
    shape: &[usize],
    values: &[f32],
) {
    assert_eq!(shape.iter().product::<usize>(), values.len());
    let offset = weights.len();
    for value in values {
        weights.extend_from_slice(&value.to_le_bytes());
    }
    let id = TensorId(tensors.len());
    tensors.push(TensorInfo {
        id,
        name: target.into(),
        dtype: DType::Float32,
        shape: shape.to_vec(),
        layout: Layout::DenseRowMajor,
        kind: TensorKind::Parameter,
        target: Some(target.into()),
        weight: Some(WeightRef {
            offset,
            byte_length: weights.len() - offset,
        }),
    });
}

fn finish(
    tensors: Vec<TensorInfo>,
    weights: Vec<u8>,
    inputs: Vec<TensorId>,
    output: TensorId,
    model_type: ModelType,
) -> Model {
    Model {
        model_type,
        tensors,
        inputs,
        outputs: vec![output],
        weights: WeightStore::from_bytes(weights),
        preprocessing: None,
        decision_threshold: Some(0.5),
    }
}

fn deterministic_input(shape: &[usize], seed: u32) -> Tensor {
    let mut state = seed;
    let values = (0..shape.iter().product())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
        })
        .collect();
    Tensor::from_f32(shape.to_vec(), values).unwrap()
}

fn packed_input(shape: &[usize], seeds: &[u32]) -> Tensor {
    let mut packed_shape = shape.to_vec();
    packed_shape[0] = seeds.len();
    let mut values = Vec::new();
    for seed in seeds {
        values.extend(deterministic_input(shape, *seed).f32_values().unwrap());
    }
    Tensor::from_f32(packed_shape, values).unwrap()
}

#[test]
fn synthetic_frame_presence_is_finite_and_repeatable() {
    let model = synthetic_model([1, 3, 32, 48]);
    let input = deterministic_input(&model.input_info()[0].shape, 0x5eed_1234);
    let mut executable = CpuBackend.compile(&model).unwrap();
    let first = executable.run_scalar(std::slice::from_ref(&input)).unwrap();
    let second = executable.run_scalar(std::slice::from_ref(&input)).unwrap();
    assert!(first.is_finite(), "{first}");
    assert_eq!(first, second);

    let scalar_one = executable.run_scalar(std::slice::from_ref(&input)).unwrap();
    let scalar_two = executable
        .run_scalar(&[deterministic_input(&[1, 3, 32, 48], 0x0bad_c0de)])
        .unwrap();
    let packed = packed_input(&[1, 3, 32, 48], &[0x5eed_1234, 0x0bad_c0de]);
    let batch = executable.run_batch_scalar(&[packed]).unwrap();
    assert_eq!(batch.len(), 2);
    assert!((batch[0] - scalar_one).abs() <= SCALAR_TOLERANCE);
    assert!((batch[1] - scalar_two).abs() <= SCALAR_TOLERANCE);

    // A batch that reaches the persistent worker pool must agree with the
    // single-sample path as well.
    let seeds = (0..8)
        .map(|index| 0x1000 + index as u32)
        .collect::<Vec<_>>();
    let wide = packed_input(&[1, 3, 32, 48], &seeds);
    let mut expected = Vec::new();
    for seed in &seeds {
        expected.push(
            executable
                .run_scalar(&[deterministic_input(&[1, 3, 32, 48], *seed)])
                .unwrap(),
        );
    }
    let batched = executable.run_batch_scalar(&[wide]).unwrap();
    for (batched, expected) in batched.iter().zip(&expected) {
        assert!(
            (batched - expected).abs() <= SCALAR_TOLERANCE,
            "{batched} vs {expected}"
        );
    }
}

#[test]
fn synthetic_frame_presence_handles_dynamic_input_shapes() {
    for shape in [[1, 3, 32, 48], [1, 3, 64, 80]] {
        let model = synthetic_model(shape);
        let mut executable = FramePresenceCpu::compile(&model).unwrap();
        assert_eq!(
            executable.region_shape(),
            [shape[2] / 8, shape[3] / 8],
            "unexpected region grid for {shape:?}"
        );
        let input = deterministic_input(&shape, 0x2222_3333);
        let frame = executable.run_nchw_bytes_with_frame(&input.data).unwrap();
        assert!(frame.presence_logit.is_finite());
        assert_eq!(
            frame.region_logits.len(),
            frame.region_shape[0] * frame.region_shape[1]
        );
        assert!(frame.region_logits.iter().all(|value| value.is_finite()));
        let repeated = executable.run_nchw_bytes(&input.data).unwrap();
        assert_eq!(frame.presence_logit, repeated);
    }
}

#[test]
fn rejects_mismatched_model_types() {
    let model = synthetic_model([1, 3, 32, 48]);
    let mismatched = Model {
        model_type: ModelType::RoiPresence,
        ..model.clone()
    };
    assert!(CpuBackend.compile(&mismatched).is_err());
    assert!(FramePresenceCpu::compile(&mismatched).is_err());
    assert!(CpuBackend.compile(&model).is_ok());
}

fn fixture(variable: &str) -> Option<PathBuf> {
    let path = std::env::var_os(variable).map(PathBuf::from)?;
    if !path.exists() {
        panic!("{variable} does not point to an existing path");
    }
    Some(path)
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn dims(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("shape array")
        .iter()
        .map(|dimension| dimension.as_u64().expect("shape dimension") as usize)
        .collect()
}

fn assert_close(actual: f32, expected: f32, label: &str) -> f32 {
    let error = (actual - expected).abs();
    let tolerance = 2e-4 + 2e-4 * expected.abs();
    assert!(
        error <= tolerance,
        "{label}: {actual} differs from {expected} by {error} (tolerance {tolerance})"
    );
    error
}

#[test]
fn real_model_matches_pytorch_reference() {
    let (Some(model_dir), Some(cases_dir)) = (
        fixture("SUBFAST_RT_FRAME_PRESENCE_DIR"),
        fixture("SUBFAST_RT_FRAME_PRESENCE_CASES"),
    ) else {
        eprintln!(
            "skipping frame-presence parity; set SUBFAST_RT_FRAME_PRESENCE_DIR and SUBFAST_RT_FRAME_PRESENCE_CASES"
        );
        return;
    };
    let model = Model::load(&model_dir).unwrap();
    let mut executable = CpuBackend.compile(&model).unwrap();
    let mut direct = FramePresenceCpu::compile(&model).unwrap();
    let cases: serde_json::Value =
        serde_json::from_slice(&std::fs::read(cases_dir.join("cases.json")).unwrap()).unwrap();
    let cases = cases.as_array().expect("cases.json must be an array");
    assert!(!cases.is_empty());

    let mut worst_presence = 0.0f32;
    let mut worst_region = 0.0f32;
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let input_dims = dims(&case["input"]["shape"]);
        let mut runtime_dims = vec![1];
        runtime_dims.extend(&input_dims);
        let input = Tensor::from_f32(
            runtime_dims,
            read_f32(&cases_dir.join(case["input"]["file"].as_str().unwrap())),
        )
        .unwrap();
        let expected_presence = case["presence_logit"].as_f64().unwrap() as f32;
        let actual_presence = executable.run_scalar(std::slice::from_ref(&input)).unwrap();
        worst_presence = worst_presence.max(assert_close(
            actual_presence,
            expected_presence,
            &format!("{name} presence logit"),
        ));

        let frame = direct.run_nchw_bytes_with_frame(&input.data).unwrap();
        assert_eq!(frame.presence_logit, actual_presence);
        let region_dims = dims(&case["region_logits"]["shape"]);
        assert_eq!(
            frame.region_shape,
            [
                region_dims[region_dims.len() - 2],
                region_dims[region_dims.len() - 1]
            ]
        );
        let expected_region =
            read_f32(&cases_dir.join(case["region_logits"]["file"].as_str().unwrap()));
        assert_eq!(expected_region.len(), frame.region_logits.len());
        for (index, (actual, expected)) in
            frame.region_logits.iter().zip(&expected_region).enumerate()
        {
            let error = assert_close(*actual, *expected, &format!("{name} region[{index}]"));
            worst_region = worst_region.max(error);
        }
        eprintln!("{name}: presence={actual_presence:.7} expected={expected_presence:.7}");
    }
    eprintln!("max abs error: presence {worst_presence:.3e}, region {worst_region:.3e}");
    assert!(worst_presence <= 2e-4 * (1.0 + 30.0));
    assert!(worst_region <= 2e-4 * (1.0 + 30.0));
}

#[cfg(feature = "gpu")]
#[test]
fn frame_presence_gpu_matches_pytorch_and_handles_odd_batches() {
    let (Some(model_dir), Some(cases_dir)) = (
        fixture("SUBFAST_RT_FRAME_PRESENCE_DIR"),
        fixture("SUBFAST_RT_FRAME_PRESENCE_CASES"),
    ) else {
        return;
    };
    if std::env::var_os("SUBFAST_RT_TEST_GPU").is_none() {
        return;
    }
    use subfast_net_rt::backend::gpu::GpuBackend;

    let model = Model::load(&model_dir).unwrap();
    let mut gpu = GpuBackend::with_max_batch_size(32)
        .unwrap()
        .compile(&model)
        .unwrap();
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(cases_dir.join("cases.json")).unwrap()).unwrap();
    assert!(!cases.is_empty());
    let mut values = Vec::new();
    let mut expected = Vec::new();
    for case in &cases {
        let mut shape = vec![1];
        shape.extend(dims(&case["input"]["shape"]));
        let input_values = read_f32(&cases_dir.join(case["input"]["file"].as_str().unwrap()));
        let input = Tensor::from_f32(shape, input_values.clone()).unwrap();
        let expected_logit = case["presence_logit"].as_f64().unwrap() as f32;
        assert_close(
            gpu.run_scalar(&[input]).unwrap(),
            expected_logit,
            case["name"].as_str().unwrap(),
        );
        expected.push(expected_logit);
        values.extend(input_values);
    }
    let mut shape = model.input_info()[0].shape.clone();
    let elements = shape[1..].iter().product::<usize>();
    // Switch between the host-interleaved Metal plan and the GPU-transpose
    // plan on the same reusable executable. Include a partial batch.
    for batch in [4, 8, 3, 16, 32, 1, 8, 32] {
        shape[0] = batch;
        let order: Vec<_> = (0..batch)
            .map(|sample| (sample + sample / 8) % expected.len())
            .collect();
        let packed_values = order
            .iter()
            .flat_map(|&case| {
                values[case * elements..(case + 1) * elements]
                    .iter()
                    .copied()
            })
            .collect();
        let packed = Tensor::from_f32(shape.clone(), packed_values).unwrap();
        let output = gpu.run_batch_scalar(&[packed]).unwrap();
        assert_eq!(output.len(), batch);
        for (actual, case) in output.into_iter().zip(order) {
            assert_close(actual, expected[case], &format!("GPU batch {batch}"));
        }
    }

    // Stride-16 context rounds up when a dimension is only divisible by eight.
    let model = synthetic_model([1, 3, 40, 56]);
    let input = deterministic_input(&[1, 3, 40, 56], 0x2222_3333);
    let expected = CpuBackend
        .compile(&model)
        .unwrap()
        .run_scalar(std::slice::from_ref(&input))
        .unwrap();
    let actual = GpuBackend::default()
        .compile(&model)
        .unwrap()
        .run_scalar(&[input])
        .unwrap();
    assert_close(actual, expected, "odd context dimensions");
}
