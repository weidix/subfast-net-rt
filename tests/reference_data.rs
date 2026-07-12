use subfast_net_rt::ir::tensor::DType;
use subfast_net_rt::model::reference::{
    ReferenceTolerance, compare_tensors, default_tolerance_for, load_case, load_tensor_file,
};
use subfast_net_rt::tensor::Tensor;

#[test]
fn compares_float32_reference_with_tolerance() {
    let actual = Tensor::from_f32(vec![2], vec![1.0, 2.00001]).unwrap();
    let expected = Tensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap();
    compare_tensors(
        &actual,
        &expected,
        ReferenceTolerance {
            atol: 1e-4,
            rtol: 1e-4,
        },
    )
    .unwrap();
}

#[test]
fn loads_reference_case_and_binary_tensor() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("test_input.bin"), 1.0f32.to_le_bytes()).unwrap();
    std::fs::write(
        tmp.path().join("test_case.json"),
        r#"{
          "input": {"file": "test_input.bin", "dtype": "float32", "shape": [1]},
          "expected_output": {"file": "expected_output.bin", "dtype": "float32", "shape": [1]}
        }"#,
    )
    .unwrap();
    let case = load_case(tmp.path().join("test_case.json")).unwrap();
    let input = load_tensor_file(tmp.path(), &case.input).unwrap();
    assert_eq!(input.f32_values().unwrap(), [1.0]);
    assert_eq!(default_tolerance_for(DType::Float16).atol, 1e-3);
}
