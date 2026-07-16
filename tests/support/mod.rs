use subfast_net_rt::ir::tensor::{DType, Layout, TensorId, TensorInfo, TensorKind, WeightRef};
use subfast_net_rt::model::weights::WeightStore;
use subfast_net_rt::model::{Model, ModelType};
use subfast_net_rt::tensor::Tensor;

const INPUT_SHAPE: [usize; 4] = [1, 3, 32, 256];
const PAIR_INPUT_SHAPE: [usize; 4] = [1, 3, 32, 256];

pub fn deterministic_input(seed: u32) -> Tensor {
    deterministic_input_for(INPUT_SHAPE, seed)
}

pub fn deterministic_input_with_shape(shape: &[usize], seed: u32) -> Tensor {
    Tensor::from_f32(shape.to_vec(), deterministic_values(shape, seed)).unwrap()
}

pub fn deterministic_pair_input(seed: u32) -> Tensor {
    deterministic_input_for(PAIR_INPUT_SHAPE, seed)
}

pub fn packed_input(seeds: &[u32]) -> Tensor {
    packed_input_for(INPUT_SHAPE, seeds)
}

#[allow(dead_code)]
pub fn packed_input_with_shape(shape: &[usize], seeds: &[u32]) -> Tensor {
    let mut packed_shape = shape.to_vec();
    packed_shape[0] = seeds.len();
    let values = seeds
        .iter()
        .flat_map(|seed| deterministic_values(shape, *seed))
        .collect();
    Tensor::from_f32(packed_shape, values).unwrap()
}

pub fn packed_pair_input(seeds: &[u32]) -> Tensor {
    packed_input_for(PAIR_INPUT_SHAPE, seeds)
}

fn deterministic_input_for(shape: [usize; 4], seed: u32) -> Tensor {
    Tensor::from_f32(shape.to_vec(), deterministic_values(&shape, seed)).unwrap()
}

fn packed_input_for(shape: [usize; 4], seeds: &[u32]) -> Tensor {
    let mut packed_shape = shape.to_vec();
    packed_shape[0] = seeds.len();
    let values = seeds
        .iter()
        .flat_map(|seed| deterministic_values(&shape, *seed))
        .collect();
    Tensor::from_f32(packed_shape, values).unwrap()
}

fn deterministic_values(shape: &[usize], seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..shape[1..].iter().product())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
        })
        .collect()
}

pub fn synthetic_presence_model() -> Model {
    synthetic_presence_model_with_shape(INPUT_SHAPE)
}

pub fn synthetic_presence_model_with_shape(shape: [usize; 4]) -> Model {
    let mut builder = ModelBuilder::new();
    let input = builder.add_tensor("x", TensorKind::UserInput, shape.to_vec());

    builder.add_conv_parameters("backbone.0", [16, 3, 5, 5]);
    builder.add_conv_parameters("backbone.2", [32, 16, 5, 5]);
    builder.add_conv_parameters("presence_head.local", [1, 32, 3, 3]);
    builder.add_parameter_values("presence_head.evidence_pool.log_scale", vec![], &[0.0]);
    builder.add_parameter_values("presence_head.evidence_pool.bias", vec![], &[0.025]);

    let output = builder.add_tensor("output", TensorKind::UserOutput, vec![1]);
    builder.finish(vec![input], output)
}

pub fn synthetic_pair_model() -> Model {
    synthetic_pair_model_with_shape(PAIR_INPUT_SHAPE)
}

pub fn synthetic_pair_model_with_shape(shape: [usize; 4]) -> Model {
    let mut builder = ModelBuilder::new();
    let left = builder.add_tensor("left", TensorKind::UserInput, shape.to_vec());
    let right = builder.add_tensor("right", TensorKind::UserInput, shape.to_vec());

    builder.add_conv_parameters("stem.0", [16, 3, 3, 3]);
    builder.add_conv_parameters("downsample.0", [32, 16, 3, 3]);
    builder.add_conv_parameters("detail.0", [32, 1, 3, 3]);
    builder.add_conv_parameters("detail.3", [32, 32, 1, 1]);
    builder.add_conv_parameters("vertical_downsample.0", [32, 1, 3, 3]);
    builder.add_conv_parameters("vertical_downsample.3", [48, 32, 1, 1]);
    builder.add_conv_parameters("context.0", [48, 1, 3, 3]);
    builder.add_conv_parameters("context.3", [48, 48, 1, 1]);
    builder.add_conv_parameters("text_mask", [1, 48, 1, 1]);
    builder.add_linear_parameters("classifier.0", [32, 144]);
    builder.add_linear_parameters("classifier.2", [1, 32]);

    let output = builder.add_tensor("output", TensorKind::UserOutput, vec![1]);
    builder.finish(vec![left, right], output)
}

struct ModelBuilder {
    tensors: Vec<TensorInfo>,
    weights: Vec<u8>,
    parameter_serial: usize,
}

impl ModelBuilder {
    fn new() -> Self {
        Self {
            tensors: Vec::new(),
            weights: Vec::new(),
            parameter_serial: 0,
        }
    }

    fn add_tensor(
        &mut self,
        name: impl Into<String>,
        kind: TensorKind,
        shape: Vec<usize>,
    ) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(TensorInfo {
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

    fn add_conv_parameters(&mut self, target: &str, weight_shape: [usize; 4]) {
        self.add_pattern_parameter(&format!("{target}.weight"), weight_shape.to_vec(), 0.0005);
        self.add_pattern_parameter(&format!("{target}.bias"), vec![weight_shape[0]], 0.002);
    }

    fn add_linear_parameters(&mut self, target: &str, weight_shape: [usize; 2]) {
        self.add_pattern_parameter(&format!("{target}.weight"), weight_shape.to_vec(), 0.0005);
        self.add_pattern_parameter(&format!("{target}.bias"), vec![weight_shape[0]], 0.002);
    }

    fn add_pattern_parameter(&mut self, target: &str, shape: Vec<usize>, scale: f32) {
        let serial = self.parameter_serial;
        self.parameter_serial += 1;
        let values = (0..shape.iter().product())
            .map(|index| {
                let bucket = (index.wrapping_mul(7) + serial.wrapping_mul(11)) % 19;
                (bucket as f32 - 9.0) * scale
            })
            .collect::<Vec<_>>();
        self.add_parameter_values(target, shape, &values);
    }

    fn add_parameter_values(&mut self, target: &str, shape: Vec<usize>, values: &[f32]) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        let offset = self.weights.len();
        for value in values {
            self.weights.extend_from_slice(&value.to_le_bytes());
        }
        let byte_length = self.weights.len() - offset;
        let id = TensorId(self.tensors.len());
        self.tensors.push(TensorInfo {
            id,
            name: target.into(),
            dtype: DType::Float32,
            shape,
            layout: Layout::DenseRowMajor,
            kind: TensorKind::Parameter,
            target: Some(target.into()),
            weight: Some(WeightRef {
                offset,
                byte_length,
            }),
        });
    }

    fn finish(self, inputs: Vec<TensorId>, output: TensorId) -> Model {
        Model {
            model_type: if inputs.len() == 1 {
                ModelType::RoiPresence
            } else {
                ModelType::RoiPairMatcher
            },
            tensors: self.tensors,
            inputs,
            outputs: vec![output],
            weights: WeightStore::from_bytes(self.weights),
        }
    }
}
