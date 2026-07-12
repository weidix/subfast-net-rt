use subfast_net_rt::ir::graph::{Graph, Node, NodeId};
use subfast_net_rt::ir::op::Op;
use subfast_net_rt::ir::tensor::{DType, Layout, TensorId, TensorInfo, TensorKind, WeightRef};
use subfast_net_rt::model::UnifiedModel;
use subfast_net_rt::model::weights::WeightStore;
use subfast_net_rt::tensor::Tensor;

const INPUT_SHAPE: [usize; 4] = [1, 3, 32, 256];

pub fn deterministic_input(seed: u32) -> Tensor {
    Tensor::from_f32(INPUT_SHAPE.to_vec(), deterministic_values(seed)).unwrap()
}

pub fn packed_input(seeds: &[u32]) -> Tensor {
    let mut shape = INPUT_SHAPE.to_vec();
    shape[0] = seeds.len();
    let values = seeds
        .iter()
        .flat_map(|seed| deterministic_values(*seed))
        .collect();
    Tensor::from_f32(shape, values).unwrap()
}

fn deterministic_values(seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..INPUT_SHAPE[1..].iter().product())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
        })
        .collect()
}

pub fn synthetic_presence_model() -> UnifiedModel {
    let mut builder = ModelBuilder::new();
    let input = builder.add_tensor("x", TensorKind::UserInput, INPUT_SHAPE.to_vec());

    builder.add_conv_parameters("backbone.0", [16, 3, 5, 5]);
    builder.add_conv_parameters("backbone.2", [32, 16, 5, 5]);
    builder.add_conv_parameters("presence_head.local", [1, 32, 3, 3]);
    builder.add_parameter_values("presence_head.evidence_pool.log_scale", vec![], &[0.0]);
    builder.add_parameter_values("presence_head.evidence_pool.bias", vec![], &[0.025]);

    let output = builder.add_tensor("output", TensorKind::UserOutput, vec![1]);
    let specs = vec![
        conv("backbone.0", [2, 2], [2, 2], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("backbone.2", [2, 2], [2, 2], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("presence_head.local", [1, 1], [2, 2], [2, 2], 1),
        op("aten.abs.default", Op::Abs),
        op("aten.amax.default", Op::ReduceMax { axes: vec![1] }),
        op("aten.gt.Scalar", Op::GreaterThan { value: 1e-8 }),
        op(
            "aten.to.dtype",
            Op::Cast {
                dtype: DType::Float32,
            },
        ),
        avg_pool([4, 4], [4, 4], [0, 0], true),
        op("aten.gt.Scalar", Op::GreaterThan { value: 0.5 }),
        op(
            "aten.to.dtype",
            Op::Cast {
                dtype: DType::Float32,
            },
        ),
        op("aten.gt.Scalar", Op::GreaterThan { value: 0.5 }),
        op(
            "aten.to.dtype",
            Op::Cast {
                dtype: DType::Float32,
            },
        ),
        avg_pool([5, 5], [1, 1], [2, 2], false),
        op("aten.mul.Tensor", Op::Mul { value: None }),
        avg_pool([5, 5], [1, 1], [2, 2], false),
        op("aten.clamp_min.default", Op::ClampMin { value: 1e-6 }),
        op("aten.div.Tensor", Op::Div),
        op("aten.gt.Scalar", Op::GreaterThan { value: 0.5 }),
        op("aten.bitwise_not.default", Op::BitwiseNot),
        op(
            "aten.masked_fill.Scalar",
            Op::MaskedFill { value: f32::MIN },
        ),
        op(
            "aten.amax.default",
            Op::ReduceMax {
                axes: vec![1, 2, 3],
            },
        ),
        op("aten.clamp_min.default", Op::ClampMin { value: -10_000.0 }),
        op("aten.softplus.default", Op::Softplus),
        op(
            "aten.add.Tensor",
            Op::Add {
                alpha: 1.0,
                value: Some(0.5),
            },
        ),
        op("aten.mul.Tensor", Op::Mul { value: None }),
        op(
            "aten.add.Tensor",
            Op::Add {
                alpha: 1.0,
                value: None,
            },
        ),
    ];
    builder.finish(vec![input], output, specs)
}

pub fn synthetic_pair_model() -> UnifiedModel {
    let mut builder = ModelBuilder::new();
    let left = builder.add_tensor("left", TensorKind::UserInput, INPUT_SHAPE.to_vec());
    let right = builder.add_tensor("right", TensorKind::UserInput, INPUT_SHAPE.to_vec());

    builder.add_conv_parameters("model.stem.0", [16, 9, 3, 3]);
    builder.add_conv_parameters("model.downsample.0", [32, 16, 3, 3]);
    builder.add_conv_parameters("model.detail.0", [32, 1, 3, 3]);
    builder.add_conv_parameters("model.detail.3", [32, 32, 1, 1]);
    builder.add_conv_parameters("model.vertical_downsample.0", [32, 1, 3, 3]);
    builder.add_conv_parameters("model.vertical_downsample.3", [48, 32, 1, 1]);
    builder.add_conv_parameters("model.context.0", [48, 1, 3, 3]);
    builder.add_conv_parameters("model.context.3", [48, 48, 1, 1]);
    builder.add_conv_parameters("model.text_mask", [1, 48, 1, 1]);
    builder.add_linear_parameters("model.classifier.0", [32, 144]);
    builder.add_linear_parameters("model.classifier.2", [1, 32]);

    let output = builder.add_tensor("output", TensorKind::UserOutput, vec![1]);
    let specs = vec![
        op("aten.sub.Tensor", Op::Sub),
        avg_pool([5, 5], [1, 1], [2, 2], false),
        op(
            "aten.add.Tensor",
            Op::Add {
                alpha: 1.0,
                value: None,
            },
        ),
        op("aten.mul.Tensor", Op::Mul { value: Some(0.5) }),
        op("aten.abs.default", Op::Abs),
        op("aten.sub.Tensor", Op::Sub),
        op("aten.abs.default", Op::Abs),
        op("aten.cat.default", Op::Cat { dim: 1 }),
        conv("model.stem.0", [2, 2], [1, 1], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("model.downsample.0", [2, 2], [1, 1], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("model.detail.0", [1, 1], [1, 1], [1, 1], 32),
        op("aten.silu_.default", Op::Silu),
        conv("model.detail.3", [1, 1], [0, 0], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("model.vertical_downsample.0", [2, 1], [1, 1], [1, 1], 32),
        op("aten.silu_.default", Op::Silu),
        conv("model.vertical_downsample.3", [1, 1], [0, 0], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("model.context.0", [1, 1], [1, 1], [1, 1], 48),
        op("aten.silu_.default", Op::Silu),
        conv("model.context.3", [1, 1], [0, 0], [1, 1], 1),
        op("aten.silu_.default", Op::Silu),
        conv("model.text_mask", [1, 1], [0, 0], [1, 1], 1),
        op("aten.sigmoid.default", Op::Sigmoid),
        op("aten.mul.Tensor", Op::Mul { value: Some(0.75) }),
        op(
            "aten.add.Tensor",
            Op::Add {
                alpha: 1.0,
                value: Some(0.25),
            },
        ),
        op("aten.mul.Tensor", Op::Mul { value: None }),
        op("aten.mean.dim", Op::ReduceMean { axes: vec![2, 3] }),
        op("aten.amax.default", Op::ReduceMax { axes: vec![2, 3] }),
        op("aten.mean.dim", Op::ReduceMean { axes: vec![2] }),
        op("aten.amax.default", Op::ReduceMax { axes: vec![-1] }),
        op("aten.cat.default", Op::Cat { dim: 1 }),
        op("aten.linear.default", Op::Linear),
        op("aten.silu_.default", Op::Silu),
        op("aten.linear.default", Op::Linear),
        op("aten.squeeze.dim", Op::Squeeze { dim: 1 }),
    ];
    builder.finish(vec![left, right], output, specs)
}

enum NodeSpec {
    Op {
        original_op: &'static str,
        op: Op,
    },
    Conv {
        target: &'static str,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    },
}

fn op(original_op: &'static str, op: Op) -> NodeSpec {
    NodeSpec::Op { original_op, op }
}

fn conv(
    target: &'static str,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> NodeSpec {
    NodeSpec::Conv {
        target,
        stride,
        padding,
        dilation,
        groups,
    }
}

fn avg_pool(
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> NodeSpec {
    op(
        "aten.avg_pool2d.default",
        Op::AvgPool2d {
            kernel,
            stride,
            padding,
            ceil_mode: false,
            count_include_pad,
        },
    )
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

    fn parameter_id(&self, target: &str) -> TensorId {
        self.tensors
            .iter()
            .find(|tensor| tensor.target.as_deref() == Some(target))
            .map(|tensor| tensor.id)
            .unwrap_or_else(|| panic!("missing synthetic parameter {target}"))
    }

    fn finish(
        mut self,
        inputs: Vec<TensorId>,
        output: TensorId,
        specs: Vec<NodeSpec>,
    ) -> UnifiedModel {
        let mut nodes = Vec::with_capacity(specs.len());
        let mut previous = inputs[0];
        let last_index = specs.len() - 1;
        for (index, spec) in specs.into_iter().enumerate() {
            let node_output = if index == last_index {
                output
            } else {
                self.add_tensor(
                    format!("synthetic_intermediate_{index}"),
                    TensorKind::Intermediate,
                    vec![1],
                )
            };
            let (name, original_op, op, node_inputs) = match spec {
                NodeSpec::Op { original_op, op } => (
                    format!("synthetic_node_{index}"),
                    original_op,
                    op,
                    vec![previous],
                ),
                NodeSpec::Conv {
                    target,
                    stride,
                    padding,
                    dilation,
                    groups,
                } => {
                    let weight = self.parameter_id(&format!("{target}.weight"));
                    let bias = self.parameter_id(&format!("{target}.bias"));
                    (
                        target.replace('.', "_"),
                        "aten.conv2d.default",
                        Op::Conv2d {
                            stride,
                            padding,
                            dilation,
                            groups,
                        },
                        vec![previous, weight, bias],
                    )
                }
            };
            nodes.push(Node {
                id: NodeId(index),
                name,
                original_op: original_op.into(),
                op,
                inputs: node_inputs,
                outputs: vec![node_output],
            });
            previous = node_output;
        }

        UnifiedModel {
            format: "subfast-net.unified-model".into(),
            format_version: 1,
            graph: Graph { nodes },
            tensors: self.tensors,
            inputs,
            outputs: vec![output],
            weights: WeightStore::from_bytes(self.weights),
        }
    }
}
