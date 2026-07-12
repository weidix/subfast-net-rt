use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use serde_json::Value;

use crate::error::{Result, RtError};
use crate::ir::graph::{Graph, Node, NodeId};
use crate::ir::op::Op;
use crate::ir::tensor::{DType, Layout, TensorId, TensorInfo, TensorKind, WeightRef};
use crate::model::schema::{RawModel, RawNode, RawTensor};
use crate::model::validate::validate_topology;
use crate::model::weights::WeightStore;

#[derive(Debug, Clone)]
pub struct UnifiedModel {
    pub format: String,
    pub format_version: u64,
    pub graph: Graph,
    pub tensors: Vec<TensorInfo>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub weights: WeightStore,
}

impl UnifiedModel {
    pub fn load(model_json: impl AsRef<Path>, model_path: impl AsRef<Path>) -> Result<Self> {
        let model_json = model_json.as_ref();
        let model_path = model_path.as_ref();
        let json_bytes = std::fs::read(model_json).map_err(|e| RtError::io(model_json, e))?;
        let raw: RawModel = serde_json::from_slice(&json_bytes)
            .map_err(|e| RtError::Json(format!("{}: {}", model_json.display(), e)))?;
        let weights = WeightStore::load(model_path)?;
        Self::from_raw(raw, weights)
    }

    pub fn input_info(&self) -> Vec<&TensorInfo> {
        self.inputs.iter().map(|id| &self.tensors[id.0]).collect()
    }

    pub fn output_info(&self) -> Vec<&TensorInfo> {
        self.outputs.iter().map(|id| &self.tensors[id.0]).collect()
    }

    fn from_raw(raw: RawModel, weights: WeightStore) -> Result<Self> {
        if raw.format != "subfast-net.unified-model" || raw.format_version != 1 {
            return Err(RtError::FormatVersion {
                format: raw.format,
                version: raw.format_version,
            });
        }
        if raw.weights.byte_order != "little" || raw.weights.layout != "dense_row_major" {
            return Err(RtError::Json(format!(
                "unsupported weights metadata: byte_order={}, layout={}",
                raw.weights.byte_order, raw.weights.layout
            )));
        }
        if raw.graph.node_count != raw.nodes.len() {
            return Err(RtError::InvalidGraph(format!(
                "graph.node_count {} does not match nodes length {}",
                raw.graph.node_count,
                raw.nodes.len()
            )));
        }

        let output_names: HashSet<String> = raw.outputs.iter().map(|t| t.name.clone()).collect();
        let mut name_to_id = HashMap::new();
        let mut tensors = Vec::with_capacity(raw.tensors.len());
        for (idx, (name, tensor)) in raw.tensors.iter().enumerate() {
            if name != &tensor.name {
                return Err(RtError::InvalidTensor {
                    tensor: name.clone(),
                    message: format!("map key does not match tensor name {}", tensor.name),
                });
            }
            let info = parse_tensor(TensorId(idx), tensor, output_names.contains(name), &weights)?;
            name_to_id.insert(name.clone(), info.id);
            tensors.push(info);
        }

        let inputs = raw
            .inputs
            .iter()
            .map(|input| {
                lookup_declared_tensor(input, &name_to_id, &tensors, TensorKind::UserInput)
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = raw
            .outputs
            .iter()
            .map(|output| {
                lookup_declared_tensor(output, &name_to_id, &tensors, TensorKind::UserOutput)
            })
            .collect::<Result<Vec<_>>>()?;

        let graph = Graph {
            nodes: raw
                .nodes
                .iter()
                .enumerate()
                .map(|(idx, node)| parse_node(idx, node, &name_to_id, &tensors))
                .collect::<Result<Vec<_>>>()?,
        };
        validate_topology(&graph, &tensors, &inputs)?;

        Ok(Self {
            format: raw.format,
            format_version: raw.format_version,
            graph,
            tensors,
            inputs,
            outputs,
            weights,
        })
    }
}

fn lookup_declared_tensor(
    raw: &crate::model::schema::RawTensorRef,
    name_to_id: &HashMap<String, TensorId>,
    tensors: &[TensorInfo],
    expected_kind: TensorKind,
) -> Result<TensorId> {
    let id = *name_to_id
        .get(&raw.name)
        .ok_or_else(|| RtError::InvalidGraph(format!("declared tensor {} is missing", raw.name)))?;
    let tensor = &tensors[id.0];
    if tensor.dtype.to_string() != raw.dtype
        || tensor.shape != raw.shape
        || tensor.layout.to_string() != raw.layout
    {
        return Err(RtError::InvalidTensor {
            tensor: raw.name.clone(),
            message: "declaration does not match tensor table".into(),
        });
    }
    match expected_kind {
        TensorKind::UserInput if tensor.kind != TensorKind::UserInput => {
            return Err(RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: "input declaration does not refer to user_input tensor".into(),
            });
        }
        TensorKind::UserOutput if tensor.kind != TensorKind::UserOutput => {
            return Err(RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: "output declaration does not refer to graph output tensor".into(),
            });
        }
        _ => {}
    }
    Ok(id)
}

fn parse_tensor(
    id: TensorId,
    raw: &RawTensor,
    is_user_output: bool,
    weights: &WeightStore,
) -> Result<TensorInfo> {
    let dtype = DType::parse(&raw.dtype).ok_or_else(|| RtError::UnsupportedDTypeLayout {
        tensor: raw.name.clone(),
        dtype: raw.dtype.clone(),
        layout: raw.layout.clone(),
    })?;
    let layout = Layout::parse(&raw.layout).ok_or_else(|| RtError::UnsupportedDTypeLayout {
        tensor: raw.name.clone(),
        dtype: raw.dtype.clone(),
        layout: raw.layout.clone(),
    })?;
    let mut kind = match raw.kind.as_str() {
        "user_input" => TensorKind::UserInput,
        "parameter" => TensorKind::Parameter,
        "buffer" => TensorKind::Buffer,
        "intermediate" => TensorKind::Intermediate,
        other => {
            return Err(RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: format!("unknown tensor kind {}", other),
            });
        }
    };
    if is_user_output {
        kind = TensorKind::UserOutput;
    }

    let expected_len =
        dtype
            .byte_len_for_shape(&raw.shape)
            .ok_or_else(|| RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: "shape byte length overflow".into(),
            })?;
    let weight = if let Some(data) = &raw.data {
        if data.file != "weights.bin" {
            return Err(RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: format!("unsupported tensor data file {}", data.file),
            });
        }
        let offset = usize::try_from(data.offset).map_err(|_| RtError::InvalidTensor {
            tensor: raw.name.clone(),
            message: "offset does not fit usize".into(),
        })?;
        let byte_length =
            usize::try_from(data.byte_length).map_err(|_| RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: "byte_length does not fit usize".into(),
            })?;
        if byte_length != expected_len {
            return Err(RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: format!(
                    "declared byte length {} does not match dtype/shape length {}",
                    byte_length, expected_len
                ),
            });
        }
        let end = offset
            .checked_add(byte_length)
            .ok_or_else(|| RtError::WeightsOutOfBounds {
                tensor: raw.name.clone(),
                offset: data.offset,
                length: data.byte_length,
                file_len: weights.total_bytes() as u64,
            })?;
        if end > weights.total_bytes() {
            return Err(RtError::WeightsOutOfBounds {
                tensor: raw.name.clone(),
                offset: data.offset,
                length: data.byte_length,
                file_len: weights.total_bytes() as u64,
            });
        }
        Some(WeightRef {
            offset,
            byte_length,
        })
    } else {
        if matches!(kind, TensorKind::Parameter | TensorKind::Buffer) {
            return Err(RtError::InvalidTensor {
                tensor: raw.name.clone(),
                message: "parameter/buffer tensor is missing data".into(),
            });
        }
        None
    };

    Ok(TensorInfo {
        id,
        name: raw.name.clone(),
        dtype,
        shape: raw.shape.clone(),
        layout,
        kind,
        target: raw.target.clone(),
        weight,
    })
}

fn parse_node(
    idx: usize,
    raw: &RawNode,
    name_to_id: &HashMap<String, TensorId>,
    tensors: &[TensorInfo],
) -> Result<Node> {
    let inputs = raw
        .inputs
        .iter()
        .map(|name| {
            name_to_id.get(name).copied().ok_or_else(|| {
                RtError::InvalidGraph(format!("node {} input {} is missing", raw.name, name))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let outputs = raw
        .outputs
        .iter()
        .map(|output| {
            name_to_id.get(&output.name).copied().ok_or_else(|| {
                RtError::InvalidGraph(format!(
                    "node {} output {} is missing",
                    raw.name, output.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if outputs.is_empty() {
        return Err(RtError::InvalidGraph(format!(
            "node {} has no outputs",
            raw.name
        )));
    }
    let op = normalize_op(raw, &inputs, tensors)?;
    Ok(Node {
        id: NodeId(idx),
        name: raw.name.clone(),
        original_op: raw.op.clone(),
        op,
        inputs,
        outputs,
    })
}

fn normalize_op(raw: &RawNode, inputs: &[TensorId], tensors: &[TensorInfo]) -> Result<Op> {
    match raw.op.as_str() {
        "aten.conv2d.default" => {
            if !(2..=3).contains(&inputs.len()) {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: format!("conv2d expects 2 or 3 inputs, got {}", inputs.len()),
                });
            }
            Ok(Op::Conv2d {
                stride: attr_usize_pair(&raw.attrs, "stride", [1, 1])?,
                padding: attr_usize_pair(&raw.attrs, "padding", [0, 0])?,
                dilation: attr_usize_pair(&raw.attrs, "dilation", [1, 1])?,
                groups: attr_usize(&raw.attrs, "groups", 1)?,
            })
        }
        "aten.batch_norm.default" => {
            if inputs.len() != 5 {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: format!("batch_norm expects 5 inputs, got {}", inputs.len()),
                });
            }
            let training = attr_bool(&raw.attrs, "training", false)?;
            if training {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: "training=true batch_norm is not supported".into(),
                });
            }
            Ok(Op::BatchNorm {
                epsilon: attr_f32(&raw.attrs, "eps", 1e-5)?,
                training,
            })
        }
        "aten.silu.default" | "aten.silu_.default" => {
            if inputs.len() != 1 {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: format!("silu expects 1 input, got {}", inputs.len()),
                });
            }
            Ok(Op::Silu)
        }
        "aten.add.Tensor" => {
            if !(1..=2).contains(&inputs.len()) {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: format!("add.Tensor expects 1 or 2 inputs, got {}", inputs.len()),
                });
            }
            Ok(Op::Add {
                alpha: attr_f32(&raw.attrs, "alpha", 1.0)?,
                value: (inputs.len() == 1)
                    .then(|| attr_f32_required(&raw.attrs, "arg1"))
                    .transpose()?,
            })
        }
        "aten.upsample_bilinear2d.vec" => {
            if inputs.len() != 1 {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: format!("upsample_bilinear2d expects 1 input, got {}", inputs.len()),
                });
            }
            Ok(Op::UpsampleBilinear2d {
                output_size: attr_usize_pair_required(&raw.attrs, "output_size")?,
                align_corners: attr_bool(&raw.attrs, "align_corners", false)?,
                scale_factors: attr_optional_f32_pair(&raw.attrs, "scale_factors")?,
            })
        }
        "aten.abs.default" => unary(raw, inputs, Op::Abs),
        "aten.amax.default" => unary(
            raw,
            inputs,
            Op::ReduceMax {
                axes: attr_isize_list(&raw.attrs, "arg1")?,
            },
        ),
        "aten.mean.dim" => unary(
            raw,
            inputs,
            Op::ReduceMean {
                axes: attr_isize_list(&raw.attrs, "arg1")?,
            },
        ),
        "aten.gt.Scalar" => unary(
            raw,
            inputs,
            Op::GreaterThan {
                value: attr_f32_required(&raw.attrs, "arg1")?,
            },
        ),
        "aten.to.dtype" => unary(
            raw,
            inputs,
            Op::Cast {
                dtype: attr_dtype(&raw.attrs, "arg1")?,
            },
        ),
        "aten.avg_pool2d.default" => unary(
            raw,
            inputs,
            Op::AvgPool2d {
                kernel: attr_usize_pair_required(&raw.attrs, "arg1")?,
                stride: attr_usize_pair(&raw.attrs, "arg2", [1, 1])?,
                padding: attr_usize_pair(&raw.attrs, "arg3", [0, 0])?,
                ceil_mode: attr_bool(&raw.attrs, "arg4", false)?,
                count_include_pad: attr_bool(&raw.attrs, "arg5", true)?,
            },
        ),
        "aten.mul.Tensor" => elementwise_optional_scalar(raw, inputs, |value| Op::Mul { value }),
        "aten.sub.Tensor" => binary(raw, inputs, Op::Sub),
        "aten.div.Tensor" => binary(raw, inputs, Op::Div),
        "aten.bitwise_not.default" => unary(raw, inputs, Op::BitwiseNot),
        "aten.masked_fill.Scalar" => binary(
            raw,
            inputs,
            Op::MaskedFill {
                value: attr_f32_required(&raw.attrs, "arg2")?,
            },
        ),
        "aten.clamp_min.default" => unary(
            raw,
            inputs,
            Op::ClampMin {
                value: attr_f32_required(&raw.attrs, "arg1")?,
            },
        ),
        "aten.softplus.default" => unary(raw, inputs, Op::Softplus),
        "aten.cat.default" => Ok(Op::Cat {
            dim: attr_isize(&raw.attrs, "arg1")?,
        }),
        "aten.sigmoid.default" => unary(raw, inputs, Op::Sigmoid),
        "aten.linear.default" => {
            if !(2..=3).contains(&inputs.len()) {
                return Err(RtError::UnsupportedOpParams {
                    node: raw.name.clone(),
                    message: format!("linear expects 2 or 3 inputs, got {}", inputs.len()),
                });
            }
            Ok(Op::Linear)
        }
        "aten.squeeze.dim" => unary(
            raw,
            inputs,
            Op::Squeeze {
                dim: attr_isize(&raw.attrs, "arg1")?,
            },
        ),
        other => Err(RtError::UnsupportedOp {
            node: raw.name.clone(),
            op: other.into(),
            inputs: format_input_info(inputs, tensors),
        }),
    }
}

fn unary(raw: &RawNode, inputs: &[TensorId], op: Op) -> Result<Op> {
    if inputs.len() != 1 {
        return Err(RtError::UnsupportedOpParams {
            node: raw.name.clone(),
            message: format!("{} expects 1 input, got {}", raw.op, inputs.len()),
        });
    }
    Ok(op)
}

fn binary(raw: &RawNode, inputs: &[TensorId], op: Op) -> Result<Op> {
    if inputs.len() != 2 {
        return Err(RtError::UnsupportedOpParams {
            node: raw.name.clone(),
            message: format!("{} expects 2 inputs, got {}", raw.op, inputs.len()),
        });
    }
    Ok(op)
}

fn elementwise_optional_scalar(
    raw: &RawNode,
    inputs: &[TensorId],
    build: impl FnOnce(Option<f32>) -> Op,
) -> Result<Op> {
    if !(1..=2).contains(&inputs.len()) {
        return Err(RtError::UnsupportedOpParams {
            node: raw.name.clone(),
            message: format!("{} expects 1 or 2 inputs, got {}", raw.op, inputs.len()),
        });
    }
    let value = (inputs.len() == 1)
        .then(|| attr_f32_required(&raw.attrs, "arg1"))
        .transpose()?;
    Ok(build(value))
}

fn format_input_info(inputs: &[TensorId], tensors: &[TensorInfo]) -> String {
    inputs
        .iter()
        .map(|id| {
            let t = &tensors[id.0];
            format!(
                "{} dtype={} shape={:?} layout={}",
                t.name, t.dtype, t.shape, t.layout
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn attr_usize(attrs: &BTreeMap<String, Value>, key: &str, default: usize) -> Result<usize> {
    match attrs.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => n.as_u64().map(|v| v as usize).ok_or_else(|| {
            RtError::Json(format!("attribute {} must be a non-negative integer", key))
        }),
        _ => Err(RtError::Json(format!(
            "attribute {} must be an integer",
            key
        ))),
    }
}

fn attr_usize_pair(
    attrs: &BTreeMap<String, Value>,
    key: &str,
    default: [usize; 2],
) -> Result<[usize; 2]> {
    match attrs.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(_) => attr_usize_pair_required(attrs, key),
    }
}

fn attr_usize_pair_required(attrs: &BTreeMap<String, Value>, key: &str) -> Result<[usize; 2]> {
    match attrs.get(key) {
        Some(Value::Array(values)) if values.len() == 2 => {
            let a = values[0].as_u64().ok_or_else(|| {
                RtError::Json(format!(
                    "attribute {}[0] must be a non-negative integer",
                    key
                ))
            })?;
            let b = values[1].as_u64().ok_or_else(|| {
                RtError::Json(format!(
                    "attribute {}[1] must be a non-negative integer",
                    key
                ))
            })?;
            Ok([a as usize, b as usize])
        }
        _ => Err(RtError::Json(format!(
            "attribute {} must be a length-2 integer array",
            key
        ))),
    }
}

fn attr_f32(attrs: &BTreeMap<String, Value>, key: &str, default: f32) -> Result<f32> {
    match attrs.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => n
            .as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| RtError::Json(format!("attribute {} must be a float", key))),
        _ => Err(RtError::Json(format!("attribute {} must be a float", key))),
    }
}

fn attr_f32_required(attrs: &BTreeMap<String, Value>, key: &str) -> Result<f32> {
    match attrs.get(key) {
        Some(Value::Number(number)) => number
            .as_f64()
            .map(|value| value as f32)
            .ok_or_else(|| RtError::Json(format!("attribute {key} must be a float"))),
        _ => Err(RtError::Json(format!("attribute {key} must be a float"))),
    }
}

fn attr_isize(attrs: &BTreeMap<String, Value>, key: &str) -> Result<isize> {
    attrs
        .get(key)
        .and_then(Value::as_i64)
        .map(|value| value as isize)
        .ok_or_else(|| RtError::Json(format!("attribute {key} must be an integer")))
}

fn attr_isize_list(attrs: &BTreeMap<String, Value>, key: &str) -> Result<Vec<isize>> {
    match attrs.get(key) {
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_i64().map(|value| value as isize).ok_or_else(|| {
                    RtError::Json(format!("attribute {key} must be an integer array"))
                })
            })
            .collect(),
        _ => Err(RtError::Json(format!(
            "attribute {key} must be an integer array"
        ))),
    }
}

fn attr_dtype(attrs: &BTreeMap<String, Value>, key: &str) -> Result<DType> {
    let value = attrs
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| RtError::Json(format!("attribute {key} must be a dtype string")))?;
    DType::parse(value).ok_or_else(|| RtError::Json(format!("unsupported dtype {value}")))
}

fn attr_bool(attrs: &BTreeMap<String, Value>, key: &str, default: bool) -> Result<bool> {
    match attrs.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        _ => Err(RtError::Json(format!(
            "attribute {} must be a boolean",
            key
        ))),
    }
}

fn attr_optional_f32_pair(attrs: &BTreeMap<String, Value>, key: &str) -> Result<Option<[f32; 2]>> {
    match attrs.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(values)) if values.len() == 2 => {
            let a = values[0]
                .as_f64()
                .ok_or_else(|| RtError::Json(format!("attribute {}[0] must be a float", key)))?;
            let b = values[1]
                .as_f64()
                .ok_or_else(|| RtError::Json(format!("attribute {}[1] must be a float", key)))?;
            Ok(Some([a as f32, b as f32]))
        }
        _ => Err(RtError::Json(format!(
            "attribute {} must be null or a length-2 float array",
            key
        ))),
    }
}
