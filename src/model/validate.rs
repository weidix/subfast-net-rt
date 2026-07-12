use std::collections::HashSet;

use crate::error::{Result, RtError};
use crate::ir::graph::Graph;
use crate::ir::tensor::{TensorId, TensorInfo, TensorKind};

pub fn validate_topology(graph: &Graph, tensors: &[TensorInfo], inputs: &[TensorId]) -> Result<()> {
    let mut available: HashSet<TensorId> = HashSet::new();
    for tensor in tensors {
        if matches!(tensor.kind, TensorKind::Parameter | TensorKind::Buffer) {
            available.insert(tensor.id);
        }
    }
    for input in inputs {
        available.insert(*input);
    }

    for node in &graph.nodes {
        for input in &node.inputs {
            if !available.contains(input) {
                let tensor = &tensors[input.0];
                return Err(RtError::InvalidGraph(format!(
                    "node {} uses tensor {} before it is available",
                    node.name, tensor.name
                )));
            }
        }
        for output in &node.outputs {
            if available.contains(output) {
                let tensor = &tensors[output.0];
                return Err(RtError::InvalidGraph(format!(
                    "node {} redefines tensor {}",
                    node.name, tensor.name
                )));
            }
            available.insert(*output);
        }
    }
    Ok(())
}
