use crate::ir::op::Op;
use crate::ir::tensor::TensorId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub name: String,
    pub original_op: String,
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
}
