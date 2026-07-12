use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct RawModel {
    pub format: String,
    pub format_version: u64,
    pub graph: RawGraph,
    pub inputs: Vec<RawTensorRef>,
    pub nodes: Vec<RawNode>,
    pub outputs: Vec<RawTensorRef>,
    pub tensors: BTreeMap<String, RawTensor>,
    pub weights: RawWeights,
}

#[derive(Debug, Deserialize)]
pub struct RawGraph {
    pub node_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct RawWeights {
    pub byte_order: String,
    pub file: String,
    pub layout: String,
}

#[derive(Debug, Deserialize)]
pub struct RawTensorRef {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub layout: String,
}

#[derive(Debug, Deserialize)]
pub struct RawTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub layout: String,
    pub kind: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub data: Option<RawData>,
}

#[derive(Debug, Deserialize)]
pub struct RawData {
    pub file: String,
    pub offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Deserialize)]
pub struct RawNode {
    pub name: String,
    pub op: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<RawTensorRef>,
    #[serde(default)]
    pub attrs: BTreeMap<String, Value>,
}
