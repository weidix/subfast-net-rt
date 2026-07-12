use crate::error::Result;
use crate::model::UnifiedModel;
use crate::tensor::Tensor;

pub mod cpu;

pub trait Backend {
    type Executable: Executable;

    fn compile(&self, model: &UnifiedModel) -> Result<Self::Executable>;
}

pub trait Executable {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
}
