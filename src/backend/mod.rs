use crate::error::Result;
use crate::model::UnifiedModel;
use crate::tensor::Tensor;

pub mod cpu;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod gpu;

#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
compile_error!("the gpu feature requires a native Metal or Vulkan target");

pub trait Backend {
    type Executable: Executable;

    fn compile(&self, model: &UnifiedModel) -> Result<Self::Executable>;
}

pub trait Executable {
    fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
}
