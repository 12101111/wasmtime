//! Implements the `wasi-nn` API for the WIT ("preview2") ABI.
//!
//! Note that `wasi-nn` is not yet included in an official "preview2" world
//! (though it could be) so by "preview2" here we mean that this can be called
//! with the component model's canonical ABI.
//!
//! This module exports its [`types`] for use throughout the crate and the
//! [`ML`] object, which exposes [`ML::add_to_linker`]. To implement all of
//! this, this module proceeds in steps:
//! 1. generate all of the WIT glue code into a `gen::*` namespace
//! 2. wire up the `gen::*` glue to the context state, delegating actual
//!    computation to a [`Backend`]
//! 3. convert some types
//!
//! [`Backend`]: crate::Backend
//! [`types`]: crate::wit::types

use std::{error::Error, fmt, hash::Hash, str::FromStr};

use crate::{ctx::WasiNnView, ExecutionContext, Graph};

/// Generate the traits and types from the `wasi-nn` WIT specification.
mod gen_ {
    wasmtime::component::bindgen!({
        world: "ml",
        path: "spec/wit/wasi-nn.wit",
        with: {
            "wasi:nn/tensor/tensor": super::Tensor,
            "wasi:nn/graph/graph": crate::Graph,
            "wasi:nn/inference/graph-execution-context": crate::ExecutionContext,
            "wasi:nn/errors/error": super::WasiNnError,
        }
    });
}
use gen_::wasi::nn as gen; // Shortcut to the module containing the types we need.

// Export the `types` used in this crate as well as `ML::add_to_linker`.
pub mod types {
    pub use gen::{
        graph::{ExecutionTarget, Graph, GraphEncoding},
        inference::GraphExecutionContext,
        tensor::{Tensor, TensorType},
    };

    use super::gen;
}
use gen::{
    errors::ErrorCode,
    tensor::{TensorData, TensorDimensions, TensorType},
};
pub use gen_::Ml as ML;
use wasmtime::component::Resource;

#[derive(Debug, Clone)]
/// All inputs and outputs to an ML inference are represented as `Tensor`s.
pub struct Tensor {
    /// Describe the size of the tensor (e.g., 2x2x2x2 -> [2, 2, 2, 2]). To
    /// represent a tensor containing a single value, use `[1]` for the
    /// tensor dimensions.
    pub dimensions: TensorDimensions,

    /// Describe the type of element in the tensor (e.g., `f32`).
    pub tensor_type: TensorType,

    /// Contains the tensor data.
    pub data: TensorData,
}

impl<T: WasiNnView> gen::tensor::HostTensor for T {
    fn new(
        &mut self,
        dimensions: TensorDimensions,
        tensor_type: TensorType,
        data: TensorData,
    ) -> wasmtime::Result<Resource<Tensor>> {
        let tensor = Tensor {
            dimensions,
            tensor_type,
            data,
        };
        let resource = self.table().push(tensor)?;
        Ok(resource)
    }

    fn dimensions(&mut self, tensor: Resource<Tensor>) -> wasmtime::Result<TensorDimensions> {
        let tensor = self.table().get(&tensor)?;
        Ok(tensor.dimensions.clone())
    }

    fn ty(&mut self, tensor: Resource<Tensor>) -> wasmtime::Result<TensorType> {
        let tensor = self.table().get(&tensor)?;
        Ok(tensor.tensor_type)
    }

    fn data(&mut self, tensor: Resource<Tensor>) -> wasmtime::Result<TensorData> {
        let tensor = self.table().get(&tensor)?;
        Ok(tensor.data.clone())
    }

    fn drop(&mut self, tensor: Resource<Tensor>) -> wasmtime::Result<()> {
        let _ = self.table().delete(tensor)?;
        Ok(())
    }
}

impl<T: WasiNnView> gen::tensor::Host for T {}

pub struct WasiNnError {
    code: ErrorCode,
    data: String,
}

impl<T: WasiNnView> gen::errors::HostError for T {
    fn new(&mut self, code: ErrorCode, data: String) -> wasmtime::Result<Resource<WasiNnError>> {
        let error = WasiNnError { code, data };
        let resource = self.table().push(error)?;
        Ok(resource)
    }

    fn code(&mut self, error: Resource<WasiNnError>) -> wasmtime::Result<ErrorCode> {
        let error = self.table().get(&error)?;
        Ok(error.code)
    }

    fn data(&mut self, error: Resource<WasiNnError>) -> wasmtime::Result<String> {
        let error = self.table().get(&error)?;
        Ok(error.data.clone())
    }

    fn drop(&mut self, error: Resource<WasiNnError>) -> wasmtime::Result<()> {
        let _ = self.table().delete(error)?;
        Ok(())
    }
}

impl<T: WasiNnView> gen::errors::Host for T {}

impl<T: WasiNnView> gen::inference::HostGraphExecutionContext for T {
    fn set_input(
        &mut self,
        exec_context: Resource<ExecutionContext>,
        name: String,
        tensor: Resource<Tensor>,
    ) -> wasmtime::Result<Result<(), Resource<WasiNnError>>> {
        let mut map = std::collections::HashMap::new();
        map.insert(exec_context.rep(), 0);
        map.insert(tensor.rep(), 1);
        let mut resources = self.table().iter_entries(map);
        let mut item0 = resources.next().unwrap();
        let mut item1 = resources.next().unwrap();
        if item0.1 == 1 && item1.1 == 0 {
            std::mem::swap(&mut item0, &mut item1)
        }
        let exec_context = item0
            .0?
            .downcast_mut::<ExecutionContext>()
            .ok_or(wasmtime::component::ResourceTableError::WrongType)?;
        let tensor = item1
            .0?
            .downcast_mut::<Tensor>()
            .ok_or(wasmtime::component::ResourceTableError::WrongType)?;
        exec_context.set_input(&name, tensor)?;
        Ok(Ok(()))
    }

    fn compute(
        &mut self,
        exec_context: Resource<ExecutionContext>,
    ) -> wasmtime::Result<Result<(), Resource<WasiNnError>>> {
        let exec_context = self.table().get_mut(&exec_context)?;
        exec_context.compute()?;
        Ok(Ok(()))
    }

    fn get_output(
        &mut self,
        exec_context: Resource<ExecutionContext>,
        name: String,
    ) -> wasmtime::Result<Result<Resource<Tensor>, Resource<WasiNnError>>> {
        let exec_context = self.table().get_mut(&exec_context)?;
        let tensor = exec_context.get_output(&name)?;
        let resource = self.table().push(tensor)?;
        Ok(Ok(resource))
    }

    fn drop(&mut self, exec_context: Resource<ExecutionContext>) -> wasmtime::Result<()> {
        let _ = self.table().delete(exec_context)?;
        Ok(())
    }
}

impl<T: WasiNnView> gen::inference::Host for T {}

impl<T: WasiNnView> gen::graph::HostGraph for T {
    fn init_execution_context(
        &mut self,
        graph: Resource<Graph>,
    ) -> wasmtime::Result<Result<Resource<ExecutionContext>, Resource<WasiNnError>>> {
        let graph = self.table().get_mut(&graph)?;
        let exec_context = graph.init_execution_context()?;
        let resource = self.table().push(exec_context)?;
        Ok(Ok(resource))
    }

    fn drop(&mut self, graph: Resource<Graph>) -> wasmtime::Result<()> {
        let _ = self.table().delete(graph)?;
        Ok(())
    }
}

impl<T: WasiNnView> gen::graph::Host for T {
    fn load(
        &mut self,
        builders: Vec<gen::graph::GraphBuilder>,
        encoding: gen::graph::GraphEncoding,
        target: gen::graph::ExecutionTarget,
    ) -> wasmtime::Result<Result<Resource<Graph>, Resource<WasiNnError>>> {
        if let Some(backend) = self.ctx().backends.get_mut(&encoding) {
            let slices = builders.iter().map(|s| s.as_slice()).collect::<Vec<_>>();
            let graph = backend.load(&slices, target.into())?;
            let resource = self.table().push(graph)?;
            Ok(Ok(resource))
        } else {
            let error = self.table().push(WasiNnError {
                code: ErrorCode::InvalidEncoding,
                data: format!("{encoding:?}"),
            })?;
            Ok(Err(error))
        }
    }

    fn load_by_name(
        &mut self,
        name: String,
    ) -> wasmtime::Result<Result<Resource<Graph>, Resource<WasiNnError>>> {
        if let Some(graph) = self.ctx().registry.get_mut(&name) {
            let graph = graph.clone();
            let resource = self.table().push(graph)?;
            Ok(Ok(resource))
        } else {
            let error = self.table().push(WasiNnError {
                code: ErrorCode::NotFound,
                data: name,
            })?;
            Ok(Err(error))
        }
    }
}

impl Hash for gen::graph::GraphEncoding {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl FromStr for gen::graph::GraphEncoding {
    type Err = GraphEncodingParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openvino" => Ok(gen::graph::GraphEncoding::Openvino),
            "onnx" => Ok(gen::graph::GraphEncoding::Onnx),
            "pytorch" => Ok(gen::graph::GraphEncoding::Pytorch),
            "tensorflow" => Ok(gen::graph::GraphEncoding::Tensorflow),
            "tensorflowlite" => Ok(gen::graph::GraphEncoding::Tensorflowlite),
            "autodetect" => Ok(gen::graph::GraphEncoding::Autodetect),
            _ => Err(GraphEncodingParseError(s.into())),
        }
    }
}
#[derive(Debug)]
pub struct GraphEncodingParseError(String);
impl fmt::Display for GraphEncodingParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown graph encoding: {}", self.0)
    }
}
impl Error for GraphEncodingParseError {}
