//! Implements the `wasi-nn` API for the WITX ("preview1") ABI.
//!
//! `wasi-nn` was never included in the official "preview1" snapshot, but this
//! module implements the ABI that is compatible with "preview1".
//!
//! The only export from this module is [`add_to_linker`]. To implement it, this
//! module proceeds in steps:
//! 1. generate all of the Wiggle glue code into a `gen::*` namespace
//! 2. wire up the `gen::*` glue to the context state, delegating actual
//!    computation to a `Backend`
//! 3. wrap up with some conversions, i.e., from `gen::*` types to this crate's
//!    [`types`].
//!
//! [`types`]: crate::wit::types

use wasmtime::component::Resource;
use wiggle::GuestPtr;

use crate::{
    backend::BackendError,
    ctx::{UsageError, WasiNnError, WasiNnResult as Result, WasiNnView},
};

pub use gen::wasi_ephemeral_nn::add_to_linker;

/// Generate the traits and types from the `wasi-nn` WITX specification.
mod gen {
    use super::*;
    wiggle::from_witx!({
        witx: ["$WASI_ROOT/wasi-nn.witx"],
        errors: { nn_errno => WasiNnError }
    });

    /// Additionally, we must let Wiggle know which of our error codes
    /// represents a successful operation.
    impl wiggle::GuestErrorType for types::NnErrno {
        fn success() -> Self {
            Self::Success
        }
    }

    /// Convert the host errors to their WITX-generated type.
    impl<'a, T: WasiNnView> types::UserErrorConversion for T {
        fn nn_errno_from_wasi_nn_error(
            &mut self,
            e: WasiNnError,
        ) -> anyhow::Result<types::NnErrno> {
            tracing::debug!("host error: {:?}", e);
            match e {
                WasiNnError::BackendError(_) => unimplemented!(),
                WasiNnError::GuestError(_) => unimplemented!(),
                WasiNnError::UsageError(_) => unimplemented!(),
            }
        }
    }
}

/// Wire up the WITX-generated trait to the `wasi-nn` host state.
impl<'a, T: WasiNnView> gen::wasi_ephemeral_nn::WasiEphemeralNn for T {
    fn load<'b>(
        &mut self,
        builders: &gen::types::GraphBuilderArray<'_>,
        encoding: gen::types::GraphEncoding,
        target: gen::types::ExecutionTarget,
    ) -> Result<gen::types::Graph> {
        let graph = if let Some(backend) = self.ctx().backends.get_mut(&encoding.into()) {
            // Retrieve all of the "builder lists" from the Wasm memory (see
            // $graph_builder_array) as slices for a backend to operate on.
            let mut slices = vec![];
            for builder in builders.iter() {
                let slice = builder?
                    .read()?
                    .as_slice()?
                    .expect("cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)");
                slices.push(slice);
            }
            let slice_refs = slices.iter().map(|s| s.as_ref()).collect::<Vec<_>>();
            backend.load(&slice_refs, target.into())?
        } else {
            return Err(UsageError::InvalidEncoding(encoding.into()).into());
        };
        let graph_id = self
            .table()
            .push(graph)
            .map_err(|e| BackendError::BackendAccess(e.into()))?;
        Ok(graph_id.rep().into())
    }

    fn load_by_name<'b>(&mut self, name: &wiggle::GuestPtr<'b, str>) -> Result<gen::types::Graph> {
        let name = name.as_str()?.unwrap();
        if let Some(graph) = self.ctx().registry.get_mut(&name) {
            let graph = graph.clone();
            let graph_id = self
                .table()
                .push(graph)
                .map_err(|e| BackendError::BackendAccess(e.into()))?;
            Ok(graph_id.rep().into())
        } else {
            return Err(UsageError::NotFound(name.to_string()).into());
        }
    }

    fn init_execution_context(
        &mut self,
        graph_id: gen::types::Graph,
    ) -> Result<gen::types::GraphExecutionContext> {
        let resource: Resource<crate::Graph> = Resource::new_own(graph_id.into());
        let exec_context = if let Ok(graph) = self.table().get_mut(&resource) {
            graph.init_execution_context()?
        } else {
            return Err(UsageError::InvalidGraphHandle.into());
        };

        let exec_context_id = self
            .table()
            .push(exec_context)
            .map_err(|e| BackendError::BackendAccess(e.into()))?;
        Ok(exec_context_id.rep().into())
    }

    fn set_input<'b>(
        &mut self,
        exec_context_id: gen::types::GraphExecutionContext,
        name: &wiggle::GuestPtr<'b, str>,
        tensor: &gen::types::Tensor<'b>,
    ) -> Result<()> {
        let resource: Resource<crate::ExecutionContext> = Resource::new_own(exec_context_id.into());
        if let Ok(exec_context) = self.table().get_mut(&resource) {
            let name = name.as_str()?.unwrap();
            let tensor = crate::wit::types::Tensor {
                dimensions: tensor.dimensions.to_vec()?,
                tensor_type: tensor.type_.into(),
                data: tensor.data.to_vec()?,
            };
            Ok(exec_context.set_input(&name, &tensor)?)
        } else {
            Err(UsageError::InvalidGraphHandle.into())
        }
    }

    fn compute(&mut self, exec_context_id: gen::types::GraphExecutionContext) -> Result<()> {
        let resource: Resource<crate::ExecutionContext> = Resource::new_own(exec_context_id.into());
        if let Ok(exec_context) = self.table().get_mut(&resource) {
            Ok(exec_context.compute()?)
        } else {
            Err(UsageError::InvalidExecutionContextHandle.into())
        }
    }

    fn get_output<'b>(
        &mut self,
        exec_context_id: gen::types::GraphExecutionContext,
        name: &wiggle::GuestPtr<'b, str>,
        out_buffer: &GuestPtr<'_, u8>,
        out_buffer_max_size: u32,
    ) -> Result<u32> {
        let resource: Resource<crate::ExecutionContext> = Resource::new_own(exec_context_id.into());
        if let Ok(exec_context) = self.table().get_mut(&resource) {
            let mut destination = out_buffer
                .as_array(out_buffer_max_size)
                .as_slice_mut()?
                .expect("cannot use with shared memories; see https://github.com/bytecodealliance/wasmtime/issues/5235 (TODO)");
            let name = name.as_str()?.unwrap();
            let tensor = exec_context.get_output(&name)?;
            let output = tensor.data;
            destination[..output.len()].copy_from_slice(&output);
            Ok(output.len() as u32)
        } else {
            Err(UsageError::InvalidGraphHandle.into())
        }
    }
}

// Implement some conversion from `witx::types::*` to this crate's version.

impl From<gen::types::ExecutionTarget> for crate::wit::types::ExecutionTarget {
    fn from(value: gen::types::ExecutionTarget) -> Self {
        match value {
            gen::types::ExecutionTarget::Cpu => crate::wit::types::ExecutionTarget::Cpu,
            gen::types::ExecutionTarget::Gpu => crate::wit::types::ExecutionTarget::Gpu,
            gen::types::ExecutionTarget::Tpu => crate::wit::types::ExecutionTarget::Tpu,
        }
    }
}
impl From<gen::types::GraphEncoding> for crate::wit::types::GraphEncoding {
    fn from(value: gen::types::GraphEncoding) -> Self {
        match value {
            gen::types::GraphEncoding::Openvino => crate::wit::types::GraphEncoding::Openvino,
            gen::types::GraphEncoding::Onnx => crate::wit::types::GraphEncoding::Onnx,
            gen::types::GraphEncoding::Tensorflow => crate::wit::types::GraphEncoding::Tensorflow,
            gen::types::GraphEncoding::Pytorch => crate::wit::types::GraphEncoding::Pytorch,
            gen::types::GraphEncoding::Tensorflowlite => {
                crate::wit::types::GraphEncoding::Tensorflowlite
            }
            gen::types::GraphEncoding::Autodetect => crate::wit::types::GraphEncoding::Autodetect,
        }
    }
}
impl From<gen::types::TensorType> for crate::wit::types::TensorType {
    fn from(value: gen::types::TensorType) -> Self {
        match value {
            gen::types::TensorType::F16 => crate::wit::types::TensorType::Fp16,
            gen::types::TensorType::F32 => crate::wit::types::TensorType::Fp32,
            gen::types::TensorType::U8 => crate::wit::types::TensorType::U8,
            gen::types::TensorType::I32 => crate::wit::types::TensorType::I32,
            gen::types::TensorType::I64 => crate::wit::types::TensorType::I64,
            gen::types::TensorType::F64 => crate::wit::types::TensorType::Fp64,
        }
    }
}
