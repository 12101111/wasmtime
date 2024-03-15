//! Implements a `wasi-nn` [`BackendInner`] using OpenVINO.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use openvino::{InferenceError, Layout, Precision, SetupError, TensorDesc};

use super::{
    read, BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner,
};
use crate::{
    wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType},
    ExecutionContext, Graph,
};

#[derive(Default)]
pub struct OpenvinoBackend(Option<openvino::Core>);
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 2 {
            return Err(BackendError::InvalidNumberOfBuilders(2, builders.len()).into());
        }

        // Construct the context if none is present; this is done lazily (i.e.
        // upon actually loading a model) because it may fail to find and load
        // the OpenVINO libraries. The laziness limits the extent of the error
        // only to wasi-nn users, not all WASI users.
        if self.0.is_none() {
            self.0.replace(openvino::Core::new(None)?);
        }

        // Read the guest array.
        let xml = &builders[0];
        let weights = &builders[1];

        // Construct OpenVINO graph structures: `cnn_network` contains the graph
        // structure, `exec_network` can perform inference.
        let core = self
            .0
            .as_mut()
            .expect("openvino::Core was previously constructed");
        let mut cnn_network = core.read_network_from_buffer(&xml, &weights)?;

        // TODO: this is a temporary workaround. We need a more elegant way to
        // specify the layout in the long run. However, without this newer
        // versions of OpenVINO will fail due to parameter mismatch.
        for i in 0..cnn_network.get_inputs_len()? {
            let name = cnn_network.get_input_name(i)?;
            cnn_network.set_input_layout(&name, Layout::NHWC)?;
        }

        let exec_network =
            core.load_network(&cnn_network, map_execution_target_to_string(target))?;
        let box_: Box<dyn BackendGraph> = Box::new(OpenvinoGraph(
            Arc::new(cnn_network),
            Arc::new(Mutex::new(exec_network)),
        ));
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for OpenvinoBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.xml"))?;
        let weights = read(&path.join("model.bin"))?;
        self.load(&[&model, &weights], target)
    }
}

struct OpenvinoGraph(
    Arc<openvino::CNNNetwork>,
    Arc<Mutex<openvino::ExecutableNetwork>>,
);

unsafe impl Send for OpenvinoGraph {}
unsafe impl Sync for OpenvinoGraph {}

impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let mut network = self.1.lock().unwrap();
        let infer_request = network.create_infer_request()?;
        let box_: Box<dyn BackendExecutionContext> =
            Box::new(OpenvinoExecutionContext(self.0.clone(), infer_request));
        Ok(box_.into())
    }
}

struct OpenvinoExecutionContext(Arc<openvino::CNNNetwork>, openvino::InferRequest);

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(&mut self, name: &str, tensor: &Tensor) -> Result<(), BackendError> {
        // Construct the blob structure. TODO: there must be some good way to
        // discover the layout here; `desc` should not have to default to NHWC.
        let precision = map_tensor_type_to_precision(tensor.tensor_type);
        let dimensions = tensor
            .dimensions
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>();
        let desc = TensorDesc::new(Layout::NHWC, &dimensions, precision);
        let blob = openvino::Blob::new(&desc, &tensor.data)?;

        // Actually assign the blob to the request.
        self.1.set_blob(name, &blob)?;
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.1.infer()?;
        Ok(())
    }

    fn get_output(&mut self, name: &str) -> Result<Tensor, BackendError> {
        let blob = self.1.get_blob(name)?;

        let desc = blob.tensor_desc()?;
        let precision = desc.precision();

        let dimensions = desc.dims().iter().map(|&d| d as u32).collect();
        let tensor_type = map_precision_to_tensor_type(precision);
        let data = blob.buffer()?.to_owned();

        Ok(Tensor {
            dimensions,
            tensor_type,
            data,
        })
    }
}

impl From<InferenceError> for BackendError {
    fn from(e: InferenceError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

impl From<SetupError> for BackendError {
    fn from(e: SetupError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
fn map_execution_target_to_string(target: ExecutionTarget) -> &'static str {
    match target {
        ExecutionTarget::Cpu => "CPU",
        ExecutionTarget::Gpu => "GPU",
        ExecutionTarget::Tpu => unimplemented!("OpenVINO does not support TPU execution targets"),
    }
}

/// Return OpenVINO's precision type for the `TensorType` enum provided by
/// wasi-nn.
fn map_tensor_type_to_precision(tensor_type: TensorType) -> openvino::Precision {
    match tensor_type {
        TensorType::Fp16 => Precision::FP16,
        TensorType::Fp32 => Precision::FP32,
        TensorType::Fp64 => Precision::FP64,
        TensorType::U8 => Precision::U8,
        TensorType::I32 => Precision::I32,
        TensorType::I64 => Precision::I64,
        TensorType::Bf16 => todo!("not yet supported in `openvino` bindings"),
    }
}

fn map_precision_to_tensor_type(precision: openvino::Precision) -> TensorType {
    match precision {
        Precision::FP16 => TensorType::Fp16,
        Precision::FP32 => TensorType::Fp32,
        Precision::FP64 => TensorType::Fp64,
        Precision::U8 => TensorType::U8,
        Precision::I32 => TensorType::I32,
        Precision::I64 => TensorType::I64,
        _ => todo!("not yet supported"),
    }
}
