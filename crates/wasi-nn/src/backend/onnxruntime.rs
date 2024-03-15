//! Implements a `wasi-nn` [`BackendInner`] using ONNX via ort.

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Context;
use ort::{inputs, GraphOptimizationLevel, Session};

use super::{BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner};
use crate::{
    backend::read,
    wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType},
    ExecutionContext, Graph,
};

#[derive(Default)]
pub struct OnnxBackend();
unsafe impl Send for OnnxBackend {}
unsafe impl Sync for OnnxBackend {}

impl BackendInner for OnnxBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Onnx
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 1 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()).into());
        }

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_memory(builders[0])?;

        let box_: Box<dyn BackendGraph> =
            Box::new(ONNXGraph(Arc::new(Mutex::new(session)), target));
        Ok(box_.into())
    }

    fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for OnnxBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.onnx"))?;
        self.load(&[&model], target)
    }
}

struct ONNXGraph(Arc<Mutex<Session>>, ExecutionTarget);

unsafe impl Send for ONNXGraph {}
unsafe impl Sync for ONNXGraph {}

impl BackendGraph for ONNXGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let session = self.0.lock().unwrap();
        let inputs = session.inputs.iter().map(|_| None).collect::<Vec<_>>();
        let input_map = session
            .inputs
            .iter()
            .enumerate()
            .map(|(idx, input)| (input.name.clone(), idx))
            .collect();
        let outputs = session.outputs.iter().map(|_| None).collect::<Vec<_>>();
        let output_map = session
            .outputs
            .iter()
            .enumerate()
            .map(|(idx, output)| (output.name.clone(), idx))
            .collect();
        let box_: Box<dyn BackendExecutionContext> = Box::new(ONNXExecutionContext {
            session: self.0.clone(),
            inputs,
            input_map,
            outputs,
            output_map,
        });
        Ok(box_.into())
    }
}

struct ONNXExecutionContext {
    session: Arc<Mutex<Session>>,
    inputs: Vec<Option<Tensor>>,
    input_map: HashMap<String, usize>,
    outputs: Vec<Option<Tensor>>,
    output_map: HashMap<String, usize>,
}

unsafe impl Send for ONNXExecutionContext {}
unsafe impl Sync for ONNXExecutionContext {}

impl BackendExecutionContext for ONNXExecutionContext {
    fn set_input(&mut self, name: &str, tensor: &Tensor) -> Result<(), BackendError> {
        let index = self.input_map.get(name).context("input name not found")?;
        self.inputs[*index].replace(tensor.clone());
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        let shaped_inputs: Vec<_> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, _o)| {
                let input = self.inputs[i].as_ref().unwrap();
                let dims = input
                    .dimensions
                    .as_slice()
                    .iter()
                    .map(|d| *d as i64)
                    .collect::<Vec<_>>();
                match input.tensor_type {
                    TensorType::Fp32 => {
                        let data = bytes_to_f32_vec(input.data.to_vec());
                        inputs![(dims, Arc::new(data.into_boxed_slice()))].unwrap()
                    }
                    _ => {
                        unimplemented!("{:?} not supported by ONNX", input.tensor_type);
                    }
                }
            })
            .flatten()
            .collect();

        let session = self.session.lock().unwrap();
        let res = session.run(shaped_inputs.as_slice())?;

        for i in 0..self.outputs.len() {
            let raw: (Vec<i64>, &[f32]) = res[i].extract_raw_tensor()?;
            let f32s = raw.1.to_vec();
            let data = f32_vec_to_bytes(f32s);
            let tensor_type = TensorType::Fp32;
            let dimensions = res[i]
                .dtype()?
                .tensor_dimensions()
                .context("dimensions of output not supported")?
                .iter().map(|&i| i as u32)
                .collect();
            let tensor = Tensor { dimensions, tensor_type, data };
            self.outputs[i].replace(tensor);
        }
        Ok(())
    }

    fn get_output(&mut self, name: &str) -> Result<Tensor, BackendError> {
        let index = self
            .output_map
            .get(name)
            .context("output name not found")?;
        let data = self.outputs[*index].as_ref().unwrap().to_owned();
        Ok(data)
    }
}

impl From<ort::Error> for BackendError {
    fn from(e: ort::Error) -> Self {
        BackendError::BackendAccess(e.into())
    }
}

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let result: Vec<u8> = chunks.iter().flatten().copied().collect();
    result
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks
        .into_iter()
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    v.into_iter().collect()
}
