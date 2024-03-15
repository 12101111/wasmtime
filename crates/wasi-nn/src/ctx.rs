//! Implements the host state for the `wasi-nn` API: [WasiNnCtx].

use crate::backend::{self, BackendError};
use crate::wit::types::GraphEncoding;
use crate::{Backend, InMemoryRegistry, Registry};
use anyhow::anyhow;
use std::{collections::HashMap, path::Path};
use thiserror::Error;
use wiggle::GuestError;

type BackendName = String;
type GraphDirectory = String;

/// Construct an in-memory registry from the available backends and a list of
/// `(<backend name>, <graph directory>)`. This assumes graphs can be loaded
/// from a local directory, which is a safe assumption currently for the current
/// model types.
pub fn preload(
    preload_graphs: &[(BackendName, GraphDirectory)],
) -> anyhow::Result<(impl IntoIterator<Item = Backend>, Registry)> {
    let mut backends = backend::list();
    let mut registry = InMemoryRegistry::new();
    for (kind, path) in preload_graphs {
        let kind_ = kind.parse()?;
        let backend = backends
            .iter_mut()
            .find(|b| b.encoding() == kind_)
            .ok_or(anyhow!("unsupported backend: {}", kind))?
            .as_dir_loadable()
            .ok_or(anyhow!("{} does not support directory loading", kind))?;
        registry.load(backend, Path::new(path))?;
    }
    Ok((backends, Registry::from(registry)))
}

pub trait WasiNnView: Send {
    fn table(&mut self) -> &mut wasmtime::component::ResourceTable;
    fn ctx(&mut self) -> &mut WasiNnCtx;
}

/// Capture the state necessary for calling into the backend ML libraries.
pub struct WasiNnCtx {
    pub(crate) backends: HashMap<GraphEncoding, Backend>,
    pub(crate) registry: Registry,
}

impl WasiNnCtx {
    /// Make a new context from the default state.
    pub fn new(backends: impl IntoIterator<Item = Backend>, registry: Registry) -> Self {
        let backends = backends.into_iter().map(|b| (b.encoding(), b)).collect();
        Self {
            backends,
            registry,
        }
    }
}

/// Possible errors while interacting with [WasiNnCtx].
#[derive(Debug, Error)]
pub enum WasiNnError {
    #[error("backend error")]
    BackendError(#[from] BackendError),
    #[error("guest error")]
    GuestError(#[from] GuestError),
    #[error("usage error")]
    UsageError(#[from] UsageError),
}

#[derive(Debug, Error)]
pub enum UsageError {
    #[error("Invalid context; has the load function been called?")]
    InvalidContext,
    #[error("Only OpenVINO's IR is currently supported, passed encoding: {0:?}")]
    InvalidEncoding(GraphEncoding),
    #[error("OpenVINO expects only two buffers (i.e. [ir, weights]), passed: {0}")]
    InvalidNumberOfBuilders(u32),
    #[error("Invalid graph handle; has it been loaded?")]
    InvalidGraphHandle,
    #[error("Invalid execution context handle; has it been initialized?")]
    InvalidExecutionContextHandle,
    #[error("Not enough memory to copy tensor data of size: {0}")]
    NotEnoughMemory(u32),
    #[error("No graph found with name: {0}")]
    NotFound(String),
}

pub(crate) type WasiNnResult<T> = std::result::Result<T, WasiNnError>;

#[cfg(test)]
mod test {
    use super::*;
    use crate::Graph;
    use crate::registry::GraphRegistry;

    #[test]
    fn example() {
        struct FakeRegistry;
        impl GraphRegistry for FakeRegistry {
            fn get_mut(&mut self, _: &str) -> Option<&mut Graph> {
                None
            }
        }

        let _ctx = WasiNnCtx::new([], Registry::from(FakeRegistry));
    }
}
