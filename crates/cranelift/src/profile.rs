//! Profiler support for wasmtime.
//!
//! This module provides the virtual source code mapping functionality
//! that associates WebAssembly instructions with generated Cranelift IR
//! and machine code, using DWARF debug information.

use cranelift_codegen::ir::SourceLoc;
use wasmtime_environ::profile::{FuncProfileInfo, InstInfo, InstKind, OpProfileInfo, map_wasmtime_file};

mod operator;
pub use operator::OperatorKind;

use crate::ProfileBuilder;

/// Encode a (line, column) pair into a `SourceLoc`.
///
/// Uses the high 24 bits for the line number and the low 8 bits for the column.
/// This allows up to ~16M ops for a function and 255 columns per line.
pub fn encode_srcloc(line: u32, column: u32) -> Option<SourceLoc> {
    if line >= (1 << 24) || column >= (1 << 8) {
        log::error!("line/column: {line}/{column} too large");
        return None;
    }
    let bits = (line << 8) | column;
    Some(SourceLoc::new(bits))
}

/// Decode a `SourceLoc` into a (line, column) pair.
pub fn decode_srcloc(loc: SourceLoc) -> (u32, u32) {
    let bits = loc.bits();
    let line = bits >> 8;
    let column = bits & 0xff;
    (line, column)
}

/// Per-function profiler state maintained during function translation.
pub struct ProfileState {
    info: FuncProfileInfo,
}

impl ProfileState {
    /// Create a new profiler state for the given function name with EntryBlock as entry_op.
    pub fn new(func_name: String, pos: u32) -> Self {
        Self::new_with_op(func_name, OperatorKind::EntryBlock, pos)
    }

    /// Create a new profiler state for the given function name with entry_op.
    pub fn new_with_op(func_name: String, entry_op: OperatorKind, pos: u32) -> Self {
        Self {
            info: FuncProfileInfo {
                func_name,
                ops: vec![OpProfileInfo {
                    op: entry_op as u32,
                    pos,
                    insts: vec![],
                }],
            },
        }
    }

    /// Append a new operator kind as a new line.
    pub fn append_opkind(&mut self, op: OperatorKind, pos: u32) {
        self.info.ops.push(OpProfileInfo {
            op: op as u32,
            pos,
            insts: vec![],
        });
    }

    /// Append an instruction to the current operator and set the builder's srcloc to the current loc.
    #[track_caller]
    pub fn append_profile_inst<B: ProfileBuilder>(&mut self, builder: &mut B, kind: InstKind) {
        let loc = std::panic::Location::caller();
        let file = map_wasmtime_file(loc.file());
        let loc = loc.line().try_into().unwrap();
        let info = InstInfo {
            kind: kind as u8,
            file,
            loc,
        };
        let insts = &mut self.info.ops.last_mut().expect("no op").insts;
        insts.push(info);
        let column = insts.len().try_into().unwrap();
        if let Some(loc) = encode_srcloc(self.current_line(), column) {
            builder.set_srcloc(loc);
        } else {
            panic!("encode error: {:?}", self.info)
        }
    }

    /// Reset the builder's srcloc to column 0.
    #[cfg(feature = "profiler")]
    pub fn append_profile_default_inst<B: crate::ProfileBuilder>(&self, builder: &mut B) {
        if let Some(loc) = encode_srcloc(self.current_line(), 0) {
            builder.set_srcloc(loc);
        } else {
            panic!("encode error: {:?}", self.info)
        }
    }

    /// Get the current line number.
    fn current_line(&self) -> u32 {
        self.info.ops.len().try_into().unwrap()
    }

    pub(crate) fn current_srcloc(&self) -> SourceLoc {
        if let Some(loc) = encode_srcloc(self.current_line(), 0) {
            loc
        } else {
            panic!("encode error: {:?}", self.info)
        }
    }

    /// Consume this state and return the collected function profile info.
    pub fn into_info(self) -> FuncProfileInfo {
        self.info
    }
}
