//! Profiler support for wasmtime.
//!
//! This module provides the virtual source code mapping functionality
//! that associates WebAssembly instructions with generated Cranelift IR
//! and machine code, using DWARF debug information.

use cranelift_codegen::ir::SourceLoc;
use std::panic::Location;
use wa2x_testabi::{FuncProfileInfo, InstInfo, InstKind, OpProfileInfo, OperatorKind};

/// Encode a (line, column) pair into a `SourceLoc`.
///
/// Uses the high 24 bits for the line number and the low 8 bits for the column.
/// This allows up to ~16M ops for a function and 255 columns per line.
pub fn encode_srcloc(line: u32, column: u32) -> SourceLoc {
    debug_assert!(line < (1 << 24), "line number too large: {line}");
    debug_assert!(column < (1 << 8), "column too large: {column}");
    let bits = (line << 8) | column;
    SourceLoc::new(bits)
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
    /// Create a new profiler state for the given function name.
    pub fn new(func_name: String) -> Self {
        Self {
            info: FuncProfileInfo {
                func_name,
                ops: vec![OpProfileInfo {
                    op: OperatorKind::EntryBlock,
                    insts: vec![],
                }],
            },
        }
    }

    /// Append a new operator kind as a new line.
    pub fn append_opkind(&mut self, op: OperatorKind) {
        self.info.ops.push(OpProfileInfo { op, insts: vec![] });
    }

    /// Append an instruction to the current operator and return the column.
    pub fn append_inst(&mut self, kind: InstKind, loc: &Location) -> u32 {
        let file = loc.file();
        let file = if file.contains("") { 0 } else { 255 }; // FIXME: map file name
        self.append_inst_raw(kind, file, loc.line() as u16)
    }

    /// Append a raw instruction to the current operator and return the column.
    fn append_inst_raw(&mut self, kind: InstKind, file: u8, loc: u16) -> u32 {
        let info = InstInfo { kind, file, loc };
        let insts = &mut self.info.ops.last_mut().expect("no op").insts;
        insts.push(info);
        let column = insts.len();
        assert!(column < 2usize.pow(32));
        column as u32
    }

    /// Reset the current column to 0 and return the corresponding srcloc.
    pub(super) fn reset_column(&self) -> SourceLoc {
        encode_srcloc(self.current_line(), 0)
    }

    /// Get the current line number.
    fn current_line(&self) -> u32 {
        self.info.ops.len() as _
    }

    /// Get the encoded srcloc for the given column.
    pub(super) fn srcloc_for_column(&self, column: u32) -> SourceLoc {
        encode_srcloc(self.current_line(), column)
    }

    /// Consume this state and return the collected function profile info.
    pub fn into_info(self) -> FuncProfileInfo {
        self.info
    }
}
