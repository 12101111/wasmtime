//! Profiler support, talking to the wa2x-profiler C API.
//!
//! This module defines the data structures describing the virtual source
//! code mapping of a compiled module, plus the FFI glue to the
//! `libwa2x_profiler` shared library which owns the postcard encoding.
//! Operator kinds are passed as the numeric `OperatorKind` discriminant
//! (the wasm opcode) computed by the Cranelift backend, so this crate does
//! not need to share a wasmparser version with the wa2x repository.

use crate::prelude::*;
use core::ffi::c_char;
use std::ffi::CString;

/// Opaque handle to a `ProfileInfo` owned by the wa2x-profiler library.
#[repr(C)]
pub struct Wa2xProfilerProfileInfo {
    _private: [u8; 0],
}

mod ffi {
    use super::Wa2xProfilerProfileInfo;
    use core::ffi::c_char;

    unsafe extern "C" {
        pub fn map_wasmtime_file(file: *const c_char) -> u8;
        pub fn profile_info_new(module_name: *const c_char) -> *mut Wa2xProfilerProfileInfo;
        pub fn profile_info_free(info: *mut Wa2xProfilerProfileInfo);
        pub fn profile_info_append_func(
            info: *mut Wa2xProfilerProfileInfo,
            func_name: *const c_char,
        ) -> u32;
        pub fn profile_info_append_opkind(
            info: *mut Wa2xProfilerProfileInfo,
            kind: u32,
            pos: u32,
        ) -> u32;
        pub fn profile_info_append_inst(
            info: *mut Wa2xProfilerProfileInfo,
            kind: u8,
            file: u8,
            loc: u16,
        ) -> u32;
        pub fn profile_info_encode(
            info: *mut Wa2xProfilerProfileInfo,
            out_len: *mut usize,
        ) -> *mut u8;
        pub fn profile_info_encoded_free(data: *mut u8, len: usize);
    }
}

/// Map a wasmtime compiler source file path to the `u8` file id used by
/// `CompilerVariant::Wasmtime` (255 when the file is not instrumented).
pub fn map_wasmtime_file(file: &str) -> u8 {
    let file = CString::new(file).expect("file name contains NUL");
    unsafe { ffi::map_wasmtime_file(file.as_ptr()) }
}

/// Profiler info for a whole module: the collected function profiles plus the
/// current line number of the module's virtual source code.
#[derive(Debug, Clone, Default)]
pub struct ProfileInfo {
    /// The current line number within the module's virtual source code.
    pub line_no: u32,
    /// The per-module profile data.
    pub module: ModuleProfileInfo,
}

/// Per-module profile data.
#[derive(Debug, Clone, Default)]
pub struct ModuleProfileInfo {
    /// Name of the module.
    pub module_name: String,
    /// Profiles of the compiled functions, in compilation order.
    pub funcs: Vec<FuncProfileInfo>,
}

/// Profile of a single function.
#[derive(Debug, Clone, Default)]
pub struct FuncProfileInfo {
    /// Name of the function.
    pub func_name: String,
    /// Operators of the function, one per line of the virtual source code.
    pub ops: Vec<OpProfileInfo>,
}

/// Profile of a single operator.
#[derive(Debug, Clone, Default)]
pub struct OpProfileInfo {
    /// The `OperatorKind` discriminant (the wasm opcode) of the operator.
    pub op: u32,
    /// Byte offset of the operator within the wasm binary.
    pub pos: u32,
    /// Machine code instructions generated for this operator.
    pub insts: Vec<InstInfo>,
}

/// A single profiled instruction.
#[derive(Debug, Clone, Default)]
pub struct InstInfo {
    /// The instruction kind.
    pub kind: u8,
    /// The compiler source file id, as returned by [`map_wasmtime_file`].
    pub file: u8,
    /// The line number within the compiler source file.
    pub loc: u16,
}

/// Instruction kinds
#[repr(u8)]
pub enum InstKind {
    /// Not classified.
    Unknown = 0,
    /// A load instruction.
    Load = 1,
    /// A store instruction.
    Store = 2,
    /// An atomic instruction.
    Atomic = 3,
    /// A switch (br_table) instruction.
    Switch = 4,
    /// A direct call instruction.
    Call = 5,
    /// An indirect call instruction.
    CallIndirect = 6,
    /// A return instruction.
    Return = 7,
    /// An unconditional jump instruction.
    Jump = 8,
    /// A conditional branch instruction.
    Branch = 9,
    /// A trap instruction.
    Trap = 10,
    /// A select instruction (RISC-V without Zicond).
    Select = 11,
    /// A small-size memory move.
    MemMove = 12,
    /// A small-size memory set.
    MemSet = 13,
}

impl ProfileInfo {
    /// Create a new empty profile info for a module with the given name.
    pub fn new(module_name: String) -> Self {
        Self {
            line_no: 0,
            module: ModuleProfileInfo {
                module_name,
                funcs: vec![],
            },
        }
    }

    /// Append a function profile, returning the line number it starts at.
    pub fn append_func_profile(&mut self, profile: FuncProfileInfo) -> u32 {
        let old = self.line_no;
        let new_line = profile.ops.len().try_into().expect("overflow");
        self.module.funcs.push(profile);
        self.line_no = self.line_no.checked_add(new_line).expect("overflow");
        old
    }

    /// Encode this profile info into the postcard-encoded `.wasmtime.profile`
    /// section payload, via the wa2x-profiler C API.
    ///
    /// Returns `None` when encoding fails.
    pub fn encode(&self) -> Option<Vec<u8>> {
        let module_name = CString::new(self.module.module_name.as_str()).ok()?;
        unsafe {
            let info = ffi::profile_info_new(module_name.as_ptr());
            if info.is_null() {
                return None;
            }
            let encoded = encode_with(info, self);
            ffi::profile_info_free(info);
            encoded
        }
    }
}

fn encode_with(info: *mut Wa2xProfilerProfileInfo, profile: &ProfileInfo) -> Option<Vec<u8>> {
    for func in &profile.module.funcs {
        let func_name = CString::new(func.func_name.as_str()).ok()?;
        unsafe {
            ffi::profile_info_append_func(info, func_name.as_ptr());
        }
        for op in &func.ops {
            unsafe {
                ffi::profile_info_append_opkind(info, op.op, op.pos);
            }
            for inst in &op.insts {
                unsafe {
                    ffi::profile_info_append_inst(info, inst.kind, inst.file, inst.loc);
                }
            }
        }
    }
    let mut len = 0;
    let data = unsafe { ffi::profile_info_encode(info, &mut len) };
    if data.is_null() {
        return None;
    }
    let encoded = unsafe { core::slice::from_raw_parts(data, len) }.to_vec();
    unsafe {
        ffi::profile_info_encoded_free(data, len);
    }
    Some(encoded)
}
