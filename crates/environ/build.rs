//! link against the wa2x-profiler C library when the `profiler` feature is enabled.

use std::path::{Path, PathBuf};

fn main() {
    if std::env::var_os("CARGO_FEATURE_PROFILER").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=WA2X_PROFILER_DIR");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");
    let dir = std::env::var_os("WA2X_PROFILER_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("CARGO_TARGET_DIR").map(|dir| Path::new(&dir).join("release")))
        .expect("can't find path to libwa2x_profiler, need WA2X_PROFILER_DIR or CARGO_TARGET_DIR");
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=wa2x_profiler");
}
