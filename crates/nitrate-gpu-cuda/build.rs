/*! @build
Compile CUDA .cu sources into PTX at build time and generate a Rust source
file that embeds the resulting PTX blobs via include_bytes!.

Behavior:
- Only runs when the crate feature `cuda` is enabled (Cargo sets CARGO_FEATURE_CUDA).
- Searches `kernels/` for `*.cu` files.
- Invokes NVCC to produce `.ptx` files into OUT_DIR.
- Generates `${OUT_DIR}/kernel_ptx.rs` with a static map of (name -> PTX bytes).

Configuration (optional):
- NVCC: set env NVCC to override the NVCC executable path. Otherwise tries:
  - $CUDA_HOME/bin/nvcc
  - $CUDA_PATH/bin/nvcc
  - "nvcc" (requires it to be in PATH)
- CUDA_ARCH: set to e.g. "sm_86" or "sm_75". Defaults to "sm_52".
*/

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Always rerun if build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Only do CUDA work when the `cuda` feature is enabled on this crate.
    let cuda_feature = env::var_os("CARGO_FEATURE_CUDA").is_some();
    if !cuda_feature {
        // Still emit a stub generated file so downstream include! doesn't fail if used.
        if let Err(e) = write_stub_generated() {
            eprintln!("cargo:warning=failed to write stub kernel_ptx.rs: {e}");
        }
        return;
    }

    // Allow explicitly skipping CUDA compilation (useful for CI without CUDA)
    if env::var("NITRATE_CUDA_SKIP").is_ok() {
        eprintln!("cargo:warning=NITRATE_CUDA_SKIP set; generating stub for CUDA kernels");
        if let Err(e) = write_stub_generated() {
            eprintln!("cargo:warning=failed to write stub kernel_ptx.rs: {e}");
        }
        return;
    }

    // Check if we're in check/clippy mode (don't need actual CUDA)
    let is_check_mode = env::var("CARGO_CFG_DOCTEST").is_ok()
        || env::var("CARGO_CFG_TEST").is_ok()
        || matches!(env::var("CARGO").ok().as_deref(), Some(path) if path.contains("clippy") || path.contains("check"));

    let kernel_dir = PathBuf::from("kernels");
    println!("cargo:rerun-if-changed={}", kernel_dir.display());

    // Collect .cu sources
    let cu_files = match list_cu_sources(&kernel_dir) {
        Ok(list) => list,
        Err(e) => {
            eprintln!("cargo:warning=failed to scan kernels/: {e}");
            // Generate stub and return
            let _ = write_stub_generated();
            return;
        }
    };

    if cu_files.is_empty() {
        eprintln!(
            "cargo:warning=no CUDA .cu sources found under {}/; skipping NVCC",
            kernel_dir.display()
        );
        // Generate a stub; allows the crate to compile without kernels present.
        let _ = write_stub_generated();
        return;
    }

    // Resolve NVCC path
    let nvcc = resolve_nvcc();

    // Check if nvcc actually exists and is executable
    let nvcc_available = Command::new(&nvcc)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if !nvcc_available {
        if is_check_mode {
            // For clippy/check, generate stub and continue
            eprintln!(
                "cargo:warning=NVCC not found at {}; generating stub for check/clippy",
                nvcc.display()
            );
            let _ = write_stub_generated();
            return;
        } else {
            // For actual builds, warn but continue with stub
            eprintln!(
                "cargo:warning=NVCC not found at {}; CUDA kernels will not be available",
                nvcc.display()
            );
            let _ = write_stub_generated();
            return;
        }
    }

    // Select architecture (default reasonable baseline)
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_string());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let mut generated_entries: Vec<(String, String)> = Vec::new();

    for cu in cu_files {
        let name = cu
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("kernel")
            .to_string();

        let ptx_out = out_dir.join(format!("{name}.ptx"));

        // Compile .cu -> .ptx
        let status = Command::new(&nvcc)
            .arg("-ptx")
            .arg("-arch")
            .arg(&cuda_arch)
            .arg("-o")
            .arg(&ptx_out)
            .arg(&cu)
            .status();

        match status {
            Ok(st) if st.success() => {
                println!(
                    "cargo:warning=nvcc compiled {} -> {}",
                    cu.display(),
                    ptx_out.display()
                );
                println!("cargo:rerun-if-changed={}", cu.display());
                generated_entries.push((name, ptx_out.to_string_lossy().to_string()));
            }
            Ok(st) => {
                eprintln!(
                    "cargo:warning=nvcc failed (exit code: {:?}) for {} using {}",
                    st.code(),
                    cu.display(),
                    nvcc.display()
                );
            }
            Err(err) => {
                eprintln!(
                    "cargo:warning=failed to spawn nvcc ({}): {}",
                    nvcc.display(),
                    err
                );
            }
        }
    }

    // Generate the Rust source that embeds the PTX blobs (or stub if none succeeded)
    if let Err(e) = write_generated(&out_dir, &generated_entries) {
        eprintln!("cargo:warning=failed to write generated kernel_ptx.rs: {e}");
    }
}

fn list_cu_sources(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(OsStr::to_str)
            .map(|e| e.eq_ignore_ascii_case("cu"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
    Ok(out)
}

fn resolve_nvcc() -> PathBuf {
    if let Ok(nvcc) = env::var("NVCC") {
        return PathBuf::from(nvcc);
    }
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let p = Path::new(&cuda_home).join("bin").join("nvcc");
        if p.exists() {
            return p;
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let p = Path::new(&cuda_path).join("bin").join("nvcc");
        if p.exists() {
            return p;
        }
    }
    // Fallback to "nvcc" in PATH (may not exist)
    PathBuf::from("nvcc")
}

fn write_stub_generated() -> Result<(), std::io::Error> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    write_generated(&out_dir, &[])
}

fn write_generated(out_dir: &Path, entries: &[(String, String)]) -> Result<(), std::io::Error> {
    let gen_path = out_dir.join("kernel_ptx.rs");
    let mut f = fs::File::create(&gen_path)?;
    writeln!(f, "// @generated by build.rs - DO NOT EDIT")?;
    writeln!(f, "#[allow(dead_code)]")?;
    writeln!(f, "pub mod nitrate_cuda_ptx {{")?;
    // Expose a list of available kernel names
    write!(f, "    pub const NAMES: &[&str] = &[")?;
    for (i, (name, _)) in entries.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "\"{}\"", name)?;
    }
    writeln!(f, "];")?;

    // For each entry, expose a const with the PTX bytes
    for (name, ptx_path) in entries {
        // Use include_bytes! with OUT_DIR path; this binds the PTX at compile time.
        writeln!(
            f,
            "    pub const {}_PTX: &'static [u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}\"));",
            ident(name),
            escape_path_for_include(ptx_path, out_dir)
        )?;
    }

    // Provide a simple accessor by name
    writeln!(
        f,
        "    pub fn get_ptx_by_name(name: &str) -> Option<&'static [u8]> {{"
    )?;
    if entries.is_empty() {
        writeln!(f, "        let _ = name; None")?;
    } else {
        writeln!(f, "        match name {{")?;
        for (name, _) in entries {
            writeln!(f, "            \"{n}\" => Some({n}_PTX),", n = ident(name))?;
        }
        writeln!(f, "            _ => None,")?;
        writeln!(f, "        }}")?;
    }
    writeln!(f, "    }}")?;
    writeln!(f, "}}")?;
    Ok(())
}

// Convert arbitrary kernel name to a valid Rust identifier suffix.
fn ident(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            s.push(c);
        } else {
            s.push('_');
        }
    }
    if s.is_empty() {
        "_kernel".to_string()
    } else if s.chars().next().unwrap().is_ascii_digit() {
        format!("_{}", s)
    } else {
        s
    }
}

// For include_bytes!(concat!(env!("OUT_DIR"), "/...")), we only need the file name component.
// Ensure we reference the file as it will exist in OUT_DIR (strip directories).
fn escape_path_for_include(ptx_path: &str, out_dir: &Path) -> String {
    // If ptx_path is already inside OUT_DIR, just use the file name.
    let p = Path::new(ptx_path);
    if let Some(fname) = p.file_name().and_then(OsStr::to_str) {
        return fname.to_string();
    }
    // Fallback to using the full path relative to OUT_DIR if possible.
    if let Ok(rel) = pathdiff::diff_paths(p, out_dir) {
        if let Some(s) = rel.to_str() {
            return s.replace('\\', "/");
        }
    }
    // Last resort: normalize separators.
    ptx_path.replace('\\', "/")
}

// Minimal pathdiff (avoid adding a build-dependency): simplistic diff for common OUT_DIR usage.
// If this doesn't produce a relative path, the caller falls back to normalized absolute string.
mod pathdiff {
    use std::path::{Component, Path, PathBuf};

    pub fn diff_paths(path: impl AsRef<Path>, base: impl AsRef<Path>) -> Result<PathBuf, ()> {
        let path = path.as_ref();
        let base = base.as_ref();

        let mut ita = base.components();
        let mut itb = path.components();

        // Find common prefix
        let mut a_stack: Vec<Component> = Vec::new();
        let mut b_stack: Vec<Component> = Vec::new();

        for c in ita.by_ref() {
            a_stack.push(c);
        }
        for c in itb.by_ref() {
            b_stack.push(c);
        }

        // Re-iterate to find divergence
        let mut a_iter = a_stack.iter();
        let mut b_iter = b_stack.iter();

        let mut common = 0usize;
        loop {
            match (a_iter.next(), b_iter.next()) {
                (Some(a), Some(b)) if a == b => common += 1,
                _ => break,
            }
        }

        // Build relative path: up for remaining a, then down for remaining b
        let mut rel = PathBuf::new();
        for _ in common..a_stack.len() {
            rel.push("..");
        }
        for c in b_stack.into_iter().skip(common) {
            rel.push(c.as_os_str());
        }
        Ok(rel)
    }
}
