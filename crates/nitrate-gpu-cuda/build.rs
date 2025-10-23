/*! @build
Compile CUDA .cu sources into a fatbin containing multiple GPU architectures.

This build script generates a fat binary that includes compiled code for multiple
GPU architectures, ensuring compatibility across a wide range of NVIDIA GPUs from
Turing to Blackwell and beyond.

Behavior:
- Only runs when the crate feature `cuda` is enabled (Cargo sets CARGO_FEATURE_CUDA).
- Searches `kernels/` for `*.cu` files.
- Invokes NVCC to produce a fatbin with multiple architectures.
- Generates `${OUT_DIR}/kernel_ptx.rs` embedding the fatbin.

Configuration (optional):
- NVCC: set env NVCC to override the NVCC executable path. Otherwise tries:
  - $CUDA_HOME/bin/nvcc
  - $CUDA_PATH/bin/nvcc
  - "nvcc" (requires it to be in PATH)
- CUDA_ARCH: Comma-separated list of architectures (e.g., "sm_75,sm_86,sm_89").
  If not set, builds for all major architectures from Turing to Blackwell.
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
    let cuda_stub_feature = env::var_os("CARGO_FEATURE_CUDA_STUB").is_some();

    if !cuda_feature && !cuda_stub_feature {
        // Still emit a stub generated file so downstream include! doesn't fail if used.
        if let Err(e) = write_stub_generated() {
            eprintln!("cargo:warning=failed to write stub kernel_ptx.rs: {e}");
        }
        return;
    }

    // For cuda-stub feature, just generate stubs without needing CUDA
    if cuda_stub_feature && !cuda_feature {
        eprintln!("cargo:warning=Building with cuda-stub feature; generating stub PTX");
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

    // Get target architectures
    let architectures = get_target_architectures();
    eprintln!(
        "cargo:warning=Building CUDA fatbin for architectures: {}",
        architectures
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let mut generated_entries: Vec<(String, String)> = Vec::new();

    for cu in cu_files {
        let name = cu
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("kernel")
            .to_string();

        // Use .fatbin extension for fat binaries
        let fatbin_out = out_dir.join(format!("{name}.fatbin"));

        // Build nvcc arguments for fatbin generation
        let mut nvcc_args = vec![
            // Generate fatbin containing multiple architectures
            "-fatbin".to_string(),
            // Optimize for speed
            "-O3".to_string(),
            // Use fast math
            "--use_fast_math".to_string(),
            // Output file
            "-o".to_string(),
            fatbin_out.to_string_lossy().to_string(),
        ];

        // Add gencode flags for each architecture
        for arch in &architectures {
            nvcc_args.push("-gencode".to_string());
            nvcc_args.push(arch.to_gencode_string());
        }

        // Add the source file
        nvcc_args.push(cu.to_string_lossy().to_string());

        // Compile the kernel
        let status = Command::new(&nvcc).args(&nvcc_args).status();

        match status {
            Ok(st) if st.success() => {
                println!(
                    "cargo:warning=nvcc compiled {} -> {} (fatbin with {} architectures)",
                    cu.display(),
                    fatbin_out.display(),
                    architectures.len()
                );
                println!("cargo:rerun-if-changed={}", cu.display());
                generated_entries.push((name, fatbin_out.to_string_lossy().to_string()));
            }
            Ok(st) => {
                eprintln!(
                    "cargo:warning=nvcc failed (exit code: {:?}) for {} using {}",
                    st.code(),
                    cu.display(),
                    nvcc.display()
                );
                eprintln!("cargo:warning=nvcc arguments: {:?}", nvcc_args);
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

    // Generate the Rust source that embeds the fatbin blobs (or stub if none succeeded)
    if let Err(e) = write_generated(&out_dir, &generated_entries) {
        eprintln!("cargo:warning=failed to write generated kernel_ptx.rs: {e}");
    }
}

/// Represents a CUDA architecture target
#[derive(Clone, Debug)]
struct CudaArch {
    /// Virtual architecture (compute capability), e.g., 75, 86, 120
    compute: u32,
    /// Real architectures to compile for (can be multiple for same compute)
    real_archs: Vec<u32>,
    /// Whether to include PTX for JIT compilation
    include_ptx: bool,
}

impl std::fmt::Display for CudaArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.include_ptx {
            write!(f, "sm_{}/compute_{}", self.real_archs[0], self.compute)
        } else {
            write!(f, "sm_{}", self.real_archs[0])
        }
    }
}

impl CudaArch {
    fn new(compute: u32, real_archs: Vec<u32>, include_ptx: bool) -> Self {
        Self {
            compute,
            real_archs,
            include_ptx,
        }
    }

    fn to_gencode_string(&self) -> String {
        let mut codes = Vec::new();

        // Add all real architectures (CUBIN)
        for real in &self.real_archs {
            codes.push(format!("sm_{}", real));
        }

        // Add PTX for JIT if requested
        if self.include_ptx {
            codes.push(format!("compute_{}", self.compute));
        }

        format!("arch=compute_{},code=[{}]", self.compute, codes.join(","))
    }
}

fn get_target_architectures() -> Vec<CudaArch> {
    // Check if CUDA_ARCH is set (comma-separated list)
    if let Ok(arch_str) = env::var("CUDA_ARCH") {
        let archs: Vec<String> = arch_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if !archs.is_empty() {
            return parse_user_architectures(&archs);
        }
    }

    // Default: comprehensive architecture support for production
    // CUDA 13 minimum supported architecture is sm_75
    vec![
        // Turing (RTX 2000, GTX 1600 series)
        CudaArch::new(75, vec![75], false),
        // Ampere (A100, A30, A40, A10)
        CudaArch::new(80, vec![80], false),
        // Ampere (RTX 3000 consumer series)
        CudaArch::new(86, vec![86], false),
        // Ampere (Jetson AGX Orin, A2)
        CudaArch::new(87, vec![87], false),
        // Ada Lovelace (RTX 4000 series, L4, L40)
        CudaArch::new(89, vec![89], false),
        // Hopper (H100, H200)
        CudaArch::new(90, vec![90], false),
        // Blackwell (RTX 5090, B100, B200) with PTX for forward compatibility
        // Include PTX so future architectures can JIT compile
        CudaArch::new(120, vec![120], true),
    ]
}

fn parse_user_architectures(archs: &[String]) -> Vec<CudaArch> {
    let mut result = Vec::new();

    for arch_str in archs {
        if let Some(suffix) = arch_str.strip_prefix("sm_") {
            // Real architecture specified
            if let Ok(num) = suffix.parse::<u32>() {
                // For user-specified architectures, don't add PTX unless it's the last one
                let include_ptx = arch_str == archs.last().unwrap();
                result.push(CudaArch::new(num, vec![num], include_ptx));
            }
        } else if let Some(suffix) = arch_str.strip_prefix("compute_") {
            // Virtual architecture specified - only PTX
            if let Ok(num) = suffix.parse::<u32>() {
                result.push(CudaArch::new(num, vec![], true));
            }
        }
    }

    if result.is_empty() {
        // Fallback to default if parsing failed
        eprintln!("cargo:warning=Failed to parse CUDA_ARCH, using defaults");
        get_target_architectures()
    } else {
        result
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

    // For each entry, expose a const with the fatbin bytes
    for (name, fatbin_path) in entries {
        // Use include_bytes! with OUT_DIR path; this binds the fatbin at compile time.
        // Note: We're calling this PTX for compatibility but it's actually a fatbin
        writeln!(
            f,
            "    pub const {}_PTX: &'static [u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}\"));",
            ident(name),
            escape_path_for_include(fatbin_path, out_dir)
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
fn escape_path_for_include(fatbin_path: &str, out_dir: &Path) -> String {
    // If fatbin_path is already inside OUT_DIR, just use the file name.
    let p = Path::new(fatbin_path);
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
    fatbin_path.replace('\\', "/")
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
