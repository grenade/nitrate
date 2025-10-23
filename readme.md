# Nitrate

Note: TLS is supported. You can use stratum+ssl:// in [pool.url], or set pool.tls=true when no scheme is specified. For testing only, pool.tls_insecure=true disables certificate verification. These map to configuration keys under [pool]: tls and tls_insecure.
CI: A GitHub Actions workflow is provided at .github/workflows/ci.yml to run fmt, clippy, tests, and build the CLI with the dummy backend.

Nitrate is a modular, Rust-based GPU miner scaffold focused on Bitcoin (Stratum v1) with a pluggable GPU backend abstraction. The core engine is algorithm-agnostic; Bitcoin-specific logic lives in a separate crate, and GPU details are behind a trait so multiple backends (CUDA/HIP/OpenCL) can be swapped in. A dummy GPU backend is provided so you can run the full stack while wiring up network/protocol and core scheduling.

Key goals:
- Clear separation of concerns (protocol, Bitcoin math, core scheduling, GPU backends, metrics, config).
- A clean path from a working dummy backend to real CUDA/HIP/OpenCL implementations.
- Strong testing surface (unit, integration, benchmarks).
- Operability (metrics, logging, config).

For an in-depth tour of the workspace and responsibilities, see Agents & Workspace Guide (agents.md). For milestones, see the Roadmap (roadmap.md).


## Features

- Stratum v1 protocol types and pool client (subscribe, authorize, set_difficulty, notify).
- Bitcoin header assembly utilities (nBits/target math, midstate, optional coinbase/merkle).
- Core engine that prepares work, schedules devices, verifies candidates on CPU, and submits shares.
- GPU backend abstraction (`nitrate-gpu-api`) with a dummy backend for development.
- Prometheus metrics exporter and configurable runtime (telemetry, logging, backoff).
- CLI to run the miner, enumerate devices, and future self-tests.


## Installation

### Pre-built Binaries (Recommended)

Download the latest release from [GitHub Releases](https://github.com/grenade/nitrate/releases):

```bash
# Download the latest release
wget https://github.com/grenade/nitrate/releases/latest/download/nitrate-cli-linux-x86_64-cuda

# Make executable
chmod +x nitrate-cli-linux-x86_64-cuda

# Run the miner
./nitrate-cli-linux-x86_64-cuda --config miner.toml
```

Requirements for pre-built binaries:
- Linux x86_64
- NVIDIA GPU with CUDA support
- CUDA driver compatible with CUDA 13.0

### Building from Source

```bash
# Clone the repository
git clone https://github.com/grenade/nitrate.git
cd nitrate

# Build with CUDA support (requires CUDA toolkit)
cargo build --release -p nitrate-cli --features gpu-cuda

# Or build with dummy backend (no GPU required)
cargo build --release -p nitrate-cli --features gpu-dummy

# Run from build directory
./target/release/nitrate-cli --config miner.toml
```

#### Build Options

Environment variables for controlling the build:

- `NVCC`: Override path to NVCC compiler (default: searches CUDA_HOME/bin, CUDA_PATH/bin, or PATH)
- `CUDA_ARCH`: Target CUDA architecture (default: `sm_52`)
- `NITRATE_CUDA_SKIP=1`: Skip CUDA compilation entirely (useful for CI/checks without CUDA toolkit)

Example:
```bash
# Build for specific CUDA architecture
CUDA_ARCH=sm_86 cargo build --release -p nitrate-cli --features gpu-cuda

# Run clippy checks without CUDA toolkit
NITRATE_CUDA_SKIP=1 cargo clippy --features nitrate-core/gpu-cuda
```

## Repository layout

Top-level files:
- `Cargo.toml` — Workspace configuration, shared profiles/lints.
- `rust-toolchain.toml` — Toolchain pin.
- `miner.toml` — Example runtime configuration.
- `agents.md` — Architecture and wiring guide.
- `roadmap.md` — Milestones, definitions of done, and acceptance tests.

Crates (under `crates/`):
- `nitrate-core` — Mining engine (job handling, scheduling, verification, share submission).
- `nitrate-proto` — Stratum v1 types and serde helpers.
- `nitrate-pool` — Pool client (Tokio, JSON-RPC framing, reconnect, job channel).
- `nitrate-btc` — Bitcoin header/target math, midstate, CPU verification.
- `nitrate-gpu-api` — Cross-backend GPU abstraction traits and types.
- `nitrate-gpu-dummy` — No-op backend used for development/testing.
- `nitrate-metrics` — Prometheus exporter (counters, gauges).
- `nitrate-config` — Loading and validating TOML + ENV into strongly typed config.
- `nitrate-utils` — Utilities (double_sha256, hex, endian helpers).
- `nitrate-bench` — Criterion microbenchmarks and perf scaffolding.

Binaries (under `bins/`):
- `nitrate-cli` — Main CLI to run the miner and list devices.
- `nitrate-devtool` — Dev tooling for job replay and kernel param sweeps (future).


## Quick start (dummy GPU backend)

1) Install Rust (matching the pinned toolchain in `rust-toolchain.toml`).

2) Prepare configuration:
    - Copy or edit `miner.toml` at the repository root (see Configuration below).

3) Run the miner (dummy backend):
    - cargo run -p nitrate-cli -- --config miner.toml --log info

4) Verify metrics:
    - curl 127.0.0.1:9100/metrics

5) List devices (no-op with dummy backend, useful once real backends are added):
    - cargo run -p nitrate-cli -- devices


## Configuration

Configuration is loaded from TOML (with environment variable overrides) into a strongly-typed `AppCfg`. The default example file is at the repository root: `miner.toml`.

Sections:

- [pool]
  - url: Pool URL (e.g., stratum+tcp://solo.ckpool.org:3333)
  - user: Wallet or wallet.worker string
  - pass: Password (often "x" or token depending on pool)
- [gpu]
  - backend: One of "dummy", or in the future "cuda" / "hip" / "opencl"
  - devices: List of device indices to use (e.g., [0, 1])
  - intensity: Tuning knob; "auto" or backend-specific values
- [runtime]
  - reconnect_backoff: Backoff range (e.g., "1s..30s")
  - stale_drop_ms: Time after which in-flight work is considered stale
  - telemetry_addr: Prometheus endpoint bind address (e.g., "0.0.0.0:9100")
  - log: Log level for `tracing` (e.g., "info", "debug")

Minimal example:

    [pool]
    url = "stratum+tcp://solo.ckpool.org:3333"
    user = "bc1q...worker1"
    pass = "x"

    [gpu]
    backend = "dummy"
    devices = [0]
    intensity = "auto"

    [runtime]
    reconnect_backoff = "1s..30s"
    stale_drop_ms = 500
    telemetry_addr = "127.0.0.1:9100"
    log = "info"


## Performance Optimizations

The miner includes GPU-specific optimizations that automatically detect and configure optimal settings for different GPU models:

### Supported GPUs and Expected Performance

- **RTX 5090**: ~100-150 GH/s per GPU (2720×512 grid, 4 nonces/thread)
- **RTX 4090**: ~50-80 GH/s per GPU (2048×512 grid, 4 nonces/thread)
- **RTX 3090**: ~30-50 GH/s per GPU (1312×512 grid, 2 nonces/thread)
- **RTX 3060**: ~10-20 GH/s per GPU (448×256 grid, 1 nonce/thread)
- **A100/H100**: Datacenter optimized configurations available

### Multi-GPU Support

To use multiple GPUs, configure the device indices in your config:

```toml
[gpu]
backend = "cuda"
devices = [0, 1]  # Use both GPU 0 and GPU 1
```

### Advanced GPU Tuning

For manual tuning, you can override the auto-detected settings:

```toml
[[gpu.device_overrides]]
device_index = 0
grid_size = 2720        # Number of blocks
block_size = 512        # Threads per block
nonces_per_thread = 4   # Nonces processed per thread
ring_capacity = 32768   # Result buffer size
```

### Performance Monitoring

Monitor your hashrate and GPU utilization:

```bash
# View hashrate from metrics endpoint
curl http://localhost:9100/metrics | grep hashrate

# Monitor GPU utilization
nvidia-smi dmon -s u

# Check for thermal throttling
nvidia-smi -q -d PERFORMANCE
```

### Optimization Details

The miner implements several key optimizations:
- **Automatic GPU detection** with optimal grid/block configurations
- **Shared memory** usage for SHA256 constants
- **CUDA intrinsics** for bit operations (`__funnelshift_r`)
- **Multi-nonce processing** per thread for better parallelism
- **Large work batches** (2B nonces) to reduce kernel launch overhead

## Architecture at a glance

- Protocol layer (`nitrate-proto`, `nitrate-pool`):
  - JSON-RPC framing, subscribe/authorize, difficulty and job events.
  - Emits job updates over an internal async channel.
  - Submit shares and report accept/reject results.

- Bitcoin layer (`nitrate-btc`):
  - nBits ↔ target conversion and share target from pool difficulty.
  - Header assembly and midstate computation (first 64 bytes).
  - Optional coinbase/merkle construction depending on pool-provided parts.
  - CPU double-SHA256 verification for candidate nonces.

- Core engine (`nitrate-core`):
  - Tracks current job, cancels on `clean_jobs`/new notify.
  - Slices nonce ranges per device/stream and launches GPU work.
  - Polls results, verifies candidates, and submits valid shares.
  - Exposes and updates telemetry.

- GPU abstraction (`nitrate-gpu-api`):
  - Defines the `GpuBackend` trait for enumerate/launch/poll lifecycles.
  - Backends implement the trait; the rest of the system remains unchanged.

- Dummy backend (`nitrate-gpu-dummy`):
  - No-op implementation to keep the pipeline runnable while protocol/core are developed.

- Metrics (`nitrate-metrics`):
  - Prometheus counters/gauges: accepted/rejected shares, hashrate, reconnects, stales, latencies.


## Development workflow

- Format, lint, and test:
    - cargo fmt
    - cargo clippy -- -D warnings
    - cargo test --workspace

- Targeted testing:
    - cargo test -p nitrate-btc
    - RUST_LOG=nitrate_pool=debug cargo run -p nitrate-cli -- --config miner.toml

- Benchmarks:
    - cargo bench -p nitrate-bench

- Observability:
    - Start the CLI and curl the metrics endpoint to validate counters, gauges, and liveness.

- Feature flags (future):
    - Real GPU backends (e.g., CUDA/HIP/OpenCL) will be opt-in features enabled at build time.


## Operational notes

- Cancellations and stales:
  - The core cancels in-flight work on `clean_jobs` or when a new job arrives.
  - Tune nonce chunk sizes for a balance between latency and throughput.

- Stability and reconnects:
  - The pool client uses timeouts, keepalive, exponential backoff, and DNS re-resolve.
  - Metrics track reconnect events.

- Security:
  - Do not log secrets (passwords, sensitive wallet data).
  - Validate JSON inputs and bounds for extranonces/branches.
  - Make share submission idempotent and rate-limited where appropriate.


## Roadmap and detailed docs

- Agents & Workspace Guide: ./agents.md
  - Deep dive into crate responsibilities, interfaces, and wiring instructions.
- Roadmap: ./roadmap.md
  - Milestones (M0..M12), definitions of done, acceptance tests, and packaging/CI plans.


## Contributing

- Rust 2021 edition.
- Keep `clippy` clean; prefer `anyhow` at app edges and `thiserror` for library errors.
- Use `tracing` spans for job and device context in logs.
- Keep public APIs small, testable, and backend-agnostic.
- Open a PR with descriptive commit messages and include tests or updates to existing tests where possible.

### Version Management

The workspace uses a single version defined in the root `Cargo.toml`:
```toml
[workspace.package]
version = "0.1.0"
```

All crates inherit this version using `version.workspace = true`. To manage versions:

```bash
# Get current version
./scripts/version.sh get

# Set new version
./scripts/version.sh set 1.2.3

# Show all crate versions
./scripts/version.sh show

# Bump version
./scripts/version.sh bump patch  # or minor/major
```

## CI/CD Notes

The CI pipeline runs on standard GitHub runners without CUDA. For CUDA-related checks:
- Clippy and format checks use `NITRATE_CUDA_SKIP=1` to generate stub PTX modules
- Release builds use the `ghcr.io/quantus-network/cuda-builder:13.0.0` Docker image with CUDA 13.0
- The build system gracefully handles missing NVCC for check/clippy operations

## Release Process

Releases are created through GitHub Actions:

1. Go to [Actions → Release workflow](https://github.com/grenade/nitrate/actions/workflows/release.yml)
2. Click "Run workflow"
3. Select version bump type (major, minor, or patch)
4. The workflow will:
   - Run tests and checks
   - Build a Linux x86_64 binary with CUDA support
   - Create a GitHub release with the binary
   - Tag the repository with the new version

Binary releases include:
- `nitrate-cli-linux-x86_64-cuda`: Standalone executable with CUDA support
- `nitrate-X.Y.Z-linux-x86_64-cuda.tar.gz`: Archive with binary and example configuration

Happy hacking!