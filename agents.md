# Nitrate Agents & Workspace Guide

This document explains how the **Nitrate** GPU miner workspace is organized, what each crate is responsible for, and how to wire up the remaining implementation (Stratum v1, Bitcoin header assembly, and real GPU backends). It’s written so a new contributor can land, run the dummy backend, and then replace pieces incrementally.

---

## 1) Workspace at a glance

```
nitrate/
├─ Cargo.toml                  # [workspace], profiles, shared lints
├─ rust-toolchain.toml
├─ .cargo/config.toml
├─ miner.toml                  # runtime config (pool/gpu/runtime)
├─ crates/
│  ├─ nitrate-core/            # mining engine (jobs, scheduling, verification)
│  ├─ nitrate-proto/           # Stratum v1 protocol types (serde/JSON)
│  ├─ nitrate-pool/            # pool client & reconnect logic
│  ├─ nitrate-btc/             # Bitcoin header/target math & verification
│  ├─ nitrate-gpu-api/         # cross-backend GPU abstraction traits
│  ├─ nitrate-gpu-dummy/       # dummy backend for dev/testing
│  ├─ nitrate-metrics/         # Prometheus metrics exporter
│  ├─ nitrate-config/          # config loading/validation
│  ├─ nitrate-utils/           # small helpers (double_sha256, hex, endian)
│  └─ nitrate-bench/           # criterion microbenches & perf scaffolding
└─ bins/
   ├─ nitrate-cli/             # main CLI binary (tokio)
   └─ nitrate-devtool/         # job replay / kernel param sweeper (future)
```

**Key idea:** the engine (`nitrate-core`) stays *algorithm-agnostic*. Bitcoin-specific logic (midstate, compact targets, coinbase/merkle) is in `nitrate-btc`. GPU specifics are hidden behind `nitrate-gpu-api`, with one or more interchangeable backends (`nitrate-gpu-dummy` today; CUDA/HIP/OpenCL later).

---

## 2) Ownership & responsibilities

### `nitrate-core`
- Orchestrates mining: receive job → prepare work → split per device/stream → launch → collect → verify → submit.
- Cancels in-flight work on `clean_jobs` or new `notify`.
- Provides backpressure and keeps CPU hot-path verification tiny.
- Talks to:
  - `nitrate-pool` for jobs & share submission.
  - `nitrate-gpu-api` (selected backend) for hashing.
  - `nitrate-btc` for header/midstate/target math.
  - `nitrate-metrics` for Prometheus.
  - `nitrate-config` to load settings at startup.

### `nitrate-proto`
- Minimal Stratum v1 message shapes (serde). It does not open sockets; it only defines types and helpers to parse/format frames.

### `nitrate-pool`
- TCP client using Tokio, line-delimited JSON-RPC framing.
- Handles `mining.subscribe`, `mining.authorize`, `mining.set_difficulty`, `mining.notify`.
- Emits jobs to `nitrate-core` via an internal async channel.
- Accepts share submissions and reports responses (accepted/rejected/stale).

### `nitrate-btc`
- Bitcoin‑specific bits:
  - `nBits ↔ target` math, share targets from pool difficulty.
  - Header assembly & midstate computation (SHA256 first chunk).
  - Coinbase/merkle construction (feature‑gated; some pools provide full header parts).
  - CPU verification (double‑SHA256) before submitting a candidate.

### `nitrate-gpu-api`
- Trait-level abstraction for GPU work:
  ```rust
  #[async_trait]
  pub trait GpuBackend {
      async fn enumerate(&self) -> Result<Vec<DeviceInfo>>;
      async fn launch(&self, device_index: u32, work: KernelWork) -> Result<()>;
      async fn poll_results(&self, device_index: u32) -> Result<Vec<FoundNonce>>;
  }
  ```
- `KernelWork` contains the **midstate**, a small **tail** (time/bits/nonce slot), **target**, and a **nonce range**.

### `nitrate-gpu-dummy`
- No-op GPU backend. Implements the trait so the rest of the stack compiles and runs while wiring network/protocol pieces.

### `nitrate-metrics`
- Prometheus exporter (Hyper 0.14) exposing counters/gauges like accepted/rejected shares, hashrate, reconnects, job latency.

### `nitrate-config`
- Loads TOML + ENV overrides and produces strongly-typed config (`AppCfg`).

### `nitrate-utils`
- `double_sha256`, byte/hex helpers, small utilities.

### `nitrate-bench`
- Criterion microbenches: kernel vs CPU reference, end-to-end latency, host‑device copies (once GPU backends land).

### `nitrate-cli`
- CLI (clap) to run miner, list devices, and (later) run self-tests.

### `nitrate-devtool`
- Offline tooling: replay captured `notify` streams; sweep kernel params and record perf/occupancy (future).

---

## 3) How to wire up the rest

This section gives you concrete TODOs and call sites to fill in.

### A) Stratum v1 (in `nitrate-pool`)

1. **Framing**: line-delimited JSON per Stratum v1. Use `BufReader::lines()`.
2. **Subscribe/Authorize**:
   - Send `mining.subscribe` with client ID (e.g., `"Nitrate/0.1"`).
   - Parse response for `extranonce1` and `extranonce2_size`.
   - Send `mining.authorize` with `user` (wallet[.worker]) and `pass`.
3. **Read loop**:
   - Handle `mining.set_difficulty` → update share target (store atomically).
   - Handle `mining.notify` → build a `Job`:
     - job id, version, prevhash, coinbase parts (or header parts), merkle branches, ntime, nbits, clean flag.
     - produce: **midstate** and **header tail** (see §B).
     - send to `nitrate-core` via mpsc channel.
   - On `clean_jobs=true`, trigger core cancellation by pushing a “fresh” job and dropping old ones.
4. **Share submission**:
   - `mining.submit` parameters: worker, job_id, extranonce2, ntime, nonce.
   - Parse response, increment metrics (accepted/rejected), and log rejections.

**Interfaces**:
- In `nitrate-pool`, expose:
  ```rust
  pub struct StratumClient { /* ... */ }
  pub async fn connect(cfg: PoolConfig) -> Result<Self>;
  pub fn job_rx(&self) -> mpsc::Receiver<Job>;  // for core to consume
  pub async fn submit_share(&self, share: Share) -> Result<SubmitResult>;
  ```

### B) Bitcoin header & midstate (in `nitrate-btc`)

- **If pool provides coinbase parts**:
  1. Construct coinbase: `coinbase1 + extranonce1 + extranonce2 + coinbase2`.
  2. `merkle_root = SHA256d(coinbase)`, then fold over merkle branches.
  3. Assemble 80‑byte header:
     - version (4), prevhash (32), merkle_root (32), time (4), bits (4), nonce (4).
- **Midstate**:
  - Precompute SHA256 state after hashing bytes 0..63 (first 64 bytes of header).
  - Pass midstate + the remaining 16 bytes (time|bits|nonce placeholder) to GPU.
- **Targets**:
  - From pool difficulty (share target), compute a **big‑endian** 32‑byte target used by kernels.
- **CPU verify**:
  - When a GPU finds a candidate, insert the nonce into the header tail, compute SHA256d(header), and compare ≤ target.

**Interfaces**:
```rust
pub struct PreparedWork {
    pub midstate: [u32; 8],
    pub tail: [u8; 12],       // time (4) | bits (4) | nonce(4 placeholder)
    pub target_be: [u8; 32],
}
pub fn prepare_from_notify(n: &Notify, diff: u64, extranonce1: &[u8], extranonce2: &[u8]) -> Result<PreparedWork>;
pub fn verify_candidate(header80: &[u8]) -> [u8; 32]; // returns hash
```

### C) Core scheduling (in `nitrate-core`)

- Maintain **current job** atomically; swapping on new `notify`.
- For each GPU device:
  - Compute a nonce slice: `(start_nonce, nonce_count)`.
  - Build `KernelWork` using `PreparedWork` from `nitrate-btc`.
  - `backend.launch(device, work).await?`.
- Periodically `poll_results(device)` and on candidate:
  - Form the `Share` payload (job_id, extranonce2, ntime, nonce).
  - Run CPU verify; if OK and ≤ share target, call `pool.submit_share(share).await`.
- Cancel/restart:
  - On `clean_jobs` or new job id, signal cancellation by advancing a generation counter checked by polling loop.

### D) Real GPU backends (replace dummy)

Implement backends as **separate crates** behind features, each implementing `GpuBackend`:

- **CUDA** (`nitrate-gpu-cuda`):
  - Use `cust` or `rustacuda` for context, module, stream management.
  - Compile `.cu` with `build.rs` → embed PTX/CUBIN (`include_bytes!`).
  - Kernel input: midstate (const), tail (12B), start_nonce, nonce_count, be‑target[32].
  - Each thread scans a stride of nonces, writes out candidates to a ring buffer.
- **ROCm/HIP** (`nitrate-gpu-hip`):
  - Mirror CUDA API using HIP FFI, compile to HSACO.
- **OpenCL/Vulkan** (`nitrate-gpu-opencl` or `nitrate-gpu-vk`):
  - More portable; lower perf but good coverage.

**Kernel perf tips**:
- Place SHA constants in const memory.
- Unroll inner rounds; avoid spills.
- Use 32‑bit operations w/ byte‑swap intrinsics where available.
- Coalesce device writes for candidates; keep them rare (share target ≫ block target).

### E) Metrics (in `nitrate-metrics`)

Expose counters and gauges:
- `shares_accepted_total`, `shares_rejected_total`
- `hashrate_ghs` (update per device and sum)
- `pool_reconnects_total`, `stales_total`
- (optional) `job_latency_ms`, `submit_latency_ms`

Mount server:
```rust
let addr: SocketAddr = cfg.runtime.telemetry_addr.parse()?;
let _h = metrics.serve(addr).await?;
```

### F) Config (in `nitrate-config`)

`miner.toml` example:
```toml
[pool]
url = "stratum+tcp://solo.ckpool.org:3333"
user = "bc1q...worker1"
pass = "x"

[gpu]
backend = "cuda"   # cuda|hip|opencl|dummy
devices = [0, 1]
intensity = "auto"

[runtime]
reconnect_backoff = "1s..30s"
stale_drop_ms = 500
telemetry_addr = "0.0.0.0:9100"
log = "info"
```

---

## 4) Implementation checklist

- [ ] `nitrate-pool`: implement subscribe/authorize, read loop, JSON parse, notify → channel.
- [ ] `nitrate-btc`: build coinbase/merkle (feature‑gated), compute midstate, target, header tail.
- [ ] `nitrate-core`: scheduling, device loops, cancellation, CPU verify, submit share.
- [ ] GPU backend(s): CUDA/ROCm/OpenCL crates implementing `GpuBackend`.
- [ ] Metrics updates throughout; basic hashrate computation (shares & timing).
- [ ] CLI subcommands: `devices`, `selftest`, `run`.
- [ ] Integration test with a mock Stratum server (rotate diff, force cleans).
- [ ] Benchmarks for midstate/sha and kernel microbenchmarks.

---

## 5) Testing strategy

- **Unit tests**:
  - Proto parsing for `set_difficulty`, `notify`, and submit responses.
  - Target math and compact difficulty conversions.
  - Coinbase/merkle (if enabled) with known vectors.
- **Integration**:
  - Mock pool server streaming scripted `notify` frames; ensure stales drop promptly.
  - End-to-end share submit path calls and metrics increments.
- **Bench**:
  - `nitrate-bench`: CPU vs kernel, host/device copies (pinned vs pageable).
- **Determinism**:
  - Freeze time/nonce ranges for reproducible tests.

---

## 6) Ops: systemd & quadlet

Provide a service unit and Podman quadlet:
- Read‑only bind of config directory.
- Restart policy with jitter backoff.
- `Environment=RUST_LOG=info` and pass `--config /etc/nitrate/miner.toml`.

Expose `/metrics` on a dedicated loopback or LAN IP. Put it behind Prometheus with scrape interval 5–15s.

---

## 7) Security notes

- Never log pool passwords or private wallet data.
- Validate JSON inputs (bounds on extranonce sizes, branch counts).
- Harden TCP dial with timeouts, keepalive, and exponential backoff.
- Make submit path idempotent per share (avoid resubmission floods).

---

## 8) Performance notes

- Drop stale jobs fast: cancel kernels via generation counters and short polling cadence.
- Tune nonce chunk size to balance latency vs throughput.
- Use pinned host buffers; batch candidate reads.
- Consider NVML/rocm-smi hooks for thermal throttling.

---

## 9) Roadmap

1. Full Stratum v1 implementation + CPU-only verification path ✅
2. CUDA backend (PTX + cust) with basic tuning
3. ROCm/HIP backend
4. OpenCL fallback
5. Stratum v2 (behind feature flag)
6. Multi-pool failover and load balancing
7. Advanced telemetry (pprof, device temps, per‑GPU GH/s) and config reloading

---

## 10) Conventions & code style

- Rust 2021, `clippy` clean.
- Prefer `anyhow` at edges, `thiserror` for library boundaries.
- Use `tracing` spans for job IDs and device indexes.
- Keep public APIs small and testable; avoid leaking backend types across crates.

---

## 11) Quick start

```bash
# Dummy backend
cargo run -p nitrate-cli -- --config miner.toml --log info

# Once CUDA backend exists
cargo run -p nitrate-cli --features cuda -- --config miner.toml
```

If you’re adding a new backend, copy `nitrate-gpu-dummy`’s structure, implement `GpuBackend`, and wire feature flags in `nitrate-core` and `nitrate-cli`.
