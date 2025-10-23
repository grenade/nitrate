# Nitrate Roadmap — From Scaffold to Working GPU Miner

This roadmap breaks the work into small, testable milestones. Each milestone has a **Definition of Done (DoD)** and a minimal **Acceptance Test (AT)** you can run locally.

> Target pool for end-to-end tests: `solo.ckpool.org` (Stratum v1).

---

## M0 — Scaffold ✅
**Done already.**
- Workspace + crates, dummy GPU backend, metrics skeleton, config loader, CLI.

---

## M1 — Stratum v1 Client (nitrate-pool)
**Goal:** Connect to pool, subscribe, authorize, parse messages, and emit jobs.

**Tasks**
- TCP client with line-delimited JSON-RPC framing (Tokio).
- `mining.subscribe` → parse `extranonce1`, `extranonce2_size`.
- `mining.authorize` (user/pass).
- Handle inbound: `mining.set_difficulty`, `mining.notify`.
- Provide `job_rx()` and `submit_share()` to core.
- Reconnect with exponential backoff + DNS re-resolve.

**DoD**: CLI logs notify/difficulty events without panics.  
**AT**:
```bash
RUST_LOG=nitrate_pool=debug cargo run -p nitrate-cli -- --config miner.toml
```

---

## M2 — Bitcoin Header & Midstate (nitrate-btc)
**Goal:** Build header/merkle (if needed), compute midstate/tail/targets, and CPU-verify candidates.

**Tasks**
- Compact difficulty (nBits) ↔ target math; compute **share target** from pool diff.
- Coinbase assembly (feature-gated; some pools provide ready parts).
- Merkle root from coinbase + branches.
- Prepare 80-byte header & compute **midstate (first 64B)**.
- Produce `PreparedWork { midstate, tail[12], target_be }`.
- `verify_candidate(header80)` (double-SHA256) vs target.

**DoD**: Unit tests with known vectors pass.  
**AT**: `cargo test -p nitrate-btc`

---

## M3 — Core Scheduling & Share Pipeline (nitrate-core)
**Goal:** Turn `Notify` into GPU work, handle cancellations, verify candidates, and submit shares.

**Tasks**
- Maintain current job (atomic generation). Cancel on `clean_jobs` or new job id.
- Nonce-slicing per device/stream and kernel launches.
- Poll results → CPU verify → `submit_share`.
- Metrics updates.

**DoD**: Core reacts to notify; metrics endpoint serves.  
**AT**:
```bash
cargo run -p nitrate-cli -- --config miner.toml
curl 127.0.0.1:9100/metrics
```

---

## M4 — CUDA Backend MVP (nitrate-gpu-cuda)
**Goal:** Real hashing on NVIDIA returning candidate nonces.

**Tasks**
- Crate with `cust` or `rustacuda` runtime.
- `build.rs` compiles `.cu` → PTX/CUBIN; embed.
- Implement `GpuBackend` (init, launch, poll).
- Kernel: SHA256d midstate-based nonce scan.
- Basic tuning params: grid/block, unroll, result ring.

**DoD**: Shares produced on low diff settings or synthetic notify.  
**AT**:
```bash
cargo run -p nitrate-cli --features cuda -- --config miner.toml
```

---

## M5 — Stability & Observability
- Stale handling on `clean_jobs=true`.
- Reconnect backoff with jitter; `pool_reconnects_total`.
- Hashrate calc per device.
- Structured logs with job/device spans.

**DoD**: 24h soak, stable RSS.  

---

## M6 — HIP/ROCm Backend (nitrate-gpu-hip)
- HIP runtime wrapper; HSACO build via `hipcc`.
- Mirror CUDA trait impl.

**DoD**: Functional hashing on AMD.  

---

## M7 — OpenCL Fallback (optional)
- Runtime compile or prebuilt binaries per vendor.
- Endian/bit-swap correctness.

**DoD**: Functional on at least two vendors.  

---

## M8 — Pool Compatibility & Failover
- Parameter variations and quirks.
- Failover list, health checks, priorities.

**DoD**: Seamless switch when primary is down.  

---

## M9 — Performance Tuning
- Occupancy tuning; minimize spills.
- Coalesced memory; batch result reads.
- Pinned buffers; cadence tuning.

**DoD**: Achieve target GH/s per GPU class.  

---

## M10 — Packaging & CI
- GitHub Actions with tests; CUDA/HIP matrix where possible.
- Release artifacts for `nitrate-cli`.
- Systemd unit + Podman quadlet in `deploy/`.

**DoD**: Download-and-run release works.  

---

## M11 — Security Hardening
- Redact secrets; input bounds checks.
- Idempotent share submissions; rate limiting.

**DoD**: Static analysis + fuzz basic parsers.  

---

## M12 — Stratum v2 (feature-flag)
- Add SV2 protocol support behind a feature flag.

**DoD**: Connects to an SV2 test server and mines.

---

## Conventions
- Labels: `M1-Pool`..`M12-SV2`
- Branching: `feat/nitrate/m0x/<topic>`
- PR gate: `cargo fmt && cargo clippy -- -D warnings && cargo test`
