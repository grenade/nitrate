use anyhow::Result;
use nitrate_btc::{double_sha256_via_midstate, prepare_from_notify_parts, NotifyParams};
use nitrate_config::AppCfg;
use nitrate_gpu_api::{GpuBackend, KernelWork};
use nitrate_metrics::Metrics;
use nitrate_pool::{PoolConfig, Share, StratumClient, StratumJob};
use std::net::SocketAddr;
use tokio::{
    signal,
    sync::mpsc,
    time::{sleep, Duration},
};
use tracing::{debug, info, warn};

#[cfg(any(feature = "gpu-cuda", feature = "gpu-cuda-stub"))]
use nitrate_gpu_cuda::CudaBackend as SelectedBackend;
#[cfg(all(
    feature = "gpu-dummy",
    not(feature = "gpu-cuda"),
    not(feature = "gpu-cuda-stub")
))]
use nitrate_gpu_dummy::DummyBackend as SelectedBackend;

pub struct Engine<B: GpuBackend + Default> {
    cfg: AppCfg,
    pool: StratumClient,
    backend: B,
    metrics: Metrics,
    generation: u64,
    job_rx: Option<mpsc::UnboundedReceiver<StratumJob>>,
    current_job: Option<StratumJob>,
    extranonce2_counter: u32,
    last_hashrate_update: std::time::Instant,
    total_hashes: u64,
}

impl<B: GpuBackend + Default> Engine<B> {
    pub async fn new(cfg: AppCfg) -> Result<Self> {
        let mut pool = StratumClient::connect(PoolConfig {
            url: cfg.pool.url.clone(),
            user: cfg.pool.user.clone(),
            pass: cfg.pool.pass.clone(),
            tls: cfg.pool.tls,
            tls_insecure: cfg.pool.tls_insecure,
        })
        .await?;

        let job_rx = pool.job_rx();
        let metrics = Metrics::new();
        Ok(Self {
            cfg,
            pool,
            backend: B::default(),
            metrics,
            generation: 0,
            job_rx,
            current_job: None,
            extranonce2_counter: 0,
            last_hashrate_update: std::time::Instant::now(),
            total_hashes: 0,
        })
    }

    pub async fn run(mut self) -> Result<()> {
        // Metrics server
        let addr: SocketAddr = self.cfg.runtime.telemetry_addr.parse()?;
        let _handle = self.metrics.serve(addr).await?;

        // Device enumeration
        let devices = self.backend.enumerate().await?;
        if devices.is_empty() {
            warn!("no GPU devices found");
        } else {
            for d in &devices {
                info!("GPU {}: {} ({} MiB)", d.index, d.name, d.memory_mb);
            }
        }

        info!("engine running. Ctrl-C to stop.");
        tokio::select! {
            _ = self.main_loop() => {},
            _ = signal::ctrl_c() => {
                info!("Ctrl-C received, shutting down");
            }
        }
        Ok(())
    }

    async fn main_loop(&mut self) -> Result<()> {
        loop {
            // Check for new jobs
            if let Some(ref mut rx) = self.job_rx {
                while let Ok(job) = rx.try_recv() {
                    info!(
                        "received job {} (diff={}, clean={})",
                        job.job_id, job.difficulty, job.notify.clean_jobs
                    );
                    self.metrics.jobs_received.inc();

                    if job.notify.clean_jobs {
                        self.generation = self.generation.wrapping_add(1);
                    }
                    self.current_job = Some(job);
                }
            }

            // Process current job if we have one
            if let Some(job) = self.current_job.clone() {
                // Generate extranonce2 for this work unit
                let extranonce2 = self.next_extranonce2(job.extranonce2_size);

                // Prepare work from Stratum notify
                let prepared = match prepare_from_notify_parts(NotifyParams {
                    version_hex_le: &job.notify.version,
                    prevhash_hex_le: &job.notify.prevhash,
                    coinbase1_hex: &job.notify.coinbase1,
                    coinbase2_hex: &job.notify.coinbase2,
                    merkle_branch_hex: &job.notify.merkle_branch,
                    ntime_hex_le: &job.notify.ntime,
                    nbits_hex_le: &job.notify.nbits,
                    extranonce1: &job.extranonce1,
                    extranonce2: &extranonce2,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        warn!("failed to prepare work: {}", e);
                        sleep(Duration::from_millis(100)).await;
                        continue;
                    }
                };

                // Calculate share target from difficulty (use test difficulty if in test mode)
                let effective_difficulty = if self.cfg.runtime.test_mode {
                    info!(
                        "TEST MODE: using difficulty {}",
                        self.cfg.runtime.test_difficulty
                    );
                    self.cfg.runtime.test_difficulty
                } else {
                    job.difficulty
                };
                let share_target = self.difficulty_to_target(effective_difficulty);

                // Create GPU work
                // Use much larger nonce range for better GPU utilization
                // RTX 5090s need massive work sizes to stay busy
                let total_nonces = 4_000_000_000u32; // 4 billion nonces (near u32 max)
                let num_devices = self.cfg.gpu.devices.len() as u32;
                let nonces_per_device = total_nonces / num_devices.max(1);

                debug!(
                    "Distributing {} nonces total, {} per device across {} GPUs",
                    total_nonces, nonces_per_device, num_devices
                );

                // Launch work on all configured devices
                for (idx, &device_id) in self.cfg.gpu.devices.iter().enumerate() {
                    let work = KernelWork {
                        generation: self.generation,
                        start_nonce: (idx as u32) * nonces_per_device,
                        nonce_count: nonces_per_device,
                        target_be: share_target,
                        header_tail: prepared.tail16,
                        midstate: prepared.midstate,
                    };

                    info!(
                        "Launching work on GPU {}: start_nonce={}, count={} ({}M)",
                        device_id,
                        work.start_nonce,
                        work.nonce_count,
                        work.nonce_count / 1_000_000
                    );

                    // Launch on this device
                    if let Err(e) = self.backend.launch(device_id, work.clone()).await {
                        warn!("launch error on device {}: {}", device_id, e);
                    } else {
                        debug!("Successfully launched work on GPU {}", device_id);
                        self.metrics.gpu_launches.inc();
                    }

                    // Track hashes for hashrate calculation
                    self.total_hashes += work.nonce_count as u64;
                }

                // Update hashrate every second
                let now = std::time::Instant::now();
                let elapsed = now.duration_since(self.last_hashrate_update).as_secs_f64();
                if elapsed >= 1.0 {
                    let hashrate_hs = (self.total_hashes as f64) / elapsed;
                    let hashrate_ghs = hashrate_hs / 1_000_000_000.0;
                    self.metrics.hashrate_gps.set(hashrate_ghs);
                    info!(
                        "hashrate: {:.2} GH/s (processed {} hashes across {} GPUs)",
                        hashrate_ghs,
                        self.total_hashes,
                        self.cfg.gpu.devices.len()
                    );
                    self.last_hashrate_update = now;
                    self.total_hashes = 0;
                }

                // Poll for results from all devices
                for &device_id in &self.cfg.gpu.devices {
                    let results = match self.backend.poll_results(device_id).await {
                        Ok(r) => r,
                        Err(e) => {
                            debug!("Failed to poll device {}: {}", device_id, e);
                            continue;
                        }
                    };

                    if !results.is_empty() {
                        debug!("GPU {} returned {} candidates", device_id, results.len());
                    }

                    self.metrics.candidates_found.inc_by(results.len() as u64);
                    for candidate in results {
                        // Build full header with found nonce
                        let mut header80 = prepared.header80;
                        header80[76..80].copy_from_slice(&candidate.nonce.to_le_bytes());

                        // Verify on CPU
                        let hash = double_sha256_via_midstate(&header80);

                        // Check against share target
                        if self.hash_meets_target(&hash, &share_target) {
                            info!("share found! nonce={:08x} hash={:?}", candidate.nonce, hash);

                            // Extract ntime from header
                            let ntime = u32::from_le_bytes([
                                header80[68],
                                header80[69],
                                header80[70],
                                header80[71],
                            ]);

                            // Submit share
                            let share = Share {
                                job_id: job.job_id.clone(),
                                extranonce2: extranonce2.clone(),
                                ntime,
                                nonce: candidate.nonce,
                            };

                            if let Err(e) = self.pool.submit_share(share).await {
                                warn!("failed to submit share: {}", e);
                            } else {
                                self.metrics.shares_ok.inc();
                            }
                        }
                    }
                }
            }

            // Brief pause before next work unit
            // Very short pause for high-end GPUs to keep them fed with work
            sleep(Duration::from_millis(1)).await;
        }
    }

    fn next_extranonce2(&mut self, size: usize) -> Vec<u8> {
        self.extranonce2_counter = self.extranonce2_counter.wrapping_add(1);
        let mut e2 = vec![0u8; size];
        let bytes = self.extranonce2_counter.to_le_bytes();
        let copy_len = size.min(bytes.len());
        e2[..copy_len].copy_from_slice(&bytes[..copy_len]);
        e2
    }

    fn difficulty_to_target(&self, _difficulty: f64) -> [u8; 32] {
        // Convert pool difficulty to 256-bit target
        // diff 1 = 0x00000000ffff0000000000000000000000000000000000000000000000000000
        // For simplicity, use diff1 target for now
        // Real implementation would divide by difficulty
        [
            0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ]
    }

    fn hash_meets_target(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
        for i in 0..32 {
            if hash[i] < target[i] {
                return true;
            }
            if hash[i] > target[i] {
                return false;
            }
        }
        true
    }
}

// Type alias handy for caller
pub type DefaultEngine = Engine<SelectedBackend>;
