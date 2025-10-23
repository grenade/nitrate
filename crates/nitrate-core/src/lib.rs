use anyhow::Result;
use nitrate_config::AppCfg;
use nitrate_gpu_api::{GpuBackend, KernelWork};
use nitrate_metrics::Metrics;
use nitrate_pool::{PoolConfig, StratumClient};
use std::net::SocketAddr;
use tokio::{
    signal,
    time::{sleep, Duration},
};
use tracing::{info, warn};

#[cfg(feature = "gpu-cuda")]
use nitrate_gpu_cuda::CudaBackend as SelectedBackend;
#[cfg(all(feature = "gpu-dummy", not(feature = "gpu-cuda")))]
use nitrate_gpu_dummy::DummyBackend as SelectedBackend;

pub struct Engine<B: GpuBackend + Default> {
    cfg: AppCfg,
    #[allow(dead_code)]
    pool: StratumClient,
    backend: B,
    metrics: Metrics,
    generation: u64,
}

impl<B: GpuBackend + Default> Engine<B> {
    pub async fn new(cfg: AppCfg) -> Result<Self> {
        let pool = StratumClient::connect(PoolConfig {
            url: cfg.pool.url.clone(),
            user: cfg.pool.user.clone(),
            pass: cfg.pool.pass.clone(),
            tls: cfg.pool.tls,
            tls_insecure: cfg.pool.tls_insecure,
        })
        .await?;

        let metrics = Metrics::new();
        Ok(Self {
            cfg,
            pool,
            backend: B::default(),
            metrics,
            generation: 0,
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
            // Stub work: launch a scan on device 0 with placeholder params.
            let work = KernelWork {
                generation: self.generation,
                start_nonce: 0,
                nonce_count: 1_000_000, // scan 1M nonces per loop
                target_be: [0u8; 32],
                header_tail: [0u8; 16],
                midstate: [0u32; 8],
            };
            self.backend.launch(0, work).await?;

            // Poll and log any candidates
            let results = self.backend.poll_results(0).await?;
            if !results.is_empty() {
                for cand in results {
                    info!(
                        "candidate found: device=0 nonce={} hash={:?}",
                        cand.nonce, cand.hash_be
                    );
                }
            }

            // Advance generation for next loop and avoid busy-spin
            self.generation = self.generation.wrapping_add(1);
            sleep(Duration::from_millis(200)).await;
        }
    }
}

// Type alias handy for caller
pub type DefaultEngine = Engine<SelectedBackend>;
