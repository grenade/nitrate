use anyhow::Result;
use nitrate_config::AppCfg;
use nitrate_gpu_api::GpuBackend;
use nitrate_metrics::Metrics;
use nitrate_pool::{PoolConfig, StratumClient};
use std::net::SocketAddr;
use tokio::signal;
use tracing::{info, warn};

#[cfg(feature = "gpu-cuda")]
use nitrate_gpu_cuda::CudaBackend as SelectedBackend;
#[cfg(all(feature = "gpu-dummy", not(feature = "gpu-cuda")))]
use nitrate_gpu_dummy::DummyBackend as SelectedBackend;

pub struct Engine<B: GpuBackend + Default> {
    cfg: AppCfg,
    pool: StratumClient,
    backend: B,
    metrics: Metrics,
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
            if let Some(_job) = self.pool.next_job().await? {
                // TODO: transform into KernelWork and launch on devices
                // self.backend.launch(0, work).await?;
            }
            // Poll for results from each device in a real implementation.
        }
    }
}

// Type alias handy for caller
pub type DefaultEngine = Engine<SelectedBackend>;
