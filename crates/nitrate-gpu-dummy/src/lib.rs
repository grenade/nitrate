use anyhow::Result;
use async_trait::async_trait;
use nitrate_gpu_api::{DeviceInfo, FoundNonce, GpuBackend, KernelWork};
use tokio::time::{sleep, Duration};
use tracing::debug;

#[derive(Clone, Default)]
pub struct DummyBackend;

#[async_trait]
impl GpuBackend for DummyBackend {
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        Ok(vec![DeviceInfo { index: 0, name: "DummyGPU".into(), memory_mb: 64 }])
    }
    async fn launch(&self, device_index: u32, work: KernelWork) -> Result<()> {
        debug!(?device_index, ?work, "Dummy launch");
        Ok(())
    }
    async fn poll_results(&self, _device_index: u32) -> Result<Vec<FoundNonce>> {
        sleep(Duration::from_millis(500)).await;
        Ok(vec![]) // never finds anything
    }
}