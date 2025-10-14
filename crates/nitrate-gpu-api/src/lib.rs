use anyhow::Result;
use async_trait::async_trait;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
    pub memory_mb: u64,
}

#[derive(Clone, Debug)]
pub struct KernelWork {
    pub start_nonce: u32,
    pub nonce_count: u32,
    pub target_be: [u8; 32],
    pub header_tail: [u8; 12], // time, bits, nonce slot
    pub midstate: [u32; 8],
}

#[derive(Clone, Debug)]
pub struct FoundNonce {
    pub nonce: u32,
    pub hash_be: [u8; 32],
}

#[async_trait]
pub trait GpuBackend: Send + Sync {
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>>;
    async fn launch(&self, _device_index: u32, _work: KernelWork) -> Result<()>;
    async fn poll_results(&self, _device_index: u32) -> Result<Vec<FoundNonce>>;
}