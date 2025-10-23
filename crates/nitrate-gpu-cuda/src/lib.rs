#![doc = "CUDA backend scaffold for Nitrate GPU miner.\n\nThis crate provides a feature-gated CUDA backend implementation of the `GpuBackend`\ntrait. When built with the `cuda` feature, it initializes the CUDA driver using\n`cust` and performs basic device enumeration. `launch` and `poll_results` are\nimplemented as stubs for now, to be filled in with a real SHA256d midstate-based\nkernel and a device-side result ring.\n\nWhen the `cuda` feature is NOT enabled, the same `CudaBackend` type is exposed,\nbut all trait methods return clear errors to indicate the backend is unavailable.\nThis keeps the crate buildable in the workspace without requiring CUDA.\n"]
#[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
use anyhow::bail;
use anyhow::Result;
use async_trait::async_trait;
use nitrate_gpu_api::{DeviceInfo, FoundNonce, GpuBackend, KernelWork};
#[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
use tracing::debug;
#[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
use tracing::warn;
#[cfg(feature = "cuda")]
use tracing::{debug, info};
#[cfg(any(feature = "cuda", feature = "cuda-stub"))]
include!(concat!(env!("OUT_DIR"), "/kernel_ptx.rs"));

#[cfg(feature = "cuda")]
mod gpu_config;
#[cfg(feature = "cuda")]
use gpu_config::{GpuConfig, GpuDatabase};

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceCandidate {
    nonce: u32,
    hash_be: [u8; 32],
    generation: u64,
}

#[cfg(feature = "cuda")]
unsafe impl cust::memory::DeviceCopy for DeviceCandidate {}

/// CUDA backend for Nitrate.
///
/// - With `cuda` feature: performs CUDA driver init and basic device enumeration.
/// - Without `cuda` feature: compiles a stub that returns errors on use.
#[derive(Clone, Debug, Default)]
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    _marker: std::marker::PhantomData<()>,
    #[cfg(feature = "cuda")]
    last_candidates: std::sync::OnceLock<std::sync::Arc<std::sync::Mutex<Vec<FoundNonce>>>>,
    #[cfg(feature = "cuda")]
    gpu_configs: std::sync::Arc<std::sync::Mutex<Vec<GpuConfig>>>,
    #[cfg(feature = "cuda")]
    gpu_database: GpuDatabase,
}

#[async_trait]
impl GpuBackend for CudaBackend {
    #[cfg(feature = "cuda")]
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        use cust::prelude::*;

        // Initialize the CUDA driver (no-op if already initialized).
        cust::init(CudaFlags::empty())?;

        // Enumerate devices and configure each one
        let num = cust::device::Device::num_devices()?;
        let mut out = Vec::with_capacity(num as usize);
        let mut configs = Vec::with_capacity(num as usize);

        for ordinal in 0..num {
            let dev = cust::device::Device::get_device(ordinal)?;
            let name = dev.name()?;
            // total_memory() returns bytes; convert to MiB for display.
            let memory_mb = (dev.total_memory()? as u64) / (1024 * 1024);

            // Get optimal configuration for this GPU
            let config = self.gpu_database.get_config(&name);
            info!(
                "GPU {}: {} ({} MiB) - using grid={}, block={}, nonces_per_thread={}",
                ordinal,
                name,
                memory_mb,
                config.grid_size,
                config.block_size,
                config.nonces_per_thread
            );
            configs.push(config);

            out.push(DeviceInfo {
                index: ordinal,
                name,
                memory_mb,
            });
        }

        // Store configurations for later use
        if let Ok(mut guard) = self.gpu_configs.lock() {
            *guard = configs;
        }

        info!("CUDA enumeration found {} device(s)", out.len());
        Ok(out)
    }

    #[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        debug!("CudaBackend running in stub mode (cuda-stub feature)");
        // Return a fake GPU for testing
        Ok(vec![DeviceInfo {
            index: 0,
            name: "CUDA Stub GPU".into(),
            memory_mb: 8192,
        }])
    }

    #[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        warn!("CudaBackend requested but built without `cuda` or `cuda-stub` features");
        Ok(vec![])
    }

    #[cfg(feature = "cuda")]
    async fn launch(&self, device_index: u32, work: KernelWork) -> Result<()> {
        use cust::prelude::*;

        // Initialize driver and bind a context to the given device.
        cust::init(CudaFlags::empty())?;
        let device = cust::device::Device::get_device(device_index)?;
        // For now we create a short-lived context; future versions should manage a long-lived
        // context and stream(s) per device.
        let _ctx = cust::context::Context::new(device)?;

        // Get GPU configuration for this device
        let config = {
            let configs = self
                .gpu_configs
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock configs: {}", e))?;
            configs
                .get(device_index as usize)
                .cloned()
                .unwrap_or_else(|| {
                    debug!("No config for device {}, using defaults", device_index);
                    GpuConfig::default()
                })
        };

        // Load PTX for the "sha256d" module generated at build time and get the kernel.
        let ptx_bytes = nitrate_cuda_ptx::get_ptx_by_name("sha256d")
            .ok_or_else(|| anyhow::anyhow!("no PTX embedded for 'sha256d'"))?;
        let ptx_str =
            std::str::from_utf8(ptx_bytes).map_err(|_| anyhow::anyhow!("PTX not valid UTF-8"))?;
        let module = Module::from_ptx(ptx_str, &[])?;
        let func = module.get_function("sha256d_scan_kernel")?;

        // Set up stream and grid using GPU-specific configuration
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        let block = config.block_size;
        let grid = config.grid_size;

        // Upload parameters for mining kernel.
        let d_mid = DeviceBuffer::<u32>::from_slice(&work.midstate)?;
        let d_tail = DeviceBuffer::<u8>::from_slice(&work.header_tail)?;
        let d_target = DeviceBuffer::<u8>::from_slice(&work.target_be)?;

        // Candidate ring buffer and write index using configured size
        let ring_capacity = config.ring_capacity;
        let d_ring: DeviceBuffer<DeviceCandidate> =
            unsafe { DeviceBuffer::uninitialized(ring_capacity as usize)? };
        let d_write_idx = DeviceBuffer::<u32>::zeroed(1)?;

        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_mid.as_device_ptr(),
                d_tail.as_device_ptr(),
                d_target.as_device_ptr(),
                work.start_nonce,
                work.nonce_count,
                work.generation,
                ring_capacity,
                d_write_idx.as_device_ptr(),
                d_ring.as_device_ptr()
            ))?;
        }
        stream.synchronize()?;

        // Drain candidates found in this launch into a shared buffer for poll_results().
        let mut write_idx_host = [0u32; 1];
        d_write_idx.copy_to(&mut write_idx_host)?;
        let total = core::cmp::min(write_idx_host[0] as usize, ring_capacity as usize);
        if total > 0 {
            let mut host_ring = vec![
                DeviceCandidate {
                    nonce: 0,
                    hash_be: [0u8; 32],
                    generation: 0
                };
                total
            ];
            d_ring.copy_to(&mut host_ring[..])?;

            let found: Vec<FoundNonce> = host_ring
                .into_iter()
                .map(|c| FoundNonce {
                    nonce: c.nonce,
                    hash_be: c.hash_be,
                })
                .collect();

            let pool = self
                .last_candidates
                .get_or_init(|| std::sync::Arc::new(std::sync::Mutex::new(Vec::new())))
                .clone();
            {
                let mut guard = match pool.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                };
                guard.extend(found);
            }
        }

        debug!(
            device_index,
            generation = work.generation,
            start = work.start_nonce,
            count = work.nonce_count,
            grid = grid,
            block = block,
            "CUDA sha256d kernel launch completed with grid={} block={}",
            grid,
            block
        );

        Ok(())
    }

    #[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
    async fn launch(&self, device_index: u32, work: KernelWork) -> Result<()> {
        debug!(
            device_index,
            generation = work.generation,
            start = work.start_nonce,
            count = work.nonce_count,
            "CUDA stub launch (no-op)"
        );
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
    async fn launch(&self, device_index: u32, _work: KernelWork) -> Result<()> {
        let _ = device_index; // silence unused warning
        bail!("CudaBackend::launch called but crate built without `cuda` or `cuda-stub` features");
    }

    #[cfg(feature = "cuda")]
    async fn poll_results(&self, device_index: u32) -> Result<Vec<FoundNonce>> {
        let _ = device_index;
        // Drain any candidates captured during the last launch.
        if let Some(pool) = self.last_candidates.get() {
            let out = {
                let mut guard = match pool.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                };
                guard.drain(..).collect()
            };
            return Ok(out);
        }
        Ok(Vec::new())
    }

    #[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
    async fn poll_results(&self, _device_index: u32) -> Result<Vec<FoundNonce>> {
        // Stub returns no results
        Ok(vec![])
    }

    #[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
    async fn poll_results(&self, device_index: u32) -> Result<Vec<FoundNonce>> {
        let _ = device_index; // silence unused warning
        bail!("CudaBackend::poll_results called but crate built without `cuda` or `cuda-stub` features");
    }
}
