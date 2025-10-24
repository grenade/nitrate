#![doc = "CUDA backend scaffold for Nitrate GPU miner.\n\nThis crate provides a feature-gated CUDA backend implementation of the `GpuBackend`\ntrait. When built with the `cuda` feature, it initializes the CUDA driver using\n`cust` and performs basic device enumeration. `launch` and `poll_results` are\nimplemented as stubs for now, to be filled in with a real SHA256d midstate-based\nkernel and a device-side result ring.\n\nWhen the `cuda` feature is NOT enabled, the same `CudaBackend` type is exposed,\nbut all trait methods return clear errors to indicate the backend is unavailable.\nThis keeps the crate buildable in the workspace without requiring CUDA.\n"]
#[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
use anyhow::bail;
use anyhow::Result;
use async_trait::async_trait;
use nitrate_config::DeviceOverride;
use nitrate_gpu_api::{ConfigurableGpuBackend, DeviceInfo, FoundNonce, GpuBackend, KernelWork};
#[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
use tracing::debug;
#[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
use tracing::warn;
#[cfg(feature = "cuda")]
use tracing::{debug, info, warn};
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
#[derive(Clone, Debug)]
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

impl CudaBackend {
    #[cfg(feature = "cuda")]
    pub fn new(device_overrides: Vec<DeviceOverride>) -> Self {
        let backend = Self {
            _marker: std::marker::PhantomData,
            last_candidates: std::sync::OnceLock::new(),
            gpu_configs: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            gpu_database: GpuDatabase::new(),
        };

        // Store device overrides for later application during enumerate()
        if !device_overrides.is_empty() {
            let mut configs = std::collections::HashMap::new();
            for override_cfg in device_overrides {
                let gpu_config = GpuConfig {
                    grid_size: override_cfg.grid_size,
                    block_size: override_cfg.block_size,
                    nonces_per_thread: override_cfg.nonces_per_thread,
                    ring_capacity: override_cfg.ring_capacity,
                };
                configs.insert(override_cfg.device_index, gpu_config);
            }
            // Convert HashMap to Vec for storage, we'll apply during enumerate()
            if let Ok(mut guard) = backend.gpu_configs.lock() {
                // We'll store the overrides as a marker - actual application happens in enumerate()
                guard.clear();
                for (device_idx, config) in configs {
                    // Pad vector to accommodate this device index
                    while guard.len() <= device_idx as usize {
                        guard.push(GpuConfig::default());
                    }
                    guard[device_idx as usize] = config;
                }
            }
        }
        backend
    }

    #[cfg(all(feature = "cuda-stub", not(feature = "cuda")))]
    pub fn new(_device_overrides: Vec<DeviceOverride>) -> Self {
        Self::default()
    }

    #[cfg(not(any(feature = "cuda", feature = "cuda-stub")))]
    pub fn new(_device_overrides: Vec<DeviceOverride>) -> Self {
        Self::default()
    }
}

#[allow(clippy::derivable_impls)]
impl Default for CudaBackend {
    fn default() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            _marker: std::marker::PhantomData,
            #[cfg(feature = "cuda")]
            last_candidates: std::sync::OnceLock::new(),
            #[cfg(feature = "cuda")]
            gpu_configs: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            #[cfg(feature = "cuda")]
            gpu_database: GpuDatabase::new(),
        }
    }
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

            // Query device properties to understand resource limits
            let max_threads_per_block =
                dev.get_attribute(cust::device::DeviceAttribute::MaxThreadsPerBlock)? as u32;
            let max_threads_per_sm = dev
                .get_attribute(cust::device::DeviceAttribute::MaxThreadsPerMultiprocessor)?
                as u32;
            // MaxBlocksPerMultiprocessor is not available in cust, use a reasonable default
            let max_blocks_per_sm = 32u32; // Conservative default for modern GPUs
            let sm_count =
                dev.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as u32;
            let max_shared_mem_per_block =
                dev.get_attribute(cust::device::DeviceAttribute::MaxSharedMemoryPerBlock)? as u32;
            let max_registers_per_block =
                dev.get_attribute(cust::device::DeviceAttribute::MaxRegistersPerBlock)? as u32;

            debug!(
                "GPU {} device limits: max_threads_per_block={}, max_threads_per_sm={}, sm_count={}, max_shared_mem_per_block={}, max_registers_per_block={}",
                ordinal,
                max_threads_per_block,
                max_threads_per_sm,
                sm_count,
                max_shared_mem_per_block,
                max_registers_per_block
            );

            // Get configuration - check if we have device overrides first
            let config = {
                let configs = self
                    .gpu_configs
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Failed to lock configs: {}", e))?;

                if !configs.is_empty() && (ordinal as usize) < configs.len() {
                    let override_config = &configs[ordinal as usize];
                    // Check if this is a meaningful override (not just defaults)
                    let default_config = GpuConfig::default();
                    if override_config.grid_size != default_config.grid_size
                        || override_config.block_size != default_config.block_size
                        || override_config.nonces_per_thread != default_config.nonces_per_thread
                        || override_config.ring_capacity != default_config.ring_capacity
                    {
                        info!(
                            "GPU {}: Using device override configuration: grid={}, block={}, nonces_per_thread={}",
                            ordinal,
                            override_config.grid_size,
                            override_config.block_size,
                            override_config.nonces_per_thread
                        );
                        override_config.clone()
                    } else {
                        // No meaningful override, use database config
                        self.gpu_database.get_config(&name)
                    }
                } else {
                    // No override for this device, use database config
                    self.gpu_database.get_config(&name)
                }
            };

            // Validate configuration against device limits
            if config.block_size > max_threads_per_block {
                warn!(
                    "GPU {}: Configured block_size {} exceeds max_threads_per_block {}! Using {}",
                    ordinal, config.block_size, max_threads_per_block, max_threads_per_block
                );
            }

            let blocks_per_sm = config.grid_size.div_ceil(sm_count);
            if blocks_per_sm > max_blocks_per_sm {
                warn!(
                    "GPU {}: Grid size {} results in {} blocks per SM, exceeding limit of {}!",
                    ordinal, config.grid_size, blocks_per_sm, max_blocks_per_sm
                );
            }

            info!(
                "GPU {}: {} ({} MiB) - using grid={}, block={}, nonces_per_thread={}, ring_capacity={}",
                ordinal,
                name,
                memory_mb,
                config.grid_size,
                config.block_size,
                config.nonces_per_thread,
                config.ring_capacity
            );

            // Log expected performance based on configuration
            let total_threads = config.grid_size * config.block_size;
            let work_per_kernel = total_threads * config.nonces_per_thread;
            info!(
                "  -> Total threads: {}, Work per kernel: {} ({} million nonces)",
                total_threads,
                work_per_kernel,
                work_per_kernel / 1_000_000
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
                    warn!("No config for device {} - using defaults! This will severely impact performance!", device_index);
                    GpuConfig::default()
                })
        };

        debug!(
            "GPU {} using config: grid={}, block={}, nonces_per_thread={}, ring_capacity={}",
            device_index,
            config.grid_size,
            config.block_size,
            config.nonces_per_thread,
            config.ring_capacity
        );

        // Load fatbin for the "sha256d" module generated at build time and get the kernel.
        // The fatbin contains compiled code for multiple GPU architectures
        let fatbin_bytes = nitrate_cuda_ptx::get_ptx_by_name("sha256d")
            .ok_or_else(|| anyhow::anyhow!("no fatbin embedded for 'sha256d'"))?;
        // Load the fatbin directly - CUDA runtime will select the best architecture
        let module = Module::from_fatbin(fatbin_bytes, &[])?;
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
                ring_capacity as usize
            ];
            // Copy the full ring buffer from device to host
            d_ring.copy_to(&mut host_ring[..])?;

            let found: Vec<FoundNonce> = host_ring
                .into_iter()
                .take(total)
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

impl ConfigurableGpuBackend for CudaBackend {
    fn new_with_config(device_overrides: Vec<DeviceOverride>) -> Self {
        Self::new(device_overrides)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nitrate_config::DeviceOverride;

    #[test]
    fn test_cuda_backend_default() {
        let _backend = CudaBackend::default();
        // Should create backend with empty configs
        #[cfg(feature = "cuda")]
        {
            let configs = _backend.gpu_configs.lock().unwrap();
            assert!(configs.is_empty());
        }
    }

    #[test]
    fn test_cuda_backend_with_device_overrides() {
        let overrides = vec![
            DeviceOverride {
                device_index: 0,
                grid_size: 1360,
                block_size: 256,
                nonces_per_thread: 1,
                ring_capacity: 32768,
                use_shared_memory: true,
            },
            DeviceOverride {
                device_index: 1,
                grid_size: 680,
                block_size: 512,
                nonces_per_thread: 2,
                ring_capacity: 16384,
                use_shared_memory: false,
            },
        ];

        let _backend = CudaBackend::new(overrides.clone());

        #[cfg(feature = "cuda")]
        {
            let configs = _backend.gpu_configs.lock().unwrap();
            assert_eq!(configs.len(), 2);

            // Check device 0 config
            assert_eq!(configs[0].grid_size, 1360);
            assert_eq!(configs[0].block_size, 256);
            assert_eq!(configs[0].nonces_per_thread, 1);
            assert_eq!(configs[0].ring_capacity, 32768);

            // Check device 1 config
            assert_eq!(configs[1].grid_size, 680);
            assert_eq!(configs[1].block_size, 512);
            assert_eq!(configs[1].nonces_per_thread, 2);
            assert_eq!(configs[1].ring_capacity, 16384);
        }
    }

    #[test]
    fn test_configurable_gpu_backend_trait() {
        let overrides = vec![DeviceOverride {
            device_index: 0,
            grid_size: 2720,
            block_size: 512,
            nonces_per_thread: 4,
            ring_capacity: 32768,
            use_shared_memory: true,
        }];

        let _backend = CudaBackend::new_with_config(overrides);

        #[cfg(feature = "cuda")]
        {
            let configs = _backend.gpu_configs.lock().unwrap();
            assert_eq!(configs.len(), 1);
            assert_eq!(configs[0].grid_size, 2720);
            assert_eq!(configs[0].block_size, 512);
            assert_eq!(configs[0].nonces_per_thread, 4);
            assert_eq!(configs[0].ring_capacity, 32768);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_database_rtx5090_config() {
        let db = GpuDatabase::new();
        let config = db.get_config("NVIDIA GeForce RTX 5090");

        // Verify the default RTX 5090 config that was causing issues
        assert_eq!(config.grid_size, 2720);
        assert_eq!(config.block_size, 512);
        assert_eq!(config.nonces_per_thread, 4);
        assert_eq!(config.ring_capacity, 32768);
    }

    #[test]
    fn test_device_override_sparse_indices() {
        // Test that we can override device 2 without overriding devices 0 and 1
        let overrides = vec![DeviceOverride {
            device_index: 2,
            grid_size: 1000,
            block_size: 128,
            nonces_per_thread: 8,
            ring_capacity: 4096,
            use_shared_memory: false,
        }];

        let _backend = CudaBackend::new(overrides);

        #[cfg(feature = "cuda")]
        {
            let configs = _backend.gpu_configs.lock().unwrap();
            assert_eq!(configs.len(), 3); // Should pad to include device 2

            // Devices 0 and 1 should have defaults
            assert_eq!(configs[0].grid_size, GpuConfig::default().grid_size);
            assert_eq!(configs[1].grid_size, GpuConfig::default().grid_size);

            // Device 2 should have the override
            assert_eq!(configs[2].grid_size, 1000);
            assert_eq!(configs[2].block_size, 128);
            assert_eq!(configs[2].nonces_per_thread, 8);
            assert_eq!(configs[2].ring_capacity, 4096);
        }
    }
}
