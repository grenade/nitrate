#![doc = "CUDA backend scaffold for Nitrate GPU miner.\n\nThis crate provides a feature-gated CUDA backend implementation of the `GpuBackend`\ntrait. When built with the `cuda` feature, it initializes the CUDA driver using\n`cust` and performs basic device enumeration. `launch` and `poll_results` are\nimplemented as stubs for now, to be filled in with a real SHA256d midstate-based\nkernel and a device-side result ring.\n\nWhen the `cuda` feature is NOT enabled, the same `CudaBackend` type is exposed,\nbut all trait methods return clear errors to indicate the backend is unavailable.\nThis keeps the crate buildable in the workspace without requiring CUDA.\n"]
use anyhow::{bail, Result};
use async_trait::async_trait;
use nitrate_gpu_api::{DeviceInfo, FoundNonce, GpuBackend, KernelWork};
use tracing::warn;
#[cfg(feature = "cuda")]
use tracing::{debug, info};

/// CUDA backend for Nitrate.
///
/// - With `cuda` feature: performs CUDA driver init and basic device enumeration.
/// - Without `cuda` feature: compiles a stub that returns errors on use.
#[derive(Clone, Default, Debug)]
pub struct CudaBackend {
    #[cfg(feature = "cuda")]
    _marker: std::marker::PhantomData<()>,
}

#[async_trait]
impl GpuBackend for CudaBackend {
    #[cfg(feature = "cuda")]
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        use cust::prelude::*;

        // Initialize the CUDA driver (no-op if already initialized).
        cust::init(CudaFlags::empty())?;

        // Enumerate devices.
        // Note: These API names are based on cust >= 0.8. Adjust if needed when wiring the full backend.
        let num = cust::device::Device::num_devices()?;
        let mut out = Vec::with_capacity(num as usize);
        for ordinal in 0..num {
            let dev = cust::device::Device::get_device(ordinal as u32)?;
            let name = dev.name()?;
            // total_memory() returns bytes; convert to MiB for display.
            let memory_mb = (dev.total_memory()? as u64) / (1024 * 1024);
            out.push(DeviceInfo {
                index: ordinal as u32,
                name,
                memory_mb,
            });
        }
        info!("CUDA enumeration found {} device(s)", out.len());
        Ok(out)
    }

    #[cfg(not(feature = "cuda"))]
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        warn!("CudaBackend requested but built without `cuda` feature");
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
        let _ctx = cust::context::Context::create_and_push(
            cust::context::ContextFlags::MAP_HOST | cust::context::ContextFlags::SCHED_AUTO,
            device,
        )?;

        debug!(
            device_index,
            generation = work.generation,
            start = work.start_nonce,
            count = work.nonce_count,
            "CUDA launch stub invoked"
        );

        // TODO: Upload parameters (midstate, header_tail[16], target_be[32]), set up a
        // device-side result ring buffer, and launch the SHA256d midstate kernel here.
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    async fn launch(&self, device_index: u32, _work: KernelWork) -> Result<()> {
        let _ = device_index; // silence unused warning
        bail!("CudaBackend::launch called but crate built without `cuda` feature");
    }

    #[cfg(feature = "cuda")]
    async fn poll_results(&self, device_index: u32) -> Result<Vec<FoundNonce>> {
        let _ = device_index;
        // TODO: Read and drain the device-side result ring. For now, return no results.
        Ok(Vec::new())
    }

    #[cfg(not(feature = "cuda"))]
    async fn poll_results(&self, device_index: u32) -> Result<Vec<FoundNonce>> {
        let _ = device_index; // silence unused warning
        bail!("CudaBackend::poll_results called but crate built without `cuda` feature");
    }
}
