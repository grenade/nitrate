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
    /// Monotonic job generation assigned by the core. Used for cancellation:
    /// backends must ensure that results from older generations are dropped
    /// and never surfaced via `poll_results`.
    pub generation: u64,

    /// Inclusive starting nonce for this slice.
    pub start_nonce: u32,

    /// Total nonces to scan in this slice.
    pub nonce_count: u32,

    /// Big-endian 32-byte share target.
    /// Compare directly against `hash_be` using lexicographic comparison (hash <= target).
    pub target_be: [u8; 32],

    /// Raw 12 bytes from the 80-byte block header:
    /// ntime (u32, little-endian) | nbits (u32, little-endian) | nonce placeholder (u32, little-endian 0).
    pub header_tail: [u8; 12],

    /// SHA-256 midstate after hashing header bytes 0..63.
    /// Represented as 8 big-endian u32 words per the SHA-256 specification.
    pub midstate: [u32; 8],
}

#[derive(Clone, Debug)]
pub struct FoundNonce {
    /// Nonce that produced the candidate.
    pub nonce: u32,

    /// Big-endian 32-byte double-SHA256(header) for this nonce.
    /// This is directly comparable to `target_be` (hash <= target).
    pub hash_be: [u8; 32],
}

/// Cancellation semantics:
/// - The core supplies a monotonically increasing `generation` via `KernelWork`.
/// - Backends must ensure `poll_results` only returns candidates from the latest
///   launched `generation` for a given device. Results from older generations must
///   be dropped internally to prevent stale submissions.
///
/// Endianness conventions:
/// - `target_be`: 32-byte big-endian target; compare directly with `hash_be`.
/// - `hash_be`: 32-byte big-endian double-SHA256(header).
/// - `header_tail`: raw 12 bytes from the 80-byte header:
///   ntime (u32 LE) | nbits (u32 LE) | nonce placeholder (u32 LE = 0).
/// - `midstate`: SHA-256 state after bytes 0..63 of the header, as 8 big-endian u32 words.
/// - `start_nonce` is inclusive; scan at most `nonce_count` nonces per launch.
#[async_trait]
pub trait GpuBackend: Send + Sync {
    async fn enumerate(&self) -> Result<Vec<DeviceInfo>>;
    async fn launch(&self, _device_index: u32, _work: KernelWork) -> Result<()>;
    async fn poll_results(&self, _device_index: u32) -> Result<Vec<FoundNonce>>;
}
