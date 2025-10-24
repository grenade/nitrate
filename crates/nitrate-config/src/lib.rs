use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config error: {0}")]
    Generic(#[from] anyhow::Error),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolCfg {
    // Pool URL supports "stratum+tcp://" and "stratum+ssl://".
    // If scheme is omitted, the pool client will use `tls` to decide transport.
    pub url: String,  // e.g., stratum+tcp://solo.ckpool.org:3333
    pub user: String, // wallet or wallet.worker
    #[serde(default = "default_password")]
    pub pass: String,
    #[serde(default)]
    pub tls: bool, // true to use TLS (stratum+ssl) when no scheme is provided
    #[serde(default)]
    pub tls_insecure: bool, // disable certificate verification (testing only)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceOverride {
    pub device_index: u32,
    pub grid_size: u32,
    pub block_size: u32,
    pub nonces_per_thread: u32,
    #[serde(default = "default_ring_capacity")]
    pub ring_capacity: u32,
    #[serde(default)]
    pub use_shared_memory: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuCfg {
    #[serde(default = "default_backend")]
    pub backend: String, // "dummy" | "cuda" | "hip" | "opencl"
    #[serde(default)]
    pub devices: Vec<u32>, // device indexes
    #[serde(default = "default_intensity")]
    pub intensity: String, // "auto"
    #[serde(default)]
    pub device_overrides: Vec<DeviceOverride>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeCfg {
    #[serde(default = "default_backoff")]
    pub reconnect_backoff: String, // human range like "1s..30s"
    #[serde(default = "default_stale_ms")]
    pub stale_drop_ms: u64,
    #[serde(default = "default_telemetry")]
    pub telemetry_addr: String,
    #[serde(default)]
    pub log: Option<String>,
    #[serde(default)]
    pub test_mode: bool, // Enable artificially easy difficulty for testing
    #[serde(default = "default_test_difficulty")]
    pub test_difficulty: f64, // Override difficulty in test mode
    #[serde(default)]
    pub performance: Option<PerformanceCfg>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceCfg {
    #[serde(default = "default_work_size_multiplier")]
    pub work_size_multiplier: f64,
    #[serde(default = "default_poll_interval_ms")]
    pub poll_interval_ms: u64,
    #[serde(default = "default_concurrent_work_items")]
    pub concurrent_work_items: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppCfg {
    pub pool: PoolCfg,
    pub gpu: GpuCfg,
    pub runtime: RuntimeCfg,
}

fn default_password() -> String {
    "x".into()
}
fn default_backend() -> String {
    "dummy".into()
}
fn default_intensity() -> String {
    "auto".into()
}
fn default_backoff() -> String {
    "1s..30s".into()
}
fn default_stale_ms() -> u64 {
    500
}
fn default_telemetry() -> String {
    "0.0.0.0:9100".into()
}
fn default_test_difficulty() -> f64 {
    0.00001 // Very easy difficulty for testing
}

fn default_ring_capacity() -> u32 {
    8192
}

fn default_work_size_multiplier() -> f64 {
    1.0
}

fn default_poll_interval_ms() -> u64 {
    50
}

fn default_concurrent_work_items() -> u32 {
    1
}

pub fn load_from_path(path: &str) -> Result<AppCfg, ConfigError> {
    let builder = config::Config::builder()
        .add_source(config::File::with_name(path))
        .add_source(config::Environment::with_prefix("MINER").separator("__"));
    let cfg = builder.build().map_err(|e| anyhow::anyhow!(e))?;
    let app: AppCfg = cfg.try_deserialize().map_err(|e| anyhow::anyhow!(e))?;
    Ok(app)
}
