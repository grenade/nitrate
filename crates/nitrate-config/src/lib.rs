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
pub struct GpuCfg {
    #[serde(default = "default_backend")]
    pub backend: String, // "dummy" | "cuda" | "hip" | "opencl"
    #[serde(default)]
    pub devices: Vec<u32>, // device indexes
    #[serde(default = "default_intensity")]
    pub intensity: String, // "auto"
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

pub fn load_from_path(path: &str) -> Result<AppCfg, ConfigError> {
    let builder = config::Config::builder()
        .add_source(config::File::with_name(path))
        .add_source(config::Environment::with_prefix("MINER").separator("__"));
    let cfg = builder.build().map_err(|e| anyhow::anyhow!(e))?;
    let app: AppCfg = cfg.try_deserialize().map_err(|e| anyhow::anyhow!(e))?;
    Ok(app)
}
