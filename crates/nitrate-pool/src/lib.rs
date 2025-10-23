use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;

use tokio::time::{sleep, Duration};
use tracing::{info, warn};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    pub url: String,
    pub user: String,
    pub pass: String,
}

pub struct StratumClient {
    pub cfg: PoolConfig,
}

impl StratumClient {
    pub async fn connect(cfg: PoolConfig) -> Result<Self> {
        // URL parsing supports:
        // - stratum+tcp://host:port  => use_tls = false
        // - stratum+ssl://host:port  => use_tls = true
        // - host:port (no scheme)    => defaults to TCP (use_tls = false)
        let (use_tls, addr) = if let Some(rest) = cfg.url.strip_prefix("stratum+tcp://") {
            (false, rest.to_string())
        } else if let Some(rest) = cfg.url.strip_prefix("stratum+ssl://") {
            (true, rest.to_string())
        } else {
            (false, cfg.url.clone())
        };
        info!("connecting to pool at {} (tls={})", addr, use_tls);
        let _stream = TcpStream::connect(&addr).await?; // not kept yet; skeleton only
                                                        // TODO: When `use_tls` is true, perform TLS handshake (e.g., rustls) over this TCP stream.
                                                        // In real impl: send subscribe/authorize and start read loop.
        Ok(Self { cfg })
    }

    pub async fn next_job(&self) -> Result<Option<String>> {
        // Skeleton: sleep and return None. Replace with channel receiving Notify.
        sleep(Duration::from_secs(5)).await;
        Ok(None)
    }

    pub async fn submit_share(&self, _nonce: u32) -> Result<()> {
        warn!("submit_share called on skeleton client (noop)");
        Ok(())
    }
}
