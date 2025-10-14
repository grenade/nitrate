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
        // Minimal URL parser for stratum+tcp://host:port
        let addr = cfg.url.strip_prefix("stratum+tcp://").unwrap_or(&cfg.url);
        info!("connecting to pool at {addr}");
        let _stream = TcpStream::connect(addr).await?; // not kept yet; skeleton only
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
