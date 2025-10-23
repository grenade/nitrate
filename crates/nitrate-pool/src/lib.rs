use anyhow::Result;
use serde::{Deserialize, Serialize};
#[cfg(feature = "tls-rustls")]
use std::sync::Arc;
use tokio::net::TcpStream;
#[cfg(feature = "tls-rustls")]
use tokio_rustls::rustls::pki_types::ServerName;
#[cfg(feature = "tls-rustls")]
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
#[cfg(feature = "tls-rustls")]
use tokio_rustls::TlsConnector;

use tokio::time::{sleep, Duration};
use tracing::{info, warn};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    pub url: String,
    pub user: String,
    pub pass: String,
    pub tls: bool,
    pub tls_insecure: bool,
}

pub struct StratumClient {
    pub cfg: PoolConfig,
}

impl StratumClient {
    pub async fn connect(cfg: PoolConfig) -> Result<Self> {
        // URL parsing supports:
        // - stratum+tcp://host:port  => use_tls = false
        // - stratum+ssl://host:port  => use_tls = true
        // - host:port (no scheme)    => uses cfg.tls to decide
        let (use_tls, addr, host_for_sni) =
            if let Some(rest) = cfg.url.strip_prefix("stratum+tcp://") {
                (
                    false,
                    rest.to_string(),
                    rest.split(':').next().unwrap_or(rest).to_string(),
                )
            } else if let Some(rest) = cfg.url.strip_prefix("stratum+ssl://") {
                (
                    true,
                    rest.to_string(),
                    rest.split(':').next().unwrap_or(rest).to_string(),
                )
            } else {
                (
                    cfg.tls,
                    cfg.url.clone(),
                    cfg.url
                        .split(':')
                        .next()
                        .unwrap_or(cfg.url.as_str())
                        .to_string(),
                )
            };
        info!("connecting to pool at {} (tls={})", addr, use_tls);
        let tcp = TcpStream::connect(&addr).await?;
        #[cfg(feature = "tls-rustls")]
        let _maybe_tls = if use_tls {
            // Insecure TLS (testing only), controlled by config flag.
            let tls_insecure = cfg.tls_insecure;

            // Build rustls client config with native roots.
            let mut roots = RootCertStore::empty();
            if let Ok(iter) = rustls_native_certs::load_native_certs() {
                for cert in iter {
                    let _ = roots.add(cert);
                }
            }
            // Note: tls_insecure currently does not disable verification. A custom verifier can be wired later.
            if tls_insecure {
                warn!("tls_insecure requested in config, but certificate verification is still enforced (no custom verifier configured)");
            }
            let config = ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth();

            let connector = TlsConnector::from(Arc::new(config));
            let server_name = ServerName::try_from(host_for_sni.clone()).map_err(|_| {
                anyhow::anyhow!(format!("invalid TLS server name: {}", host_for_sni))
            })?;
            let _tls_stream = connector.connect(server_name, tcp).await?;
            Some(())
        } else {
            None
        };
        #[cfg(not(feature = "tls-rustls"))]
        {
            if use_tls {
                warn!("TLS requested (stratum+ssl) but nitrate-pool was built without the tls-rustls feature; falling back to plain TCP");
            }
        }
        // In a full implementation, keep the stream (TCP or TLS) and start the read/write protocol loops.
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
