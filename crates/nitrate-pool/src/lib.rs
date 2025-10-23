use anyhow::Result;
use nitrate_proto::MiningNotify;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
#[cfg(feature = "tls-rustls")]
use tokio_rustls::rustls::pki_types::ServerName;
#[cfg(feature = "tls-rustls")]
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
#[cfg(feature = "tls-rustls")]
use tokio_rustls::TlsConnector;
use tracing::{debug, error, info, warn};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    pub url: String,
    pub user: String,
    pub pass: String,
    pub tls: bool,
    pub tls_insecure: bool,
}

// Job information sent to core
#[derive(Clone, Debug)]
pub struct StratumJob {
    pub job_id: String,
    pub notify: MiningNotify,
    pub difficulty: f64,
    pub extranonce1: Vec<u8>,
    pub extranonce2_size: usize,
}

// Share submission
#[derive(Clone, Debug)]
pub struct Share {
    pub job_id: String,
    pub extranonce2: Vec<u8>,
    pub ntime: u32,
    pub nonce: u32,
}

pub struct StratumClient {
    #[allow(dead_code)]
    cfg: PoolConfig,
    job_tx: mpsc::UnboundedSender<StratumJob>,
    job_rx: Option<mpsc::UnboundedReceiver<StratumJob>>,
    share_tx: Option<mpsc::UnboundedSender<Share>>,
    #[allow(dead_code)]
    request_id: Arc<AtomicU64>,
}

fn parse_notify(params: &[Value]) -> Result<MiningNotify> {
    if params.len() < 9 {
        return Err(anyhow::anyhow!("notify params too short"));
    }

    Ok(MiningNotify {
        job_id: params[0].as_str().unwrap_or("").to_string(),
        prevhash: params[1].as_str().unwrap_or("").to_string(),
        coinbase1: params[2].as_str().unwrap_or("").to_string(),
        coinbase2: params[3].as_str().unwrap_or("").to_string(),
        merkle_branch: params[4]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        version: params[5].as_str().unwrap_or("").to_string(),
        nbits: params[6].as_str().unwrap_or("").to_string(),
        ntime: params[7].as_str().unwrap_or("").to_string(),
        clean_jobs: params[8].as_bool().unwrap_or(false),
    })
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err(anyhow::anyhow!("hex string has odd length"));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let from_hex = |c: u8| -> Result<u8> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(10 + (c - b'a')),
            b'A'..=b'F' => Ok(10 + (c - b'A')),
            _ => Err(anyhow::anyhow!("invalid hex character")),
        }
    };
    let mut i = 0usize;
    while i < bytes.len() {
        let hi = from_hex(bytes[i])?;
        let lo = from_hex(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

impl StratumClient {
    pub async fn connect(cfg: PoolConfig) -> Result<Self> {
        // URL parsing
        let (use_tls, addr, _host_for_sni) =
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

        // Establish connection (TCP or TLS) and split into reader/writer
        let tcp = TcpStream::connect(&addr).await?;

        #[cfg(feature = "tls-rustls")]
        let (reader, mut write_half) = if use_tls {
            let tls_insecure = cfg.tls_insecure;

            let mut roots = RootCertStore::empty();
            if let Ok(iter) = rustls_native_certs::load_native_certs() {
                for cert in iter {
                    let _ = roots.add(cert);
                }
            }
            if tls_insecure {
                warn!("tls_insecure requested but certificate verification is still enforced");
            }

            let config = ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth();

            let connector = TlsConnector::from(Arc::new(config));
            let server_name = ServerName::try_from(_host_for_sni.clone()).map_err(|_| {
                anyhow::anyhow!(format!("invalid TLS server name: {}", _host_for_sni))
            })?;
            let tls_stream = connector.connect(server_name, tcp).await?;
            let (r, w) = tokio::io::split(tls_stream);
            (BufReader::new(r), w)
        } else {
            let (r, w) = tokio::io::split(tcp);
            (BufReader::new(r), w)
        };

        #[cfg(not(feature = "tls-rustls"))]
        let (reader, mut write_half) = {
            if use_tls {
                warn!("TLS requested but nitrate-pool built without tls-rustls feature");
            }
            let (r, w) = tokio::io::split(tcp);
            (BufReader::new(r), w)
        };

        let (job_tx, job_rx) = mpsc::unbounded_channel();
        let (share_tx, share_rx) = mpsc::unbounded_channel();
        let request_id = Arc::new(AtomicU64::new(1));

        // Subscribe
        let sub_id = request_id.fetch_add(1, Ordering::Relaxed);
        let subscribe_msg = json!({
            "id": sub_id,
            "method": "mining.subscribe",
            "params": ["Nitrate/0.1"]
        });
        let frame = format!("{}\n", subscribe_msg);
        write_half.write_all(frame.as_bytes()).await?;
        write_half.flush().await?;

        // Authorize
        let auth_id = request_id.fetch_add(1, Ordering::Relaxed);
        let authorize_msg = json!({
            "id": auth_id,
            "method": "mining.authorize",
            "params": [&cfg.user, &cfg.pass]
        });
        let frame = format!("{}\n", authorize_msg);
        write_half.write_all(frame.as_bytes()).await?;
        write_half.flush().await?;

        // Spawn protocol tasks
        let client = Self {
            cfg: cfg.clone(),
            job_tx,
            job_rx: Some(job_rx),
            share_tx: Some(share_tx),
            request_id: request_id.clone(),
        };

        // Spawn read loop
        let job_tx_clone = client.job_tx.clone();
        let request_id_clone = request_id.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::read_loop(reader, job_tx_clone, request_id_clone).await {
                error!("read loop error: {}", e);
            }
        });

        // Spawn write loop for share submissions
        tokio::spawn(async move {
            if let Err(e) = Self::write_loop(write_half, share_rx).await {
                error!("write loop error: {}", e);
            }
        });

        Ok(client)
    }

    pub fn job_rx(&mut self) -> Option<mpsc::UnboundedReceiver<StratumJob>> {
        self.job_rx.take()
    }

    pub async fn submit_share(&self, share: Share) -> Result<()> {
        if let Some(ref tx) = self.share_tx {
            tx.send(share)?;
        }
        Ok(())
    }

    async fn read_loop(
        mut reader: BufReader<impl tokio::io::AsyncRead + Unpin>,
        job_tx: mpsc::UnboundedSender<StratumJob>,
        _request_id: Arc<AtomicU64>,
    ) -> Result<()> {
        let mut line = String::new();
        let mut current_difficulty = 1.0;
        let mut extranonce1 = Vec::new();
        let mut extranonce2_size = 4usize;

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                warn!("pool connection closed");
                break;
            }

            let msg: Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    debug!("failed to parse JSON: {} (line: {})", e, line.trim());
                    continue;
                }
            };

            // Handle responses to our requests
            if let Some(id) = msg.get("id") {
                if let Some(result) = msg.get("result") {
                    debug!("response id={}: {:?}", id, result);

                    // Parse subscribe response for extranonce
                    if let Some(arr) = result.as_array() {
                        if arr.len() >= 2 {
                            // arr[1] is extranonce1 (hex)
                            if let Some(e1_hex) = arr.get(1).and_then(|v| v.as_str()) {
                                extranonce1 = hex_decode(e1_hex).unwrap_or_default();
                                info!("extranonce1: {:?}", extranonce1);
                            }
                            // arr[2] is extranonce2_size
                            if let Some(e2_size) = arr.get(2).and_then(|v| v.as_u64()) {
                                extranonce2_size = e2_size as usize;
                                info!("extranonce2_size: {}", extranonce2_size);
                            }
                        }
                    }
                }
            }

            // Handle mining methods
            if let Some(method) = msg.get("method").and_then(|v| v.as_str()) {
                match method {
                    "mining.set_difficulty" => {
                        if let Some(params) = msg.get("params").and_then(|v| v.as_array()) {
                            if let Some(diff) = params.get(0).and_then(|v| v.as_f64()) {
                                current_difficulty = diff;
                                info!("difficulty set to {}", current_difficulty);
                            }
                        }
                    }
                    "mining.notify" => {
                        if let Some(params) = msg.get("params").and_then(|v| v.as_array()) {
                            if let Ok(notify) = parse_notify(params) {
                                let job = StratumJob {
                                    job_id: notify.job_id.clone(),
                                    notify,
                                    difficulty: current_difficulty,
                                    extranonce1: extranonce1.clone(),
                                    extranonce2_size,
                                };
                                debug!("new job: {}", job.job_id);
                                let _ = job_tx.send(job);
                            }
                        }
                    }
                    _ => {
                        debug!("unknown method: {}", method);
                    }
                }
            }
        }

        Ok(())
    }

    async fn write_loop(
        mut writer: impl tokio::io::AsyncWrite + Unpin,
        mut share_rx: mpsc::UnboundedReceiver<Share>,
    ) -> Result<()> {
        let mut request_id = 100u64;

        while let Some(share) = share_rx.recv().await {
            request_id += 1;

            // Format extranonce2 and ntime as hex
            let e2_hex = hex_encode(&share.extranonce2);
            let ntime_hex = format!("{:08x}", share.ntime);
            let nonce_hex = format!("{:08x}", share.nonce);

            let submit_msg = json!({
                "id": request_id,
                "method": "mining.submit",
                "params": [
                    "worker",  // worker name (often ignored)
                    &share.job_id,
                    &e2_hex,
                    &ntime_hex,
                    &nonce_hex
                ]
            });

            let frame = format!("{}\n", submit_msg);
            writer.write_all(frame.as_bytes()).await?;
            writer.flush().await?;
            debug!("submitted share for job {}", share.job_id);
        }

        Ok(())
    }
}
