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

// Type alias for pool connection components
type PoolConnection = (
    BufReader<Box<dyn tokio::io::AsyncRead + Unpin + Send>>,
    Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
);

pub struct StratumClient {
    #[allow(dead_code)]
    cfg: PoolConfig,
    #[allow(dead_code)]
    job_tx: mpsc::UnboundedSender<StratumJob>,
    job_rx: Option<mpsc::UnboundedReceiver<StratumJob>>,
    share_queue: Arc<tokio::sync::Mutex<Vec<Share>>>,
    #[allow(dead_code)]
    request_id: Arc<AtomicU64>,
    connected: Arc<tokio::sync::RwLock<bool>>,
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
    if !s.len().is_multiple_of(2) {
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
        let request_id = Arc::new(AtomicU64::new(1));
        let share_queue = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let connected = Arc::new(tokio::sync::RwLock::new(true));

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
            job_tx: job_tx.clone(),
            job_rx: Some(job_rx),
            share_queue: share_queue.clone(),
            request_id: request_id.clone(),
            connected: connected.clone(),
        };

        // Spawn connection manager that handles reconnection
        let cfg_clone = cfg.clone();
        let job_tx_clone = job_tx.clone();
        let share_queue_clone = share_queue.clone();
        let request_id_clone = request_id.clone();
        let connected_clone = connected.clone();

        // Box the initial connection to match the expected types
        let boxed_reader = BufReader::new(
            Box::new(reader.into_inner()) as Box<dyn tokio::io::AsyncRead + Unpin + Send>
        );
        let boxed_writer = Box::new(write_half) as Box<dyn tokio::io::AsyncWrite + Unpin + Send>;

        tokio::spawn(async move {
            Self::connection_manager(
                cfg_clone,
                job_tx_clone,
                share_queue_clone,
                request_id_clone,
                connected_clone,
                Some((boxed_reader, boxed_writer)),
            )
            .await;
        });

        Ok(client)
    }

    pub fn job_rx(&mut self) -> Option<mpsc::UnboundedReceiver<StratumJob>> {
        self.job_rx.take()
    }

    pub async fn submit_share(&self, share: Share) -> Result<()> {
        // Queue the share for submission
        let mut queue = self.share_queue.lock().await;
        queue.push(share);

        // If disconnected, shares will be submitted when connection is restored
        if !*self.connected.read().await {
            debug!("Share queued for submission when connection is restored");
        }

        Ok(())
    }

    async fn connection_manager(
        cfg: PoolConfig,
        job_tx: mpsc::UnboundedSender<StratumJob>,
        share_queue: Arc<tokio::sync::Mutex<Vec<Share>>>,
        request_id: Arc<AtomicU64>,
        connected: Arc<tokio::sync::RwLock<bool>>,
        mut initial_conn: Option<PoolConnection>,
    ) {
        let mut backoff_ms = 1000;
        const MAX_BACKOFF_MS: u64 = 30000;

        loop {
            // Use initial connection if available, otherwise reconnect
            let (reader, write_half) = if let Some(conn) = initial_conn.take() {
                conn
            } else {
                // Mark as disconnected
                *connected.write().await = false;

                // Wait with exponential backoff
                tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);

                info!("Attempting to reconnect to pool at {}", cfg.url);

                // Try to reconnect
                match Self::establish_connection(&cfg).await {
                    Ok(conn) => {
                        info!("Successfully reconnected to pool");
                        backoff_ms = 1000; // Reset backoff
                        conn
                    }
                    Err(e) => {
                        error!("Failed to reconnect: {}. Retrying in {}ms", e, backoff_ms);
                        continue;
                    }
                }
            };

            // Mark as connected
            *connected.write().await = true;

            // Spawn read loop
            let job_tx_clone = job_tx.clone();
            let request_id_clone = request_id.clone();
            let connected_clone = connected.clone();
            let read_handle = tokio::spawn(async move {
                let result = Self::read_loop(reader, job_tx_clone, request_id_clone).await;
                *connected_clone.write().await = false;
                result
            });

            // Spawn write loop with share queue
            let share_queue_clone = share_queue.clone();
            let connected_clone = connected.clone();
            let write_handle = tokio::spawn(async move {
                let result = Self::write_loop_with_queue(write_half, share_queue_clone).await;
                *connected_clone.write().await = false;
                result
            });

            // Wait for either task to complete (indicating disconnection)
            tokio::select! {
                result = read_handle => {
                    match result {
                        Ok(Ok(())) => info!("Read loop completed normally"),
                        Ok(Err(e)) => error!("Read loop error: {}", e),
                        Err(e) => error!("Read loop panic: {}", e),
                    }
                }
                result = write_handle => {
                    match result {
                        Ok(Ok(())) => info!("Write loop completed normally"),
                        Ok(Err(e)) => error!("Write loop error: {}", e),
                        Err(e) => error!("Write loop panic: {}", e),
                    }
                }
            }

            warn!("Pool connection lost, will attempt reconnection");
        }
    }

    async fn establish_connection(cfg: &PoolConfig) -> Result<PoolConnection> {
        // Parse URL manually - expecting format like "stratum+tcp://host:port"
        let url_str = &cfg.url;
        let url_str = url_str
            .strip_prefix("stratum+tcp://")
            .or_else(|| url_str.strip_prefix("stratum+ssl://"))
            .or_else(|| url_str.strip_prefix("stratum+tls://"))
            .unwrap_or(url_str);

        let (host, port) = if let Some(colon_idx) = url_str.rfind(':') {
            let host = &url_str[..colon_idx];
            let port_str = &url_str[colon_idx + 1..];
            let port = port_str.parse::<u16>().unwrap_or(3333);
            (host, port)
        } else {
            (url_str, 3333u16)
        };

        let stream = TcpStream::connect((host, port)).await?;

        #[cfg(feature = "tls-rustls")]
        let (reader, mut write_half): (
            BufReader<Box<dyn tokio::io::AsyncRead + Unpin + Send>>,
            Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
        ) = {
            if cfg.tls {
                let connector = crate::tls::create_tls_connector(cfg.tls_insecure)?;
                let server_name = ServerName::try_from(host.to_string())
                    .map_err(|_| anyhow::anyhow!("invalid DNS name: {}", host))?;
                let tls_stream = connector.connect(server_name, stream).await?;
                let (read_half, write_half) = tokio::io::split(tls_stream);
                (
                    BufReader::new(
                        Box::new(read_half) as Box<dyn tokio::io::AsyncRead + Unpin + Send>
                    ),
                    Box::new(write_half) as Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
                )
            } else {
                let (read_half, write_half) = stream.into_split();
                (
                    BufReader::new(
                        Box::new(read_half) as Box<dyn tokio::io::AsyncRead + Unpin + Send>
                    ),
                    Box::new(write_half) as Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
                )
            }
        };

        #[cfg(not(feature = "tls-rustls"))]
        let (reader, mut write_half): (
            BufReader<Box<dyn tokio::io::AsyncRead + Unpin + Send>>,
            Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
        ) = {
            let (read_half, write_half) = stream.into_split();
            (
                BufReader::new(Box::new(read_half) as Box<dyn tokio::io::AsyncRead + Unpin + Send>),
                Box::new(write_half) as Box<dyn tokio::io::AsyncWrite + Unpin + Send>,
            )
        };

        // Send subscribe message
        let subscribe_msg = json!({
            "id": 1,
            "method": "mining.subscribe",
            "params": ["nitrate/0.1.0"]
        });
        let frame = format!("{}\n", subscribe_msg);
        write_half.write_all(frame.as_bytes()).await?;
        write_half.flush().await?;

        // Authorize
        let authorize_msg = json!({
            "id": 2,
            "method": "mining.authorize",
            "params": [&cfg.user, &cfg.pass]
        });
        let frame = format!("{}\n", authorize_msg);
        write_half.write_all(frame.as_bytes()).await?;
        write_half.flush().await?;

        Ok((reader, write_half))
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
                            if let Some(diff) = params.first().and_then(|v| v.as_f64()) {
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

    async fn write_loop_with_queue(
        mut writer: impl tokio::io::AsyncWrite + Unpin,
        share_queue: Arc<tokio::sync::Mutex<Vec<Share>>>,
    ) -> Result<()> {
        let mut request_id = 100u64;
        let mut check_interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

        loop {
            check_interval.tick().await;

            // Get shares from queue
            let shares = {
                let mut queue = share_queue.lock().await;
                if queue.is_empty() {
                    continue;
                }
                queue.drain(..).collect::<Vec<_>>()
            };

            // Submit each share
            for share in shares {
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
                if let Err(e) = writer.write_all(frame.as_bytes()).await {
                    // Put share back in queue for retry
                    share_queue.lock().await.insert(0, share);
                    return Err(anyhow::anyhow!("Failed to write share: {}", e));
                }
                if let Err(e) = writer.flush().await {
                    // Put share back in queue for retry
                    share_queue.lock().await.insert(0, share);
                    return Err(anyhow::anyhow!("Failed to flush: {}", e));
                }
                info!(
                    "submitted share (nonce={:08x}) for job {}",
                    share.nonce, share.job_id
                );
            }
        }
    }
}
