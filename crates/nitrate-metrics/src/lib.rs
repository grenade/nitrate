use anyhow::Result;
use bytes::Bytes;
use http::{Request, Response};
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::service::Service;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use prometheus::{Encoder, Gauge, IntCounter, Registry, TextEncoder};
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tracing::info;

pub struct Metrics {
    pub registry: Registry,
    pub shares_ok: IntCounter,
    pub shares_rejected: IntCounter,
    pub jobs_received: IntCounter,
    pub gpu_launches: IntCounter,
    pub candidates_found: IntCounter,
    pub hashrate_gps: Gauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let shares_ok = IntCounter::new("shares_accepted_total", "Accepted shares").unwrap();
        let shares_rejected = IntCounter::new("shares_rejected_total", "Rejected shares").unwrap();
        let jobs_received = IntCounter::new("jobs_received_total", "Mining jobs received").unwrap();
        let gpu_launches = IntCounter::new("gpu_launches_total", "GPU kernel launches").unwrap();
        let candidates_found =
            IntCounter::new("candidates_found_total", "Candidate nonces found").unwrap();
        let hashrate_gps = Gauge::new("hashrate_ghs", "Hashrate (GH/s)").unwrap();
        registry.register(Box::new(shares_ok.clone())).unwrap();
        registry
            .register(Box::new(shares_rejected.clone()))
            .unwrap();
        registry.register(Box::new(jobs_received.clone())).unwrap();
        registry.register(Box::new(gpu_launches.clone())).unwrap();
        registry
            .register(Box::new(candidates_found.clone()))
            .unwrap();
        registry.register(Box::new(hashrate_gps.clone())).unwrap();
        Self {
            registry,
            shares_ok,
            shares_rejected,
            jobs_received,
            gpu_launches,
            candidates_found,
            hashrate_gps,
        }
    }

    pub async fn serve(&self, addr: SocketAddr) -> Result<JoinHandle<()>> {
        let listener = TcpListener::bind(addr).await?;
        info!("metrics listening on http://{addr}/");

        let service = MetricsService {
            registry: Arc::new(self.registry.clone()),
        };

        let handle = tokio::spawn(async move {
            loop {
                let (stream, _addr) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(e) => {
                        eprintln!("accept error: {e}");
                        continue;
                    }
                };

                let io = TokioIo::new(stream);
                let service = service.clone();

                tokio::spawn(async move {
                    let builder = Builder::new(TokioExecutor::new());
                    if let Err(e) = builder.serve_connection(io, service).await {
                        eprintln!("connection error: {e}");
                    }
                });
            }
        });

        Ok(handle)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct MetricsService {
    registry: Arc<Registry>,
}

impl Service<Request<Incoming>> for MetricsService {
    type Response = Response<Full<Bytes>>;
    type Error = anyhow::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn call(&self, _req: Request<Incoming>) -> Self::Future {
        let registry = self.registry.clone();
        Box::pin(async move {
            let encoder = TextEncoder::new();
            let metric_families = registry.gather();
            let mut buffer = Vec::new();
            encoder.encode(&metric_families, &mut buffer)?;
            let body = Full::new(Bytes::from(buffer));
            Ok(Response::new(body))
        })
    }
}
