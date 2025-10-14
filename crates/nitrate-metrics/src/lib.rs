use anyhow::Result;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use prometheus::{Encoder, Gauge, IntCounter, Registry, TextEncoder};
use std::net::SocketAddr;
use tokio::task::JoinHandle;
use tracing::info;

pub struct Metrics {
    pub registry: Registry,
    pub shares_ok: IntCounter,
    pub shares_rejected: IntCounter,
    pub hashrate_gps: Gauge,
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        let shares_ok = IntCounter::new("shares_accepted_total", "Accepted shares").unwrap();
        let shares_rejected = IntCounter::new("shares_rejected_total", "Rejected shares").unwrap();
        let hashrate_gps = Gauge::new("hashrate_ghs", "Hashrate (GH/s)").unwrap();
        registry.register(Box::new(shares_ok.clone())).unwrap();
        registry
            .register(Box::new(shares_rejected.clone()))
            .unwrap();
        registry.register(Box::new(hashrate_gps.clone())).unwrap();
        Self {
            registry,
            shares_ok,
            shares_rejected,
            hashrate_gps,
        }
    }

    pub async fn serve(&self, addr: SocketAddr) -> Result<JoinHandle<()>> {
        let registry = self.registry.clone();
        let make_svc = make_service_fn(move |_| {
            let registry = registry.clone();
            async move {
                Ok::<_, hyper::Error>(service_fn(move |_req: Request<Body>| {
                    let registry = registry.clone();
                    async move {
                        let encoder = TextEncoder::new();
                        let metric_families = registry.gather();
                        let mut buffer = Vec::new();
                        encoder.encode(&metric_families, &mut buffer).unwrap();
                        Ok::<_, hyper::Error>(Response::new(Body::from(buffer)))
                    }
                }))
            }
        });
        let server = Server::bind(&addr).serve(make_svc);
        info!("metrics listening on http://{addr}/");
        let handle = tokio::spawn(async move {
            if let Err(e) = server.await {
                eprintln!("metrics server error: {e}");
            }
        });
        Ok(handle)
    }
}
