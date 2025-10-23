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

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
