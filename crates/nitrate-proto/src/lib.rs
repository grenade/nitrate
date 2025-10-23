use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeReq(pub [serde_json::Value; 2]); // minimal stub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizeReq(pub [serde_json::Value; 2]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotifyParams {
    pub job_id: String,
    pub clean_jobs: bool,
    // Other fields omitted for brevity; real Stratum has many.
}

#[derive(Debug, Clone)]
pub enum ProtoEvent {
    SetDifficulty(u64),
    Notify(NotifyParams),
    KeepAlive,
}
