use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscribeReq(pub [serde_json::Value; 2]); // minimal stub
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizeReq(pub [serde_json::Value; 2]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningNotify {
    // Stratum v1 mining.notify parameters (hex-encoded strings as sent by pools)
    pub job_id: String,
    pub prevhash: String,
    pub coinbase1: String,
    pub coinbase2: String,
    pub merkle_branch: Vec<String>,
    pub version: String, // 4-byte header version (hex)
    pub nbits: String,   // compact target (hex)
    pub ntime: String,   // current time (hex)
    pub clean_jobs: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetDifficulty {
    // Stratum v1 mining.set_difficulty parameter (often a floating value)
    pub difficulty: f64,
}

#[derive(Debug, Clone)]
pub enum ProtoEvent {
    SetDifficulty(SetDifficulty),
    Notify(MiningNotify),
    KeepAlive,
}
