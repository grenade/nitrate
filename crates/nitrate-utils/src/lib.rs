use sha2::{Digest, Sha256};

pub fn double_sha256(data: &[u8]) -> [u8; 32] {
    let first = Sha256::digest(data);
    let second = Sha256::digest(&first);
    let mut out = [0u8; 32];
    out.copy_from_slice(&second);
    out
}

pub fn be_bytes_to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}
