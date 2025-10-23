use anyhow::Result;
use nitrate_utils::double_sha256;

/// Very small placeholder until real header assembly is implemented.
pub fn quick_verify_share(header: &[u8]) -> Result<[u8; 32]> {
    Ok(double_sha256(header))
}
