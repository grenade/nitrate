use anyhow::Result;
#[cfg(test)]
use nitrate_utils::be_bytes_to_hex;
use nitrate_utils::double_sha256;

/// Prepared GPU work derived from Stratum notify fields and extranonces.
#[derive(Clone, Debug)]
pub struct PreparedWork {
    /// SHA-256 state after hashing header bytes 0..63, as 8 big-endian u32 words.
    pub midstate: [u32; 8],
    /// Last 16 bytes of the 80-byte header:
    /// merkle_root tail (4 bytes in header order) | ntime (4 LE) | nbits (4 LE) | nonce placeholder (4 LE = 0)
    pub tail16: [u8; 16],
    /// 32-byte big-endian share target (block target if no share target provided).
    pub target_be: [u8; 32],
    /// Full 80-byte header with nonce set to 0 (little-endian fields in header as per Bitcoin).
    pub header80: [u8; 80],
}

/// Build raw coinbase transaction bytes: coinbase1 || extranonce1 || extranonce2 || coinbase2.
pub fn build_coinbase_bytes(
    coinbase1_hex: &str,
    extranonce1: &[u8],
    extranonce2: &[u8],
    coinbase2_hex: &str,
) -> Result<Vec<u8>> {
    let mut out = hex_decode(coinbase1_hex)?;
    out.extend_from_slice(extranonce1);
    out.extend_from_slice(extranonce2);
    out.extend_from_slice(&hex_decode(coinbase2_hex)?);
    Ok(out)
}

/// Compute merkle root (big-endian) from coinbase txid (big-endian) and merkle branch items
/// provided as hex (little-endian byte order as in Stratum v1).
pub fn merkle_root_be_from_branch(
    coinbase_txid_be: [u8; 32],
    merkle_branch_hex: &[String],
) -> Result<[u8; 32]> {
    // Start from little-endian internal representation for folding.
    let mut h_le = {
        let mut t = coinbase_txid_be;
        t.reverse();
        t
    };
    for br_hex in merkle_branch_hex {
        let br_le = hex_decode(br_hex)?;
        // Ensure 32 bytes
        if br_le.len() != 32 {
            return Err(anyhow::anyhow!(
                "merkle branch item must be 32 bytes, got {}",
                br_le.len()
            ));
        }
        // Concatenate h_le || br_le and hash
        let mut concat = Vec::with_capacity(64);
        concat.extend_from_slice(&h_le);
        concat.extend_from_slice(&br_le);
        let mut hh = double_sha256(&concat);
        // Next iteration expects little-endian
        hh.reverse();
        h_le.copy_from_slice(&hh);
    }
    // Convert final little-endian root to big-endian for header assembly
    h_le.reverse();
    let mut root_be = [0u8; 32];
    root_be.copy_from_slice(&h_le);
    Ok(root_be)
}

/// Prepare midstate/tail/target/header from Stratum notify parts.
/// All hex inputs are as provided by typical Stratum v1:
/// - prevhash: 32-byte hash in little-endian hex
/// - version, ntime, nbits: 4-byte fields in little-endian hex
/// - coinbase1/2: transaction slices in hex (raw bytes)
/// - merkle_branch: array of 32-byte hashes in little-endian hex
pub fn prepare_from_notify_parts(
    version_hex_le: &str,
    prevhash_hex_le: &str,
    coinbase1_hex: &str,
    coinbase2_hex: &str,
    merkle_branch_hex: &[String],
    ntime_hex_le: &str,
    nbits_hex_le: &str,
    extranonce1: &[u8],
    extranonce2: &[u8],
) -> Result<PreparedWork> {
    // Parse fixed fields
    let version = parse_u32_le_hex(version_hex_le)?;
    let ntime = parse_u32_le_hex(ntime_hex_le)?;
    let nbits = parse_u32_le_hex(nbits_hex_le)?;

    // prevhash: provided as little-endian hex, convert to big-endian bytes for header assembly
    let mut prevhash_le = hex_decode(prevhash_hex_le)?;
    if prevhash_le.len() != 32 {
        return Err(anyhow::anyhow!("prevhash must be 32 bytes"));
    }
    prevhash_le.reverse();
    let mut prevhash_be = [0u8; 32];
    prevhash_be.copy_from_slice(&prevhash_le);

    // Coinbase and merkle root
    let coinbase = build_coinbase_bytes(coinbase1_hex, extranonce1, extranonce2, coinbase2_hex)?;
    let coinbase_txid_be = double_sha256(&coinbase);
    let merkle_root_be = merkle_root_be_from_branch(coinbase_txid_be, merkle_branch_hex)?;

    // Assemble header with nonce = 0 (placeholder)
    let header80 = assemble_header(version, prevhash_be, merkle_root_be, ntime, nbits, 0);
    let mut first64 = [0u8; 64];
    first64.copy_from_slice(&header80[0..64]);
    let mut tail16 = [0u8; 16];
    tail16.copy_from_slice(&header80[64..80]);

    let midstate = sha256_midstate_first64(&first64);
    let target_be = nbits_to_target_be(nbits);

    Ok(PreparedWork {
        midstate,
        tail16,
        target_be,
        header80,
    })
}

/// Decode hex string into bytes.
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

/// Parse a 4-byte little-endian hex string into u32.
fn parse_u32_le_hex(s: &str) -> Result<u32> {
    let b = hex_decode(s)?;
    if b.len() != 4 {
        return Err(anyhow::anyhow!(
            "expected 4-byte hex, got {} bytes",
            b.len()
        ));
    }
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Assemble an 80-byte Bitcoin block header from components.
/// Inputs:
/// - version: u32 (will be written LE)
/// - prevhash_be: 32-byte previous block hash in big-endian display order
/// - merkle_root_be: 32-byte merkle root in big-endian display order
/// - ntime: u32 (LE in header)
/// - nbits: u32 (LE in header)
/// - nonce: u32 (LE in header)
///
/// The header format (80 bytes):
/// version[4-LE] | prevhash[32-LE] | merkle[32-LE] | ntime[4-LE] | nbits[4-LE] | nonce[4-LE]
pub fn assemble_header(
    version: u32,
    prevhash_be: [u8; 32],
    merkle_root_be: [u8; 32],
    ntime: u32,
    nbits: u32,
    nonce: u32,
) -> [u8; 80] {
    let mut h = [0u8; 80];
    // version
    put_le_u32(&mut h[0..4], version);
    // prevhash: header stores little-endian (reverse of display/be)
    for i in 0..32 {
        h[4 + i] = prevhash_be[31 - i];
    }
    // merkle root: header stores little-endian (reverse of display/be)
    for i in 0..32 {
        h[36 + i] = merkle_root_be[31 - i];
    }
    // tail
    put_le_u32(&mut h[68..72], ntime);
    put_le_u32(&mut h[72..76], nbits);
    put_le_u32(&mut h[76..80], nonce);
    h
}

/// Compute big-endian 32-byte target from compact nBits.
pub fn nbits_to_target_be(nbits: u32) -> [u8; 32] {
    let e = ((nbits >> 24) & 0xff) as usize;
    let m = (nbits & 0x007f_ffff) as u32; // ignore sign bit
    let mut tgt = [0u8; 32];

    if e <= 3 {
        // Shift mantissa right by whole bytes and place in the last 3 bytes.
        let shift = 8 * (3 - e);
        let val = m >> shift;
        tgt[29] = ((val >> 16) & 0xff) as u8;
        tgt[30] = ((val >> 8) & 0xff) as u8;
        tgt[31] = (val & 0xff) as u8;
    } else if e < 32 {
        // Place the 3-byte mantissa starting at (32 - e) for big-endian representation.
        let offset = 32 - e;
        if offset < 30 {
            // Ensure we have room for 3 bytes
            tgt[offset] = ((m >> 16) & 0xff) as u8;
            if offset + 1 < 32 {
                tgt[offset + 1] = ((m >> 8) & 0xff) as u8;
            }
            if offset + 2 < 32 {
                tgt[offset + 2] = (m & 0xff) as u8;
            }
        }
    }
    // else: exponent too large, target stays all zeros (impossible difficulty)

    tgt
}

/// Compute SHA-256 midstate after hashing the first 64 bytes of a header.
/// Returns 8 big-endian u32 words as defined by SHA-256.
pub fn sha256_midstate_first64(header_first64: &[u8; 64]) -> [u32; 8] {
    let mut state = sha256_iv();
    sha256_compress(&mut state, header_first64);
    state
}

/// Compute SHA-256(header80) using the provided midstate and the last 16 bytes (ntime|nbits|nonce LE).
/// Returns 32-byte big-endian digest.
pub fn sha256_from_midstate(midstate: [u32; 8], tail16: &[u8; 16]) -> [u8; 32] {
    // Build the second block for 80-byte message:
    // tail16 (16B) + 0x80 + zeros ... + length(80*8=640) in 64-bit BE at the end.
    let mut block2 = [0u8; 64];
    block2[0..16].copy_from_slice(tail16);
    block2[16] = 0x80;
    // bytes 17..55 are zero by default
    let bit_len: u64 = 80 * 8;
    block2[56..64].copy_from_slice(&bit_len.to_be_bytes());

    let mut st = midstate;
    sha256_compress(&mut st, &block2);

    // Output digest as big-endian bytes
    let mut out = [0u8; 32];
    for (i, w) in st.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

/// Compute double SHA-256 of an 80-byte header via midstate path.
pub fn double_sha256_via_midstate(header80: &[u8; 80]) -> [u8; 32] {
    let mut first64 = [0u8; 64];
    first64.copy_from_slice(&header80[0..64]);
    let mut tail16 = [0u8; 16];
    tail16.copy_from_slice(&header80[64..80]);
    let mid = sha256_midstate_first64(&first64);
    let h1 = sha256_from_midstate(mid, &tail16);
    // Second SHA-256 over 32-byte result
    sha256_32bytes(&h1)
}

/// Very small placeholder kept for compatibility; uses regular double_sha256.
pub fn quick_verify_share(header: &[u8]) -> Result<[u8; 32]> {
    Ok(double_sha256(header))
}

/* ==== Internal SHA-256 primitives (single-block compress) ==== */

#[inline(always)]
fn rotr(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}
#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}
#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}
#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
    rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
}
#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
    rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
}
#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
    rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)
}
#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
    rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10)
}

fn sha256_iv() -> [u32; 8] {
    [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ]
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    // Prepare message schedule
    let mut w = [0u32; 64];
    for (i, chunk) in block.chunks_exact(4).enumerate().take(16) {
        w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for t in 16..64 {
        w[t] = small_sigma1(w[t - 2])
            .wrapping_add(w[t - 7])
            .wrapping_add(small_sigma0(w[t - 15]))
            .wrapping_add(w[t - 16]);
    }

    // Initialize working variables
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for t in 0..64 {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[t])
            .wrapping_add(w[t]);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

/// SHA-256 over exactly 32 bytes of input (single-block path).
fn sha256_32bytes(input32: &[u8; 32]) -> [u8; 32] {
    // Build single block: data (32B) + 0x80 + zeros ... + bitlen(256)
    let mut block = [0u8; 64];
    block[0..32].copy_from_slice(input32);
    block[32] = 0x80;
    // zeros until 56
    block[56..64].copy_from_slice(&(32u64 * 8).to_be_bytes());

    let mut st = sha256_iv();
    sha256_compress(&mut st, &block);

    let mut out = [0u8; 32];
    for (i, w) in st.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

fn put_le_u32(dst: &mut [u8], v: u32) {
    dst.copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex_to_bytes(s: &str) -> Vec<u8> {
        let s = s.trim();
        let mut out = Vec::with_capacity(s.len() / 2);
        let mut it = s.as_bytes().chunks_exact(2);
        for pair in &mut it {
            let hi = (pair[0] as char).to_digit(16).unwrap();
            let lo = (pair[1] as char).to_digit(16).unwrap();
            out.push(((hi << 4) | lo) as u8);
        }
        out
    }

    #[test]
    fn genesis_header_and_hash() {
        // Genesis block components
        let version = 1u32;
        let prevhash_be = [0u8; 32];
        // Merkle root (big-endian display)
        let merkle_be = {
            let s = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b";
            let v = hex_to_bytes(s);
            let mut a = [0u8; 32];
            a.copy_from_slice(&v);
            a
        };
        let ntime = 1231006505u32; // 0x495fab29
        let nbits = 0x1d00ffffu32;
        let nonce = 2083236893u32; // 0x7c2bac1d

        let header = assemble_header(version, prevhash_be, merkle_be, ntime, nbits, nonce);
        assert_eq!(header.len(), 80);
        // Known genesis hash (display big-endian hex)
        let genesis_hash_be_hex =
            "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f";
        let dh = double_sha256(&header);
        // Reverse bytes to compare against display hex
        let mut dh_rev = dh;
        dh_rev.reverse();
        assert_eq!(be_bytes_to_hex(&dh_rev), genesis_hash_be_hex);

        // Check via midstate path
        let dh_mid = double_sha256_via_midstate(&header);
        assert_eq!(dh_mid, dh);
    }

    #[test]
    fn nbits_to_target_basic() {
        // For 0x1d00ffff, the target starts with 0x00ffff * 2^(8*(0x1d-3))
        let target = nbits_to_target_be(0x1d00ffff);
        // The first three bytes should be 00 00 00 (high zeros), there will be non-zeros later.
        assert_eq!(target[0], 0x00);
        assert_eq!(target[1], 0x00);
        assert_eq!(target[2], 0x00);
        // For 0x1d00ffff (e=0x1d), mantissa bytes (00 ff ff) land at indices 3..5 in BE.
        assert_eq!(target[3], 0x00);
        assert_eq!(target[4], 0xff);
        assert_eq!(target[5], 0xff);
    }
}
