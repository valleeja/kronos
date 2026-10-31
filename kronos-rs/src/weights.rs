//! Loading model folders from disk, with size limits and optional SHA-256 pinning.

use std::fs;
use std::path::Path;

use candle_core::Device;
use candle_nn::VarBuilder;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use crate::model::{Kronos, KronosConfig, KronosTokenizer, TokenizerConfig, DTYPE};

/// Largest weight file accepted (Kronos-base is about 0.4 GB).
pub const MAX_WEIGHTS_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let size = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if size > limit {
        return Err(format!("{}: file is larger than the {limit} byte limit", path.display()));
    }
    fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// Reads `config.json` and `model.safetensors` from a Hugging Face model folder.
/// If `expected_sha256` is given, the weights must match it exactly.
fn load_dir<C: DeserializeOwned>(
    dir: &Path,
    expected_sha256: Option<&str>,
    device: &Device,
) -> Result<(C, VarBuilder<'static>, String), String> {
    let config_bytes = read_limited(&dir.join("config.json"), MAX_CONFIG_BYTES)?;
    let config: C = serde_json::from_slice(&config_bytes).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
    let weights_path = dir.join("model.safetensors");
    let weights = read_limited(&weights_path, MAX_WEIGHTS_BYTES)?;
    let digest = sha256_hex(&weights);
    if let Some(expected) = expected_sha256 {
        if !digest.eq_ignore_ascii_case(expected.trim()) {
            return Err(format!(
                "{}: SHA-256 mismatch (expected {expected}, got {digest}). Refusing to load.",
                weights_path.display()
            ));
        }
    }
    let vb = VarBuilder::from_buffered_safetensors(weights, DTYPE, device)
        .map_err(|e| format!("{}: {e}", weights_path.display()))?;
    Ok((config, vb, digest))
}

/// Loads the tokenizer; returns it with the SHA-256 of its weights.
pub fn load_tokenizer(dir: &Path, expected_sha256: Option<&str>, device: &Device) -> Result<(KronosTokenizer, String), String> {
    let (config, vb, digest) = load_dir::<TokenizerConfig>(dir, expected_sha256, device)?;
    let tokenizer = KronosTokenizer::new(config, vb).map_err(|e| format!("tokenizer: {e}"))?;
    Ok((tokenizer, digest))
}

/// Loads the predictor; returns it with the SHA-256 of its weights.
pub fn load_model(dir: &Path, expected_sha256: Option<&str>, device: &Device) -> Result<(Kronos, String), String> {
    let (config, vb, digest) = load_dir::<KronosConfig>(dir, expected_sha256, device)?;
    let model = Kronos::new(config, vb).map_err(|e| format!("model: {e}"))?;
    Ok((model, digest))
}
