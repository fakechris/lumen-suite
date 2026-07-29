//! Model file layout probes.
//!
//! This crate never resolves default directories, environment overrides, or
//! downloads — that responsibility belongs to the consumer (e.g. the
//! `lumen-models` crate). Everything here operates on a directory the caller
//! already chose.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Explicit SenseVoice model files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenseVoiceModelPaths {
    /// `model.int8.onnx` / `model.onnx` / `sensevoice.onnx`.
    pub model: PathBuf,
    /// `tokens.txt`.
    pub tokens: PathBuf,
}

impl SenseVoiceModelPaths {
    /// Probe `dir` for the known SenseVoice file layout.
    pub fn discover(dir: &Path) -> Option<Self> {
        Some(Self {
            model: sensevoice_model_path(dir)?,
            tokens: sensevoice_tokens_path(dir)?,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.model.is_file() && self.tokens.is_file()
    }
}

/// Explicit Whisper model files (sherpa-onnx encoder/decoder layout).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokens: PathBuf,
}

impl WhisperModelPaths {
    /// Probe `dir` for the known Whisper file layout.
    pub fn discover(dir: &Path) -> Option<Self> {
        Some(Self {
            encoder: whisper_encoder_path(dir)?,
            decoder: whisper_decoder_path(dir)?,
            tokens: whisper_tokens_path(dir)?,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.encoder.is_file() && self.decoder.is_file() && self.tokens.is_file()
    }
}

/// Explicit offline Paraformer model files (`model.onnx` + `tokens.txt`).
///
/// Directory convention (resolved by the consumer / lumen-models):
/// `<models>/paraformer/offline/{model.onnx,tokens.txt}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParaformerOfflineModelPaths {
    /// `model.int8.onnx` / `model.onnx` / `model.quant.onnx`.
    pub model: PathBuf,
    /// `tokens.txt`.
    pub tokens: PathBuf,
}

impl ParaformerOfflineModelPaths {
    /// Probe `dir` for the known offline Paraformer file layout.
    pub fn discover(dir: &Path) -> Option<Self> {
        Some(Self {
            model: paraformer_offline_model_path(dir)?,
            tokens: paraformer_tokens_path(dir)?,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.model.is_file() && self.tokens.is_file()
    }
}

/// Explicit streaming Paraformer model files (encoder/decoder + `tokens.txt`).
///
/// Directory convention (resolved by the consumer / lumen-models):
/// `<models>/paraformer/streaming/{encoder.onnx,decoder.onnx,tokens.txt}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParaformerStreamingModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokens: PathBuf,
}

impl ParaformerStreamingModelPaths {
    /// Probe `dir` for the known streaming Paraformer file layout.
    pub fn discover(dir: &Path) -> Option<Self> {
        Some(Self {
            encoder: matching_file(dir, "encoder", ".onnx")?,
            decoder: matching_file(dir, "decoder", ".onnx")?,
            tokens: paraformer_tokens_path(dir)?,
        })
    }

    pub fn is_ready(&self) -> bool {
        self.encoder.is_file() && self.decoder.is_file() && self.tokens.is_file()
    }
}

pub fn paraformer_offline_ready(dir: &Path) -> bool {
    paraformer_offline_model_path(dir).is_some() && paraformer_tokens_path(dir).is_some()
}

pub fn paraformer_streaming_ready(dir: &Path) -> bool {
    matching_file(dir, "encoder", ".onnx").is_some()
        && matching_file(dir, "decoder", ".onnx").is_some()
        && paraformer_tokens_path(dir).is_some()
}

pub fn paraformer_offline_model_path(dir: &Path) -> Option<PathBuf> {
    for name in [
        "model.int8.onnx",
        "model.onnx",
        "model.quant.onnx",
        "paraformer.onnx",
    ] {
        let path = dir.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    // Fall back to any *paraformer*.onnx that is not an encoder/decoder split.
    matching_file(dir, "paraformer", ".onnx")
}

pub fn paraformer_tokens_path(dir: &Path) -> Option<PathBuf> {
    let path = dir.join("tokens.txt");
    path.is_file().then_some(path)
}

pub fn sensevoice_ready(dir: &Path) -> bool {
    sensevoice_model_path(dir).is_some() && sensevoice_tokens_path(dir).is_some()
}

pub fn whisper_ready(dir: &Path) -> bool {
    whisper_encoder_path(dir).is_some()
        && whisper_decoder_path(dir).is_some()
        && whisper_tokens_path(dir).is_some()
}

/// Qwen3-ASR MLX snapshot readiness (config + weights + tokenizer files).
pub fn qwen_ready(dir: &Path) -> bool {
    dir.join("config.json").is_file()
        && (dir.join("model.safetensors").is_file() || qwen_sharded_weights_ready(dir))
        && dir.join("vocab.json").is_file()
        && dir.join("merges.txt").is_file()
}

fn qwen_sharded_weights_ready(dir: &Path) -> bool {
    let Ok(contents) = std::fs::read(dir.join("model.safetensors.index.json")) else {
        return false;
    };
    let Ok(index) = serde_json::from_slice::<serde_json::Value>(&contents) else {
        return false;
    };
    let Some(weight_map) = index.get("weight_map").and_then(|value| value.as_object()) else {
        return false;
    };
    let shards: HashSet<&str> = weight_map
        .values()
        .filter_map(|value| value.as_str())
        .collect();
    !shards.is_empty()
        && shards.iter().all(|shard| {
            let path = Path::new(shard);
            path.components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
                && dir.join(path).is_file()
        })
}

pub fn sensevoice_model_path(dir: &Path) -> Option<PathBuf> {
    for name in ["model.int8.onnx", "model.onnx", "sensevoice.onnx"] {
        let path = dir.join(name);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

pub fn sensevoice_tokens_path(dir: &Path) -> Option<PathBuf> {
    let path = dir.join("tokens.txt");
    path.is_file().then_some(path)
}

pub fn whisper_encoder_path(dir: &Path) -> Option<PathBuf> {
    matching_file(dir, "encoder", ".onnx")
}

pub fn whisper_decoder_path(dir: &Path) -> Option<PathBuf> {
    matching_file(dir, "decoder", ".onnx")
}

pub fn whisper_tokens_path(dir: &Path) -> Option<PathBuf> {
    matching_file(dir, "tokens", ".txt").or_else(|| {
        let path = dir.join("tokens.txt");
        path.is_file().then_some(path)
    })
}

fn matching_file(dir: &Path, contains: &str, suffix: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains(contains) && name.ends_with(suffix) {
            return Some(entry.path());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("lumen-asr-engine-{name}-{nonce}"))
    }

    #[test]
    fn sensevoice_discovery_finds_int8_model() {
        let dir = temp_dir("sv");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"tokens").unwrap();

        let paths = SenseVoiceModelPaths::discover(&dir).unwrap();
        assert_eq!(paths.model, dir.join("model.int8.onnx"));
        assert!(paths.is_ready());
        assert!(sensevoice_ready(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn whisper_discovery_matches_prefixed_names() {
        let dir = temp_dir("wh");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tiny.en-encoder.onnx"), b"e").unwrap();
        std::fs::write(dir.join("tiny.en-decoder.onnx"), b"d").unwrap();
        std::fs::write(dir.join("tiny.en-tokens.txt"), b"t").unwrap();

        let paths = WhisperModelPaths::discover(&dir).unwrap();
        assert!(paths.is_ready());
        assert!(whisper_ready(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn missing_dir_is_not_ready() {
        let dir = temp_dir("missing");
        assert!(!sensevoice_ready(&dir));
        assert!(!whisper_ready(&dir));
        assert!(!qwen_ready(&dir));
        assert!(!paraformer_offline_ready(&dir));
        assert!(!paraformer_streaming_ready(&dir));
        assert!(SenseVoiceModelPaths::discover(&dir).is_none());
        assert!(ParaformerOfflineModelPaths::discover(&dir).is_none());
        assert!(ParaformerStreamingModelPaths::discover(&dir).is_none());
    }

    #[test]
    fn paraformer_offline_discovery_finds_model_and_tokens() {
        let dir = temp_dir("pf-off");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.int8.onnx"), b"m").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"t").unwrap();

        let paths = ParaformerOfflineModelPaths::discover(&dir).unwrap();
        assert_eq!(paths.model, dir.join("model.int8.onnx"));
        assert!(paths.is_ready());
        assert!(paraformer_offline_ready(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn paraformer_streaming_discovery_matches_encoder_decoder() {
        let dir = temp_dir("pf-stream");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("encoder.onnx"), b"e").unwrap();
        std::fs::write(dir.join("decoder.onnx"), b"d").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"t").unwrap();

        let paths = ParaformerStreamingModelPaths::discover(&dir).unwrap();
        assert!(paths.is_ready());
        assert!(paraformer_streaming_ready(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }
}
