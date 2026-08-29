//! Qwen3-ASR sherpa-onnx engine tests.
//!
//! CI-safe cases (no model files) cover config validation, path resolution,
//! and error paths. The end-to-end decode test is `#[ignore]`d by default —
//! run it against an installed model with:
//!
//! ```sh
//! LUMEN_QWEN_MODEL_DIR=/path/to/qwen3-sherpa \
//! cargo test -p lumen-asr-engine --test qwen_sherpa -- --ignored
//! ```

use lumen_asr_engine::{AsrEngine, AsrRequest, QwenAsr, QwenAsrConfig};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn config(model_dir: impl Into<PathBuf>) -> QwenAsrConfig {
    QwenAsrConfig::product(model_dir, Some("zh".into()), Duration::from_secs(300))
}

fn temp_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("lumen-asr-engine-qwen-{name}-{nonce}"))
}

/// Write the installed sherpa layout (`conv_frontend.onnx`,
/// `encoder/decoder.int8.onnx`, `tokenizer/*`) with placeholder bytes.
fn write_fake_model(dir: &Path) {
    std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
    std::fs::write(dir.join("conv_frontend.onnx"), b"cf").unwrap();
    std::fs::write(dir.join("encoder.int8.onnx"), b"e").unwrap();
    std::fs::write(dir.join("decoder.int8.onnx"), b"d").unwrap();
    std::fs::write(dir.join("tokenizer/vocab.json"), b"{}").unwrap();
    std::fs::write(dir.join("tokenizer/merges.txt"), b"m").unwrap();
    std::fs::write(dir.join("tokenizer/tokenizer_config.json"), b"{}").unwrap();
}

#[test]
fn readiness_tracks_the_sherpa_layout() {
    let dir = temp_dir("ready");
    let engine = QwenAsr::new(config(&dir));
    assert!(!engine.is_ready());
    write_fake_model(&dir);
    assert!(engine.is_ready());
    assert!(engine.is_supported());
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn missing_model_is_a_config_error_not_a_crash() {
    let engine = QwenAsr::new(config("/nonexistent/qwen3-sherpa"));
    let err = engine
        .transcribe(AsrRequest::new(vec![0.0f32; 1_600], 16_000))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not configured"), "{err}");
}

/// End-to-end: transcribe one second of near-silence with the real model.
/// Asserts protocol health (a result with diagnostics), not the text.
#[tokio::test]
#[ignore = "requires an installed Qwen3-ASR sherpa model (set LUMEN_QWEN_MODEL_DIR)"]
async fn qwen_sherpa_round_trip() {
    let model_dir =
        std::env::var("LUMEN_QWEN_MODEL_DIR").expect("set LUMEN_QWEN_MODEL_DIR to run this test");
    let engine = QwenAsr::new(config(model_dir));
    assert!(engine.is_ready());

    let samples: Vec<f32> = (0..16_000)
        .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.01)
        .collect();
    let first = engine
        .transcribe(AsrRequest::new(samples.clone(), 16_000))
        .await
        .expect("first request");
    // Cold start: the recognizer was created for this request.
    assert_eq!(first.diagnostics.worker_reused, Some(false));

    // Second request on a warm engine reuses the loaded recognizer.
    let second = engine
        .transcribe(AsrRequest::new(samples, 16_000))
        .await
        .expect("second request");
    assert_eq!(second.diagnostics.worker_reused, Some(true));

    assert!(engine.unload());
    assert!(!engine.unload());
}
