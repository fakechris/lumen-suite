//! Qwen MLX worker integration tests.
//!
//! These require a local Python environment with `mlx` / Qwen3-ASR deps and a
//! downloaded model snapshot, so they are `#[ignore]`d by default. Run with:
//!
//! ```sh
//! LUMEN_QWEN_PYTHON=/path/to/venv/bin/python \
//! LUMEN_QWEN_MODEL_DIR=/path/to/Qwen3-ASR-snapshot \
//! cargo test -p lumen-asr-engine --test qwen_worker -- --ignored
//! ```

use lumen_asr_engine::{AsrEngine, AsrRequest, QwenAsr, QwenAsrConfig};
use std::time::Duration;

fn env_config() -> Option<QwenAsrConfig> {
    let python = std::env::var("LUMEN_QWEN_PYTHON").ok()?;
    let model_dir = std::env::var("LUMEN_QWEN_MODEL_DIR").ok()?;
    Some(QwenAsrConfig::product(
        python,
        model_dir,
        Some("zh".into()),
        Duration::from_secs(300),
    ))
}

/// End-to-end: spawn the embedded worker, transcribe one second of near-silence.
/// We only assert protocol health (a response with diagnostics), not the text.
#[tokio::test]
#[ignore = "requires local Python + mlx + Qwen model (set LUMEN_QWEN_PYTHON / LUMEN_QWEN_MODEL_DIR)"]
async fn qwen_worker_round_trip() {
    let config =
        env_config().expect("set LUMEN_QWEN_PYTHON and LUMEN_QWEN_MODEL_DIR to run this test");
    let engine = QwenAsr::new(config);

    let samples: Vec<f32> = (0..16_000)
        .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.01)
        .collect();
    let result = engine
        .transcribe(AsrRequest::new(samples, 16_000))
        .await
        .expect("worker round trip");

    assert_eq!(result.diagnostics.worker_reused, Some(false));
    assert!(engine.unload());
}

/// Second request on a warm engine must reuse the persistent worker.
#[tokio::test]
#[ignore = "requires local Python + mlx + Qwen model (set LUMEN_QWEN_PYTHON / LUMEN_QWEN_MODEL_DIR)"]
async fn qwen_worker_is_reused_across_requests() {
    let config =
        env_config().expect("set LUMEN_QWEN_PYTHON and LUMEN_QWEN_MODEL_DIR to run this test");
    let engine = QwenAsr::new(config);
    let samples = vec![0.0f32; 8_000];

    let first = engine
        .transcribe(AsrRequest::new(samples.clone(), 16_000))
        .await
        .expect("first request");
    assert_eq!(first.diagnostics.worker_reused, Some(false));

    let second = engine
        .transcribe(AsrRequest::new(samples, 16_000))
        .await
        .expect("second request");
    assert_eq!(second.diagnostics.worker_reused, Some(true));
    assert!(engine.unload());
}
