//! Multi-stream streaming Paraformer integration tests.
//!
//! These require a downloaded streaming Paraformer model directory
//! (`encoder.onnx` + `decoder.onnx` + `tokens.txt`), so they are `#[ignore]`d
//! by default. Run with:
//!
//! ```sh
//! LUMEN_STREAMING_PARAFORMER_DIR=/path/to/streaming-paraformer \
//! cargo test -p lumen-asr-engine --test streaming_multi -- --ignored
//! ```

use lumen_asr_engine::{StreamingAsrEngine, StreamingRecognizer};

fn model_dir() -> String {
    std::env::var("LUMEN_STREAMING_PARAFORMER_DIR")
        .expect("set LUMEN_STREAMING_PARAFORMER_DIR to run this test")
}

/// Quiet 440 Hz tone; we assert decoding health across streams, not the text.
fn tone(seconds: f32) -> Vec<f32> {
    let n = (16_000.0 * seconds) as usize;
    (0..n)
        .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.01)
        .collect()
}

/// One recognizer (one model load), two independent streams decoded
/// per-stream — the mic + system-audio shape used by dual-track live
/// transcription.
#[test]
#[ignore = "requires streaming Paraformer model (set LUMEN_STREAMING_PARAFORMER_DIR)"]
fn two_streams_share_one_recognizer() {
    let recognizer = StreamingRecognizer::from_dir(model_dir()).expect("load recognizer");

    let mut mic = recognizer.new_stream();
    let mut system = recognizer.new_stream();

    let chunk = tone(0.2);
    for _ in 0..5 {
        mic.accept_waveform(&chunk, 16_000);
        system.accept_waveform(&chunk, 16_000);
        mic.decode();
        system.decode();
    }
    mic.input_finished();
    system.input_finished();
    mic.decode();
    system.decode();

    // Results must be independently retrievable; near-silence usually decodes
    // to empty text, so only protocol health is asserted.
    let _ = (mic.result(), system.result());
    let _ = (mic.is_endpoint(), system.is_endpoint());
    mic.reset();
    system.reset();
}

/// Same shape via the batch decode path.
#[test]
#[ignore = "requires streaming Paraformer model (set LUMEN_STREAMING_PARAFORMER_DIR)"]
fn decode_batch_drains_both_streams() {
    let recognizer = StreamingRecognizer::from_dir(model_dir()).expect("load recognizer");

    let mut mic = recognizer.new_stream();
    let mut system = recognizer.new_stream();

    let chunk = tone(1.0);
    mic.accept_waveform(&chunk, 16_000);
    system.accept_waveform(&chunk, 16_000);
    recognizer.decode_batch(&mut [&mut mic, &mut system]);

    let _ = (mic.result(), system.result());
}

/// A `StreamingStream` keeps the shared model alive after the recognizer
/// handle is dropped, and still works as a `Box<dyn StreamingAsrEngine>`.
#[test]
#[ignore = "requires streaming Paraformer model (set LUMEN_STREAMING_PARAFORMER_DIR)"]
fn stream_outlives_recognizer_handle() {
    let recognizer = StreamingRecognizer::from_dir(model_dir()).expect("load recognizer");
    let mut engine: Box<dyn StreamingAsrEngine> = Box::new(recognizer.new_stream());
    drop(recognizer);

    engine.accept_waveform(&tone(0.5), 16_000);
    engine.decode();
    let _ = engine.result();
}
