# lumen-asr-engine

Shared ASR engine layer for Lumen products (lumen-asr, lumen-navi, future
lumen-cut). One `AsrEngine` trait over local sherpa-onnx engines
(SenseVoice / Whisper), the local Qwen3-ASR MLX Python worker (with shadow
analysis), and OpenAI-compatible HTTP transcription.

## Design rules

- **No path resolution, no downloads.** Engines receive model directories or
  explicit file paths (`SenseVoiceModelPaths`, `WhisperModelPaths`,
  `QwenAsrConfig::model_dir`) from the caller. Default-directory discovery,
  `LUMEN_MODELS_DIR` handling, legacy-root scanning, and model downloads live
  in the consumer (the `lumen-models` crate) — this crate only probes
  readiness of a directory it is given.
- **No audio capture.** `AudioCapture` (cpal microphone handling) stays in
  lumen-asr; this crate only ships pure audio helpers (WAV codec, resampling).
- **macOS Speech.framework stays consumer-side.** `EngineKind::Speech`
  represents it for config/status; `build_engine` returns `Ok(None)` so the
  consumer (lumen-navi platform layer) supplies its own engine.

## Features

| Feature | Default | Pulls in | Enables |
|---------|---------|----------|---------|
| `sherpa` | yes | `sherpa-onnx` 1.13.4 (downloads a prebuilt static lib on first build; needs network) | `SenseVoiceSherpaAsr` / `WhisperAsr` inference. Without it the types still exist but `transcribe` returns `AsrError::Unsupported` and `is_supported()` is false. |
| `cloud` | no | `reqwest` (rustls) | `OpenAiAudioAsr` / `OpenAiAudioConfig`, and the `EngineKind::OpenAiAudio` branch of `build_engine`. |
| — (always) | | `tokio` (process/io), `tempfile` | `QwenAsr` local MLX worker. The worker script `src/qwen_worker.py` is embedded via `include_str!` (`PRODUCT_WORKER`); actual inference additionally needs a local Python with `mlx` and a model snapshot at runtime. |

## Unified API sketch

```rust
#[async_trait]
pub trait AsrEngine: Send + Sync {
    fn id(&self) -> AsrEngineId;                     // runtime identity
    fn is_supported(&self) -> bool;                  // build/host capability probe
    fn max_audio_bytes(&self) -> Option<usize>;      // enforced by transcribe_wav
    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError>;      // PCM path (lumen-asr style)
    async fn transcribe_wav(&self, audio: &[u8], locale: &str) -> Result<AsrResult, AsrError>; // WAV path (lumen-navi style, default impl)
}
```

- `AsrRequest { samples, sample_rate, hotwords, language_hint }` — build with
  `AsrRequest::new(samples, rate)` + builders.
- `AsrResult { text, engine: AsrEngineId, engine_label, language, confidence, diagnostics }`
  — `engine_label` is navi's free-form transcript label, `diagnostics` is
  lumen-asr's `AsrRuntimeDiagnostics` (incl. Qwen metrics / shadow output).
- `EngineKind` (config selector): `SenseVoice | Whisper | Qwen | Speech | OpenAiAudio`.
- Qwen shadow analysis is preserved: `QwenAsr::transcribe_with_shadow(req, Some(QwenShadowRequest { .. }))`.

## Migration: lumen-asr → lumen-asr-engine

| Old (`lumen_asr::` / `lumen_core::`) | New (`lumen_asr_engine::`) |
|---|---|
| `lumen_asr::AsrEngine` | `AsrEngine` (same shape; `transcribe` unchanged, plus `is_supported` / `max_audio_bytes` / `transcribe_wav` defaults) |
| `AsrRequest { samples, sample_rate, hotwords }` | `AsrRequest::new(samples, rate).with_hotwords(v)` (new optional `language_hint` field) |
| `AsrResult { text, engine, language, diagnostics }` | Same fields, plus `engine_label` and `confidence` |
| `lumen_core::AsrEngineId` | `AsrEngineId` (adds `OpenAiAudio`, `Speech` variants; HTTP results are now `OpenAiAudio` instead of `Other`) |
| `lumen_core::{AsrRuntimeDiagnostics, AsrTokenEvidence, Qwen*}` | Re-defined here verbatim; import from `lumen_asr_engine` |
| `AsrError` | Same, plus `AudioTooLarge`, `InvalidAudio`, `Unsupported` variants |
| `SenseVoiceSherpaAsr::new(dir)` / `.with_language` / `.is_ready` / `.model_dir()` | Unchanged (`model_dir()` now returns `PathBuf`), plus `from_paths(SenseVoiceModelPaths)` / `.with_max_audio_bytes` |
| `WhisperAsr::new(dir)` | Unchanged, plus `from_paths(WhisperModelPaths)` |
| `QwenAsr`, `QwenAsrConfig`, `QwenShadowRequest`, `QwenShadowTerm`, `transcribe_with_shadow`, `activate` / `unload` | Unchanged (worker script now ships with this crate) |
| `OpenAiAudioAsr`, `OpenAiAudioConfig` | Feature `cloud`; config gains `max_audio_bytes`, `engine_label` |
| `prepare_for_asr(&CaptureResult)` | `audio::prepare_for_asr(&samples, sample_rate)` (capture stays product-side) |
| `resample_linear` | `audio::resample_linear` (identical for valid inputs; now also guards `to_hz == 0`) |
| `EngineKind::{SenseVoice, Qwen, Whisper}` + `parse` | Superset enum; all old aliases still parse to the same variants |
| `sensevoice_status()` / `whisper_status()` / `qwen_status()` | `probe_status(kind, Some(&dir))` — the caller passes the dir (resolution moved to lumen-models) |
| `sensevoice_ready` / `whisper_ready` / `qwen_ready` (per-dir probes) | Unchanged |
| `paths::{default_*_dir, lumen_models_dir, scan_model_candidates, ...}`, `ModelInstallLock`, `AudioCapture` | **Not ported** — path resolution / download / capture belong to lumen-models & product crates |

## Migration: lumen-navi → lumen-asr-engine

| Old (`lumen_asr_engine::` in navi / `lumen_platform::`) | New (`lumen_asr_engine::`) |
|---|---|
| `lumen_platform::AsrEngine::transcribe(&[u8], locale)` | `AsrEngine::transcribe_wav(&[u8], locale)` (same decode → resample → infer pipeline) |
| `lumen_platform::AsrEngine::is_supported()` | `AsrEngine::is_supported()` |
| `lumen_platform::AsrResult { text, confidence, language, engine: String }` | `AsrResult` — the string label is `engine_label`; typed id in `engine` |
| `PlatformError::Message` / `Unsupported` | `AsrError::{Inference, NotConfigured, InvalidAudio, AudioTooLarge}` / `AsrError::Unsupported` |
| `EngineKind::{SenseVoice, Whisper, Speech, OpenAiAudio}` | Same variants, plus `Qwen` (local MLX worker, new capability for navi) |
| `EngineKind::parse("qwen" \| "qwen3-asr")` → `OpenAiAudio` | **Behavior change:** now → `Qwen` (local). Cloud names `qwen_asr` / `qwen-asr` / `qwen_asr_0.8b` still → `OpenAiAudio`. Navi callers that mean "HTTP Qwen" should store `openai_audio`/`qwen_asr` |
| `EngineBuildConfig { kind, models_root, model_dir, locale, max_audio_bytes, http_* }` | Same minus `models_root`; `model_dir` is now **required** for local engines (resolve it first via lumen-models). Adds `qwen_python`, `qwen_timeout_ms` |
| `build_engine(&cfg) -> Result<Option<Arc<dyn AsrEngine>>, String>` | Same signature & `Speech → Ok(None)` contract; no longer falls back to default model dirs |
| `engine_status` / `engine_status_with_root` | `probe_status(kind, model_dir)` — no default-dir fallback; `EngineStatus.kind` is typed `EngineKind` |
| `SenseVoiceSherpaAsr` / `WhisperAsr` builders | Unchanged |
| `OpenAiAudioAsr` / `OpenAiAudioConfig` | Unchanged (feature `cloud`); locale hint now flows via `AsrRequest::language_hint` in `transcribe_wav` |
| `wav::{decode_wav_pcm_s16le, resample_linear, prepare_for_offline_asr, samples_to_wav_mono_i16, DecodedPcm}` | `audio::` module, same signatures (errors are `AsrError` instead of `PlatformError`) |
| `sensevoice_ready` / `whisper_ready` | Unchanged |
| `download::*`, `paths::*` (default dirs, discovery, `LUMEN_MODELS_DIR`), `ModelInstallLock` | **Not ported** — moves to lumen-models |
| macOS `MacSpeechAsr` | Stays in navi's platform layer; wrap it to implement this crate's trait |

## Behavior differences vs. the originals

1. `EngineKind::parse`: bare `qwen` / `qwen3_asr` / `qwen3-asr` / `local_qwen`
   now always mean the **local** MLX worker (lumen-asr semantics). lumen-navi
   previously mapped `qwen`/`qwen3-asr` to its HTTP path.
2. No implicit model-dir fallback anywhere: `build_engine` / `probe_status`
   error (or report not-ready) instead of scanning shared/legacy roots.
3. HTTP engine results carry `AsrEngineId::OpenAiAudio` (lumen-asr used
   `Other`); serialized results change from `"other"` to `"openai_audio"`.
4. lumen-asr's HTTP engine gains navi's stricter behavior: response error body
   included in the error message, `max_audio_bytes` enforced (default 8 MiB),
   `engine_label` configurable.
5. `resample_linear` merges both variants (navi's zero-`to_hz` guard +
   non-empty output clamp); identical output for valid inputs.
6. Offline engines report `is_supported() = sherpa feature && model files
   ready` (navi semantics); lumen-asr had no such probe.
7. The `shared_model_contract_matches_cluster_v1` tests were not ported — the
   contract doc lives in the product repos; re-assert it there.

## Testing

```sh
cargo test -p lumen-asr-engine                       # unit tests (no models needed)
cargo test -p lumen-asr-engine --features cloud      # + HTTP engine compile/tests
# Qwen worker integration (needs Python + mlx + model snapshot):
LUMEN_QWEN_PYTHON=... LUMEN_QWEN_MODEL_DIR=... \
  cargo test -p lumen-asr-engine --test qwen_worker -- --ignored
```
