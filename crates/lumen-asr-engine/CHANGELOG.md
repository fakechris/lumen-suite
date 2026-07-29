# Changelog

All notable changes to `lumen-asr-engine` are documented here.

## Unreleased

### Added

- **Offline Paraformer engine** (`ParaformerAsr`, `EngineKind::Paraformer`,
  `AsrEngineId::Paraformer`, label `paraformer`) via sherpa-onnx 1.13.4
  (existing `sherpa` feature; no new dependency).
  - **Word/token-level timestamps** surfaced on the new
    `AsrResult.words: Vec<WordTiming { word, start, end }>` field.
  - **Hotwords / contextual biasing** from `AsrRequest.hotwords`, injected
    per-request via sherpa `create_stream_with_hotwords`
    (`modified_beam_search`; `with_hotwords_score` / `with_modeling_unit`
    tunables).
  - Wired into `build_engine` and `probe_status`; `EngineKind::parse` accepts
    `paraformer` / `paraformer_offline` / `paraformer-offline` / `funasr`.
- **Streaming Paraformer engine** (`StreamingParaformerAsr`) plus a new
  object-safe `StreamingAsrEngine` trait (`accept_waveform` / `decode` /
  `result` / `partial_text` / `is_endpoint` / `reset` / `input_finished`) and
  `StreamingResult { text, is_final }`, over sherpa's `OnlineRecognizer`
  (encoder + decoder + `tokens.txt`).
- Model-path probes for both layouts: `ParaformerOfflineModelPaths`,
  `ParaformerStreamingModelPaths`, `paraformer_offline_ready`,
  `paraformer_streaming_ready` (convention
  `<models>/paraformer/{offline,streaming}/…`; resolution stays consumer-side).
- `AsrResult.words` field — optional, `#[serde(default, skip_serializing_if)]`
  so existing transcript payloads are byte-for-byte unchanged and legacy JSON
  still deserializes.

### Unchanged

- SenseVoice / Whisper / Qwen / OpenAI-HTTP engines and their behavior are
  untouched (Paraformer is purely additive).
