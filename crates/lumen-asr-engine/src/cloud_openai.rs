//! OpenAI-compatible batch audio transcription
//! (`POST {base_url}/audio/transcriptions`, multipart WAV).
//!
//! Merged from lumen-asr `cloud_openai.rs` and lumen-navi `openai_http.rs`
//! (navi added `max_audio_bytes`, `engine_label`, and error-body reporting).

use crate::audio::samples_to_wav_mono_i16;
use crate::{AsrEngine, AsrEngineId, AsrError, AsrRequest, AsrResult};
use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct OpenAiAudioConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub timeout: Duration,
    /// Optional ISO-639-1 language hint (e.g. `zh`, `en`). A per-request
    /// [`AsrRequest::language_hint`] wins over this.
    pub language: Option<String>,
    pub max_audio_bytes: usize,
    /// Engine label written into results (e.g. `openai_audio`, `qwen_asr`).
    pub engine_label: String,
    /// Optional prompt hint to guide model style or vocabulary context.
    pub prompt: Option<String>,
}

impl Default for OpenAiAudioConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-transcribe".into(),
            timeout: Duration::from_secs(120),
            language: None,
            max_audio_bytes: 8 * 1024 * 1024,
            engine_label: "openai_audio".into(),
            prompt: None,
        }
    }
}

pub struct OpenAiAudioAsr {
    client: reqwest::Client,
    config: OpenAiAudioConfig,
}

impl OpenAiAudioAsr {
    pub fn new(config: OpenAiAudioConfig) -> Result<Self, AsrError> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| AsrError::Inference(format!("http client: {e}")))?;
        Ok(Self { client, config })
    }
}

#[async_trait]
impl AsrEngine for OpenAiAudioAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::OpenAiAudio
    }

    fn is_supported(&self) -> bool {
        !self.config.base_url.trim().is_empty() && !self.config.model.trim().is_empty()
    }

    fn max_audio_bytes(&self) -> Option<usize> {
        Some(self.config.max_audio_bytes)
    }

    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError> {
        if req.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }
        if !self.is_supported() {
            return Err(AsrError::Unsupported(
                "openai_audio base_url/model not configured".into(),
            ));
        }
        let wav = samples_to_wav_mono_i16(&req.samples, req.sample_rate);
        if wav.len() > self.config.max_audio_bytes {
            return Err(AsrError::AudioTooLarge {
                actual: wav.len(),
                max: self.config.max_audio_bytes,
            });
        }
        let base = self.config.base_url.trim_end_matches('/');
        let url = format!("{base}/audio/transcriptions");

        let part = Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| AsrError::Inference(e.to_string()))?;
        let mut form = Form::new()
            .part("file", part)
            .text("model", self.config.model.clone());

        let is_diarize = self.config.model.contains("diarize");
        if is_diarize {
            form = form.text("response_format", "diarized_json");
            form = form.text("chunking_strategy", "auto");
        } else {
            let prompt_text = if !req.hotwords.is_empty() {
                Some(req.hotwords.join(", "))
            } else {
                self.config.prompt.clone()
            };
            if let Some(p) = prompt_text.filter(|s| !s.trim().is_empty()) {
                form = form.text("prompt", p);
            }
            if !req.hotwords.is_empty() && self.config.model.contains("gpt-transcribe") {
                form = form.text("keywords", req.hotwords.join(", "));
            }
        }

        let lang = req
            .language_hint
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| self.config.language.clone().filter(|s| !s.is_empty()));
        if let Some(lang) = &lang {
            form = form.text("language", lang.clone());
        }

        let mut builder = self.client.post(&url).multipart(form);
        if !self.config.api_key.is_empty() {
            builder = builder.bearer_auth(&self.config.api_key);
        }

        let resp = builder
            .send()
            .await
            .map_err(|e| AsrError::Inference(format!("http: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AsrError::Inference(format!(
                "provider rejected request with status {status}: {body}"
            )));
        }
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|_| AsrError::Inference("malformed provider response".into()))?;

        let text = if is_diarize {
            if let Some(segs) = v.get("segments").and_then(|s| s.as_array()) {
                let mut formatted = String::new();
                for seg in segs {
                    let spk = seg.get("speaker").and_then(|s| s.as_str()).unwrap_or("");
                    let seg_text = seg.get("text").and_then(|t| t.as_str()).unwrap_or("").trim();
                    if !seg_text.is_empty() {
                        if !formatted.is_empty() {
                            formatted.push('\n');
                        }
                        if !spk.is_empty() {
                            formatted.push_str(&format!("{spk}: {seg_text}"));
                        } else {
                            formatted.push_str(seg_text);
                        }
                    }
                }
                if !formatted.is_empty() {
                    formatted
                } else {
                    v.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string()
                }
            } else {
                v.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string()
            }
        } else {
            v.get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .trim()
                .to_string()
        };

        let mut result = AsrResult::new(text, AsrEngineId::OpenAiAudio);
        result.engine_label = self.config.engine_label.clone();
        result.language = lang;
        result.diagnostics.model = Some(self.config.model.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_to_gpt_transcribe() {
        let cfg = OpenAiAudioConfig::default();
        assert_eq!(cfg.model, "gpt-transcribe");
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
        assert!(cfg.prompt.is_none());
    }

    #[test]
    fn parse_diarized_json_segments() {
        let json: serde_json::Value = serde_json::json!({
            "text": "Hello world. How are you?",
            "segments": [
                {
                    "speaker": "A",
                    "text": "Hello world."
                },
                {
                    "speaker": "B",
                    "text": "How are you?"
                }
            ]
        });

        let segs = json.get("segments").unwrap().as_array().unwrap();
        let mut formatted = String::new();
        for seg in segs {
            let spk = seg.get("speaker").and_then(|s| s.as_str()).unwrap_or("");
            let seg_text = seg.get("text").and_then(|t| t.as_str()).unwrap_or("").trim();
            if !seg_text.is_empty() {
                if !formatted.is_empty() {
                    formatted.push('\n');
                }
                if !spk.is_empty() {
                    formatted.push_str(&format!("{spk}: {seg_text}"));
                } else {
                    formatted.push_str(seg_text);
                }
            }
        }
        assert_eq!(formatted, "A: Hello world.\nB: How are you?");
    }
}

