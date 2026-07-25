//! Qwen3-ASR via a persistent local MLX Python worker (stdin/stdout
//! JSON-lines protocol). Ported from lumen-asr, including the shadow
//! analysis (hotword rescoring) capability.
//!
//! The worker script ships with this crate (`src/qwen_worker.py`) and is
//! embedded via `include_str!`; callers may override it with
//! [`QwenAsrConfig::worker_script`] for development.

use crate::diagnostics::{AsrTokenEvidence, QwenRuntimeMetrics, QwenShadowDiagnostics};
use crate::{AsrEngine, AsrEngineId, AsrError, AsrRequest, AsrResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

/// The MLX worker script carried by this crate.
pub const PRODUCT_WORKER: &str = include_str!("qwen_worker.py");
const MAX_WORKER_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_SHADOW_TERMS: usize = 64;
const MAX_SHADOW_TERM_CHARS: usize = 64;
const MAX_SHADOW_SOURCE_CHARS: usize = 32;
const MAX_SHADOW_TOTAL_TERM_CHARS: usize = 4096;
static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize)]
pub struct QwenShadowTerm {
    pub surface: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct QwenShadowRequest {
    pub schema_version: u32,
    pub enabled: bool,
    pub terms: Vec<QwenShadowTerm>,
    pub max_spans_per_chunk: u32,
    pub beam_width: u32,
    pub beam_depth: u32,
}

impl Default for QwenShadowRequest {
    fn default() -> Self {
        Self {
            schema_version: 1,
            enabled: true,
            terms: Vec::new(),
            max_spans_per_chunk: 2,
            beam_width: 4,
            beam_depth: 4,
        }
    }
}

impl QwenShadowRequest {
    /// Apply the product's fixed compute and payload limits before serialization.
    pub fn bounded(mut self) -> Self {
        let mut total_chars = 0usize;
        let mut terms = Vec::with_capacity(self.terms.len().min(MAX_SHADOW_TERMS));
        for term in self.terms {
            if terms.len() >= MAX_SHADOW_TERMS || total_chars >= MAX_SHADOW_TOTAL_TERM_CHARS {
                break;
            }
            let surface = truncate_chars(term.surface.trim(), MAX_SHADOW_TERM_CHARS);
            if surface.is_empty() {
                continue;
            }
            let remaining = MAX_SHADOW_TOTAL_TERM_CHARS - total_chars;
            let surface = truncate_chars(&surface, remaining);
            if surface.is_empty()
                || terms
                    .iter()
                    .any(|existing: &QwenShadowTerm| existing.surface == surface)
            {
                continue;
            }
            total_chars += surface.chars().count();
            terms.push(QwenShadowTerm {
                surface,
                source: truncate_chars(term.source.trim(), MAX_SHADOW_SOURCE_CHARS),
            });
        }

        self.schema_version = 1;
        self.terms = terms;
        self.max_spans_per_chunk = self.max_spans_per_chunk.clamp(1, 2);
        self.beam_width = self.beam_width.clamp(1, 4);
        self.beam_depth = self.beam_depth.clamp(1, 4);
        self
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[derive(Debug, Clone)]
pub struct QwenAsrConfig {
    pub python_executable: PathBuf,
    /// Empty → use the embedded [`PRODUCT_WORKER`] script.
    pub worker_script: PathBuf,
    /// MLX model snapshot directory (resolved by the caller / lumen-models).
    pub model_dir: PathBuf,
    pub language: Option<String>,
    pub timeout: Duration,
    /// Test/development worker flags. Product callers leave this empty.
    pub extra_args: Vec<String>,
}

impl QwenAsrConfig {
    pub fn product(
        python_executable: impl Into<PathBuf>,
        model_dir: impl Into<PathBuf>,
        language: Option<String>,
        timeout: Duration,
    ) -> Self {
        Self {
            python_executable: python_executable.into(),
            worker_script: PathBuf::new(),
            model_dir: model_dir.into(),
            language,
            timeout,
            extra_args: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct QwenAsr {
    config: QwenAsrConfig,
    worker: Arc<Mutex<Option<QwenWorker>>>,
    active: Arc<AtomicBool>,
    lifecycle_generation: Arc<AtomicU64>,
}

impl QwenAsr {
    pub fn new(config: QwenAsrConfig) -> Self {
        Self {
            config,
            worker: Arc::new(Mutex::new(None)),
            active: Arc::new(AtomicBool::new(true)),
            lifecycle_generation: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn model_dir(&self) -> &Path {
        &self.config.model_dir
    }

    pub fn python_executable(&self) -> &Path {
        &self.config.python_executable
    }

    pub fn activate(&self) {
        self.lifecycle_generation.fetch_add(1, Ordering::SeqCst);
        self.active.store(true, Ordering::SeqCst);
    }

    /// Release the loaded model when the user switches to another ASR engine.
    ///
    /// If a request is in flight, it finishes normally and releases the model
    /// before another request can reuse the worker.
    pub fn unload(&self) -> bool {
        self.active.store(false, Ordering::SeqCst);
        self.lifecycle_generation.fetch_add(1, Ordering::SeqCst);
        let Ok(mut guard) = self.worker.try_lock() else {
            return true;
        };
        if let Some(worker) = guard.take() {
            schedule_worker_stop(worker);
        }
        true
    }

    async fn start_worker(&self) -> Result<QwenWorker, AsrError> {
        let mut command = Command::new(&self.config.python_executable);
        command.arg("-u");
        if self.config.worker_script.as_os_str().is_empty() {
            command.arg("-c").arg(PRODUCT_WORKER);
        } else {
            command.arg(&self.config.worker_script);
        }
        command.arg("--model").arg(&self.config.model_dir);
        if let Some(language) = self
            .config
            .language
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            command.arg("--language").arg(language);
        }
        command.args(&self.config.extra_args);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command
            .spawn()
            .map_err(|error| AsrError::NotConfigured(format!("Qwen worker: {error}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AsrError::Inference("Qwen worker stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AsrError::Inference("Qwen worker stdout unavailable".into()))?;
        Ok(QwenWorker {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    async fn transcribe_path(
        &self,
        generation: u64,
        request_id: u64,
        path: &Path,
        shadow: Option<QwenShadowRequest>,
    ) -> Result<(WorkerResponse, bool), AsrError> {
        let result = self
            .transcribe_path_inner(generation, request_id, path, shadow)
            .await;
        if self.lifecycle_generation.load(Ordering::SeqCst) != generation
            && !self.active.load(Ordering::SeqCst)
        {
            let mut guard = self.worker.lock().await;
            if let Some(mut worker) = guard.take() {
                let _ = worker.child.kill().await;
            }
        }
        result
    }

    async fn transcribe_path_inner(
        &self,
        generation: u64,
        request_id: u64,
        path: &Path,
        shadow: Option<QwenShadowRequest>,
    ) -> Result<(WorkerResponse, bool), AsrError> {
        let mut guard = self.worker.lock().await;
        if self.lifecycle_generation.load(Ordering::SeqCst) != generation
            || !self.active.load(Ordering::SeqCst)
        {
            return Err(AsrError::NotConfigured(
                "Qwen engine was deselected before transcription started".into(),
            ));
        }
        let worker_reused = guard.is_some();
        if !worker_reused {
            *guard = Some(self.start_worker().await?);
        }
        let worker = guard.as_mut().expect("worker initialized");
        let request = WorkerRequest {
            id: request_id,
            audio_path: path.display().to_string(),
            shadow: shadow.map(QwenShadowRequest::bounded),
        };
        let mut encoded = serde_json::to_vec(&request)
            .map_err(|error| AsrError::Inference(format!("encode Qwen request: {error}")))?;
        encoded.push(b'\n');

        let exchange = async {
            worker.stdin.write_all(&encoded).await?;
            worker.stdin.flush().await?;
            let mut line = Vec::new();
            let bytes = (&mut worker.stdout)
                .take((MAX_WORKER_RESPONSE_BYTES + 1) as u64)
                .read_until(b'\n', &mut line)
                .await?;
            if bytes == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Qwen worker exited",
                ));
            }
            if bytes > MAX_WORKER_RESPONSE_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Qwen worker response exceeded 1 MiB",
                ));
            }
            Ok::<Vec<u8>, std::io::Error>(line)
        };

        let line = match tokio::time::timeout(self.config.timeout, exchange).await {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => {
                let _ = worker.child.kill().await;
                *guard = None;
                return Err(AsrError::Inference(format!("Qwen worker I/O: {error}")));
            }
            Err(_) => {
                let _ = worker.child.kill().await;
                *guard = None;
                return Err(AsrError::Inference(format!(
                    "Qwen worker timed out after {}s",
                    self.config.timeout.as_secs()
                )));
            }
        };
        let response: WorkerResponse = match serde_json::from_slice(&line) {
            Ok(response) => response,
            Err(error) => {
                if let Some(worker) = guard.as_mut() {
                    let _ = worker.child.kill().await;
                }
                *guard = None;
                return Err(AsrError::Inference(format!(
                    "invalid Qwen response: {error}"
                )));
            }
        };
        if response.id != request_id {
            if let Some(worker) = guard.as_mut() {
                let _ = worker.child.kill().await;
            }
            *guard = None;
            return Err(AsrError::Inference(format!(
                "Qwen response id mismatch: expected {request_id}, got {}",
                response.id
            )));
        }
        if let Some(error) = response.error.as_deref().filter(|value| !value.is_empty()) {
            if let Some(worker) = guard.as_mut() {
                let _ = worker.child.kill().await;
            }
            *guard = None;
            return Err(AsrError::Inference(error.to_owned()));
        }
        Ok((response, worker_reused))
    }

    pub async fn transcribe_with_shadow(
        &self,
        req: AsrRequest,
        shadow: Option<QwenShadowRequest>,
    ) -> Result<AsrResult, AsrError> {
        if req.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }
        let generation = self.lifecycle_generation.load(Ordering::SeqCst);
        if !self.active.load(Ordering::SeqCst) {
            return Err(AsrError::NotConfigured("Qwen engine is not active".into()));
        }
        let request_id = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let audio_file = tokio::task::spawn_blocking(move || {
            let mut audio_file = tempfile::Builder::new()
                .prefix("lumen-qwen-")
                .suffix(".wav")
                .tempfile()
                .map_err(|error| AsrError::Inference(format!("create Qwen audio: {error}")))?;
            crate::audio::write_wav_mono_i16(&mut audio_file, &req.samples, req.sample_rate)
                .and_then(|_| audio_file.flush())
                .map_err(|error| AsrError::Inference(format!("write Qwen audio: {error}")))?;
            Ok::<_, AsrError>(audio_file)
        })
        .await
        .map_err(|error| AsrError::Inference(format!("prepare Qwen audio task: {error}")))??;
        let (response, worker_reused) = self
            .transcribe_path(generation, request_id, audio_file.path(), shadow)
            .await?;
        let (model, model_revision) = crate::model_identity_from_path(&self.config.model_dir);
        let WorkerResponse {
            text,
            language,
            token_evidence,
            qwen_metrics,
            qwen_shadow,
            ..
        } = response;
        let mut result = AsrResult::new(text.unwrap_or_default(), self.id());
        result.language = language;
        result.diagnostics.worker_reused = Some(worker_reused);
        result.diagnostics.model = model;
        result.diagnostics.model_revision = model_revision;
        result.diagnostics.token_evidence = token_evidence;
        result.diagnostics.qwen = qwen_metrics;
        result.diagnostics.qwen_shadow = qwen_shadow;
        Ok(result)
    }
}

#[async_trait]
impl AsrEngine for QwenAsr {
    fn id(&self) -> AsrEngineId {
        AsrEngineId::Qwen3Asr
    }

    async fn transcribe(&self, req: AsrRequest) -> Result<AsrResult, AsrError> {
        self.transcribe_with_shadow(req, None).await
    }
}

#[derive(Serialize)]
struct WorkerRequest {
    id: u64,
    audio_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    shadow: Option<QwenShadowRequest>,
}

#[derive(Deserialize)]
struct WorkerResponse {
    id: u64,
    text: Option<String>,
    language: Option<String>,
    error: Option<String>,
    #[serde(default)]
    token_evidence: Vec<AsrTokenEvidence>,
    #[serde(default)]
    qwen_metrics: Option<QwenRuntimeMetrics>,
    #[serde(default)]
    qwen_shadow: Option<QwenShadowDiagnostics>,
}

struct QwenWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

fn schedule_worker_stop(mut worker: QwenWorker) {
    if let Ok(runtime) = tokio::runtime::Handle::try_current() {
        runtime.spawn(async move {
            let _ = worker.child.kill().await;
        });
    } else {
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match runtime {
                Ok(runtime) => runtime.block_on(async move {
                    let _ = worker.child.kill().await;
                }),
                Err(_) => {
                    let _ = worker.child.start_kill();
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_worker_script_is_present() {
        assert!(PRODUCT_WORKER.contains("argparse"));
        assert!(PRODUCT_WORKER.len() > 10_000);
    }

    #[test]
    fn shadow_request_bounded_dedupes_and_clamps() {
        let request = QwenShadowRequest {
            terms: vec![
                QwenShadowTerm {
                    surface: "  Lumen  ".into(),
                    source: "dict".into(),
                },
                QwenShadowTerm {
                    surface: "Lumen".into(),
                    source: "dict".into(),
                },
                QwenShadowTerm {
                    surface: "   ".into(),
                    source: "dict".into(),
                },
            ],
            max_spans_per_chunk: 99,
            beam_width: 0,
            beam_depth: 99,
            ..QwenShadowRequest::default()
        }
        .bounded();

        assert_eq!(request.terms.len(), 1);
        assert_eq!(request.terms[0].surface, "Lumen");
        assert_eq!(request.max_spans_per_chunk, 2);
        assert_eq!(request.beam_width, 1);
        assert_eq!(request.beam_depth, 4);
    }

    #[test]
    fn shadow_request_bounded_caps_term_count() {
        let terms = (0..200)
            .map(|i| QwenShadowTerm {
                surface: format!("term-{i}"),
                source: "dict".into(),
            })
            .collect();
        let request = QwenShadowRequest {
            terms,
            ..QwenShadowRequest::default()
        }
        .bounded();
        assert!(request.terms.len() <= 64);
    }
}
