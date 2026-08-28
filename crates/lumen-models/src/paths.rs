//! Resolve offline ASR model directories shared by all Lumen applications.
//!
//! **Shared cluster root** — all Lumen apps (asr, navi, future) install and
//! discover models under one place so users do not re-download per product:
//!
//! - macOS: `~/Library/Application Support/Lumen/models/`
//! - other: `~/.lumen/models/`
//! - override: env [`ENV_LUMEN_MODELS_DIR`] or an explicit
//!   `models_root` argument (e.g. Navi's `asr.models_root` config).
//!
//! Per-app legacy paths (`LumenAsr/models`, `LumenNavi/models`, the original
//! 闪电说 `Shandianshuo/models`, dot directories, coli caches) are still
//! **discovered** and selectable — including packages under non-canonical
//! folder names such as `sensevoice-small`; new downloads always go to the
//! shared root, and legacy models are never moved, copied, or deleted.
//!
//! [`resolve_sensevoice_dir`] / [`resolve_qwen_asr_dir`] are the unified
//! "shared root first, legacy fallback" entry points consumed by all Lumen
//! products (asr, cut, translation, navi).
//!
//! Behavior is specified by `contracts/SHARED_MODELS_CONTRACT.md` (cluster
//! contract v1); see the contract hash test in `lib.rs`.

use std::collections::HashSet;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Env var for the shared Lumen models root (cluster-wide).
pub const ENV_LUMEN_MODELS_DIR: &str = "LUMEN_MODELS_DIR";
/// Cluster-wide SenseVoice directory override.
pub const ENV_LUMEN_SENSEVOICE_DIR: &str = "LUMEN_SENSEVOICE_DIR";
/// Cluster-wide Whisper directory override.
pub const ENV_LUMEN_WHISPER_DIR: &str = "LUMEN_WHISPER_DIR";
/// Navi-era compatibility override (checked after [`ENV_LUMEN_SENSEVOICE_DIR`]).
pub const ENV_LUMEN_NAVI_SENSEVOICE_DIR: &str = "LUMEN_NAVI_SENSEVOICE_DIR";
/// Navi-era compatibility override (checked after [`ENV_LUMEN_WHISPER_DIR`]).
pub const ENV_LUMEN_NAVI_WHISPER_DIR: &str = "LUMEN_NAVI_WHISPER_DIR";

/// A discovered (or planned) model directory shown to users.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelCandidate {
    /// `"sensevoice"`, `"whisper"` or `"qwen"`.
    pub engine: String,
    pub path: PathBuf,
    pub label: String,
    pub ready: bool,
    /// `"env"`, `"lumen-shared"`, `"legacy-lumen-asr"`, `"legacy-lumen-navi"`,
    /// `"legacy-shandianshuo"`, `"coli-cache"`, `"lumen-asr"` or
    /// `"huggingface-cache"`.
    pub source: String,
}

/// Resolve the user home directory: `HOME` → `USERPROFILE` →
/// `HOMEDRIVE + HOMEPATH` → system temp dir as last resort.
pub fn user_home_dir() -> PathBuf {
    for key in ["HOME", "USERPROFILE"] {
        if let Some(path) = nonempty_env_path(key) {
            return path;
        }
    }
    match (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH")) {
        (Some(drive), Some(path)) if !drive.is_empty() && !path.is_empty() => {
            let mut home = PathBuf::from(drive);
            home.push(path);
            home
        }
        _ => std::env::temp_dir(),
    }
}

/// Read `key` as a path, ignoring unset, empty, and whitespace-only values.
fn nonempty_env_path(key: &str) -> Option<PathBuf> {
    let value = std::env::var_os(key)?;
    match value.to_str() {
        Some(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
        }
        None => (!value.is_empty()).then(|| PathBuf::from(value)),
    }
}

/// Shared models root for the Lumen app cluster.
///
/// Priority: `LUMEN_MODELS_DIR` → platform default
/// (`~/Library/Application Support/Lumen/models` on macOS, `~/.lumen/models`
/// elsewhere).
pub fn lumen_models_dir() -> PathBuf {
    lumen_models_dir_with_override(None)
}

/// Like [`lumen_models_dir`], but an explicit non-empty `override_root`
/// (typically from app config such as Navi's `asr.models_root`) wins over the
/// environment variable and the platform default.
pub fn lumen_models_dir_with_override(override_root: Option<&Path>) -> PathBuf {
    if let Some(root) = override_root.filter(|path| !path.as_os_str().is_empty()) {
        return root.to_path_buf();
    }
    if let Some(root) = nonempty_env_path(ENV_LUMEN_MODELS_DIR) {
        return root;
    }
    let home = user_home_dir();
    #[cfg(target_os = "macos")]
    {
        home.join("Library/Application Support/Lumen/models")
    }
    #[cfg(not(target_os = "macos"))]
    {
        home.join(".lumen/models")
    }
}

/// Compatibility alias for callers written before the cluster-wide path rename.
#[deprecated(note = "prefer lumen_models_dir")]
pub fn app_models_dir() -> PathBuf {
    lumen_models_dir()
}

/// Canonical install / default lookup dir for SenseVoice under the cluster root.
pub fn shared_sensevoice_dir(models_root: Option<&Path>) -> PathBuf {
    lumen_models_dir_with_override(models_root).join("sensevoice")
}

/// Canonical install / default lookup dir for Whisper under the cluster root.
pub fn shared_whisper_dir(models_root: Option<&Path>) -> PathBuf {
    lumen_models_dir_with_override(models_root).join("whisper")
}

/// Canonical install / default lookup dir for the **offline** Paraformer model
/// under the cluster root: `<models>/paraformer/offline`.
pub fn shared_paraformer_offline_dir(models_root: Option<&Path>) -> PathBuf {
    lumen_models_dir_with_override(models_root)
        .join("paraformer")
        .join("offline")
}

/// Canonical install / default lookup dir for the **streaming** Paraformer
/// model under the cluster root: `<models>/paraformer/streaming`.
pub fn shared_paraformer_streaming_dir(models_root: Option<&Path>) -> PathBuf {
    lumen_models_dir_with_override(models_root)
        .join("paraformer")
        .join("streaming")
}

/// Canonical install dir for the Silero VAD model under the cluster root:
/// `<models>/silero-vad`.
pub fn shared_silero_vad_dir(models_root: Option<&Path>) -> PathBuf {
    lumen_models_dir_with_override(models_root).join("silero-vad")
}

/// Offline Paraformer dir under the shared cluster root (`paraformer/offline`).
///
/// Paraformer has no legacy per-app layout or env override, so this is simply
/// the shared install target — ready or not.
pub fn default_paraformer_offline_dir() -> PathBuf {
    default_paraformer_offline_dir_with_root(None)
}

pub fn default_paraformer_offline_dir_with_root(models_root: Option<&Path>) -> PathBuf {
    shared_paraformer_offline_dir(models_root)
}

/// Streaming Paraformer dir under the shared cluster root
/// (`paraformer/streaming`).
pub fn default_paraformer_streaming_dir() -> PathBuf {
    default_paraformer_streaming_dir_with_root(None)
}

pub fn default_paraformer_streaming_dir_with_root(models_root: Option<&Path>) -> PathBuf {
    shared_paraformer_streaming_dir(models_root)
}

/// Silero VAD dir under the shared cluster root (`silero-vad`).
///
/// Silero VAD has no legacy per-app layout or env override, so this is simply
/// the shared install target — ready or not.
pub fn default_silero_vad_dir() -> PathBuf {
    shared_silero_vad_dir(None)
}

/// Pre-cluster per-app roots, scanned on every platform so upgrades never
/// force a re-download (contract §5).
///
/// `Shandianshuo/models` is the original 闪电说 (pre-rename) app-support root;
/// scanning it lets long-time users keep models they installed under the old
/// product name (e.g. `Shandianshuo/models/sensevoice-small`) without a
/// re-download. It is a superset of the contract's minimum list — discovery
/// only, never moved or deleted.
pub fn legacy_model_roots(home: &Path) -> Vec<PathBuf> {
    vec![
        home.join("Library/Application Support/LumenAsr/models"),
        home.join("Library/Application Support/LumenNavi/models"),
        home.join("Library/Application Support/Shandianshuo/models"),
        home.join(".lumen-asr/models"),
        home.join(".lumen-navi/models"),
    ]
}

fn legacy_source(root: &Path) -> &'static str {
    let path = root.to_string_lossy();
    if path.contains("Shandianshuo") {
        "legacy-shandianshuo"
    } else if path.contains("LumenAsr") || path.contains(".lumen-asr") {
        "legacy-lumen-asr"
    } else {
        "legacy-lumen-navi"
    }
}

/// First-level subdirectories of `root` that pass `ready` (skipping `extract`
/// temp dirs). Used to discover model packages whose directory name is not the
/// canonical engine name — e.g. legacy `sensevoice-small` under
/// `Shandianshuo/models`, or a user-renamed folder under the shared root.
fn ready_subdirs(root: &Path, ready: fn(&Path) -> bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && !entry.file_name().to_string_lossy().contains("extract")
                && ready(&path)
            {
                out.push(path);
            }
        }
    }
    out
}

/// Resolve the SenseVoice dir: env → shared cluster dir → other ready
/// candidates → shared dir as the install target (contract §3).
pub fn default_sensevoice_dir() -> PathBuf {
    default_sensevoice_dir_with_root(None)
}

pub fn default_sensevoice_dir_with_root(models_root: Option<&Path>) -> PathBuf {
    for key in [ENV_LUMEN_SENSEVOICE_DIR, ENV_LUMEN_NAVI_SENSEVOICE_DIR] {
        if let Some(path) = nonempty_env_path(key) {
            return path;
        }
    }
    let shared = shared_sensevoice_dir(models_root);
    if sensevoice_ready(&shared) {
        return shared;
    }
    for (path, _) in sensevoice_discovery_paths(models_root) {
        if path != shared && sensevoice_ready(&path) {
            return path;
        }
    }
    // Empty shared path = default download / config target (one place for all apps).
    shared
}

/// Resolve the Whisper dir: env → shared cluster dir → other ready
/// candidates → shared dir as the install target (contract §3).
pub fn default_whisper_dir() -> PathBuf {
    default_whisper_dir_with_root(None)
}

pub fn default_whisper_dir_with_root(models_root: Option<&Path>) -> PathBuf {
    for key in [ENV_LUMEN_WHISPER_DIR, ENV_LUMEN_NAVI_WHISPER_DIR] {
        if let Some(path) = nonempty_env_path(key) {
            return path;
        }
    }
    let shared = shared_whisper_dir(models_root);
    if whisper_ready(&shared) {
        return shared;
    }
    for (path, _) in whisper_discovery_paths(models_root) {
        if path != shared && whisper_ready(&path) {
            return path;
        }
    }
    shared
}

/// Resolve the Qwen3-ASR snapshot dir: app dir → huggingface cache → app dir.
pub fn default_qwen_dir() -> PathBuf {
    let app_dir = qwen_app_model_dir();
    if qwen_ready(&app_dir) {
        return app_dir;
    }
    for (path, _) in qwen_discovery_paths(None) {
        if path != app_dir && qwen_ready(&path) {
            return path;
        }
    }
    app_dir
}

/// Resolve the **first existing and ready** SenseVoice directory for any Lumen
/// product (asr dictation + meeting, cut, translation, navi) to consume, or
/// `None` when no usable model is installed anywhere.
///
/// Unlike [`default_sensevoice_dir_with_root`] (which returns the shared
/// download target even when nothing is installed yet), this returns a path
/// only when a real, ready model exists. Resolution order:
///
/// 1. Engine env overrides (`LUMEN_SENSEVOICE_DIR`, then the Navi-era
///    `LUMEN_NAVI_SENSEVOICE_DIR`) — used only if the pointed-at dir is ready.
/// 2. Shared cluster dir `lumen_models_dir()/sensevoice`.
/// 3. Any other ready first-level subdir under the shared root.
/// 4. Legacy fallbacks (read-only, never moved): `LumenAsr/models`,
///    `LumenNavi/models`, `Shandianshuo/models` (incl. `sensevoice-small`),
///    dot-dir variants, and known `.coli/models` packages.
///
/// `override_root` (e.g. Navi's `asr.models_root`) overrides the shared root.
pub fn resolve_sensevoice_dir(override_root: Option<&Path>) -> Option<PathBuf> {
    for key in [ENV_LUMEN_SENSEVOICE_DIR, ENV_LUMEN_NAVI_SENSEVOICE_DIR] {
        if let Some(path) = nonempty_env_path(key) {
            if sensevoice_ready(&path) {
                return Some(path);
            }
        }
    }
    sensevoice_discovery_paths(override_root)
        .into_iter()
        .map(|(path, _)| path)
        .find(|path| sensevoice_ready(path))
}

/// Resolve the **first existing and ready** Qwen3-ASR snapshot directory for
/// any Lumen product to consume, or `None` when none is installed.
///
/// Resolution order:
///
/// 1. Shared cluster dir `lumen_models_dir()/qwen3-asr-0.6b-8bit` and any other
///    ready first-level subdir under the shared root.
/// 2. Legacy fallbacks (read-only, never moved): `LumenAsr/models/qwen3-asr-0.6b-8bit`,
///    any ready `*qwen*` subdir under `Shandianshuo/models` / other legacy roots.
/// 3. The Hugging Face MLX snapshot cache
///    (`~/.cache/huggingface/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit`).
///
/// `override_root` overrides the shared root.
pub fn resolve_qwen_asr_dir(override_root: Option<&Path>) -> Option<PathBuf> {
    qwen_discovery_paths(override_root)
        .into_iter()
        .map(|(path, _)| path)
        .find(|path| qwen_ready(path))
}

fn sensevoice_discovery_paths(models_root: Option<&Path>) -> Vec<(PathBuf, &'static str)> {
    shared_engine_discovery_paths(
        models_root,
        "sensevoice",
        sensevoice_ready,
        &[
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        ],
    )
}

fn whisper_discovery_paths(models_root: Option<&Path>) -> Vec<(PathBuf, &'static str)> {
    shared_engine_discovery_paths(
        models_root,
        "whisper",
        whisper_ready,
        &["sherpa-onnx-whisper-tiny.en", "sherpa-onnx-whisper-base.en"],
    )
}

/// Canonical Qwen3-ASR snapshot directory name (matches the MLX model id).
const QWEN_DIR_NAME: &str = "qwen3-asr-0.6b-8bit";

/// Shared canonical dir first, then any other ready first-level subdir under
/// the shared root, then legacy roots (canonical join plus any ready subdir,
/// e.g. `sensevoice-small`), then known coli cache packages.
fn shared_engine_discovery_paths(
    models_root: Option<&Path>,
    canonical_name: &str,
    ready: fn(&Path) -> bool,
    coli_package_names: &[&'static str],
) -> Vec<(PathBuf, &'static str)> {
    let shared_root = lumen_models_dir_with_override(models_root);
    let mut paths = vec![(shared_root.join(canonical_name), "lumen-shared")];
    for path in ready_subdirs(&shared_root, ready) {
        if path.file_name() != Some(OsStr::new(canonical_name)) {
            paths.push((path, "lumen-shared"));
        }
    }
    let home = user_home_dir();
    for root in legacy_model_roots(&home) {
        let source = legacy_source(&root);
        paths.push((root.join(canonical_name), source));
        for path in ready_subdirs(&root, ready) {
            if path.file_name() != Some(OsStr::new(canonical_name)) {
                paths.push((path, source));
            }
        }
    }
    for name in coli_package_names {
        paths.push((home.join(".coli/models").join(name), "coli-cache"));
    }
    paths
}

/// Shared root (canonical + ready subdirs) first, then legacy app dirs
/// (`LumenAsr` and any ready `*qwen*` subdir under `Shandianshuo` / other
/// legacy roots), then the Hugging Face MLX snapshot cache.
fn qwen_discovery_paths(models_root: Option<&Path>) -> Vec<(PathBuf, &'static str)> {
    let shared_root = lumen_models_dir_with_override(models_root);
    let mut paths = vec![(shared_root.join(QWEN_DIR_NAME), "lumen-shared")];
    for path in ready_subdirs(&shared_root, qwen_ready) {
        if path.file_name() != Some(OsStr::new(QWEN_DIR_NAME)) {
            paths.push((path, "lumen-shared"));
        }
    }
    let home = user_home_dir();
    // Canonical legacy app dir (LumenAsr / .lumen-asr).
    paths.push((qwen_app_model_dir(), "lumen-asr"));
    // Any ready qwen package sitting under a legacy root under a non-canonical
    // name (e.g. `Shandianshuo/models/qwen3-asr-...`).
    for root in legacy_model_roots(&home) {
        let source = legacy_source(&root);
        for path in ready_subdirs(&root, qwen_ready) {
            paths.push((path, source));
        }
    }
    let snapshots = home
        .join(".cache/huggingface/hub")
        .join("models--mlx-community--Qwen3-ASR-0.6B-8bit")
        .join("snapshots");
    if let Ok(entries) = std::fs::read_dir(snapshots) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                paths.push((path, "huggingface-cache"));
            }
        }
    }
    paths
}

fn qwen_app_model_dir() -> PathBuf {
    let home = user_home_dir();
    #[cfg(target_os = "macos")]
    {
        home.join("Library/Application Support/LumenAsr/models")
            .join(QWEN_DIR_NAME)
    }
    #[cfg(not(target_os = "macos"))]
    {
        home.join(".lumen-asr/models").join(QWEN_DIR_NAME)
    }
}

/// Scan known locations for ready (or placeholder shared) model dirs.
///
/// Users can pick any ready path; new downloads land under the shared root.
/// Covers all engines — filter on [`ModelCandidate::engine`] if your app only
/// supports a subset.
pub fn scan_model_candidates() -> Vec<ModelCandidate> {
    scan_model_candidates_with_root(None)
}

pub fn scan_model_candidates_with_root(models_root: Option<&Path>) -> Vec<ModelCandidate> {
    let mut out = Vec::new();
    for key in [ENV_LUMEN_SENSEVOICE_DIR, ENV_LUMEN_NAVI_SENSEVOICE_DIR] {
        if let Some(path) = nonempty_env_path(key) {
            push_candidate(&mut out, "sensevoice", path, "env", false);
        }
    }
    for key in [ENV_LUMEN_WHISPER_DIR, ENV_LUMEN_NAVI_WHISPER_DIR] {
        if let Some(path) = nonempty_env_path(key) {
            push_candidate(&mut out, "whisper", path, "env", false);
        }
    }
    let shared_sensevoice = shared_sensevoice_dir(models_root);
    for (path, source) in sensevoice_discovery_paths(models_root) {
        let install_target = path == shared_sensevoice;
        push_candidate(&mut out, "sensevoice", path, source, install_target);
    }
    let shared_whisper = shared_whisper_dir(models_root);
    for (path, source) in whisper_discovery_paths(models_root) {
        let install_target = path == shared_whisper;
        push_candidate(&mut out, "whisper", path, source, install_target);
    }
    for (path, source) in qwen_discovery_paths(models_root) {
        push_candidate(&mut out, "qwen", path, source, false);
    }
    let mut seen = HashSet::new();
    out.retain(|candidate| seen.insert((candidate.engine.clone(), candidate.path.clone())));
    out.sort_by(|left, right| {
        candidate_score(right)
            .cmp(&candidate_score(left))
            .then_with(|| left.path.cmp(&right.path))
    });
    out
}

fn push_candidate(
    candidates: &mut Vec<ModelCandidate>,
    engine: &str,
    path: PathBuf,
    source: &str,
    install_target: bool,
) {
    let ready = match engine {
        "sensevoice" => sensevoice_ready(&path),
        "whisper" => whisper_ready(&path),
        "qwen" => qwen_ready(&path),
        _ => false,
    };
    if !ready && !install_target {
        return;
    }
    let name = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    let label = if ready {
        format!("{name} · {source}")
    } else {
        format!("{engine} · {source} — 下载目标（全 Lumen 应用共享）")
    };
    candidates.push(ModelCandidate {
        engine: engine.into(),
        path,
        label,
        ready,
        source: source.into(),
    });
}

fn candidate_score(candidate: &ModelCandidate) -> i32 {
    i32::from(candidate.ready) * 10
        + i32::from(candidate.source == "lumen-shared") * 5
        + i32::from(candidate.source == "env") * 8
}

/// SenseVoice readiness: one of the known model files plus `tokens.txt`
/// (contract §4).
pub fn sensevoice_ready(dir: &Path) -> bool {
    sensevoice_model_path(dir).is_some() && sensevoice_tokens_path(dir).is_some()
}

/// Whisper readiness: `*encoder*.onnx` + `*decoder*.onnx` + `*tokens*.txt`
/// (contract §4).
pub fn whisper_ready(dir: &Path) -> bool {
    whisper_encoder_path(dir).is_some()
        && whisper_decoder_path(dir).is_some()
        && whisper_tokens_path(dir).is_some()
}

/// Qwen3-ASR MLX snapshot readiness: config + (single or sharded) weights +
/// tokenizer assets.
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

/// Offline Paraformer readiness: one known model file plus `tokens.txt`.
pub fn paraformer_offline_ready(dir: &Path) -> bool {
    paraformer_offline_model_path(dir).is_some() && paraformer_tokens_path(dir).is_some()
}

/// Streaming Paraformer readiness: `*encoder*.onnx` + `*decoder*.onnx` +
/// `tokens.txt`.
pub fn paraformer_streaming_ready(dir: &Path) -> bool {
    paraformer_encoder_path(dir).is_some()
        && paraformer_decoder_path(dir).is_some()
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

pub fn paraformer_encoder_path(dir: &Path) -> Option<PathBuf> {
    matching_file(dir, "encoder", ".onnx")
}

pub fn paraformer_decoder_path(dir: &Path) -> Option<PathBuf> {
    matching_file(dir, "decoder", ".onnx")
}

pub fn paraformer_tokens_path(dir: &Path) -> Option<PathBuf> {
    let path = dir.join("tokens.txt");
    path.is_file().then_some(path)
}

/// Silero VAD readiness: the single `silero_vad.onnx` model file.
pub fn silero_vad_ready(dir: &Path) -> bool {
    silero_vad_model_path(dir).is_some()
}

pub fn silero_vad_model_path(dir: &Path) -> Option<PathBuf> {
    let path = dir.join("silero_vad.onnx");
    path.is_file().then_some(path)
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
    use crate::test_support::{temp_dir, EnvGuard, ENV_LOCK};

    // --- readiness ---------------------------------------------------------

    #[test]
    fn not_ready_empty_dir() {
        let dir = temp_dir("empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!sensevoice_ready(&dir));
        assert!(!whisper_ready(&dir));
        assert!(!qwen_ready(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn whisper_discovery_matches_prefixed_names() {
        let dir = temp_dir("whisper-prefixed");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tiny.en-encoder.onnx"), b"e").unwrap();
        std::fs::write(dir.join("tiny.en-decoder.onnx"), b"d").unwrap();
        std::fs::write(dir.join("tiny.en-tokens.txt"), b"t").unwrap();

        assert!(whisper_ready(&dir));
        assert_eq!(
            whisper_encoder_path(&dir),
            Some(dir.join("tiny.en-encoder.onnx"))
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn qwen_ready_requires_model_config_and_tokenizer_assets() {
        let root = temp_dir("qwen-ready");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), b"{}").unwrap();
        std::fs::write(root.join("model.safetensors"), b"model").unwrap();
        assert!(!qwen_ready(&root));

        std::fs::write(root.join("tokenizer_config.json"), b"{}").unwrap();
        assert!(!qwen_ready(&root));
        std::fs::write(root.join("vocab.json"), b"{}").unwrap();
        assert!(!qwen_ready(&root));
        std::fs::write(root.join("merges.txt"), b"").unwrap();
        assert!(qwen_ready(&root));

        std::fs::remove_file(root.join("model.safetensors")).unwrap();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            br#"{
                "weight_map": {
                    "encoder": "model-00001-of-00002.safetensors",
                    "decoder": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();
        assert!(!qwen_ready(&root));
        std::fs::write(root.join("model-00001-of-00002.safetensors"), b"model").unwrap();
        assert!(!qwen_ready(&root));
        std::fs::write(root.join("model-00002-of-00002.safetensors"), b"model").unwrap();
        assert!(qwen_ready(&root));
        let _ = std::fs::remove_dir_all(root);
    }

    // --- roots and layout --------------------------------------------------

    #[test]
    fn legacy_roots_cover_macos_and_dot_directory_layouts() {
        let home = Path::new("/home/alice");

        assert_eq!(
            legacy_model_roots(home),
            vec![
                home.join("Library/Application Support/LumenAsr/models"),
                home.join("Library/Application Support/LumenNavi/models"),
                home.join("Library/Application Support/Shandianshuo/models"),
                home.join(".lumen-asr/models"),
                home.join(".lumen-navi/models"),
            ]
        );
    }

    #[test]
    fn paraformer_shared_dirs_are_under_models_root() {
        let root = temp_dir("pf-root");
        assert_eq!(
            shared_paraformer_offline_dir(Some(&root)),
            root.join("paraformer").join("offline")
        );
        assert_eq!(
            shared_paraformer_streaming_dir(Some(&root)),
            root.join("paraformer").join("streaming")
        );
        assert_eq!(
            default_paraformer_offline_dir_with_root(Some(&root)),
            root.join("paraformer").join("offline")
        );
        assert_eq!(
            default_paraformer_streaming_dir_with_root(Some(&root)),
            root.join("paraformer").join("streaming")
        );
    }

    #[test]
    fn paraformer_offline_ready_requires_model_and_tokens() {
        let dir = temp_dir("pf-offline-ready");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!paraformer_offline_ready(&dir));
        std::fs::write(dir.join("model.int8.onnx"), b"m").unwrap();
        assert!(!paraformer_offline_ready(&dir));
        std::fs::write(dir.join("tokens.txt"), b"t").unwrap();
        assert!(paraformer_offline_ready(&dir));
        assert_eq!(
            paraformer_offline_model_path(&dir),
            Some(dir.join("model.int8.onnx"))
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn silero_vad_ready_requires_model_file() {
        let dir = temp_dir("silero-vad-ready");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!silero_vad_ready(&dir));
        std::fs::write(dir.join("silero_vad.onnx"), b"model").unwrap();
        assert!(silero_vad_ready(&dir));
        assert_eq!(
            silero_vad_model_path(&dir),
            Some(dir.join("silero_vad.onnx"))
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn silero_vad_shared_dir_is_under_models_root() {
        let root = temp_dir("silero-vad-root");
        assert_eq!(shared_silero_vad_dir(Some(&root)), root.join("silero-vad"));
    }

    #[test]
    fn paraformer_streaming_ready_requires_encoder_decoder_tokens() {
        let dir = temp_dir("pf-streaming-ready");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("encoder.int8.onnx"), b"e").unwrap();
        std::fs::write(dir.join("decoder.int8.onnx"), b"d").unwrap();
        assert!(!paraformer_streaming_ready(&dir));
        std::fs::write(dir.join("tokens.txt"), b"t").unwrap();
        assert!(paraformer_streaming_ready(&dir));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn override_models_root_install_path() {
        let root = temp_dir("models-root-override");
        assert_eq!(lumen_models_dir_with_override(Some(&root)), root);
        assert_eq!(shared_sensevoice_dir(Some(&root)), root.join("sensevoice"));
        // Download target is always the shared subdir under models_root,
        // even when legacy caches exist on the machine.
        assert_eq!(shared_whisper_dir(Some(&root)), root.join("whisper"));
    }

    // --- discovery / candidates -------------------------------------------

    #[test]
    fn shared_root_discovers_ready_model_in_custom_subdir() {
        let root = temp_dir("shared-custom");
        let custom = root.join("sherpa-sensevoice-custom");
        std::fs::create_dir_all(&custom).unwrap();
        std::fs::write(custom.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(custom.join("tokens.txt"), b"tokens").unwrap();

        let candidates = scan_model_candidates_with_root(Some(&root));

        assert!(candidates.iter().any(|candidate| {
            candidate.engine == "sensevoice"
                && candidate.path == custom
                && candidate.source == "lumen-shared"
                && candidate.ready
        }));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn missing_shared_targets_are_still_listed_for_installation() {
        let root = temp_dir("shared-placeholder");
        let candidates = scan_model_candidates_with_root(Some(&root));

        assert!(candidates.iter().any(|candidate| {
            candidate.engine == "sensevoice"
                && candidate.path == root.join("sensevoice")
                && !candidate.ready
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.engine == "whisper"
                && candidate.path == root.join("whisper")
                && !candidate.ready
        }));
    }

    #[test]
    fn model_candidate_serializes_path_as_string() {
        let candidate = ModelCandidate {
            engine: "sensevoice".into(),
            path: PathBuf::from("/tmp/models/sensevoice"),
            label: "sensevoice · lumen-shared".into(),
            ready: true,
            source: "lumen-shared".into(),
        };
        let json = serde_json::to_value(&candidate).unwrap();
        assert_eq!(json["path"], "/tmp/models/sensevoice");
        let back: ModelCandidate = serde_json::from_value(json).unwrap();
        assert_eq!(back, candidate);
    }

    // --- environment overrides --------------------------------------------

    #[test]
    fn models_root_env_override_and_priority() {
        let _lock = ENV_LOCK.lock().unwrap();
        let root = temp_dir("env-root");
        let _guard = EnvGuard::set(ENV_LUMEN_MODELS_DIR, root.as_os_str());

        // Env wins over the platform default…
        assert_eq!(lumen_models_dir(), root);
        assert_eq!(shared_sensevoice_dir(None), root.join("sensevoice"));
        // …but an explicit override (app config) wins over env.
        let explicit = temp_dir("env-root-explicit");
        assert_eq!(lumen_models_dir_with_override(Some(&explicit)), explicit);
    }

    #[test]
    fn blank_models_root_env_is_ignored() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _guard = EnvGuard::set(ENV_LUMEN_MODELS_DIR, "   ");
        let resolved = lumen_models_dir();
        assert_ne!(resolved, PathBuf::from("   "));
        assert!(resolved.to_string_lossy().contains("models"));
    }

    #[test]
    fn sensevoice_env_override_beats_ready_shared_dir() {
        let _lock = ENV_LOCK.lock().unwrap();
        let root = temp_dir("env-sv-root");
        let shared = root.join("sensevoice");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(shared.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(shared.join("tokens.txt"), b"tokens").unwrap();

        let forced = temp_dir("env-sv-forced");
        let _guard = EnvGuard::set(ENV_LUMEN_SENSEVOICE_DIR, forced.as_os_str());
        assert_eq!(default_sensevoice_dir_with_root(Some(&root)), forced);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn navi_compat_env_vars_are_honored_after_cluster_vars() {
        let _lock = ENV_LOCK.lock().unwrap();
        let navi_dir = temp_dir("env-navi-sv");
        let _navi = EnvGuard::set(ENV_LUMEN_NAVI_SENSEVOICE_DIR, navi_dir.as_os_str());
        let _cleared = EnvGuard::unset(ENV_LUMEN_SENSEVOICE_DIR);
        let root = temp_dir("env-navi-root");
        assert_eq!(default_sensevoice_dir_with_root(Some(&root)), navi_dir);

        // The cluster-wide var still wins when both are set.
        let cluster_dir = temp_dir("env-cluster-sv");
        let _cluster = EnvGuard::set(ENV_LUMEN_SENSEVOICE_DIR, cluster_dir.as_os_str());
        assert_eq!(default_sensevoice_dir_with_root(Some(&root)), cluster_dir);
    }

    #[test]
    fn whisper_env_override_beats_shared_dir() {
        let _lock = ENV_LOCK.lock().unwrap();
        let forced = temp_dir("env-wh-forced");
        let _guard = EnvGuard::set(ENV_LUMEN_WHISPER_DIR, forced.as_os_str());
        let root = temp_dir("env-wh-root");
        assert_eq!(default_whisper_dir_with_root(Some(&root)), forced);
    }

    // --- legacy fallback ---------------------------------------------------

    #[test]
    fn legacy_dot_directory_model_is_used_when_shared_root_is_empty() {
        let _lock = ENV_LOCK.lock().unwrap();
        let fake_home = temp_dir("legacy-home");
        let legacy = fake_home.join(".lumen-navi/models/sensevoice");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("model.onnx"), b"model").unwrap();
        std::fs::write(legacy.join("tokens.txt"), b"tokens").unwrap();

        let _home = EnvGuard::set("HOME", fake_home.as_os_str());
        let _sv = EnvGuard::unset(ENV_LUMEN_SENSEVOICE_DIR);
        let _navi_sv = EnvGuard::unset(ENV_LUMEN_NAVI_SENSEVOICE_DIR);

        let empty_root = temp_dir("legacy-empty-root");
        assert_eq!(default_sensevoice_dir_with_root(Some(&empty_root)), legacy);

        let candidates = scan_model_candidates_with_root(Some(&empty_root));
        assert!(candidates.iter().any(|candidate| {
            candidate.engine == "sensevoice"
                && candidate.path == legacy
                && candidate.source == "legacy-lumen-navi"
                && candidate.ready
        }));
        let _ = std::fs::remove_dir_all(fake_home);
    }

    #[test]
    fn ready_shared_dir_beats_legacy_models() {
        let _lock = ENV_LOCK.lock().unwrap();
        let fake_home = temp_dir("legacy-home-beaten");
        let legacy = fake_home.join(".lumen-asr/models/sensevoice");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("model.onnx"), b"model").unwrap();
        std::fs::write(legacy.join("tokens.txt"), b"tokens").unwrap();

        let _home = EnvGuard::set("HOME", fake_home.as_os_str());
        let _sv = EnvGuard::unset(ENV_LUMEN_SENSEVOICE_DIR);
        let _navi_sv = EnvGuard::unset(ENV_LUMEN_NAVI_SENSEVOICE_DIR);

        let root = temp_dir("legacy-shared-root");
        let shared = root.join("sensevoice");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(shared.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(shared.join("tokens.txt"), b"tokens").unwrap();

        assert_eq!(default_sensevoice_dir_with_root(Some(&root)), shared);
        let _ = std::fs::remove_dir_all(fake_home);
        let _ = std::fs::remove_dir_all(root);
    }

    // --- unified resolve (shared root + legacy fallback) -------------------

    /// Writes a ready SenseVoice layout into `dir`.
    fn write_sensevoice(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.int8.onnx"), b"model").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"tokens").unwrap();
    }

    /// Writes a ready Qwen3-ASR (single-weight) layout into `dir`.
    fn write_qwen(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), b"model").unwrap();
        std::fs::write(dir.join("vocab.json"), b"{}").unwrap();
        std::fs::write(dir.join("merges.txt"), b"").unwrap();
    }

    /// Clears all inputs that could leak the tester's real machine into resolve.
    fn isolate_home(fake_home: &Path) -> (EnvGuard, EnvGuard, EnvGuard, EnvGuard) {
        (
            EnvGuard::set("HOME", fake_home.as_os_str()),
            EnvGuard::unset("USERPROFILE"),
            EnvGuard::unset(ENV_LUMEN_SENSEVOICE_DIR),
            EnvGuard::unset(ENV_LUMEN_NAVI_SENSEVOICE_DIR),
        )
    }

    #[test]
    fn resolve_sensevoice_prefers_shared_root() {
        let _lock = ENV_LOCK.lock().unwrap();
        let fake_home = temp_dir("resolve-sv-shared-home");
        let _iso = isolate_home(&fake_home);

        let root = temp_dir("resolve-sv-shared-root");
        let shared = root.join("sensevoice");
        write_sensevoice(&shared);

        assert_eq!(resolve_sensevoice_dir(Some(&root)), Some(shared));
        let _ = std::fs::remove_dir_all(fake_home);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_sensevoice_falls_back_to_shandianshuo_small() {
        let _lock = ENV_LOCK.lock().unwrap();
        let fake_home = temp_dir("resolve-sv-sdss-home");
        let _iso = isolate_home(&fake_home);
        // Legacy 闪电说 layout: models live under `sensevoice-small`.
        let legacy =
            fake_home.join("Library/Application Support/Shandianshuo/models/sensevoice-small");
        write_sensevoice(&legacy);

        // Shared root exists but has no model → must fall back in-place.
        let empty_root = temp_dir("resolve-sv-sdss-root");
        std::fs::create_dir_all(&empty_root).unwrap();

        assert_eq!(
            resolve_sensevoice_dir(Some(&empty_root)),
            Some(legacy.clone())
        );

        // And it shows up as a selectable candidate with the new source label.
        let candidates = scan_model_candidates_with_root(Some(&empty_root));
        assert!(candidates.iter().any(|c| {
            c.engine == "sensevoice"
                && c.path == legacy
                && c.source == "legacy-shandianshuo"
                && c.ready
        }));
        let _ = std::fs::remove_dir_all(fake_home);
        let _ = std::fs::remove_dir_all(empty_root);
    }

    #[test]
    fn resolve_sensevoice_none_when_nothing_installed() {
        let _lock = ENV_LOCK.lock().unwrap();
        let fake_home = temp_dir("resolve-sv-none-home");
        std::fs::create_dir_all(&fake_home).unwrap();
        let _iso = isolate_home(&fake_home);

        let empty_root = temp_dir("resolve-sv-none-root");
        std::fs::create_dir_all(&empty_root).unwrap();

        assert_eq!(resolve_sensevoice_dir(Some(&empty_root)), None);
        let _ = std::fs::remove_dir_all(fake_home);
        let _ = std::fs::remove_dir_all(empty_root);
    }

    #[test]
    fn resolve_qwen_prefers_shared_then_shandianshuo_then_none() {
        let _lock = ENV_LOCK.lock().unwrap();
        let fake_home = temp_dir("resolve-qwen-home");
        std::fs::create_dir_all(&fake_home).unwrap();
        let _iso = isolate_home(&fake_home);

        // (a) nothing installed → None.
        let root = temp_dir("resolve-qwen-root");
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(resolve_qwen_asr_dir(Some(&root)), None);

        // (b) only legacy Shandianshuo/models/<name> has a ready qwen.
        let legacy =
            fake_home.join("Library/Application Support/Shandianshuo/models/qwen3-asr-0.6b-8bit");
        write_qwen(&legacy);
        assert_eq!(resolve_qwen_asr_dir(Some(&root)), Some(legacy.clone()));

        // (c) shared root wins once a ready qwen is installed there.
        let shared = root.join("qwen3-asr-0.6b-8bit");
        write_qwen(&shared);
        assert_eq!(resolve_qwen_asr_dir(Some(&root)), Some(shared));
        let _ = std::fs::remove_dir_all(fake_home);
        let _ = std::fs::remove_dir_all(root);
    }
}
