# lumen-models

Lumen 产品簇的共享模型层：ASR 模型的**路径解析**、**就绪校验**、**跨进程安装锁**与 **SenseVoice 下载安装**。统一了此前在 `lumen-asr/crates/lumen-asr` 与 `lumen-navi/crates/lumen-asr-engine` 中重复实现的 `paths.rs` / `install_lock.rs` / `download.rs`。

行为契约见 [`contracts/SHARED_MODELS_CONTRACT.md`](../../contracts/SHARED_MODELS_CONTRACT.md)（自 v1.1 起 canonical source 在 lumen-suite）。`lib.rs` 中的 `shared_model_contract_matches_cluster_v1` 测试将契约正文 pin 到 cluster v1 哈希（FNV-1a 64 = `0xc87789f4de205e71`）。

分工：本 crate 负责"选哪个目录 / 怎么装"；同 workspace 的 `lumen-asr-engine` 只做"给定目录里的模型文件探测与推理"（其 `model_paths.rs` 不做默认目录解析）。

## Features

- `download`（默认开启）：SenseVoice 包下载安装（Rust HTTP streaming +
  bzip2/tar 解压，不依赖系统命令）。关闭后仅保留路径/锁逻辑。

## 迁移映射

### lumen-asr（`crates/lumen-asr`，模块 `paths` / `install_lock`）

| 旧（`lumen_asr::…`） | 新（`lumen_models::…`） | 备注 |
|---|---|---|
| `user_home_dir` | `user_home_dir` | 同名同义 |
| `lumen_models_dir` / `lumen_models_dir_with_override` | 同名 | 同义 |
| `app_models_dir` | `app_models_dir` | 已标 `#[deprecated]`，请改用 `lumen_models_dir` |
| `shared_sensevoice_dir` / `shared_whisper_dir` | 同名 | 同义 |
| `legacy_model_roots` | `legacy_model_roots` | 同义 |
| `default_sensevoice_dir(_with_root)` | 同名 | 新增 `LUMEN_NAVI_SENSEVOICE_DIR` 兜底（在 `LUMEN_SENSEVOICE_DIR` 之后） |
| `default_whisper_dir(_with_root)` | 同名 | 同上（`LUMEN_NAVI_WHISPER_DIR`） |
| `default_qwen_dir` / `qwen_ready` | 同名 | 同义 |
| `sensevoice_ready` / `whisper_ready` | 同名 | 同义 |
| `sensevoice_model_path` / `sensevoice_tokens_path` / `whisper_{encoder,decoder,tokens}_path` | 同名 | 同义 |
| `scan_model_candidates(_with_root)` / `ModelCandidate` | 同名 | `ModelCandidate` 增加 serde 派生；额外扫描 `LUMEN_NAVI_*` env |
| `ModelInstallLock` / `SENSEVOICE_INSTALL_LOCK_NAME`（`install_lock`） | 同名 | 同义 |
| `ENV_LUMEN_MODELS_DIR` | `ENV_LUMEN_MODELS_DIR` | 新增 `ENV_LUMEN_{SENSEVOICE,WHISPER}_DIR`、`ENV_LUMEN_NAVI_*` 常量 |

### lumen-navi（`crates/lumen-asr-engine`，模块 `paths` / `install_lock` / `download`）

| 旧（`lumen_asr_engine::…`） | 新（`lumen_models::…`） | 备注 |
|---|---|---|
| `paths::*`（同上表各函数） | 同名 | 同义；`default_*_dir` 对 `LUMEN_NAVI_*` 的兼容已内置 |
| `ModelCandidate { path: String, … }` | `ModelCandidate { path: PathBuf, … }` | serde JSON 形状不变（`PathBuf` 序列化为字符串）；Rust 调用点需 `path.display().to_string()` 或直接用 `PathBuf` |
| `scan_model_candidates(_with_root)` | 同名 | 现在**包含 qwen 候选**（source `lumen-asr` / `huggingface-cache`）；只支持 sensevoice/whisper 的 UI 请按 `engine` 过滤 |
| `ModelInstallLock` | `ModelInstallLock` | 同义 |
| `download_sensevoice_package` | `download_sensevoice_package`（feature `download`） | 错误类型由 `String` 改为 `DownloadError`（`.to_string()` 可得等价消息）；进度语义不变 |
| `DownloadProgress` / `SENSEVOICE_ARCHIVE_URL` / `SENSEVOICE_ARCHIVE_NAME` / `default_models_root` | 同名（feature `download`） | 同义 |

## 快速上手

```rust
use lumen_models as models;

// 解析（不存在时返回规范安装目标）
let sv_dir = models::default_sensevoice_dir();
if !models::sensevoice_ready(&sv_dir) {
    let cancel = std::sync::atomic::AtomicBool::new(false);
    models::download_sensevoice_package(&models::default_models_root(), &cancel, |p| {
        eprintln!("[{}] {}", p.phase, p.message);
    })?;
}
```
