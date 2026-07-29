# Lumen Voice — 会议转录模式（Meeting）实施计划

状态：v1，2026-07-28 确认。落地在 lumen-asr（产品）+ suite/diar-rs（引擎）。

## 锁定的决策

- **定位**：macOS beta 先跑通。用 diar-rs 现成方案（含 CC-BY-NC-4.0 分割模型）——**仅限非商用 beta / dogfood**。商用发布前须换 MIT 分割模型（见 ADR-0001，末尾"商用前置"）。
- **平台**：仅 macOS（因 diar-rs / MLX；SenseVoice 本身跨平台）。**v1 只采集本机麦克风，不做系统音频 / loopback**——面向"人都在同一房间、一个 mic 收全场"的场景，多说话人靠 diar-rs 从单路 mic 录音分离。捕获远程通话对方声音（系统音频）留作**未来增强**（需 ScreenCaptureKit/CoreAudio process-tap + Screen Recording 权限，v1 不做）。
- **MVP 功能**：核心链路 + 会议纪要 + 音频播放器 + 说话人注册（全含）。
- **产品归属**：lumen-asr 的第二模式（Dictation / Meetings），非新仓库（见 PRODUCT_MATRIX.md）。

## 架构要点

会议是**并行新管线**，不是听写栈的扩展（听写栈是"一句短话→单字符串→注入"，无法承载长录多说话人）。复用的是套路（cpal 线程、AsrEngine trait、迁移框架、lumen-transcript.v1、tab 壳），不是管线本身。

关键设计：**diar 优先分段**。diar-rs 内部自带 VAD 分段，用它先切出说话人轮次（既是语音/静音分段，又带说话人标签），再把每个轮次的音频喂给 AsrEngine 转录，拼成多 segment 的 lumen-transcript.v1。一次解决两个现有缺口：ASR 的 8MB≈4.4 分钟单请求上限（轮次都很短，天然绕过），和说话人归属。

链路（v1，单路 mic）：
```
麦克风(连续录音) ─→ wav ─→ diar-rs(分段+说话人)
                              │
                每轮次音频 ─→ AsrEngine 转录
                              │
          多segment+speaker ─→ lumen-transcript.v1
                              │
        存储(v6) ── 会议库UI ── 纪要(LLM) ── 导出Cut
```

## 分阶段

### Stage M1：会议数据骨架（存储 v6 + 会议会话类型）
**Goal**：定义会议实体与数据模型，不含音频/diar/UI。存储 schema v6 加：`meetings`（或 sessions.kind）、`transcript_segments`(start/end/text/speaker_id/channel)、`speakers`(label 可改名, embedding ref)、`meeting_summaries`。新增会议会话类型（长录、多段、暂停/继续、无注入），与听写 Session 并列。
**Success**：`cargo test` 全绿；schema v6 迁移测试；会议实体的增删查改单测。纯 Rust，无 UI。
**归属**：lumen-asr（crates/lumen-store、lumen-core）
**Status**：Complete（2026-07-28，asr PR #38）

### Stage M2：diar-rs 可嵌入 + 离线转录管线
**状态**: Complete（2026-07-28）。M2a：diar-rs vendored fbank（suite 92cf565，env -u PYTHON cargo build 通过）。M2b：新建 crates/lumen-meeting（asr PR #39 合并），纯组装逻辑单测绿、diar-rs 双门控（cfg macos + diarize feature）、失败回滚。离线管线 wav→diar→逐段ASR→多段transcript→v6 入库就绪（真模型集成测试 #[ignore]）。
**Goal**：(a) 让 diar-rs 可作为 Rust 依赖被 asr 干净构建——vendored kaldi-native-fbank C++ 源码编译，去掉 build.rs 的 Python 定位（ADR-0001 阻塞项 2）；(b) 给定一个**预录 wav 文件**，跑通 diar-rs 分段+说话人 → 每轮次 AsrEngine 转录 → 拼成多 segment lumen-transcript.v1 → 存 v6。先不接 live 采集，用测试音频。
**Success**：`cargo test`；asr 能 `cargo build`（无需 PYTHON env）；离线跑通样例会议 wav，产出带说话人的多段转录并入库。
**归属**：suite/diar-rs（vendored fbank）+ lumen-asr（管线接入）
**Status**：Not Started

### Stage M3：Live 连续录音（麦克风）
**Goal**：新增**独立的连续录音路径**（不改动听写现有的 hold-to-talk `AudioCapture`，避免回归听写）：cpal 连续采集 → 分块**增量写入 wav 文件**、暂停/继续、时长无上限、内存不随时长增长。录制生命周期（start/recording/paused/stop→finalize），产出会议 wav 喂 M2 管线。仅用**已有 mic 权限**。
**模式互斥（模式仲裁器）**：会议是长时**独占**采集模式。开始录制时**挂起听写全局热键**（注销 / 或 handler no-op 并提示"会议录制中"），停止后恢复。原因：① cpal 单输入设备，麦克风独占；② 听写会向光标注入文字，开会时误触会污染文档。v1 互斥；"边开会边听写记笔记"（同一路音频 fan-out 双消费）留作未来增强。
**启动方式**：会议**不靠热键**，由 UI 显式"开始会议"触发（M3 先用 Tauri 命令 start/stop 驱动、可测；精致按钮与录制状态 UI 在 M4）。
**Success**：连续录 30–60 分钟稳定落盘、内存有界；录制期间听写热键被挂起、停止后恢复；产出 wav 走 M2 出带说话人转录。
**归属**：lumen-asr（crates/lumen-asr 新录音器、apps/desktop 的热键仲裁 + start/stop 命令）
**Status**：Complete（2026-07-28，asr PR #40）。独立连续录音器（cpal 流式写 wav、内存有界、Drop 收尾）+ 模式仲裁器（双向互斥：会议挂起听写热键、听写中拒开会议，任何失败路径都恢复麦克风/热键）+ start/stop/pause/resume 命令。audio.rs 零改动。
（未来增强：系统音频/loopback 采集远程对方声音——新增 SystemAudioCapture 平台端口 + Screen Recording 权限，不在 v1。）

### Stage M4：会议库 UI + 播放器 + 纪要 + 导出
**Goal**：desktop 加 `"meeting"` TabId 与会议库面板（录音列表、带说话人的转录视图、说话人重命名）。音频播放器 + 点击 segment 跳转。会议纪要：lumen-prompts 加 minutes intent（摘要/行动项/决议），走既有 corrector LLM 层。"Export to Cut"（多 segment transcript）。
**Success**：端到端可用：开会 → 转录 → 查看/改说话人 → 出纪要 → 导出 Cut 打开无需重转。
**归属**：lumen-asr（apps/desktop、lumen-prompts、export）
**Status**：Not Started

### Stage M5：说话人注册（跨会议身份）
**Goal**：给 diar-rs 加 enrollment/voiceprint（基于现有 WeSpeaker 256-d embedding：注册库存 `~/Library/Application Support/Lumen/identity/`，聚类后与注册 embedding 余弦匹配）。会议里"这是 Chris"的注册/识别 UI。跨会议身份成为共享资产（Voice 会议 / Cut 说话人标注 / Navi 记忆归属）。
**Success**：注册一次后，后续会议自动把该说话人标为已知身份。
**归属**：suite/diar-rs（enrollment）+ lumen-asr（UI、identity 存储）
**Status**：Not Started

## 商用前置（beta → 付费之间必须做）

- 换掉 diar-rs 的 CC-BY-NC 分割模型：自训 MIT `microsoft/wavlm-base` 分割头，或把 pyannote 的 MIT segmentation 导成 ONNX。见 ADR-0001。
- 复核 identity/embedding 存储的隐私与加密（可复用 lumen-context 的 ChaCha20+Keychain 封存模式）。
