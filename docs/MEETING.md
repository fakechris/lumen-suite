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
**Status**：Complete（2026-07-28，见本段顶部"状态"）

### Stage M3：Live 连续录音（麦克风）
**Goal**：新增**独立的连续录音路径**（不改动听写现有的 hold-to-talk `AudioCapture`，避免回归听写）：cpal 连续采集 → 分块**增量写入 wav 文件**、暂停/继续、时长无上限、内存不随时长增长。录制生命周期（start/recording/paused/stop→finalize），产出会议 wav 喂 M2 管线。仅用**已有 mic 权限**。
**模式互斥（模式仲裁器）**：会议是长时**独占**采集模式。开始录制时**挂起听写全局热键**（注销 / 或 handler no-op 并提示"会议录制中"），停止后恢复。原因：① cpal 单输入设备，麦克风独占；② 听写会向光标注入文字，开会时误触会污染文档。v1 互斥；"边开会边听写记笔记"（同一路音频 fan-out 双消费）留作未来增强。
**启动方式**：会议**不靠热键**，由 UI 显式"开始会议"触发（M3 先用 Tauri 命令 start/stop 驱动、可测；精致按钮与录制状态 UI 在 M4）。
**Success**：连续录 30–60 分钟稳定落盘、内存有界；录制期间听写热键被挂起、停止后恢复；产出 wav 走 M2 出带说话人转录。
**归属**：lumen-asr（crates/lumen-asr 新录音器、apps/desktop 的热键仲裁 + start/stop 命令）
**Status**：Complete（2026-07-28，asr PR #40）。独立连续录音器（cpal 流式写 wav、内存有界、Drop 收尾）+ 模式仲裁器（双向互斥：会议挂起听写热键、听写中拒开会议，任何失败路径都恢复麦克风/热键）+ start/stop/pause/resume 命令。audio.rs 零改动。
（未来增强：系统音频/loopback 采集远程对方声音——新增 SystemAudioCapture 平台端口 + Screen Recording 权限，不在 v1。）

### Stage M4：会议库 UI + 播放器 + 纪要 + 导出 —— 详见 docs/MEETING_M4_UX.md
**UX 基准（2026-07-28 用户确认）**：列表+详情页；详情 = Granola 纪要优先 + Otter 转录/播放联动 + Descript-lite 说话人修正；不做多轨时间线。三个 v1 裁决：①录制窗口无实时逐字稿/说话人计数（离线架构）；②纪要为结构化 JSON（条目带 source 时间戳，可点跳原文）；③JSON 导出 = lumen-transcript.v1（= Cut 导入格式）。
**子阶段**：M4a 后端闭环（stop→自动转录→结构化 minutes→状态推进；说话人 reassign/merge；4 预设导出；均可测）→ M4b 列表+详情壳+纪要页 → M4c 逐字稿阅读器+底部播放器+说话人修正 UI → M4d 录制窗口+导出面板。
**Success**：端到端：开会 → 停止自动转录出说话人分段 → 纪要（条目可点跳原文）→ 逐字稿阅读+回听 → 改/合并说话人 → 4 预设导出（含 Cut 可导入 JSON）。
**归属**：lumen-asr（apps/desktop、src-tauri、lumen-prompts、lumen-store、export）
**Status**：M4a Complete（2026-07-28，asr PR #41）——说话人 reassign/merge（同会议校验）、get_meeting_detail、4 预设导出（含 lumen-transcript.v1=Cut 格式）、结构化 minutes（JSON+容错解析+LLM 生成）、process_meeting 状态机（+Transcribing/Summarizing）。**M4a-2 Complete**（asr PR #43）：diarize feature 接进 macOS desktop 构建、stop 后台专用线程跑 process_meeting（!Send Store→独立 SQLite 连接）、process_meeting_now 命令、缺模型/非 macOS 标 failed。**M4b Complete**（asr PR #42）：会议 tab+列表（搜索/过滤/分组）+详情+结构化纪要页+最小逐字稿。diar 模型已放置 `~/Library/Application Support/Lumen/models/diar/`。顺带根治 qwen_worker Windows flaky（500ms→5s）。**M4a-3 Complete**（asr PR #44）：内联“开始/停止会议”按钮+录制态计时（暂停留 M4d）、failure_reason（schema v7）+ 缺模型/平台不支持引导、无 LLM 时纪要页提示配置。**会议从 UI 端到端可用**。M4c（逐字稿播放器联动+说话人修正 UI）、M4d（正式录制窗口+暂停+导出面板）、M5（说话人注册）未开工。

### Stage M6：实时逐字稿 + Paraformer 引擎（2026-07-29 决策）
**背景/决策**：用户 dogfood 反馈"录制时是黑盒,要实时看到转录"。且经模型对比,**会议改用 Paraformer**（经 sherpa-onnx,不换框架）——它对会议三大需求都强:①真流式(streaming Paraformer→实时字幕)②原生时间戳(点句回听 M4c + 纪要跳原文)③热词(复用个人词典,人名术语更准);SenseVoice 的情绪/音频事件对会议无用。**听写继续用 SenseVoice**（快省够用,不动）。
**架构**：双层——**录制中** streaming Paraformer 按 VAD 逐语音段出实时粗稿(无说话人标签,diar 是后处理重活);**停止后** offline Paraformer(带时间戳+热词)精转录 + diar-rs 分说话人对齐,产出最终带说话人逐字稿替换实时版。
**子阶段**：
- **P1** Paraformer 引擎接入 lumen-asr-engine（suite）:offline(时间戳+热词) + streaming 两种,经 sherpa-onnx online/offline 识别 API,模型经 lumen-models 解析。
- **P2-hotword（决策 A，2026-07-29）**：sherpa 的 Paraformer **不支持引擎级热词**（只对 transducer 有效）——已把引擎改 greedy+安全忽略热词。热词改走**转录后词典纠正**：复用 corrector/lumen-dictionary,对会议逐字稿做后处理纠正人名/术语。（选项 B=换 zipformer 原生热词,暂不做。）
- **P2** 会议 offline 管线切到 Paraformer（lumen-meeting）:带词级时间戳(喂 M4c 回听)+ 热词(复用 lumen-dictionary);diar-rs 说话人与 Paraformer 词时间戳对齐(仿 lumen-cut)。
- **P3** 实时层:录制中后台 streaming Paraformer 按 VAD 逐段 → Tauri 事件透出实时逐字稿到录制界面。
- **P4** 崩溃恢复:启动时检测残留"录制中"会议 → 收尾 wav + 转录(前半段不丢)。
**商用前置**：核实 Paraformer(FunASR)模型许可证可商用(与 diar 的 CC-BY-NC 一起过)。
**Status**：P1 开工（2026-07-29）

### Stage M5：说话人注册（跨会议身份）
**Goal**：给 diar-rs 加 enrollment/voiceprint（基于现有 WeSpeaker 256-d embedding：注册库存 `~/Library/Application Support/Lumen/identity/`，聚类后与注册 embedding 余弦匹配）。会议里"这是 Chris"的注册/识别 UI。跨会议身份成为共享资产（Voice 会议 / Cut 说话人标注 / Navi 记忆归属）。
**Success**：注册一次后，后续会议自动把该说话人标为已知身份。
**归属**：suite/diar-rs（enrollment）+ lumen-asr（UI、identity 存储）
**Status**：Not Started

## 商用前置（beta → 付费之间必须做）

- 换掉 diar-rs 的 CC-BY-NC 分割模型：自训 MIT `microsoft/wavlm-base` 分割头，或把 pyannote 的 MIT segmentation 导成 ONNX。见 ADR-0001。
- 复核 identity/embedding 存储的隐私与加密（可复用 lumen-context 的 ChaCha20+Keychain 封存模式）。
