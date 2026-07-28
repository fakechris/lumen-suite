# lumen-suite 平台化实施计划

## Stage 1: 契约与共享 crate 落地（本仓库内，不动产品仓库）
**Goal**: contracts/（provider 目录、transcript schema、models 契约）+ crates/lumen-models + crates/lumen-asr-engine，全部编译、测试通过
**Success Criteria**: `cargo test` 全绿；JSON 契约通过 schema 校验；每个 crate 附迁移对照表
**Status**: Complete（2026-07-26；diar-rs 亦已 subtree 并入 `diar/`）

## Stage 2: 建远程仓库，消费方切换 git dependency
**Goal**: lumen-suite 推送到 github.com/fakechris/lumen-suite（已转 public）；lumen-asr 与 lumen-navi 迁移到 `lumen-models` + `lumen-asr-engine`（git dep），删除仓库内重复实现
**Success Criteria**: CI 绿；行为无回归（paths 契约测试沿用）
**Status**: Complete（navi 已合入 main；asr 走 PR #28，净删 ~5000 行重复代码）

## Stage 3: lumen-cut 跟进迁移
**Goal**: cut 的模型 runtime 接入 `Lumen/models/` 共享目录与 lumen-models；translation 消费 provider-catalog.v1.json
**Success Criteria**: 三个产品共享同一份模型磁盘存储；provider 目录单一事实源
**Status**: Complete（2026-07-27；cut PR #20：共享 Qwen snapshot 发现；translation PR #7：PROVIDER_CATALOG 从契约派生 + no_thinking 注入。剩余副本：Swift PopClip 的 Preferences.swift）

## Stage 4: transcript 交换落地
**Goal**: navi/asr 导出 lumen-transcript.v1；cut 增加 "Import Lumen Transcript" 入口
**Success Criteria**: navi 录制的会议在 cut 中打开无需重新 ASR
**Status**: Complete（2026-07-28）。suite 加 `lumen-transcript` crate（serde 类型+schema 校验，三方共享）。
  navi 导出（已合入 main，含 ARCHITECTURE/AUDIO_PRODUCT/PLAN 文档修正）；asr 导出（PR #32，
  导出文本优先级 pasted→corrected→asr_raw 以保住学习基线）；cut 导入（PR #21，CLI + Tauri）；
  Swift PopClip 接 catalog（PR #8）——provider 目录四份副本全部归一。

## Stage 4.5: lumen-context 迁移（asr→navi 解耦）
**Goal**: 把 lumen-context crate 从 lumen-navi 迁到 lumen-suite（纯迁移，API 不变），asr 与 navi 都改依赖 suite；
  后续再拆 lumen-context-macos / lumen-context-windows。消除 asr 对 navi 仓库（UNLICENSED）的耦合，
  解决 Windows 编译与 SignPath 开源审查问题。
**Success Criteria**: asr 与 navi 均不再跨依赖对方仓库；两者共同依赖 lumen-suite
**Status**: Complete（2026-07-28）。lumen-context 经 git subtree 带历史迁入 `crates/lumen-context`（suite rev c33c13e，
  workspace 成员，macOS 35 测试绿）。asr 改指向 suite（PR #33 已合并，四依赖统一 pin c33c13e，源码零改动，
  Cargo.toml/lock 内 lumen-navi 零命中，macOS 全量 CI 绿）——asr→navi 耦合根除。
  navi 侧：codex/context-capture-foundation 分支删除仓库内副本、改指向 suite（PR #6，base 为该特性分支，
  6 个消费者 crate 本地 cargo check 全绿；该分支无 CI，靠本地验证）——待该分支自行落地时合入。
  后续可选：拆 lumen-context-macos / lumen-context-windows（缝已在 operational.rs 的平台 trait 预留）。

## Stage 5: 会议模式（lumen-asr）— 详见 docs/MEETING.md
**Goal**: diar-rs 接入 + minutes + 会议库 UI + 说话人注册。**决策（2026-07-28）**：macOS beta 先跑通
  （diar-rs 含 CC-BY-NC 模型，仅非商用 beta）；**仅 macOS**；**v1 只采本机麦克风，不做系统音频/loopback**；
  MVP 含核心链路 + 纪要 + 播放器 + 说话人注册。
**Success Criteria**: 端到端：单 mic 连续录音 → diar 分说话人 → 分段转录 → 查看/改名 → 纪要 → 导出 Cut
**子阶段**（见 docs/MEETING.md）：M1 数据骨架/存储v6 → M2 diar-rs 可嵌入+离线管线 → M3 连续录音 →
  M4 会议库UI+播放器+纪要+导出 → M5 说话人注册
**Status**: 计划就绪，待开工 M1
