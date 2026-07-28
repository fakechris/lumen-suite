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
**Status**: 调研中

## Stage 5: 会议模式（lumen-asr）
**Goal**: 系统音频采集 + diar-rs 接入 + minutes prompts + 会议库 UI（见 docs/PRODUCT_MATRIX.md 与 ADR-0001）
**Success Criteria**: 端到端：开会 → 双流录音 → 分说话人转录 → 纪要 → 导出到 Cut
**Status**: Not Started（前置：ADR-0001 阻塞项 1、2）
