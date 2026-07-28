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
**Status**: Not Started

## Stage 5: 会议模式（lumen-asr）
**Goal**: 系统音频采集 + diar-rs 接入 + minutes prompts + 会议库 UI（见 docs/PRODUCT_MATRIX.md 与 ADR-0001）
**Success Criteria**: 端到端：开会 → 双流录音 → 分说话人转录 → 纪要 → 导出到 Cut
**Status**: Not Started（前置：ADR-0001 阻塞项 1、2）
