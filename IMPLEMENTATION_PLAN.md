# lumen-suite 平台化实施计划

## Stage 1: 契约与共享 crate 落地（本仓库内，不动产品仓库）
**Goal**: contracts/（provider 目录、transcript schema、models 契约）+ crates/lumen-models + crates/lumen-asr-engine，全部编译、测试通过
**Success Criteria**: `cargo test` 全绿；JSON 契约通过 schema 校验；每个 crate 附迁移对照表
**Status**: In Progress

## Stage 2: 建远程仓库，消费方切换 git dependency
**Goal**: lumen-suite 推送到 github.com/fakechris/lumen-suite（建议 private 起步）；lumen-asr 首先迁移到 `lumen-models` + `lumen-asr-engine`（git dep），删除本仓库内重复实现
**Success Criteria**: lumen-asr CI 绿；行为无回归（paths 契约测试沿用）
**Status**: Not Started（等待用户创建 remote 并 push）

## Stage 3: lumen-navi、lumen-cut 跟进迁移
**Goal**: navi 的 lumen-asr-engine 内部 crate 替换为共享版；cut 的模型 runtime 接入 `Lumen/models/` 共享目录与 lumen-models
**Success Criteria**: 三个产品共享同一份模型磁盘存储（Qwen3-ASR 不再重复占盘）
**Status**: Not Started

## Stage 4: transcript 交换落地
**Goal**: navi/asr 导出 lumen-transcript.v1；cut 增加 "Import Lumen Transcript" 入口
**Success Criteria**: navi 录制的会议在 cut 中打开无需重新 ASR
**Status**: Not Started

## Stage 5: 会议模式（lumen-asr）
**Goal**: 系统音频采集 + diar-rs 接入 + minutes prompts + 会议库 UI（见 docs/PRODUCT_MATRIX.md 与 ADR-0001）
**Success Criteria**: 端到端：开会 → 双流录音 → 分说话人转录 → 纪要 → 导出到 Cut
**Status**: Not Started（前置：ADR-0001 阻塞项 1、2）
