# lumen-suite

Lumen 产品簇的**平台仓库**：契约（contracts）+ 共享 Rust crate。产品仓库（lumen-asr / lumen-cut / lumen-translation / lumen-navi / diar-rs）保持独立，通过 git dependency 与数据契约消费本仓库。

> 策略：产品层分开，能力层统一。"重复三次才抽取"——本仓库只收纳已在 ≥2 个产品中重复实现、且接口已稳定的能力。

## 内容

```
contracts/
  SHARED_MODELS_CONTRACT.md        共享模型目录契约（canonical，自 lumen-asr 迁入）
  provider-catalog.v1.json         LLM provider 目录（数据即契约，四语言消费方共享）
  provider-catalog.schema.json
  lumen-transcript.v1.schema.json  转录交换格式（含说话人/词级 timing）
crates/
  lumen-models                     模型下载/路径解析/安装锁（统一 asr/navi/cut 三份实现）
  lumen-asr-engine                 ASR 引擎层（sherpa-onnx SenseVoice/Whisper + MLX Qwen + cloud）
docs/
  PRODUCT_MATRIX.md                产品矩阵与仓库布局决策
  ADR-0001-diarization.md          说话人分离技术收敛决策
```

## 规则

- crate 一律宽松许可（MIT），保证 AGPL 产品（lumen-cut）可消费
- 契约演进 additive-only；JSON 契约必须过 schema 校验
- 共享层不使用 `lumen-core` 命名（该名已被 lumen-asr 内部 crate 占用）
- 产品仓库不得反向依赖：suite → 产品 的依赖方向禁止
