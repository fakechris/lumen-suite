# ADR-0001: 说话人分离（diarization）技术收敛

状态：Accepted，2026-07-25

## 背景

产品簇内存在两套互不相通的 diarization 实现：

1. **lumen-cut**：pyannote.audio 3.4.0 Python sidecar（`pyannote/speaker-diarization-3.1`，torch 2.5.1，MPS/CPU），已在产品内跑通，含说话人→段落分配（`diarize/assign.rs`）与 UI。
2. **diar-rs**（2026-07-26 起并入本仓库 `diar/crates/diar-rs`）：纯 Rust/ONNX（ort）管线——DiariZen WavLM 分割 + WeSpeaker ResNet34 embedding + PLDA + AHC 聚类，lib + CLI + C FFI，无 torch 依赖。当前无任何消费方。

## 决策

**短期（现在）**：lumen-cut 继续使用 pyannote sidecar，不动。它能跑、有质量保底，替换没有产品收益。

**战略（会议模式上线路径）**：投资 diar-rs 作为簇内统一 diarization 引擎，lumen-asr 会议模式作为其第一个消费方（与 lumen-models/lumen-asr-engine 一样，通过对 lumen-suite 的 git dependency 以 Rust crate 方式接入，不走 C FFI）。理由：无 Python/torch 运行时依赖、启动快、体积小、适合常驻菜单栏 app；cut 的 pyannote sidecar 依赖 ~2GB torch 运行时，不适合 Voice 产品形态。

**收敛条件**（满足后 cut 再迁移到 diar-rs，此前不迁）：
- diar-rs 在会议真值集上 frame_acc/DER 不劣于 pyannote 3.1
- 下述阻塞项全部解决

## diar-rs 的阻塞项（按优先级）

1. **许可证**：分割模型 DiariZen WavLM-base 为 CC-BY-NC-4.0（禁商用）。商业化前必须替换——候选：pyannote/segmentation-3.0（MIT）导出 ONNX，或自训。embedding 模型 WeSpeaker 为 CC-BY-4.0，可商用，无需动。
2. **构建可移植性**：`build.rs` 通过导入 Python 包 `kaldi_native_fbank` 定位原生库——共享 crate 不可接受。改为 vendored 源码编译（cc crate 直接编 kaldi-native-fbank 的 C++ 源）或纯 Rust fbank 实现。
3. **说话人注册（enrollment/voiceprint）缺失**：会议场景需要跨会话身份（"这是 Chris"）。基于现有 WeSpeaker 256-d embedding 增加注册库（存 `~/Library/Application Support/Lumen/identity/`，写入契约），聚类后与注册 embedding 做余弦匹配。
4. VBx 聚类修复（vbx.rs 已存在但未接入，AHC 为当前路径）——质量优化项，非阻塞。

## 后果

- 短期内簇里同时存在两套 diarization，这是有意的（避免为统一而统一拖慢 cut）。
- diar-rs 增加 enrollment 后，说话人身份成为跨产品共享资产（Voice 会议、Cut 说话人标注、Navi 记忆归属）。
