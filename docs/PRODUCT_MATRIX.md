# Lumen 产品矩阵与仓库布局

状态：v1，2026-07-25 确认

## 原则

产品层分开（独立 App、独立下载、独立权限申请），能力层统一（lumen-suite 契约 + 共享 crate，git dependency 消费——该机制已被 `lumen-context`（asr ← navi）和 `SHARED_MODELS_CONTRACT.md`（asr ↔ navi）验证）。

## 产品矩阵

| 产品 | 仓库 | 定位 | 用户心智 |
|---|---|---|---|
| **Lumen Voice** | lumen-asr | AI 语音输入 + **会议转录**（两种模式：Dictation / Meetings） | "对着它说话，它替我写字、替我记会" |
| **Lumen Cut** | lumen-cut | transcript-first 音视频编辑器 | "转录稿即时间线，改文字即剪片" |
| **Lumen Translate** | lumen-translation | 全场景语言层（网页/PDF/字幕/划词） | "看什么都是双语" |
| **Lumen Navi** | lumen-navi | 持续上下文 / 个人记忆（长期方向，不催熟） | "我的第二记忆" |

## 会议转录的归属：lumen-asr（决策）

会议功能（多说话人识别 + 会议纪要）**在 lumen-asr 内产品化为第二模式，不新建仓库**。理由：

1. **底座复用率最高**：音频采集（cpal）、ASR 引擎（sherpa/MLX Qwen）、LLM 纠正与 prompt 层（lumen-corrector/lumen-prompts）、模型管理、托盘/热键 UX 全部已存在于 lumen-asr。
2. **市场先例**：Superwhisper、MacWhisper、VoiceInk 都是"听写 + 会议转录"一体的 voice 工具，用户心智统一为"声音进，文字出"，不稀释定位。
3. **不放 navi**：navi 的价值主张是 always-on 环境捕获，信任门槛最高、产品最早期；把可快速商业化的会议功能绑在它上面会拖慢变现。navi 后续通过共享存储契约*索引*会议产物即可。
4. **不放 cut**：cut 是编辑器，是会议转录的*下游*——通过 `lumen-transcript.v1` 导入会议稿做精修/出片/出字幕，而非采集端。
5. **给 diar-rs 第一个消费方**：会议模式接入 diar-rs 做说话人分离（见 ADR-0001）。

会议模式需要新增的能力（均为 lumen-asr 内的增量）：
- 系统音频采集（对方声音）：macOS ScreenCaptureKit audio / CoreAudio process tap（macOS 14.4+）
- 说话人分离：diar-rs（战略路径）或 pyannote sidecar（短期兜底），见 ADR-0001
- 会议纪要：lumen-prompts 新增 minutes intent（摘要/行动项/决议），走既有 corrector LLM 层
- 会议库 UI：录音列表、说话人重命名、导出 `lumen-transcript.v1` → Cut 导入

## 仓库布局（确认版）

```
lumen-suite        平台仓库：contracts/（provider 目录、transcript schema、models 契约）
                   + crates/（lumen-models、lumen-asr-engine，宽松许可 MIT）
lumen-asr          产品：Lumen Voice（听写 + 会议）
lumen-cut          产品：编辑器（AGPL，消费 suite 的 MIT crate 无问题）
lumen-translation  产品：语言层（TS monorepo，消费 contracts/ 的 JSON 数据）
lumen-navi         产品：持续上下文（继续导出 lumen-context 供 asr 消费）
diar-rs            独立引擎库：保持独立仓库，不合并（理由见下）
```

### diar-rs 为什么不合并进其他仓库

- 它是可嵌入引擎（rlib + cdylib + staticlib + C FFI），有独立的发布节奏和潜在开源社区价值；
- 许可证隔离：MIT 引擎不应并进 AGPL 的 cut；并进 asr 会把 kaldi-native-fbank C++ 依赖和 Python lab 拖进产品仓库；
- 消费方式与其他共享 crate 一致：git dependency。
- 命名可后续演进为 `lumen-diar`，非必需。

### 命名注意

lumen-asr 仓库内已有名为 `lumen-core` 的 crate（会话状态机）。共享层一律以 `lumen-models` / `lumen-asr-engine` 等具体名称命名，**不使用 `lumen-core` 作为平台层名称**。

## 商业化排序（维持既定判断）

1. **Lumen Voice**（听写先行，会议模式为付费差异化）
2. **Lumen Cut**（AI 视频编辑窗口期）
3. Translate 保持开源获客 / 生态入口
4. Navi 技术储备，不催熟

## 明确不做

- 不合并成单一超级 App（权限灾难、转化率、发版节奏耦合）
- 不做跨语言统一 LLM router（共享 provider *数据*，不共享路由*代码*）
- 不做统一存储引擎（各产品 schema 服务各自领域；只统一数据目录约定与交换格式）
