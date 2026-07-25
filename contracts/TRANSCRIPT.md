# lumen-transcript.v1 — Lumen 转录交换格式

统一的转录交换格式，让 Lumen 产品簇内的转录产出方（lumen-navi、lumen-asr、diar-rs）
产出的结果能被消费方（lumen-cut）直接导入 —— "Import from Navi/Voice"，不用重新 ASR，
并承载 diarization 说话人信息。

- Schema: [`lumen-transcript.v1.schema.json`](./lumen-transcript.v1.schema.json)（JSON Schema draft 2020-12）
- 判别字段: 顶层 `"schema": "lumen-transcript.v1"`
- 命名约定: **snake_case**（与 lumen-cut 现有 sidecar 格式 `asr_out.v1` / `diarize_out.v1` 一致；
  lumen-cut 的 `doc.json` 用 camelCase，但那是它的内部项目文件，不是交换边界）
- 时间单位: **秒，浮点**，相对媒体起点
- 演进策略: **additive**。所有非必填字段宽松；未知字段消费方必须容忍（导入可忽略，
  转存应保留）。破坏性变更 → 新文件 `lumen-transcript.v2`。

---

## 1. 设计目标与非目标

### 目标

1. **一次 ASR，处处可用** — navi 的后台转录、lumen-asr 的听写会话，导出后 lumen-cut
   直接建项目，跳过整个 ASR sidecar 流程。
2. **承载说话人** — diarization 结果（diar-rs `Turn`、pyannote `diarize_out.v1`）以
   `speakers` 表 + `segments[].speaker` 的形式随转录一起流转。
3. **词级 timing 可选可传** — 有词级时间戳的引擎（lumen-cut 的 sidecar ASR）不丢信息；
   没有的（navi、lumen-asr）也能产出合法文档。
4. **媒体只引用不内嵌** — `media.path` + `media.content_hash`（blake3），文件小、可入库、可 diff。
5. **宽松、可增量演进** — 只有 `schema`、`segments`、`segment.{start,end,text}`、
   `speaker.id`、`word.{word,start,end}` 是必填；其余全部可选，`additionalProperties: true`。

### 非目标

- **不是编辑模型** — 不承载 lumen-cut 的 soft-cut/hidden/章节/导出设置等项目状态；
  那是 `doc.json` 的职责。导入是单向的（interchange → 项目）。
- **不承载音频/视频字节** — 永不内嵌媒体。
- **不做声纹** — `speaker.voiceprint` / `speaker.enrollment` 仅保留扩展位，v1 生产方
  不得产出、消费方必须忽略。
- **不统一置信度语义** — `confidence` 只保证 [0,1] 区间，跨引擎不可比。
- **不是流式协议** — 这是落盘交换文件，不是实时增量事件流。

---

## 2. 字段映射

### 2.1 navi `transcript.v1`（derived doc）→ `lumen-transcript.v1`

来源：`/Users/chris/source/lumen-navi/crates/lumen-process/src/transcribe_worker.rs`
的 `transcript_body_json()`（每个 `audio_chunk.v1` 事件一条 derived 记录）。

| navi transcript.v1（per chunk） | lumen-transcript.v1 | 说明 |
|---|---|---|
| `payload_version` | —（被 `schema` 取代） | 导出器负责版本翻译 |
| `event_id` | `segments[i].id` | 每个 chunk 导出为一个 segment，chunk 事件 id 即 segment id |
| `text` | `segments[i].text` | 空文本的 chunk 本来就不落库，天然满足非空 |
| `confidence` | `segments[i].confidence` | 引擎级 0–1，语义不跨引擎可比 |
| `language` | `provenance.language`（一致时）；否则 `segments[i].language` | chunk 间语言不一致时降级到 per-segment |
| `engine` | `provenance.engine` | 多 chunk 引擎不一致罕见；取多数并在 `provenance.extra` 记录 |
| `audio_bytes` | `media.bytes` | 单 chunk 导出时直接映射；整 session 导出为各 chunk 之和 |
| `audio_blake3` | `media.content_hash` = `"blake3:" + hex` | 整 session 导出时用拼接后音频的 hash |
| —（无） | `segments[i].start` / `end` | **合成**：navi derived doc 不含时间。导出器用 `SourceEvent.ts`（chunk 事件时间）相对 session 起点 + chunk 音频时长（bytes / sample_rate / 2）推算，见 §4 阻抗失配 #1 |
| —（无） | `words` | navi 无词级 timing，省略 |
| —（无） | `speakers` | navi 暂无 diarization；后续接 diar-rs 后填充 |
| — | `provenance.app` = `"lumen-navi"` | |

### 2.2 `lumen-transcript.v1` → lumen-cut `asr_out.v1` / `Doc`

目标：`/Users/chris/source/lumen-cut/src-tauri/src/asr/mod.rs`（`AsrOutV1` 及
`impl From<AsrOutV1> for Doc`）、`/Users/chris/source/lumen-cut/src-tauri/src/data/doc.rs`。

推荐路径：新写 `From<LumenTranscriptV1> for Doc`（不绕道 `AsrOutV1`，因为 interchange
比 `AsrOutV1` 多 speaker 表、translations、confidence）。分组规则：**连续同 speaker 的
segment 合并为一个 `Paragraph`**（speaker 均为 None 时按停顿阈值或全部并入单段落，
与 sidecar 的成段行为对齐）；每个 segment 成为一个 `Sentence`。

| lumen-transcript.v1 | cut `AsrOutV1` | cut `Doc` | 说明 |
|---|---|---|---|
| `schema` | `schema_version: 1` | `schema: 1` | |
| `provenance.language` | `language` | `meta.language` | |
| `media.duration_seconds` | `duration_seconds` | `media.duration_seconds` | 缺失时取 max(segments[].end) |
| `media.path` | — | `media.path` | interchange 有 path 是相对 sidecar 流程的**增益**（sidecar 导入后 path 为空串需另行 rebind） |
| `media.sample_rate` / `channels` | — | `media.sample_rate` / `channels` | 缺失时 cut 保持 `From<AsrOutV1>` 现状（16000/1）或置 None |
| `media.content_hash` | — | —（可存入 doc.json 未知字段区） | 用于导入时校验选中的媒体文件 |
| segment 分组（连续同 speaker） | `paragraphs[]` | `paragraphs[]`（`Paragraph.id` 顺序生成） | |
| `segments[].speaker` | `paragraphs[].speaker` | `Paragraph.speaker` | **存 display_name 还是 id？** 存 `display_name`（有则用，无则用 id），因为 cut 的 speaker 就是展示字符串（`data/speakers.rs` 直接对 `paragraph.speaker` 做 rename/merge） |
| `segments[].text` | `sentences[].text` | `Sentence.text`（`Sentence.id` = `"p{pi}s{si}"` 生成） | |
| `segments[].words[].word` | `words[].text` | `Word.text`（`Word.id` = `"w{n}"` 全局递增） | 字段名不同：interchange 用 `word`，cut 用 `text` |
| `segments[].words[].start/end` | `words[].start/end` | `Word.start/end` | 同为秒浮点，直传 |
| `segments[].words` 缺失 | — | **合成单词**：一个横跨 `[start, end]` 的伪 `Word`，text = 整句文本 | cut 的时间轴/字幕/diarize-assign 都从 word 时间推段落时间，见 §4 #3 |
| `segments[].translations[lang]` | — | `translations[lang][sentence_id] = TranslationGroup { text, source_words: [], source_text: Some(segment.text) }` | `source_words` 留空（陈旧检测降级用 `source_text` 兜底），见 §4 #6 |
| `segments[].confidence` / `words[].confidence` | — | **丢弃**（或写入 doc.json 顶层未知字段区留档） | cut Doc 无 confidence 字段，见 §4 #5 |
| `speakers[].display_name` | — | 参与上面的 speaker 字符串解析 | |
| `speakers[].voiceprint/enrollment` | — | 忽略（v1 保留位） | |
| `provenance.*` | — | 可整体存入 doc.json 顶层未知字段（`importProvenance`），cut 的保留逻辑会带着走 | |

### 2.3 diar-rs `Turn` → `speakers` / `segments`

来源：`/Users/chris/source/diar-rs/crates/diar-rs/src/pipeline.rs`
（`Turn { start: f64, end: f64, speaker: u32 }`，`DiarizeResult.timeline`）。

| diar-rs | lumen-transcript.v1 | 说明 |
|---|---|---|
| `Turn.speaker: u32`（0 基聚类标签） | `speaker.id` = `"S{n+1}"`；`segments[].speaker` 同 | 与 diar-rs 自己的 abs-timeline 输出（`Speaker{n+1}`，见 `io.rs`）对齐；pyannote 路线则保留 `"SPEAKER_00"` 原样 |
| `DiarizeResult.talk_sec` 的 keys | `speakers[]` 表（每个出现过的标签一条，`display_name` 缺省） | |
| `Turn.start` / `end` | `segments[].start` / `end` | 同为秒浮点，直传 |
| —（无文本） | `segments[].text = ""` | **speaker-only 文档**：diar-rs 单独导出时 text 为空串（schema 允许）。更常见的用法是**合并器**：把 Turn 按最大时间重叠打到已有转录的 segments 上（算法同 cut `diarize/assign.rs` 的 `match_paragraph`，coverage ≥ 0.5、margin ≥ 0.15 才写入） |
| `DiarizeResult.method` | `provenance.engine` = `"diar-rs/open"`，`method` 全文入 `provenance.extra.method` | |
| `n_turns/n_chunks/n_xvec/elapsed_sec/frame_hz` | `provenance.extra.*`（可选） | 诊断信息，opaque |

### 2.4 lumen-asr `SessionRecord` → `lumen-transcript.v1`

来源：`/Users/chris/source/lumen-asr/crates/lumen-core/src/types.rs`。

| lumen-asr | lumen-transcript.v1 | 说明 |
|---|---|---|
| `corrected`（否则 `asr_raw`） | 单个 segment 的 `text` | 导出取**校正后**文本；原始文本入 `provenance.extra.asr_raw` 备查。`pasted` 不导出（那是插入行为的产物） |
| `audio_path` | `media.path` | |
| —（无时长字段） | `segments[0].start = 0`，`end = media.duration_seconds` | 需探测 wav 头取时长，见 §4 #7 |
| `asr_engine` | `provenance.engine` | |
| `corrector_engine` | `provenance.extra.corrector_engine` | |
| `created_at` | `provenance.created_at` | |
| `id` | `segments[0].id` = session uuid | |
| `focus`（app/窗口） | `provenance.extra.focus`（可选） | 隐私敏感，默认建议不导出 |
| — | `provenance.app` = `"lumen-asr"` | |

---

## 3. 示例

### 3.1 纯转录（lumen-navi 导出一个音频 session，无说话人、无词级 timing）

```json
{
  "schema": "lumen-transcript.v1",
  "provenance": {
    "app": "lumen-navi",
    "app_version": "0.3.0",
    "engine": "speech",
    "language": "zh-CN",
    "created_at": "2026-07-25T09:30:00Z"
  },
  "media": {
    "duration_seconds": 92.5,
    "sample_rate": 16000,
    "channels": 1,
    "bytes": 2960000,
    "content_hash": "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
  },
  "segments": [
    {
      "id": "0d2c8a1e-6b7f-4a2e-9c1d-8f3b5a7e2c10",
      "start": 0.0,
      "end": 30.0,
      "text": "今天先对一下上周的进展，然后排后面两周的计划。",
      "confidence": 0.91
    },
    {
      "id": "7a4f2b90-13cd-4e58-8a6b-2d9e0c4f1b33",
      "start": 30.0,
      "end": 61.5,
      "text": "存储层的迁移已经完成，索引重建还差最后一步。",
      "confidence": 0.88
    },
    {
      "id": "c91d3e57-2f80-4b1a-b7c4-6e5a8d0f9a21",
      "start": 61.5,
      "end": 92.5,
      "text": "剩下的风险主要在回放兼容性上，需要再加一轮测试。",
      "confidence": 0.86
    }
  ]
}
```

### 3.2 会议场景（说话人 + 词级 timing + 翻译扩展位）

```json
{
  "schema": "lumen-transcript.v1",
  "provenance": {
    "app": "lumen-cut",
    "engine": "pyannote",
    "model": "pyannote/speaker-diarization-3.1",
    "language": "zh",
    "created_at": "2026-07-25T10:12:34Z"
  },
  "media": {
    "path": "/Users/chris/Movies/standup-2026-07-25.mp4",
    "duration_seconds": 7.2,
    "sample_rate": 44100,
    "channels": 2
  },
  "speakers": [
    { "id": "SPEAKER_00", "display_name": "Alice" },
    { "id": "SPEAKER_01" }
  ],
  "segments": [
    {
      "id": "seg-1",
      "start": 0.32,
      "end": 2.9,
      "text": "大家好，我们开始今天的站会。",
      "speaker": "SPEAKER_00",
      "confidence": 0.94,
      "words": [
        { "word": "大家", "start": 0.32, "end": 0.78 },
        { "word": "好", "start": 0.78, "end": 1.02 },
        { "word": "我们", "start": 1.35, "end": 1.7 },
        { "word": "开始", "start": 1.7, "end": 2.1 },
        { "word": "今天", "start": 2.1, "end": 2.45 },
        { "word": "的", "start": 2.45, "end": 2.55 },
        { "word": "站会", "start": 2.55, "end": 2.9 }
      ],
      "translations": {
        "en": "Hi everyone, let's start today's standup."
      }
    },
    {
      "id": "seg-2",
      "start": 3.4,
      "end": 7.1,
      "text": "好的，我先说，昨天把导入功能收尾了。",
      "speaker": "SPEAKER_01",
      "confidence": 0.9,
      "words": [
        { "word": "好的", "start": 3.4, "end": 3.8, "confidence": 0.97 },
        { "word": "我", "start": 4.05, "end": 4.2 },
        { "word": "先", "start": 4.2, "end": 4.4 },
        { "word": "说", "start": 4.4, "end": 4.62 },
        { "word": "昨天", "start": 5.0, "end": 5.4 },
        { "word": "把", "start": 5.4, "end": 5.55 },
        { "word": "导入", "start": 5.55, "end": 6.0 },
        { "word": "功能", "start": 6.0, "end": 6.45 },
        { "word": "收尾", "start": 6.45, "end": 6.9 },
        { "word": "了", "start": 6.9, "end": 7.1 }
      ],
      "translations": {
        "en": "Sure, I'll go first — I wrapped up the import feature yesterday."
      }
    }
  ]
}
```

---

## 4. 阻抗失配与取舍

1. **navi 没有 segment 时间**。`transcript.v1` derived doc 只有整 chunk 的文本，时间信息
   在 `SourceEvent.ts`（事件层）而不在转录体里。取舍：`start/end` 保持**必填**（cut 的
   时间轴导入离不开它），由 navi 导出器合成 —— chunk 起点 = 事件 ts − session 起点，
   时长 = `audio_bytes / (sample_rate × 2)`（16 kHz mono s16）。代价：段内无词级精度，
   段边界精度取决于 chunk 切分。
2. **navi 单 chunk 单文本 vs 交换格式句子级 segment**。navi 的一个 chunk（约 30 s）导出
   为一个粗 segment，不强行断句。cut 导入后可用其现有断句/重排能力细化。
3. **cut 的一切时间都挂在 word 上**。`Doc` 的段落起止、字幕 cue、`diarize/assign.rs` 的
   speaker 重叠匹配全部从 `Word.start/end` 推导，而 `Sentence.words` 无处存句级时间。
   取舍：导入无词 timing 的 segment 时**合成一个横跨 [start,end] 的伪 word**（text = 句文本）。
   字幕逐词高亮在这种项目里退化为整句高亮 —— 可接受，且与 rebind/soft-cut 兼容。
4. **diar-rs speaker 是 u32，其他都是 string**。统一为 string id；diar-rs 导出映射为
   `"S{n+1}"`（与它自己 abs-timeline 的 `Speaker1` 编号一致），pyannote 保留 `SPEAKER_00`。
   聚类标签跨文件不稳定 —— 因此 `display_name` 与 id 分离，重跑 diarization 只会换 id，
   人工命名挂在表上可迁移。
5. **confidence 单向丢失**。navi/lumen-asr 产 confidence，cut `Doc` 无处存放。取舍：导入时
   丢弃（或整体塞进 doc.json 顶层未知字段区留档，cut 的 unknown-field 保留机制会带着走）。
   不为它扩展 cut 数据模型 —— 剪辑流程用不到。
6. **翻译粒度**。cut 的翻译按 `(lang, group_id)` 组织且依赖 `source_words` 词 id 做陈旧
   检测；交换格式只有 per-segment `lang → text`（词 id 是 cut 内部生成的，交换层不可能有）。
   取舍：导入时 group_id 取生成的 sentence id，`source_words` 留空、`source_text` 填
   segment 原文 —— 陈旧检测降级为文本比对，功能不缺失只是粒度变粗。
7. **lumen-asr 无时长、无分段**。`SessionRecord` 只有 `audio_path` 和三份文本
   （raw/corrected/pasted）。取舍：导出单 segment `[0, duration]`，duration 靠读 wav 头；
   文本取 `corrected`（用户实际认可的版本），`asr_raw` 进 `provenance.extra`。
8. **speaker 存 id 还是 display_name 进 cut**。cut 的 `paragraph.speaker` 就是展示字符串
   （rename/merge 直接改写它）。取舍：导入时写 `display_name ?? id`，不在 cut 里维护
   id→name 双层结构；交换文件本身保留双层，信息不丢。
9. **大小写风格**。navi derived / cut sidecar（asr_out/diarize_out）是 snake_case，cut
   `doc.json` 是 camelCase。交换格式选 snake_case：它站在 sidecar 输出那一边（生产边界），
   camelCase 是 cut 项目文件的内政。

---

## 5. 各产品接入点建议

### lumen-navi（产出方）

- 新增 `crates/lumen-process/src/transcript_export.rs`：
  `export_session_transcript(store, session_id) -> LumenTranscriptV1`。
  按 `audio_session.v1` 找到 session 内全部 `audio_chunk.v1` 事件，读各自的
  `transcript.v1` derived（复用 `transcribe_worker.rs` 的 `DERIVED_TRANSCRIPT_V1` 常量），
  按事件 ts 排序合成 segments（时间合成规则见 §4 #1）。
- 序列化类型建议放 `crates/lumen-types/src/`（新 `transcript.rs` 模块），worker 与导出器共用。
- CLI/UI 入口：navi 的 session 详情处加 "Export transcript…"，落盘
  `<session>.lumen-transcript.json`。

### lumen-asr（产出方）

- 新增 `crates/lumen-core/src/export.rs`：`SessionRecord -> LumenTranscriptV1`
  （映射见 §2.4），从历史记录 UI 挂 "Export"。

### diar-rs（产出方 / 增强方）

- `crates/diar-rs/src/io.rs` 加 `write_lumen_transcript(result, path)`：speaker-only 文档
  （text 为空串）。
- 更有用的是合并模式：`merge_into_transcript(turns, &mut transcript)` —— 按最大重叠给已有
  segments 打 `speaker`，并生成 `speakers` 表（阈值直接沿用 cut `assign.rs` 的
  `MIN_SPEAKER_COVERAGE = 0.5` / `MIN_SPEAKER_MARGIN = 0.15`）。

### lumen-cut（消费方）

- `src-tauri/src/asr/mod.rs` 旁新增 `src-tauri/src/import/mod.rs`：
  - `LumenTranscriptV1` serde 类型 + `TryFrom<LumenTranscriptV1> for Doc`
    （分组/伪 word/翻译映射见 §2.2；放 import 模块而不是塞进 `From<AsrOutV1>` 旁边，
    因为它还要写 `translations` 和处理 speaker 表）。
  - 导入时若 `media.path` 存在且文件在，直接绑定；否则走现有 rebind 流程
    （`src-tauri/src/data/rebind.rs`），可用 `media.content_hash` 校验。
- Tauri command `import_transcript(path)` + CLI 子命令 `lumen-cut import <file>`，
  UI 上呈现为 "Import from Navi/Voice"。
- 说话人：导入后 `data/speakers.rs` 的 rename/merge 原样可用，无需改动。

### 校验

任何一方的 CI 里可用：

```bash
python3 -c "import json, jsonschema, sys; \
  jsonschema.validate(json.load(open(sys.argv[1])), \
  json.load(open('contracts/lumen-transcript.v1.schema.json')))" <file>
```
