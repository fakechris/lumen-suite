# Lumen Provider Catalog（provider-catalog.v1.json）

Lumen 产品簇统一的 LLM / 机器翻译 / ASR provider 目录 —— **"provider catalog as data"**。
一份 JSON 数据 + 一份 JSON Schema，替代四处各自维护、已经发散的硬编码预设。

- 数据：`contracts/provider-catalog.v1.json`（26 个 provider 条目）
- Schema：`contracts/provider-catalog.schema.json`（JSON Schema draft 2020-12）

## 1. 目的

四个代码位置各自维护 provider 预设，且已互相发散（见 §6 差异清单）：

| # | 消费方 | 语言 | 源文件 |
|---|--------|------|--------|
| 1 | lumen-translation engines | TypeScript | `packages/engines/src/providers.ts`（`PROVIDER_CATALOG`，最完整底本） |
| 2 | lumen-cut | TypeScript | `src/llmProviders.ts`（`LLM_PROVIDER_PRESETS`） |
| 3 | lumen-asr desktop | Rust | `apps/desktop/src-tauri/src/provider_presets.rs`；行为 quirk 参考 `crates/lumen-corrector/src/openai_compat.rs` 的 `inject_no_thinking_params` |
| 4 | lumen-translation popclip | Swift | `apps/popclip-window/LumenTranslation/Preferences.swift`（`Providers.catalog`） |

本目录以 lumen-translation 的 `PROVIDER_CATALOG`（13 家国产/聚合厂商）为主底本，用 cut / asr / popclip 补充
OpenAI、Anthropic、Gemini、Ollama、LM Studio、StepFun、MiMo、免费 MT（Google/Microsoft）、
Soniox 及本地 ASR 引擎，并把 asr 的 per-provider 行为标志（关闭 thinking 的注入方式等）沉淀为 `quirks` 数据。

**不收录的条目**：各消费方 UI 的哨兵项（cut 的 `custom`、asr 的 `openai_compatible`/`none`/ASR `custom`）
不是真实 provider，由各 App 自行保留；catalog 只含真实厂商/引擎。

## 2. 顶层结构

```json
{
  "spec": "lumen.provider-catalog/v1",   // 格式判别符，v1 内不变
  "version": "1.0.0",                    // 数据 semver
  "generated_from": [...],               // 来源溯源
  "providers": [ ... ]
}
```

## 3. 字段语义

### Provider 条目

| 字段 | 必填 | 语义 |
|------|------|------|
| `id` | 是 | 规范稳定 id（`^[a-z0-9_]+$`）。v1 内**永不改名、永不删除** |
| `aliases` | 否 | 四个消费方历史上用过的 id（如 `glm-cn`、`zhipu`、`volcengine`、`google`），供迁移映射 |
| `display_name` | 是 | `{ en, zh? }` 双语显示名 |
| `api_style` | 是 | 线协议：`openai_compat` / `anthropic`（Messages API）/ `ollama`（openai_compat + Ollama 原生透传参数）/ `google_translate` / `microsoft_translator` / `asr_only`（无 chat 协议，见 `asr.api_style`）/ `local`（进程内引擎，无网络端点） |
| `region` | 是 | `cn`（大陆运营）/ `global`（海外）/ `both`（有独立 cn+global 端点）/ `local`（本机） |
| `capabilities` | 是 | `chat` / `audio_transcription` / `translation` 的子集 |
| `endpoints` | 条件必填 | `{ cn?, global?, local? }`，每项 `{ base_url, notes? }`。**openai_compat/anthropic/ollama 的 base_url 不含 chat 路径**（拼接 `chat_path`）；MT 类 base_url 即完整端点。`local`/`asr_only` 风格可省略 |
| `chat_path` | 否 | 拼在 base_url 后的 chat 路径。默认 `/chat/completions`；anthropic 为 `/messages` |
| `default_model` | 否 | 推荐默认模型（选取原则：取最新维护的来源；分歧见 §6） |
| `models` | 否 | UI 下拉候选（非穷举，允许用户手填） |
| `needs_key` | 是 | 是否需要 API key |
| `auth` | 否 | `{ header, value_template }`，`{key}` 被替换为用户 key。**缺省 = `Authorization: Bearer {key}`**。目前仅 anthropic 覆盖为 `x-api-key: {key}` |
| `extra_headers` | 否 | 每次请求都发送的静态头（OpenRouter 归因头、`anthropic-version`） |
| `docs_url` | 否 | 取 key / 文档页 |
| `quirks` | 否 | per-provider 行为标志，见下 |
| `asr` | 否 | 语音转写子配置，见下 |
| `notes` | 否 | 人类可读备注（含模型退役、snapshot 过期等运营事实） |

### `quirks`（源自 lumen-corrector `inject_no_thinking_params` + 各 TS 源）

| 字段 | 语义 |
|------|------|
| `no_thinking.strategy` | 目前只有 `body_params` |
| `no_thinking.body_params` | 深合并进 chat 请求体以关闭思考链的 JSON。例：ollama `{"think":false,"options":{...}}`；qwen `{"enable_thinking":false,"think":false}`；deepseek/minimax `{"thinking":{"type":"disabled"}}`（minimax 另加 `"reasoning_split":true`）；gemini `{"extra_body":{"google":{"thinking_config":{"thinking_budget":0}}}}`；openrouter `{"reasoning":{"effort":"none","exclude":true}}` |
| `no_thinking.model_filter` | 大小写不敏感子串数组；存在时仅当模型名命中才注入（deepseek: `reasoner`/`r1`/`thinking`）。缺省 = 无条件注入 |
| `thinking_not_disableable_models` | 参数被忽略、无法关思考的模型（MiniMax M2.x / `*-highspeed`），响应侧必须清洗 |
| `strip_thinking_tags` | 响应可能含 `<think>`/`<thinking>` 块，客户端需剥离（对照 `openai_compat.rs` 的 `strip_thinking_tags`；lumen-asr 实际对所有响应都跑一遍，此标志标记"已知会出现"的厂商） |
| `legacy_endpoints` | 仍被部分消费方使用的废弃端点完整 URL（MiniMax `text/chatcompletion_v2`，含旧海外域名 `api.minimax.chat`） |
| `attribution_headers_configurable` | `extra_headers` 是可被调用方覆盖的中性默认值（OpenRouter `HTTP-Referer`/`X-Title`，换更高限流） |

### `asr`

`{ api_style: openai_audio | websocket | http_batch | local, base_url?, default_model?, models?, status: wired | config_only, notes? }`
`openai_audio` = `POST {base_url}/audio/transcriptions`；`status` 沿用 lumen-asr 语义（`config_only` = UI 可选但客户端未接）。

## 4. 四个消费方如何消费

原则：**JSON 是唯一事实源；各仓库要么直接加载，要么由脚本从 JSON 生成本地代码，禁止手改生成物或回到手写字面量。**
分发建议：各仓库 vendor 一份 `provider-catalog.v1.json`（copy，不是重写），CI 里比对与 lumen-suite 主本的哈希，漂移即报错。

### 4.1 lumen-translation（TS, `packages/engines`）

`tsconfig` 开 `resolveJsonModule`，直接 import，再写一个 ~30 行 adapter 把 catalog 条目映射回现有
`ProviderPreset` 形状（`endpoint = endpoints.cn.base_url + chat_path`，`overseasEndpoint` 同理），
`createProviderEngine` 一行不用改：

```ts
import catalog from "../../../contracts/provider-catalog.v1.json";
export const PROVIDER_CATALOG: ProviderPreset[] = catalog.providers
  .filter(p => p.capabilities.includes("chat"))
  .map(toPreset);
```

### 4.2 lumen-cut（TS）

同上：import JSON + adapter 到 `LlmProviderPreset`。`inferLlmProvider` 改为同时匹配
`endpoints.*` 与 `quirks.legacy_endpoints`，并利用 `aliases` 兼容已存储的旧 id（`minimax-cn`、`glm-global` 等）。

### 4.3 lumen-asr（Rust）

编译期嵌入 + serde，一次解析（推荐，零运行时 IO、无 build.rs 复杂度）：

```rust
static CATALOG: Lazy<Catalog> = Lazy::new(|| {
    serde_json::from_str(include_str!("../../../contracts/provider-catalog.v1.json"))
        .expect("provider catalog is validated in CI")
});
```

`llm_presets()` / `asr_presets()` 变成对 CATALOG 的过滤视图（`capabilities` 含 `chat` → LLM 预设；
含 `audio_transcription` 且有 `asr` 块 → ASR 预设）。
`inject_no_thinking_params` 改为数据驱动：按 provider 查 `quirks.no_thinking`，命中 `model_filter`
（无 filter 则无条件）就把 `body_params` 深合并进请求体——URL/模型名字符串探测只留作
自定义端点（catalog 未命中）时的 fallback。加一个单测断言：数据驱动结果 == 现有硬编码逻辑输出。

### 4.4 popclip-window（Swift）

把 JSON 作为 bundle resource 加入 Xcode target，`Codable` 解码：

```swift
struct Catalog: Decodable { let providers: [CatalogProvider] }
let url = Bundle.main.url(forResource: "provider-catalog.v1", withExtension: "json")!
let catalog = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: url))
```

`Providers.catalog` 变成 catalog 的过滤视图（保留其"精选短名单"策略：可只取
`google_translate`、`microsoft_translator` + 指定 id 白名单）。`endpointCN/endpointOverseas`
映射自 `endpoints.cn/global`。若不想引入运行时解码，也可用 build phase 脚本从 JSON 生成
`Providers.generated.swift`（生成物进 .gitignore）。

## 5. 版本演进规则（additive only）

`version` 是数据 semver；`spec`（`lumen.provider-catalog/v1`）与文件名中的 `v1` 是格式主版本。

**v1 内允许（minor/patch bump）**：
- 新增 provider、新增可选字段值（models 追加、补 docs_url/quirks/asr）；
- 修正明显数据错误（endpoint 域名、退役模型下架）→ patch；
- 标记弃用：不删条目/字段，用 `notes` 写明弃用与替代（如 MiniMax `legacy_endpoints`）。

**v1 内禁止**：
- 删除 provider、改 `id`、删已有字段、改字段含义、缩小枚举；
- 把可选字段改为必填。

需要破坏性变更时：新建 `provider-catalog.v2.json` + `spec: lumen.provider-catalog/v2`，
v1 冻结并与 v2 并存一个迁移周期。Schema 收紧（如给 `models` 加约束）视为破坏性变更。
CI 要求：每次改动跑 `python3 -m jsonschema`（或等价）校验数据 vs schema，并跑各消费方 adapter 单测。

## 6. 四处现存差异清单（合并时发现，供后续修复）

按 provider 列出；`translation` = providers.ts，`cut` = llmProviders.ts，`asr` = provider_presets.rs，`swift` = Preferences.swift。

1. **GLM/智谱 — id 三套、模型代际脱节**：id 为 `glm`（translation/swift）、`glm-cn`+`glm-global`（cut）、`zhipu`（asr）。默认模型 `glm-4-flash`（translation/asr/swift）vs `glm-5.2`/`glm-5.1`（cut）；translation/swift 只列 glm-4 家族，cut 只列 glm-5 家族。仅 cut 收录了 Z.AI 海外端点 `api.z.ai`。
2. **Kimi — 默认模型三分**：`moonshot-v1-8k`（translation/swift）、`kimi-latest`（cut）、`kimi-k2.5`（asr）；`kimi-k2.5/k2.6` 仅 asr 有。目录取 `kimi-latest`。
3. **MiniMax — API 代际、海外域名、默认模型全部分裂**：translation/swift 用旧版 `POST /v1/text/chatcompletion_v2`，海外域名 `api.minimax.chat`；cut/asr 用新版 `/v1/chat/completions`，海外域名 `api.minimax.io`（仅 cut 有海外条目）。默认模型 `MiniMax-Text-01` + abab 家族（translation/swift，已过时）vs `MiniMax-M3`（cut/asr）。cut 还拆成 `minimax-cn`/`minimax-global` 两个 id。目录取新版端点为主，旧版下沉到 `quirks.legacy_endpoints`。
4. **OpenAI — 默认与候选模型不一致**：默认 `gpt-4.1-mini`（cut）vs `gpt-4o-mini`（asr/swift）；swift 仍列 `gpt-4-turbo`/`gpt-3.5-turbo` 旧模型。translation 完全没有 OpenAI 条目。
5. **Anthropic — 三种互相矛盾的接入方式**：cut 用原生 `/v1/messages`（正确）；asr 把 `api.anthropic.com/v1` 当 openai_compatible 用（源码注释自认是占位，实际 Anthropic 不提供 `/chat/completions`，大概率不工作）；swift 干脆经 OpenRouter 路由（默认 `anthropic/claude-3.5-sonnet`，模型已老旧）。模型命名也分裂：`claude-sonnet-4-5`（cut）vs `claude-sonnet-4-6`/`claude-opus-4-6`（asr）。另外四处均未实现原生 `x-api-key` + `anthropic-version` 鉴权，目录中的 `auth` 为规范补全。
6. **豆包/火山 — id 与默认模型不同**：`doubao` + `doubao-pro-32k`（translation，注释明确选非日期别名避免 snapshot 过期）vs `volcengine` + `doubao-seed-2-0-lite`（asr，且列了会过期的 `-260428` snapshot）。
7. **Qwen — 默认模型与候选不同**：`qwen-plus`（translation/cut）vs `qwen-turbo`（asr）；`qwen3.5-flash`/`qwen3.6-flash` 仅 asr 有，`qwen2.5-72b-instruct` 仅 translation 有。
8. **SiliconFlow — 海外端点与默认模型**：`api.siliconflow.com` 海外端点仅 translation 有；默认 `deepseek-ai/DeepSeek-V3`（translation/cut）vs `DeepSeek-V3.2`（asr）。
9. **OpenRouter — 默认模型与归因头**：默认 `openai/gpt-4o-mini`（translation/asr）vs `openai/gpt-5.2`（cut）。`HTTP-Referer`/`X-Title` 归因头只有 translation（及 swift 的 anthropic-via-openrouter）发送，cut/asr 不发 → 白白损失 OpenRouter 的限流优待。
10. **Ollama — 地址、默认模型、quirk 覆盖不一致**：`localhost:11434` + `qwen3:8b`（cut）vs `127.0.0.1:11434` + `qwen3.5:9b`（asr）。只有 asr 注入 `think:false` + `options{num_ctx,num_predict}`；cut 对本地思考模型不做任何处理。
11. **no_thinking quirk 只存在于 asr**：`inject_no_thinking_params`（Ollama/Qwen/DeepSeek-reasoner/MiniMax/Gemini/OpenRouter 六类注入）只在 lumen-corrector 实现；translation/cut/swift 对同样的厂商发"思考全开"的请求，多花 token、拖慢翻译/剪辑场景。响应侧 `<think>` 剥离同样只有 asr 有。
12. **DeepSeek — 模型列表小分歧**：`deepseek-coder` 仅 translation/swift 有；`deepseek-v4-flash` 仅 asr 有；cut 只列 chat+reasoner。
13. **单一来源独有的 provider**：混元/文心/星火/百川/零一万物仅 translation 有；Gemini/StepFun/MiMo/LM Studio/Soniox 及本地 ASR 引擎仅 asr 有；免费 Google/Microsoft 翻译仅 swift 有。swift 只覆盖 translation 13 家中的 4 家（kimi/glm/minimax/deepseek）。
14. **needs_key/docs 元数据不齐**：docs_url 只有 translation/swift 维护；cut/asr 均无（asr 用自由文本 `notes` 兜底）。
15. **temperature / system prompt 不在任何 catalog 内且各处硬编码不同**：translation 的 `ProviderEngineOptions` 允许 per-call `temperature/systemPrompt`（无默认值持久化）；asr corrector 硬编码 `temperature=0.3`（clamp 0.01–1.0）+ `max_tokens=1024` + 内置 prompt 组装；swift Preferences 完全不持久化 prompt/temperature。这类"调用参数默认值"刻意**不进** v1 目录（属产品策略非厂商事实），但四处应各自显式声明默认值并对齐命名。

**目录默认模型选取原则**（有分歧时）：优先"最新且仍维护"的来源，其余全部并入 `models[]` 保留 —— 具体取舍见上表各条。
