# lumen-transcript

Shared serde types and validation for the **`lumen-transcript.v1`** interchange
format — the transcript exchange file that lets Lumen producers
(**lumen-navi**, **lumen-asr**, **diar-rs**) hand results to the consumer
(**lumen-cut**, "Import from Navi/Voice") without each side hand-writing the
structs.

## Relationship to the schema

The **canonical definition** of the format is the JSON Schema at
[`contracts/lumen-transcript.v1.schema.json`](../../contracts/lumen-transcript.v1.schema.json)
(design rationale and field mappings in
[`contracts/TRANSCRIPT.md`](../../contracts/TRANSCRIPT.md)). This crate is that
schema's Rust binding:

- The schema file is embedded via `include_str!` and exposed as
  `lumen_transcript::SCHEMA_JSON`.
- The top-level `schema` field always serializes as the constant
  `SCHEMA_ID` (`"lumen-transcript.v1"`) and deserialization rejects any other
  value.
- The format evolves additively (`additionalProperties: true`): unknown fields
  at every level are preserved through `unknown_fields`
  (`#[serde(flatten)]`), so re-emitting a document never drops data.

If the schema and this crate ever disagree, the schema wins.

## Usage

From inside this workspace:

```toml
[dependencies]
lumen-transcript = { path = "../lumen-transcript" }
```

From the product repos (lumen-navi, lumen-asr, lumen-cut, diar-rs):

```toml
[dependencies]
lumen-transcript = { git = "https://github.com/fakechris/lumen-suite", package = "lumen-transcript" }
```

### Producing

```rust
use lumen_transcript::{Media, Provenance, Segment, Speaker, TranscriptV1, Word};

let doc = TranscriptV1::from_timed_texts([(0.0, 30.0, "今天先对一下上周的进展。")])
    .with_provenance(Provenance::new("lumen-navi"))
    .with_media(Media { duration_seconds: Some(30.0), ..Media::default() })
    .with_speakers(vec![Speaker::new("S1").with_display_name("Alice")]);

let json = doc.to_json_string_pretty()?;
```

Segments carry optional detail via builder methods:
`with_id`, `with_speaker`, `with_confidence`, `with_language`,
`with_words(vec![Word::new("好的", 3.4, 3.8)])`, `with_translation("en", "…")`.

### Consuming

```rust
let doc = lumen_transcript::TranscriptV1::from_json_str(&json)?;
// Fails if `schema` is missing or not "lumen-transcript.v1",
// or any required field (segments, start/end/text, …) is absent.
```

### Validation (feature `validate`, off by default)

```toml
lumen-transcript = { path = "../lumen-transcript", features = ["validate"] }
```

```rust
let value: serde_json::Value = serde_json::from_str(&json)?;
lumen_transcript::validate(&value)?; // Err(Vec<String>) lists every violation
```

## Development

```sh
cargo test -p lumen-transcript
cargo test -p lumen-transcript --features validate
```

Tests round-trip the two canonical examples embedded in
`contracts/TRANSCRIPT.md`, so the crate stays honest against the contract.
