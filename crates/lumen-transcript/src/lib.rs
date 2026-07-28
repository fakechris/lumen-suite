//! Shared serde types for the `lumen-transcript.v1` interchange format.
//!
//! The canonical definition of the format is the JSON Schema at
//! `contracts/lumen-transcript.v1.schema.json` (embedded here as
//! [`SCHEMA_JSON`]); this crate is its Rust binding, shared by the
//! producers (lumen-navi, lumen-asr, diar-rs) and the consumer (lumen-cut)
//! so nobody hand-writes the structs.
//!
//! Design notes:
//! - All times are seconds from the start of the referenced media (`f64`).
//! - The format evolves additively (`additionalProperties: true`): unknown
//!   fields at every level are captured in `unknown_fields` via
//!   `#[serde(flatten)]` and survive a deserialize → serialize round trip.
//! - The top-level `schema` field is the format discriminator. It always
//!   serializes as [`SCHEMA_ID`] and deserialization fails for any other
//!   value (a future v2 gets its own file and its own types).

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{Error as _, Unexpected};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

/// The format discriminator value: the only accepted top-level `"schema"`.
pub const SCHEMA_ID: &str = "lumen-transcript.v1";

/// The canonical JSON Schema (draft 2020-12) for this format, embedded from
/// `contracts/lumen-transcript.v1.schema.json` in the repository.
pub const SCHEMA_JSON: &str = include_str!("../../../contracts/lumen-transcript.v1.schema.json");

/// Marker for the top-level `"schema"` field.
///
/// Serializes as the constant [`SCHEMA_ID`]; deserialization rejects any
/// other value, so a `TranscriptV1` can only ever represent a v1 document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SchemaTag;

impl Serialize for SchemaTag {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(SCHEMA_ID)
    }
}

impl<'de> Deserialize<'de> for SchemaTag {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        if value == SCHEMA_ID {
            Ok(SchemaTag)
        } else {
            Err(D::Error::invalid_value(Unexpected::Str(&value), &SCHEMA_ID))
        }
    }
}

impl fmt::Display for SchemaTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(SCHEMA_ID)
    }
}

/// A complete `lumen-transcript.v1` document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptV1 {
    /// Always `"lumen-transcript.v1"`; see [`SchemaTag`].
    pub schema: SchemaTag,
    /// Where this transcript came from. Informational only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    /// Reference to the source media (never embedded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<Media>,
    /// Speaker table referenced by `segments[].speaker`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speakers: Option<Vec<Speaker>>,
    /// Time-ordered transcript units (roughly sentence/cue granularity).
    pub segments: Vec<Segment>,
    /// Unknown top-level fields, preserved for round-tripping.
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

impl TranscriptV1 {
    /// A minimal valid document containing the given segments.
    pub fn new(segments: Vec<Segment>) -> Self {
        Self {
            schema: SchemaTag,
            provenance: None,
            media: None,
            speakers: None,
            segments,
            unknown_fields: Map::new(),
        }
    }

    /// Build a document from `(start, end, text)` triples, one segment each.
    pub fn from_timed_texts<I, S>(items: I) -> Self
    where
        I: IntoIterator<Item = (f64, f64, S)>,
        S: Into<String>,
    {
        Self::new(
            items
                .into_iter()
                .map(|(start, end, text)| Segment::new(start, end, text))
                .collect(),
        )
    }

    /// Set the provenance block (builder style).
    #[must_use]
    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Set the media reference (builder style).
    #[must_use]
    pub fn with_media(mut self, media: Media) -> Self {
        self.media = Some(media);
        self
    }

    /// Set the speaker table (builder style).
    #[must_use]
    pub fn with_speakers(mut self, speakers: Vec<Speaker>) -> Self {
        self.speakers = Some(speakers);
        self
    }

    /// Parse a document from a JSON string, rejecting non-v1 documents.
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Serialize to a compact JSON string.
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json_string_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Where a transcript came from. Purely informational.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Provenance {
    /// Producing application, e.g. `"lumen-navi"`, `"lumen-asr"`, `"diar-rs"`.
    pub app: String,
    /// Producing application version, e.g. `"0.4.2"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_version: Option<String>,
    /// ASR/diarization engine identifier, e.g. `"sensevoice_sherpa"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// Stable model name without local filesystem paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Immutable model revision/commit when the runtime exposes one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    /// Primary language as a BCP-47 tag, e.g. `"zh-CN"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// RFC 3339 timestamp of when this document was produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Free-form engine diagnostics. Opaque to consumers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, Value>>,
    /// Unknown fields, preserved for round-tripping.
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

impl Provenance {
    /// Provenance with only the required `app` set.
    pub fn new(app: impl Into<String>) -> Self {
        Self {
            app: app.into(),
            ..Self::default()
        }
    }
}

/// Reference to the source media. The media itself is never embedded.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Media {
    /// Path to the source audio/video file, when one exists on disk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Total media duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Audio sample rate in Hz, e.g. 16000.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Audio channel count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u32>,
    /// Size of the referenced media in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Algorithm-prefixed content hash, e.g. `"blake3:9f8a…"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Unknown fields, preserved for round-tripping.
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

/// An entry in the speaker table.
///
/// The reserved v1 slots `voiceprint`/`enrollment` (producers MUST NOT emit,
/// consumers MUST ignore) are intentionally not modeled; if present they are
/// carried through `unknown_fields`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speaker {
    /// Stable speaker id within this document, e.g. `"S1"`, `"SPEAKER_00"`.
    pub id: String,
    /// Human-assigned name, e.g. `"Alice"`. Absent until labeled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Unknown fields (including reserved slots), preserved for round-tripping.
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

impl Speaker {
    /// A speaker with only the required `id` set.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: None,
            unknown_fields: Map::new(),
        }
    }

    /// Set the human-assigned display name (builder style).
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }
}

/// One time-ordered transcript unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// Optional stable segment id assigned by the producer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Segment start, seconds from media start.
    pub start: f64,
    /// Segment end, seconds from media start. MUST be `>= start`.
    pub end: f64,
    /// Transcript text. May be empty for speaker-only (diarization) segments.
    pub text: String,
    /// Speaker id referencing `speakers[].id`. Absent when unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// Engine confidence in `[0, 1]`; comparable only within one engine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Per-segment BCP-47 language override (code-switching).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Optional word-level timing tiling the segment in order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,
    /// Translations of `text`, keyed by BCP-47 language tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub translations: Option<BTreeMap<String, String>>,
    /// Unknown fields, preserved for round-tripping.
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

impl Segment {
    /// A segment with the required fields set and everything else absent.
    pub fn new(start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            id: None,
            start,
            end,
            text: text.into(),
            speaker: None,
            confidence: None,
            language: None,
            words: None,
            translations: None,
            unknown_fields: Map::new(),
        }
    }

    /// Set the producer-assigned segment id (builder style).
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the speaker id (builder style).
    #[must_use]
    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        self.speaker = Some(speaker.into());
        self
    }

    /// Set the engine confidence (builder style).
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set the per-segment language override (builder style).
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set word-level timing (builder style).
    #[must_use]
    pub fn with_words(mut self, words: Vec<Word>) -> Self {
        self.words = Some(words);
        self
    }

    /// Add one translation keyed by BCP-47 language tag (builder style).
    #[must_use]
    pub fn with_translation(mut self, lang: impl Into<String>, text: impl Into<String>) -> Self {
        self.translations
            .get_or_insert_with(BTreeMap::new)
            .insert(lang.into(), text.into());
        self
    }
}

/// Word-level timing within a segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    /// Word surface form (for CJK engines possibly one character/token).
    pub word: String,
    /// Word start, seconds from media start.
    pub start: f64,
    /// Word end, seconds from media start. MUST be `>= start`.
    pub end: f64,
    /// Optional per-word confidence in `[0, 1]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Unknown fields, preserved for round-tripping.
    #[serde(flatten)]
    pub unknown_fields: Map<String, Value>,
}

impl Word {
    /// A word with the required fields set.
    pub fn new(word: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            word: word.into(),
            start,
            end,
            confidence: None,
            unknown_fields: Map::new(),
        }
    }

    /// Set the per-word confidence (builder style).
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

#[cfg(feature = "validate")]
mod schema_validation {
    use std::sync::OnceLock;

    use jsonschema::Validator;
    use serde_json::Value;

    fn validator() -> &'static Validator {
        static VALIDATOR: OnceLock<Validator> = OnceLock::new();
        VALIDATOR.get_or_init(|| {
            let schema: Value =
                serde_json::from_str(crate::SCHEMA_JSON).expect("embedded schema is valid JSON");
            jsonschema::validator_for(&schema).expect("embedded schema compiles")
        })
    }

    /// Validate a JSON value against the embedded `lumen-transcript.v1`
    /// JSON Schema. Returns every violation as a human-readable
    /// `"<instance path>: <message>"` string.
    ///
    /// Only available with the `validate` feature.
    pub fn validate(value: &Value) -> Result<(), Vec<String>> {
        let errors: Vec<String> = validator()
            .iter_errors(value)
            .map(|err| format!("{}: {}", err.instance_path, err))
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(feature = "validate")]
pub use schema_validation::validate;
