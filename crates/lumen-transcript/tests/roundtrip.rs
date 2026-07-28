//! Round-trip and validation tests driven by the canonical examples in
//! `contracts/TRANSCRIPT.md`.

use lumen_transcript::{
    Media, Provenance, SchemaTag, Segment, Speaker, TranscriptV1, Word, SCHEMA_ID,
};
use serde_json::{json, Value};

const TRANSCRIPT_MD: &str = include_str!("../../../contracts/TRANSCRIPT.md");

/// Extract every fenced ```json block from TRANSCRIPT.md.
fn example_documents() -> Vec<Value> {
    TRANSCRIPT_MD
        .split("```json")
        .skip(1)
        .map(|rest| {
            let body = rest
                .split("```")
                .next()
                .expect("fenced block has a closing fence");
            serde_json::from_str(body).expect("example block is valid JSON")
        })
        .collect()
}

fn roundtrip(value: &Value) -> Value {
    let doc: TranscriptV1 =
        serde_json::from_value(value.clone()).expect("example deserializes into TranscriptV1");
    serde_json::to_value(&doc).expect("TranscriptV1 serializes")
}

#[test]
fn transcript_md_has_two_examples() {
    assert_eq!(example_documents().len(), 2);
}

#[test]
fn examples_roundtrip_losslessly() {
    for (i, example) in example_documents().iter().enumerate() {
        assert_eq!(&roundtrip(example), example, "example {} round trip", i + 1);
    }
}

#[test]
fn examples_map_onto_typed_fields() {
    let docs: Vec<TranscriptV1> = example_documents()
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();

    // Example 1: navi export, no speakers, no words.
    let navi = &docs[0];
    assert_eq!(navi.provenance.as_ref().unwrap().app, "lumen-navi");
    assert_eq!(navi.segments.len(), 3);
    assert!(navi.speakers.is_none());
    assert!(navi.segments[0].words.is_none());
    assert_eq!(
        navi.media.as_ref().unwrap().content_hash.as_deref().unwrap(),
        "blake3:af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    );
    assert!(navi.unknown_fields.is_empty());

    // Example 2: meeting with speakers, words, translations.
    let meeting = &docs[1];
    let speakers = meeting.speakers.as_ref().unwrap();
    assert_eq!(speakers[0].display_name.as_deref(), Some("Alice"));
    assert!(speakers[1].display_name.is_none());
    let seg = &meeting.segments[0];
    assert_eq!(seg.speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(seg.words.as_ref().unwrap().len(), 7);
    assert_eq!(
        seg.translations.as_ref().unwrap().get("en").unwrap(),
        "Hi everyone, let's start today's standup."
    );
    let word = &meeting.segments[1].words.as_ref().unwrap()[0];
    assert_eq!(word.word, "好的");
    assert_eq!(word.confidence, Some(0.97));
}

#[test]
fn unknown_fields_are_preserved_at_every_level() {
    let mut example = example_documents().remove(1);
    example["x_top"] = json!({"nested": [1, 2, 3]});
    example["provenance"]["x_provenance"] = json!("p");
    example["media"]["x_media"] = json!(42);
    example["speakers"][0]["x_speaker"] = json!(true);
    example["speakers"][0]["voiceprint"] = json!({"reserved": "slot"});
    example["segments"][0]["x_segment"] = json!(null);
    example["segments"][0]["words"][0]["x_word"] = json!(1.5);

    assert_eq!(roundtrip(&example), example);
}

#[test]
fn missing_required_fields_fail() {
    let mut no_segments = example_documents().remove(0);
    no_segments.as_object_mut().unwrap().remove("segments");
    assert!(serde_json::from_value::<TranscriptV1>(no_segments).is_err());

    let mut no_schema = example_documents().remove(0);
    no_schema.as_object_mut().unwrap().remove("schema");
    assert!(serde_json::from_value::<TranscriptV1>(no_schema).is_err());

    let mut no_text = example_documents().remove(0);
    no_text["segments"][0].as_object_mut().unwrap().remove("text");
    assert!(serde_json::from_value::<TranscriptV1>(no_text).is_err());

    let mut no_start = example_documents().remove(0);
    no_start["segments"][0]
        .as_object_mut()
        .unwrap()
        .remove("start");
    assert!(serde_json::from_value::<TranscriptV1>(no_start).is_err());

    let mut no_speaker_id = example_documents().remove(1);
    no_speaker_id["speakers"][0]
        .as_object_mut()
        .unwrap()
        .remove("id");
    assert!(serde_json::from_value::<TranscriptV1>(no_speaker_id).is_err());

    let mut no_word_end = example_documents().remove(1);
    no_word_end["segments"][0]["words"][0]
        .as_object_mut()
        .unwrap()
        .remove("end");
    assert!(serde_json::from_value::<TranscriptV1>(no_word_end).is_err());
}

#[test]
fn wrong_schema_discriminator_is_rejected() {
    let mut wrong = example_documents().remove(0);
    wrong["schema"] = json!("lumen-transcript.v2");
    let err = serde_json::from_value::<TranscriptV1>(wrong).unwrap_err();
    assert!(
        err.to_string().contains(SCHEMA_ID),
        "error should mention the expected discriminator: {err}"
    );
}

#[test]
fn schema_field_serializes_as_constant() {
    let doc = TranscriptV1::new(vec![Segment::new(0.0, 1.0, "hi")]);
    let value = serde_json::to_value(&doc).unwrap();
    assert_eq!(value["schema"], json!(SCHEMA_ID));
    assert_eq!(doc.schema, SchemaTag);
}

#[test]
fn builders_produce_expected_json() {
    let doc = TranscriptV1::from_timed_texts([(0.0, 1.5, "hello"), (1.5, 3.0, "world")])
        .with_provenance(Provenance::new("lumen-asr"))
        .with_media(Media {
            path: Some("/tmp/a.wav".into()),
            duration_seconds: Some(3.0),
            ..Media::default()
        })
        .with_speakers(vec![Speaker::new("S1").with_display_name("Alice")]);

    let value = serde_json::to_value(&doc).unwrap();
    assert_eq!(value["segments"][1]["text"], json!("world"));
    assert_eq!(value["speakers"][0]["display_name"], json!("Alice"));
    // Absent optionals must not serialize as null.
    assert!(value["segments"][0].get("speaker").is_none());
    assert!(value["provenance"].get("engine").is_none());
    assert!(value["media"].get("content_hash").is_none());

    // JSON string helpers round trip.
    let parsed = TranscriptV1::from_json_str(&doc.to_json_string().unwrap()).unwrap();
    assert_eq!(parsed, doc);
}

#[test]
fn segment_and_word_builders_roundtrip() {
    let seg = Segment::new(0.32, 2.9, "大家好")
        .with_id("seg-1")
        .with_speaker("SPEAKER_00")
        .with_confidence(0.94)
        .with_language("zh")
        .with_words(vec![
            Word::new("大家", 0.32, 0.78),
            Word::new("好", 0.78, 1.02).with_confidence(0.8),
        ])
        .with_translation("en", "Hi everyone");
    let doc = TranscriptV1::new(vec![seg]);
    let value = serde_json::to_value(&doc).unwrap();
    let back: TranscriptV1 = serde_json::from_value(value).unwrap();
    assert_eq!(back, doc);
}

#[cfg(feature = "validate")]
mod validate {
    use super::*;
    use lumen_transcript::validate;

    #[test]
    fn examples_pass_schema_validation() {
        for (i, example) in example_documents().iter().enumerate() {
            validate(example).unwrap_or_else(|errs| {
                panic!("example {} failed schema validation: {errs:?}", i + 1)
            });
        }
    }

    #[test]
    fn constructed_documents_pass_schema_validation() {
        let doc = TranscriptV1::from_timed_texts([(0.0, 1.0, "hello")])
            .with_provenance(Provenance::new("lumen-navi"))
            .with_speakers(vec![Speaker::new("S1")]);
        let value = serde_json::to_value(&doc).unwrap();
        validate(&value).expect("constructed document validates against the schema");
    }

    #[test]
    fn invalid_documents_fail_schema_validation() {
        // Out-of-range confidence and wrong discriminator both violate the schema.
        let mut bad = example_documents().remove(0);
        bad["segments"][0]["confidence"] = json!(1.5);
        bad["schema"] = json!("lumen-transcript.v2");
        let errs = validate(&bad).unwrap_err();
        assert!(errs.len() >= 2, "expected multiple violations: {errs:?}");
    }
}
