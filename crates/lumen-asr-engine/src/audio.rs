//! Pure audio helpers: WAV decode/encode and linear resampling.
//!
//! Merges `lumen-asr/src/audio.rs` (resample only — microphone capture stays
//! product-side) and `lumen-navi/src/wav.rs` (WAV codec).

use crate::AsrError;
use std::io::Write;

/// Target sample rate expected by all offline engines.
pub const ASR_TARGET_SAMPLE_RATE: u32 = 16_000;

/// Decoded mono f32 samples in [-1, 1].
#[derive(Debug, Clone)]
pub struct DecodedPcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Parse a RIFF/WAVE PCM s16le blob. Multi-channel is averaged to mono.
pub fn decode_wav_pcm_s16le(audio: &[u8]) -> Result<DecodedPcm, AsrError> {
    if audio.len() < 44 {
        return Err(AsrError::InvalidAudio("wav too short".into()));
    }
    if &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return Err(AsrError::InvalidAudio("not a RIFF/WAVE blob".into()));
    }

    let mut offset = 12usize;
    let mut channels: u16 = 1;
    let mut sample_rate: u32 = ASR_TARGET_SAMPLE_RATE;
    let mut bits_per_sample: u16 = 16;
    let mut audio_format: u16 = 1;
    let mut data: Option<&[u8]> = None;

    while offset + 8 <= audio.len() {
        let id = &audio[offset..offset + 4];
        let size = u32::from_le_bytes(audio[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let body_start = offset + 8;
        let body_end = body_start.saturating_add(size);
        if body_end > audio.len() {
            return Err(AsrError::InvalidAudio("wav chunk overflow".into()));
        }
        let body = &audio[body_start..body_end];
        if id == b"fmt " {
            if body.len() < 16 {
                return Err(AsrError::InvalidAudio("fmt chunk too short".into()));
            }
            audio_format = u16::from_le_bytes(body[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(body[2..4].try_into().unwrap()).max(1);
            sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(body[14..16].try_into().unwrap());
        } else if id == b"data" {
            data = Some(body);
        }
        // chunks are word-aligned
        offset = body_end + (size % 2);
    }

    if audio_format != 1 {
        return Err(AsrError::InvalidAudio(format!(
            "unsupported wav format {audio_format} (need PCM)"
        )));
    }
    if bits_per_sample != 16 {
        return Err(AsrError::InvalidAudio(format!(
            "unsupported bits_per_sample {bits_per_sample} (need 16)"
        )));
    }
    let data = data.ok_or_else(|| AsrError::InvalidAudio("wav missing data chunk".into()))?;
    if data.len() < 2 {
        return Err(AsrError::InvalidAudio("empty wav data".into()));
    }

    let frame_bytes = 2 * channels as usize;
    let frames = data.len() / frame_bytes;
    let mut samples = Vec::with_capacity(frames);
    for i in 0..frames {
        let base = i * frame_bytes;
        let mut acc = 0i32;
        for ch in 0..channels as usize {
            let o = base + ch * 2;
            let s = i16::from_le_bytes([data[o], data[o + 1]]);
            acc += s as i32;
        }
        let mono = (acc / channels as i32) as i16;
        samples.push(mono as f32 / 32768.0);
    }

    if sample_rate == 0 {
        sample_rate = ASR_TARGET_SAMPLE_RATE;
    }

    Ok(DecodedPcm {
        samples,
        sample_rate,
    })
}

/// Linear resample mono f32 to `to_hz`.
pub fn resample_linear(samples: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if samples.is_empty() || from_hz == 0 || to_hz == 0 {
        return Vec::new();
    }
    if from_hz == to_hz {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    let out_len = ((samples.len() as f64) / ratio).floor().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(samples.len().saturating_sub(1));
        let t = (src - i0 as f64) as f32;
        let a = samples[i0.min(samples.len() - 1)];
        let b = samples[i1];
        out.push(a + (b - a) * t);
    }
    out
}

/// Normalize raw capture samples to 16 kHz mono for ASR engines.
///
/// Replaces `lumen_asr::prepare_for_asr(&CaptureResult)` — pass the capture's
/// sample buffer and rate directly (mic capture stays product-side).
pub fn prepare_for_asr(samples: &[f32], sample_rate: u32) -> Vec<f32> {
    resample_linear(samples, sample_rate, ASR_TARGET_SAMPLE_RATE)
}

/// Decode a WAV blob and resample to 16 kHz mono for offline engines.
pub fn prepare_for_offline_asr(audio: &[u8]) -> Result<DecodedPcm, AsrError> {
    let decoded = decode_wav_pcm_s16le(audio)?;
    let samples = resample_linear(&decoded.samples, decoded.sample_rate, ASR_TARGET_SAMPLE_RATE);
    Ok(DecodedPcm {
        samples,
        sample_rate: ASR_TARGET_SAMPLE_RATE,
    })
}

/// Encode mono f32 samples as WAV s16le (HTTP multipart upload, worker temp files).
pub fn samples_to_wav_mono_i16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    // Vec sink cannot fail short of the RIFF u32 limit, which write_wav_mono_i16 checks.
    write_wav_mono_i16(&mut out, samples, sample_rate).expect("in-memory wav encode");
    out
}

/// Stream mono f32 samples as WAV s16le into `output`.
///
/// Errors when the payload would exceed the RIFF u32 size limit.
pub fn write_wav_mono_i16(
    output: &mut impl Write,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let sample_rate = if sample_rate == 0 {
        ASR_TARGET_SAMPLE_RATE
    } else {
        sample_rate
    };
    let data_len = samples.len().saturating_mul(2);
    let data_len_u32 = u32::try_from(data_len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "WAV input exceeds the RIFF size limit",
        )
    })?;
    let mut output = std::io::BufWriter::new(output);
    output.write_all(b"RIFF")?;
    output.write_all(&36u32.saturating_add(data_len_u32).to_le_bytes())?;
    output.write_all(b"WAVEfmt ")?;
    output.write_all(&16u32.to_le_bytes())?;
    output.write_all(&1u16.to_le_bytes())?;
    output.write_all(&1u16.to_le_bytes())?;
    output.write_all(&sample_rate.to_le_bytes())?;
    output.write_all(&sample_rate.saturating_mul(2).to_le_bytes())?;
    output.write_all(&2u16.to_le_bytes())?;
    output.write_all(&16u16.to_le_bytes())?;
    output.write_all(b"data")?;
    output.write_all(&data_len_u32.to_le_bytes())?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    // From lumen-navi wav.rs (pcm_s16le_to_wav came from lumen-platform; we
    // roundtrip through our own encoder instead).
    #[test]
    fn roundtrip_pcm_wav() {
        let samples: Vec<f32> = (0..1600).map(|i| ((i % 100) as f32) / 100.0).collect();
        let wav = samples_to_wav_mono_i16(&samples, 16_000);
        let dec = decode_wav_pcm_s16le(&wav).unwrap();
        assert_eq!(dec.sample_rate, 16_000);
        assert_eq!(dec.samples.len(), samples.len());
    }

    // From lumen-asr audio.rs.
    #[test]
    fn resample_identity() {
        let s = vec![0.0, 0.5, 1.0];
        assert_eq!(resample_linear(&s, 16000, 16000), s);
    }

    // From lumen-asr audio.rs.
    #[test]
    fn resample_down() {
        let s = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_linear(&s, 32000, 16000);
        assert!(out.len() >= 2);
    }

    // From lumen-navi wav.rs.
    #[test]
    fn resample_halves_length() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let out = resample_linear(&samples, 32_000, 16_000);
        assert!((out.len() as i32 - 50).abs() <= 1);
    }

    // From lumen-asr lib.rs (`prepare_resamples`), adapted to the new signature.
    #[test]
    fn prepare_resamples() {
        let out = prepare_for_asr(&[0.0, 1.0, 0.0, -1.0], 32_000);
        assert!(!out.is_empty());
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode_wav_pcm_s16le(&[0u8; 10]).is_err());
        assert!(decode_wav_pcm_s16le(&[0u8; 64]).is_err());
    }
}
