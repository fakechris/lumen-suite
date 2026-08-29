//! Streaming Ogg-Opus recording and decoding for meeting tracks.
//!
//! [`OpusSink`] is the Opus twin of [`crate::meeting_recorder::WavSink`]: same
//! contract shape (`create` / `write_samples` / `finalize` / `samples_written`
//! / `sample_rate`), but samples are Opus-encoded **while recording** and muxed
//! into an Ogg container, so a one-hour meeting costs ~10× less disk than the
//! raw PCM16 WAV (~24 kbps mono at 16 kHz instead of ~1.9 MB/min per track).
//!
//! Layout choices (per Ogg Opus, RFC 7845):
//!
//! - The encoder always runs at [`OPUS_SAMPLE_RATE`] (16 kHz, mono, VoIP
//!   application mode, VBR on, ~24 kbps). Input at any other rate is linearly
//!   resampled to 16 kHz first (speech transcription quality, same trade-off
//!   the silero VAD path already makes).
//! - Granule positions and the OpusHead pre-skip are in 48 kHz units, as the
//!   spec requires; one 20 ms frame (320 samples at 16 kHz) is 960 granule
//!   units. Pre-skip is the encoder's reported lookahead, the value libopus
//!   recommends for this field.
//! - A page is ended roughly every second of audio, so a **crashed** stream is
//!   still decodable up to the last completed page — the Opus analogue of
//!   [`crate::meeting_recorder::repair_wav_header`]'s salvage. Unlike `WavSink`
//!   there is no header to repair: the OpusHead/OpusTags pages are written
//!   up-front on `create`, and `finalize` only appends the EOS page.
//!
//! [`decode_opus_to_pcm`] is the symmetric reader, and [`pcm_to_wav_bytes`]
//! renders decoded samples back into in-memory WAV bytes so downstream
//! features (playback, range editing, ASR) keep working without ffmpeg.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use ogg::writing::{PacketWriteEndInfo, PacketWriter};
use ogg::PacketReader;
use opus::{Application, Bitrate, Channels, Decoder, Encoder};

/// Sample rate the Opus encoder (and decoder) always runs at. Meeting audio
/// is speech for transcription; 16 kHz mono at ~24 kbps is the size/quality
/// sweet spot (~10× smaller than the PCM16 WAVs).
pub const OPUS_SAMPLE_RATE: u32 = 16_000;

/// Opus frame length: 20 ms — the standard VoIP frame, 320 samples at 16 kHz.
const FRAME_SAMPLES: usize = (OPUS_SAMPLE_RATE as usize) / 50;

/// Target bitrate: 24 kbps VBR, speech at 16 kHz mono.
const OPUS_BITRATE: i32 = 24_000;

/// End an Ogg page every this many audio packets (~1 s at 20 ms frames), so a
/// crashed recording loses at most ~1 s of audio and pages stay small.
const PACKETS_PER_PAGE: u32 = 50;

/// Granule units per 16 kHz sample (granule positions are 48 kHz-based).
const GRANULES_PER_SAMPLE: u64 = 48_000 / OPUS_SAMPLE_RATE as u64;

/// Ogg stream serial used for the single logical stream. Any constant works;
/// the value only has to be unique within a physical stream.
const STREAM_SERIAL: u32 = 1;

/// Incremental mono Ogg-Opus writer — see the module comment for the layout.
///
/// Crash-safety parity with [`crate::meeting_recorder::WavSink`]: a take that
/// is never finalized (process killed) is still a decodable Ogg-Opus stream up
/// to the last completed page (~1 s granularity); [`finalize`](OpusSink::finalize)
/// only writes the trailing EOS page with the exact end granule position, so
/// players report the true duration and can trim the codec pre-skip and final
/// padding. There is no header-repair equivalent to run after a crash.
pub struct OpusSink {
    writer: PacketWriter<BufWriter<File>>,
    encoder: Encoder,
    /// Native rate of the samples handed to [`write_samples`](OpusSink::write_samples).
    input_rate: u32,
    /// Samples fed in so far, at `input_rate` (what `samples_written` reports,
    /// so duration math matches `WavSink`).
    samples_written: u64,
    /// Resampler to 16 kHz, `None` when the input already is 16 kHz.
    resampler: Option<LinearResampler>,
    /// 16 kHz samples not yet filling a whole frame.
    pending: Vec<f32>,
    /// Total 16 kHz samples received (excludes final zero-padding), used for
    /// the end-of-stream granule position.
    received_16k: u64,
    /// Codec pre-skip in 48 kHz granule units (encoder lookahead × 3).
    preskip: u64,
    /// Granule position to stamp on the next audio packet.
    next_granule: u64,
    /// Audio packets written since the last page boundary.
    packets_on_page: u32,
    /// Whether the EOS page has been written.
    ended: bool,
}

impl OpusSink {
    /// Create the file and immediately write the OpusHead and OpusTags header
    /// pages (each alone on its own page, per RFC 7845). Input at a rate other
    /// than 16 kHz is transparently resampled.
    pub fn create(path: impl AsRef<Path>, sample_rate: u32) -> io::Result<Self> {
        if sample_rate == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "opus sink sample rate must be > 0",
            ));
        }
        let mut encoder = Encoder::new(OPUS_SAMPLE_RATE, Channels::Mono, Application::Voip)
            .map_err(opus_err("create encoder"))?;
        encoder
            .set_bitrate(Bitrate::Bits(OPUS_BITRATE))
            .map_err(opus_err("set bitrate"))?;
        encoder.set_vbr(true).map_err(opus_err("enable vbr"))?;
        // libopus: OPUS_GET_LOOKAHEAD "should be used" for the Ogg pre-skip.
        let lookahead = encoder
            .get_lookahead()
            .map_err(opus_err("query lookahead"))?;
        let preskip = lookahead.max(0) as u64 * GRANULES_PER_SAMPLE;

        let file = File::create(path)?;
        let mut writer = PacketWriter::new(BufWriter::new(file));

        // OpusHead: magic, version 1, mono, pre-skip, *input* rate (16 kHz,
        // informational — granule math is always 48 kHz), gain 0, mapping 0.
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.push(1);
        head.push(1);
        head.extend_from_slice(&(preskip as u16).to_le_bytes());
        head.extend_from_slice(&OPUS_SAMPLE_RATE.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(0);
        writer.write_packet(
            head.into_boxed_slice(),
            STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )?;

        // OpusTags: empty vendor string, zero user comments.
        let mut tags = Vec::with_capacity(16);
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&0u32.to_le_bytes());
        tags.extend_from_slice(&0u32.to_le_bytes());
        writer.write_packet(
            tags.into_boxed_slice(),
            STREAM_SERIAL,
            PacketWriteEndInfo::EndPage,
            0,
        )?;

        Ok(Self {
            writer,
            encoder,
            input_rate: sample_rate,
            samples_written: 0,
            resampler: (sample_rate != OPUS_SAMPLE_RATE).then(|| LinearResampler::new(sample_rate)),
            pending: Vec::with_capacity(FRAME_SAMPLES * 2),
            received_16k: 0,
            preskip,
            next_granule: preskip,
            packets_on_page: 0,
            ended: false,
        })
    }

    /// Encode and append mono `f32` samples in `[-1, 1]` at the input rate
    /// given to [`create`](OpusSink::create).
    pub fn write_samples(&mut self, samples: &[f32]) -> io::Result<()> {
        if self.ended {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "opus sink already finalized",
            ));
        }
        self.samples_written += samples.len() as u64;
        match &mut self.resampler {
            Some(resampler) => {
                let mut scratch = Vec::with_capacity(samples.len() / 2 + 1);
                resampler.push(samples, &mut scratch);
                self.received_16k += scratch.len() as u64;
                self.pending.extend_from_slice(&scratch);
            }
            None => {
                self.received_16k += samples.len() as u64;
                self.pending.extend_from_slice(samples);
            }
        }
        self.encode_full_frames(false)
    }

    /// Encode every complete buffered frame. When `stream_end` is set, a
    /// trailing partial frame is zero-padded to a full frame and the EOS page
    /// is stamped with the true end granule (excluding the padding), so
    /// decoders trim exactly back to the real length.
    fn encode_full_frames(&mut self, stream_end: bool) -> io::Result<()> {
        loop {
            if self.pending.len() < FRAME_SAMPLES {
                if !stream_end {
                    return Ok(());
                }
                // Zero-pad the final partial frame (or synthesize one silent
                // frame when the take ended exactly on a frame boundary — or
                // captured no audio at all — so the stream always has a real
                // EOS audio page). The EOS granule below excludes the padding,
                // so decoders trim this frame back away.
                self.pending.resize(FRAME_SAMPLES, 0.0);
            }
            let is_last = stream_end && self.pending.len() == FRAME_SAMPLES;
            // Granule position: pre-skip + samples through the end of this
            // frame, in 48 kHz units. For the final (possibly padded) frame
            // use the true sample count so the EOS page records real duration.
            let granule = if is_last {
                self.preskip + self.received_16k * GRANULES_PER_SAMPLE
            } else {
                self.next_granule += FRAME_SAMPLES as u64 * GRANULES_PER_SAMPLE;
                self.next_granule
            };
            let mut packet = [0u8; 4000];
            let len = self
                .encoder
                .encode_float(&self.pending[..FRAME_SAMPLES], &mut packet)
                .map_err(opus_err("encode"))?;
            self.pending.drain(..FRAME_SAMPLES);
            self.packets_on_page += 1;
            let end_info = if is_last {
                PacketWriteEndInfo::EndStream
            } else if self.packets_on_page >= PACKETS_PER_PAGE {
                self.packets_on_page = 0;
                PacketWriteEndInfo::EndPage
            } else {
                PacketWriteEndInfo::NormalPacket
            };
            self.writer
                .write_packet(packet[..len].into(), STREAM_SERIAL, end_info, granule)?;
            if is_last {
                self.ended = true;
                return Ok(());
            }
        }
    }

    /// Number of mono input samples written so far (input-rate domain, same
    /// contract as `WavSink::samples_written`).
    pub fn samples_written(&self) -> u64 {
        self.samples_written
    }

    /// Input sample rate this sink was created with (same contract as
    /// `WavSink::sample_rate`; the on-disk stream is always 16 kHz Opus).
    pub fn sample_rate(&self) -> u32 {
        self.input_rate
    }

    /// Flush the encoder (zero-padding the final partial frame), write the EOS
    /// page, and return the total number of input samples written.
    pub fn finalize(mut self) -> io::Result<u64> {
        if !self.ended {
            self.encode_full_frames(true)?;
        }
        self.writer.into_inner().flush()?;
        Ok(self.samples_written)
    }
}

fn opus_err(what: &'static str) -> impl FnOnce(opus::Error) -> io::Error {
    move |e| io::Error::other(format!("opus {what}: {e}"))
}

/// Chunk-boundary-aware linear resampler (input rate → 16 kHz). Stateless
/// per-chunk resampling would glitch at every boundary and drift in length, so
/// this carries the fractional read position and the previous chunk's last
/// sample across calls. Speech-grade, matching the VAD path's resampler.
struct LinearResampler {
    /// Input samples to step per output sample (`input_rate / 16000`).
    ratio: f64,
    /// Read position relative to the start of the *next* chunk (may be
    /// negative by < 1 to interpolate from the previous chunk's tail).
    phase: f64,
    /// Last sample of the previous chunk, for boundary interpolation.
    tail: Option<f32>,
}

impl LinearResampler {
    fn new(input_rate: u32) -> Self {
        Self {
            ratio: f64::from(input_rate) / f64::from(OPUS_SAMPLE_RATE),
            phase: 0.0,
            tail: None,
        }
    }

    /// Resample `chunk`, appending to `out`. Samples whose interpolation would
    /// need the *next* chunk's first sample are deferred to the next call.
    fn push(&mut self, chunk: &[f32], out: &mut Vec<f32>) {
        if chunk.is_empty() {
            return;
        }
        let tail = self.tail;
        let at = |i: isize| -> f32 {
            if i < 0 {
                tail.unwrap_or(chunk[0])
            } else {
                chunk[i as usize]
            }
        };
        let mut pos = self.phase;
        // Stop before the last input sample: interpolating there needs the
        // next chunk's first sample, so defer it (and everything after) via
        // `phase` to the next call.
        while pos < (chunk.len() - 1) as f64 {
            let i0 = pos.floor() as isize;
            let t = (pos - i0 as f64) as f32;
            let a = at(i0);
            let b = at(i0 + 1);
            out.push(a + (b - a) * t);
            pos += self.ratio;
        }
        self.phase = pos - chunk.len() as f64;
        self.tail = chunk.last().copied();
    }
}

/// Decode an Ogg-Opus file (as produced by [`OpusSink`]) to mono `f32` samples
/// at [`OPUS_SAMPLE_RATE`]. Returns `(samples, 16000)`.
///
/// The codec pre-skip is removed from the front and the output is trimmed to
/// the EOS page's granule position, so the returned length matches the number
/// of samples originally encoded (within one frame). A truncated stream (app
/// killed mid-recording) decodes cleanly up to the last completed page.
pub fn decode_opus_to_pcm(path: impl AsRef<Path>) -> io::Result<(Vec<f32>, u32)> {
    let file = File::open(path)?;
    let mut reader = PacketReader::new(file);
    let mut decoder: Option<Decoder> = None;
    let mut preskip: u64 = 0;
    let mut last_granule: u64 = 0;
    let mut pcm: Vec<f32> = Vec::new();

    loop {
        let packet = match reader.read_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // A truncated tail (app killed mid-recording) leaves a partial
            // page the Ogg layer rejects; salvage what decoded cleanly instead
            // of failing the whole read.
            Err(e) => {
                tracing::warn!(error = %e, "opus stream truncated; returning decoded prefix");
                break;
            }
        };
        let data = &packet.data;
        if data.starts_with(b"OpusHead") {
            if data.len() < 19 || data[9] != 1 {
                return Err(invalid("unsupported OpusHead (not mono version-1)"));
            }
            preskip = u64::from(u16::from_le_bytes([data[10], data[11]]));
            decoder = Some(
                Decoder::new(OPUS_SAMPLE_RATE, Channels::Mono)
                    .map_err(opus_err("create decoder"))?,
            );
            continue;
        }
        if data.starts_with(b"OpusTags") {
            continue;
        }
        if data.is_empty() {
            // Zero-length EOS marker page (exact-frame-boundary takes).
            continue;
        }
        let decoder = decoder
            .as_mut()
            .ok_or_else(|| invalid("audio packet before OpusHead"))?;
        // Worst case at 16 kHz is a 60 ms frame = 960 samples.
        let mut frame = [0.0f32; 960];
        let n = decoder
            .decode_float(data, &mut frame, false)
            .map_err(opus_err("decode"))?;
        pcm.extend_from_slice(&frame[..n]);
        // `absgp_page` is the granule position of the page this packet's page
        // ended at (0 when the page ends mid-packet); the last nonzero one
        // marks the true end of the stream.
        let gp = packet.absgp_page();
        if gp > 0 {
            last_granule = gp;
        }
    }
    if decoder.is_none() {
        return Err(invalid("no OpusHead found (not an Ogg-Opus file)"));
    }

    // Remove the codec pre-skip from the front (converted to 16 kHz samples).
    let skip = ((preskip * u64::from(OPUS_SAMPLE_RATE)) / 48_000) as usize;
    let skip = skip.min(pcm.len());
    pcm.drain(..skip);
    // Trim to the true length recorded by the final page's granule position.
    if last_granule >= preskip && last_granule > 0 {
        let valid = ((last_granule - preskip) * u64::from(OPUS_SAMPLE_RATE) / 48_000) as usize;
        pcm.truncate(valid);
    }
    Ok((pcm, OPUS_SAMPLE_RATE))
}

fn invalid(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Render mono `f32` samples in `[-1, 1]` as a complete PCM16 WAV byte string
/// (RIFF header + body, sizes filled in up-front). Lets the desktop hand WAV
/// bytes to playback/editing without ffmpeg once a take is Opus on disk.
pub fn pcm_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);

    let channels: u16 = 1;
    let bits: u16 = 16;
    let block_align: u16 = channels * (bits / 8);
    let byte_rate: u32 = sample_rate * u32::from(block_align);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sine at `freq` Hz, amplitude 0.3, `len` samples at `rate`.
    fn sine(len: usize, rate: u32, freq: f32) -> Vec<f32> {
        (0..len)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|&s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Best normalized cross-correlation between `a` and `b` over lag ±`max_lag`.
    fn best_correlation(a: &[f32], b: &[f32], max_lag: isize) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len() as isize;
        let mut best = f32::MIN;
        for lag in -max_lag..=max_lag {
            let (mut dot, mut ea, mut eb) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..n {
                let j = i + lag;
                if j < 0 || j >= n {
                    continue;
                }
                let (x, y) = (a[i as usize] as f64, b[j as usize] as f64);
                dot += x * y;
                ea += x * x;
                eb += y * y;
            }
            if ea > 0.0 && eb > 0.0 {
                best = best.max((dot / (ea.sqrt() * eb.sqrt())) as f32);
            }
        }
        best
    }

    #[test]
    fn opus_roundtrip_sine_and_silence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("take.opus");
        let rate = OPUS_SAMPLE_RATE;

        // 1 s sine, 0.5 s silence, 1 s sine — fed in 100 ms chunks like the
        // capture callback would.
        let mut signal = sine(rate as usize, rate, 440.0);
        signal.extend(std::iter::repeat_n(0.0, rate as usize / 2));
        signal.extend(sine(rate as usize, rate, 440.0));
        let total = signal.len();

        let mut sink = OpusSink::create(&path, rate).unwrap();
        for chunk in signal.chunks(rate as usize / 10) {
            sink.write_samples(chunk).unwrap();
        }
        assert_eq!(sink.samples_written(), total as u64);
        assert_eq!(sink.finalize().unwrap(), total as u64);

        let (pcm, decoded_rate) = decode_opus_to_pcm(&path).unwrap();
        assert_eq!(decoded_rate, rate);
        // Lossy codec + frame padding: within one 20 ms frame of the input.
        let drift = (pcm.len() as i64 - total as i64).abs();
        assert!(
            drift <= FRAME_SAMPLES as i64,
            "len {} vs {}",
            pcm.len(),
            total
        );

        // Sine segments survive loudly, the silence segment stays quiet.
        assert!(rms(&pcm[..rate as usize]) > 0.1);
        let silence = &pcm[rate as usize..rate as usize * 3 / 2];
        assert!(rms(silence) < 0.01, "silence rms {}", rms(silence));
        // Waveform fidelity (allowing a small codec lag): strong correlation.
        let corr = best_correlation(
            &pcm[rate as usize * 3 / 2..],
            &signal[rate as usize * 3 / 2..],
            200,
        );
        assert!(corr > 0.9, "correlation {corr}");

        // ~24 kbps VBR: well under PCM16 (256 kbps at 16 kHz mono).
        let bytes = std::fs::metadata(&path).unwrap().len();
        let pcm_bytes = total as u64 * 2;
        assert!(bytes * 5 < pcm_bytes, "opus {bytes} vs pcm {pcm_bytes}");
    }

    #[test]
    fn opus_duration_matches_input_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("duration.opus");
        // Odd length (2.37 s) exercises final-frame zero-padding; the EOS
        // granule must still record the true length.
        let total = (2.37 * OPUS_SAMPLE_RATE as f64) as usize;
        let signal = sine(total, OPUS_SAMPLE_RATE, 330.0);

        let mut sink = OpusSink::create(&path, OPUS_SAMPLE_RATE).unwrap();
        sink.write_samples(&signal).unwrap();
        sink.finalize().unwrap();

        let (pcm, rate) = decode_opus_to_pcm(&path).unwrap();
        let decoded_seconds = pcm.len() as f64 / rate as f64;
        let true_seconds = total as f64 / OPUS_SAMPLE_RATE as f64;
        assert!(
            (decoded_seconds - true_seconds).abs() < 0.05,
            "decoded {decoded_seconds}s vs true {true_seconds}s"
        );
    }

    #[test]
    fn opus_multipage_stream_stays_sample_aligned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.opus");
        // 3 minutes: 9000 frames → ~180 pages at 1 s per page.
        let total = 180 * OPUS_SAMPLE_RATE as usize;
        let signal = sine(total, OPUS_SAMPLE_RATE, 440.0);

        let mut sink = OpusSink::create(&path, OPUS_SAMPLE_RATE).unwrap();
        for chunk in signal.chunks(1600) {
            sink.write_samples(chunk).unwrap();
        }
        sink.finalize().unwrap();

        let (pcm, rate) = decode_opus_to_pcm(&path).unwrap();
        assert_eq!(rate, OPUS_SAMPLE_RATE);
        let drift = (pcm.len() as i64 - total as i64).abs();
        assert!(
            drift <= FRAME_SAMPLES as i64,
            "len {} vs {}",
            pcm.len(),
            total
        );
        // Alignment holds at the *end* of the multi-page stream (a granule
        // bug would accumulate drift across pages).
        let tail = OPUS_SAMPLE_RATE as usize;
        let corr = best_correlation(
            &pcm[pcm.len() - tail..],
            &signal[signal.len() - tail..],
            200,
        );
        assert!(corr > 0.9, "tail correlation {corr}");
    }

    #[test]
    fn opus_sink_resamples_non_16k_input() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wideband.opus");
        // 48 kHz mic capture: resampled to 16 kHz before encoding.
        let total = 48_000; // 1 s
        let signal = sine(total, 48_000, 440.0);

        let mut sink = OpusSink::create(&path, 48_000).unwrap();
        assert_eq!(sink.sample_rate(), 48_000);
        for chunk in signal.chunks(4800) {
            sink.write_samples(chunk).unwrap();
        }
        assert_eq!(sink.samples_written(), total as u64);
        assert_eq!(sink.finalize().unwrap(), total as u64);

        let (pcm, rate) = decode_opus_to_pcm(&path).unwrap();
        assert_eq!(rate, OPUS_SAMPLE_RATE);
        // ~1 s at 16 kHz; linear resampling may be off by a couple of samples.
        let expected = OPUS_SAMPLE_RATE as usize;
        let drift = (pcm.len() as i64 - expected as i64).abs();
        assert!(drift <= FRAME_SAMPLES as i64 + 4, "len {}", pcm.len());
        assert!(rms(&pcm) > 0.1);
    }

    #[test]
    fn opus_empty_take_is_valid_zero_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.opus");
        let sink = OpusSink::create(&path, OPUS_SAMPLE_RATE).unwrap();
        assert_eq!(sink.finalize().unwrap(), 0);

        let (pcm, rate) = decode_opus_to_pcm(&path).unwrap();
        assert_eq!(rate, OPUS_SAMPLE_RATE);
        assert!(pcm.is_empty(), "decoded {} samples", pcm.len());
    }

    #[test]
    fn opus_unfinalized_stream_decodes_completed_prefix() {
        // Crash simulation: write samples and drop the sink without finalize.
        // No EOS page, but completed pages must still decode.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("crashed.opus");
        let total = 5 * OPUS_SAMPLE_RATE as usize;
        let signal = sine(total, OPUS_SAMPLE_RATE, 440.0);

        let mut sink = OpusSink::create(&path, OPUS_SAMPLE_RATE).unwrap();
        sink.write_samples(&signal).unwrap();
        drop(sink);

        let (pcm, _) = decode_opus_to_pcm(&path).unwrap();
        // Everything up to the last page boundary (~1 s granularity) survives.
        assert!(
            pcm.len() >= 3 * OPUS_SAMPLE_RATE as usize,
            "len {}",
            pcm.len()
        );
        assert!(pcm.len() <= total + FRAME_SAMPLES, "len {}", pcm.len());
        assert!(rms(&pcm) > 0.1);
    }

    #[test]
    fn opus_sink_rejects_zero_sample_rate() {
        let dir = tempfile::tempdir().unwrap();
        let err = match OpusSink::create(dir.path().join("bad.opus"), 0) {
            Err(e) => e,
            Ok(_) => panic!("zero sample rate must fail"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn decode_rejects_non_opus_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-audio.opus");
        std::fs::write(&path, b"definitely not ogg").unwrap();
        let err = decode_opus_to_pcm(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn pcm_to_wav_bytes_produces_a_valid_wav() {
        let samples = [0.0f32, 1.0, -1.0, 0.5];
        let bytes = pcm_to_wav_bytes(&samples, 16_000);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + samples.len() * 2);

        // Round-trip through a standard WAV reader.
        let mut cursor = io::Cursor::new(bytes);
        let mut reader = hound::WavReader::new(&mut cursor).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().bits_per_sample, 16);
        let decoded: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
        assert_eq!(decoded, vec![0, 32767, -32767, 16383]);
    }

    /// Manual verification aid (not part of CI): produces a real .opus file at
    /// a stable path so `ffprobe` can be run against it during development.
    /// Run with: cargo test -p lumen-audio opus_produce_ffprobe_sample -- --ignored
    #[test]
    #[ignore]
    fn opus_produce_ffprobe_sample() {
        let path = std::path::PathBuf::from("/tmp/lumen-opus-sample.opus");
        let total = 5 * OPUS_SAMPLE_RATE as usize;
        let signal = sine(total, OPUS_SAMPLE_RATE, 440.0);
        let mut sink = OpusSink::create(&path, OPUS_SAMPLE_RATE).unwrap();
        sink.write_samples(&signal).unwrap();
        sink.finalize().unwrap();
        eprintln!("wrote {}", path.display());
    }
}
