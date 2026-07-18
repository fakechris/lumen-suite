"""Kaldi-native fbank matching WeSpeaker / EmbResnet34 input.

WeSpeaker ResNet34 expects 80-dim log-mel, 25ms window / 10ms shift @ 16 kHz.
kaldi-native-fbank is the closest open implementation to FbankFast.
"""

from __future__ import annotations

import numpy as np


def compute_fbank_knf(
    pcm: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    dither: float = 0.0,
    snip_edges: bool = True,
    subtract_mean: bool = True,
    int16_scale: bool = False,
) -> np.ndarray:
    """Return float32 [T, 80] log-mel fbank.

    Args:
      int16_scale: If True and |pcm|≤1.5, multiply by 32768 before knf
        (legacy float→int16 frontend path). Default False: knf on float
        [-1,1] empirically clusters better with this emb.onnx export.
    """
    import kaldi_native_fbank as knf

    x = np.asarray(pcm, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 1.5:
        # Native C++: pcm_float * 32768 → int16 → FbankFast
        # knf + emb.onnx: float [-1,1] is the better-calibrated path here
        wav = (x * 32768.0) if int16_scale else x
    else:
        # already int16-range float
        wav = x if int16_scale else (x / 32768.0)

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sample_rate)
    opts.frame_opts.frame_length_ms = float(frame_length_ms)
    opts.frame_opts.frame_shift_ms = float(frame_shift_ms)
    opts.frame_opts.dither = float(dither)
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.window_type = "hamming"  # WeSpeaker ResNet34 common
    opts.frame_opts.snip_edges = bool(snip_edges)
    opts.mel_opts.num_bins = int(num_mel_bins)
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0  # Kaldi: nyquist + offset
    opts.use_energy = False
    opts.use_log_fbank = True
    opts.use_power = True

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, wav.astype(np.float32).tolist())
    fbank.input_finished()

    frames = []
    for i in range(fbank.num_frames_ready):
        frames.append(fbank.get_frame(i))
    if not frames:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    feats = np.stack(frames, axis=0).astype(np.float32)
    if subtract_mean and feats.shape[0] > 0:
        feats = feats - feats.mean(axis=0, keepdims=True)
    return feats
