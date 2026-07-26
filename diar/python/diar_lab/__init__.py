"""Open-source speaker diarization lab (diar_lab).

Pipeline (open ONNX weights):
  - Kaldi-native fbank (WeSpeaker-compatible)
  - Official BUT VBx algorithm (Landini et al. 2022)
  - AHC init + VB-HMM refine
"""

from .pipeline import diarize, DiarizationResult

__all__ = ["diarize", "DiarizationResult"]
