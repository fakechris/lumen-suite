#include "knf_c_api.h"

#include <string>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"

struct KnfOnlineFbank {
  knf::OnlineFbank *impl;
};

extern "C" {

KnfOnlineFbank *knf_online_fbank_new(float samp_freq, float frame_length_ms,
                                     float frame_shift_ms, float dither,
                                     float preemph, int remove_dc,
                                     const char *window_type, int snip_edges,
                                     int num_bins, float low_freq,
                                     float high_freq, int use_energy,
                                     int use_log_fbank, int use_power) {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = samp_freq;
  opts.frame_opts.frame_length_ms = frame_length_ms;
  opts.frame_opts.frame_shift_ms = frame_shift_ms;
  opts.frame_opts.dither = dither;
  opts.frame_opts.preemph_coeff = preemph;
  opts.frame_opts.remove_dc_offset = remove_dc != 0;
  opts.frame_opts.window_type = window_type ? window_type : "hamming";
  opts.frame_opts.round_to_power_of_two = true;
  opts.frame_opts.snip_edges = snip_edges != 0;

  opts.mel_opts.num_bins = num_bins;
  opts.mel_opts.low_freq = low_freq;
  opts.mel_opts.high_freq = high_freq;
  opts.mel_opts.is_librosa = false;

  opts.use_energy = use_energy != 0;
  opts.use_log_fbank = use_log_fbank != 0;
  opts.use_power = use_power != 0;

  auto *f = new KnfOnlineFbank;
  f->impl = new knf::OnlineFbank(opts);
  return f;
}

void knf_online_fbank_accept(KnfOnlineFbank *f, float sampling_rate,
                             const float *waveform, int n) {
  if (!f || !f->impl || !waveform || n <= 0) return;
  f->impl->AcceptWaveform(sampling_rate, waveform, n);
}

void knf_online_fbank_input_finished(KnfOnlineFbank *f) {
  if (!f || !f->impl) return;
  f->impl->InputFinished();
}

int knf_online_fbank_num_frames(const KnfOnlineFbank *f) {
  if (!f || !f->impl) return 0;
  return f->impl->NumFramesReady();
}

int knf_online_fbank_dim(const KnfOnlineFbank *f) {
  if (!f || !f->impl) return 0;
  return f->impl->Dim();
}

int knf_online_fbank_get_frame(const KnfOnlineFbank *f, int i, float *out) {
  if (!f || !f->impl || !out) return -1;
  if (i < 0 || i >= f->impl->NumFramesReady()) return -1;
  const float *frame = f->impl->GetFrame(i);
  int dim = f->impl->Dim();
  for (int d = 0; d < dim; ++d) out[d] = frame[d];
  return 0;
}

void knf_online_fbank_free(KnfOnlineFbank *f) {
  if (!f) return;
  delete f->impl;
  delete f;
}

}  // extern "C"
