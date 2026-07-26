/* Minimal C API over knf::OnlineFbank (kaldi-native-fbank). */
#ifndef DIAR_KNF_C_API_H_
#define DIAR_KNF_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KnfOnlineFbank KnfOnlineFbank;

/* Create OnlineFbank with WeSpeaker-style options (see Python fbank.py). */
KnfOnlineFbank *knf_online_fbank_new(float samp_freq, float frame_length_ms,
                                     float frame_shift_ms, float dither,
                                     float preemph, int remove_dc,
                                     const char *window_type, int snip_edges,
                                     int num_bins, float low_freq,
                                     float high_freq, int use_energy,
                                     int use_log_fbank, int use_power);

void knf_online_fbank_accept(KnfOnlineFbank *f, float sampling_rate,
                             const float *waveform, int n);
void knf_online_fbank_input_finished(KnfOnlineFbank *f);
int knf_online_fbank_num_frames(const KnfOnlineFbank *f);
int knf_online_fbank_dim(const KnfOnlineFbank *f);
/* Copy frame i into out[0..dim). Returns 0 on success, -1 on error. */
int knf_online_fbank_get_frame(const KnfOnlineFbank *f, int i, float *out);
void knf_online_fbank_free(KnfOnlineFbank *f);

#ifdef __cplusplus
}
#endif

#endif /* DIAR_KNF_C_API_H_ */
