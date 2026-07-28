# Vendored upstream sources

This directory contains third-party C/C++ sources vendored into `diar-rs` and
compiled statically by `build.rs`. No changes were made to the upstream source
files (they are copied verbatim); only test files and build-system files were
omitted.

## kaldi-native-fbank

- Repository: https://github.com/csukuangfj/kaldi-native-fbank
- Commit: `b09e686fe2084732ddd30d1ef80acfc0f13eaf01` (Release v1.22.3)
- License: Apache-2.0 (see `LICENSE-kaldi-native-fbank.Apache-2.0.txt`)
- Vendored path: `kaldi-native-fbank/csrc/`
- Contents: all non-test `*.cc` / `*.h` from upstream `kaldi-native-fbank/csrc/`.
  Excluded: `test-*.cc` (gtest-dependent test files), `CMakeLists.txt`,
  `CPPLINT.cfg`.

## kissfft

- Repository: https://github.com/mborgerding/kissfft
- Commit: `febd4caeed32e33ad8b2e0bb5ea77542c40f18ec`
- License: BSD-3-Clause (see `LICENSE-kissfft.BSD-3-Clause.txt`)
- Vendored path: `kissfft/`
- Contents: `kiss_fft.c`, `kiss_fft.h`, `kiss_fftr.c`, `kiss_fftr.h`,
  `_kiss_fft_guts.h`. This is the FFT backend used by
  `kaldi-native-fbank/csrc/rfft.cc`.
