// Vendored kaldi-native-fbank (Apache-2.0) + kissfft (BSD-3-Clause) are
// compiled statically from source under native/knf/. This removes the former
// build-time dependency on a Python-located `kaldi-native-fbank-core` dylib
// (and its Unix-only rpath), so `cargo build` works on a clean machine with no
// Python and no preinstalled kaldi-native-fbank. See native/knf/UPSTREAM.md.

use std::path::Path;

fn main() {
    let knf_csrc = Path::new("native/knf/kaldi-native-fbank/csrc");
    let kissfft = Path::new("native/knf/kissfft");

    println!("cargo:rerun-if-changed=native/knf_c_api.cpp");
    println!("cargo:rerun-if-changed=native/knf_c_api.h");
    println!("cargo:rerun-if-changed=native/knf");

    // kissfft: C sources, default (float) scalar. Headers carry extern "C",
    // so the C++ side links against them directly.
    cc::Build::new()
        .include(kissfft)
        .file(kissfft.join("kiss_fft.c"))
        .file(kissfft.join("kiss_fftr.c"))
        .warnings(false)
        .compile("kissfft");

    // kaldi-native-fbank core + our C shim, compiled as C++. Only the fbank
    // feature chain is needed (the shim uses knf::OnlineFbank); the mfcc /
    // whisper / stft recognizers are intentionally not compiled.
    let knf_cc = [
        "feature-fbank.cc",
        "feature-functions.cc",
        "feature-window.cc",
        "kaldi-math.cc",
        "log.cc",
        "mel-computations.cc",
        "online-feature.cc",
        "rfft.cc",
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++14")
        // native/ for knf_c_api.h; native/knf for "kaldi-native-fbank/csrc/*.h";
        // kissfft dir for rfft.cc's "kiss_fftr.h".
        .include("native")
        .include("native/knf")
        .include(kissfft)
        // Match kaldi-native-fbank's CMake compile definitions.
        .define("KNF_ENABLE_CHECK", "1")
        .define("KNF_HAVE_EXECINFO_H", "1")
        .define("KNF_HAVE_CXXABI_H", "1")
        .warnings(false);
    for f in knf_cc {
        build.file(knf_csrc.join(f));
    }
    build.file("native/knf_c_api.cpp");
    build.compile("knf_c_api");
}
