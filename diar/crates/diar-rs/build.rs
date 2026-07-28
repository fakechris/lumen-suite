use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/knf_c_api.cpp");
    println!("cargo:rerun-if-changed=native/knf_c_api.h");
    println!("cargo:rerun-if-env-changed=KNF_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KNF_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=PYTHON");

    let (lib_dir, include_dir) = knf_paths();
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=kaldi-native-fbank-core");
    // Unix loaders can find the Python package's shared library in place.
    // MSVC does not understand `-Wl,-rpath`; Windows packaging must place the
    // corresponding DLL next to the final executable instead.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    cc::Build::new()
        .cpp(true)
        .std("c++14")
        .file("native/knf_c_api.cpp")
        .include("native")
        .include(&include_dir)
        .compile("knf_c_api");

    println!(
        "cargo:warning=knf lib={} include={}",
        lib_dir.display(),
        include_dir.display()
    );
}

fn knf_paths() -> (PathBuf, PathBuf) {
    if let (Ok(lib), Ok(inc)) = (env::var("KNF_LIB_DIR"), env::var("KNF_INCLUDE_DIR")) {
        return (PathBuf::from(lib), PathBuf::from(inc));
    }
    let python = env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let out = Command::new(&python)
        .args([
            "-c",
            r#"
import pathlib
try:
    import kaldi_native_fbank as knf
    p = pathlib.Path(knf.__file__).resolve().parent
    print(p / "lib")
    print(p / "include")
except Exception as e:
    raise SystemExit(f"kaldi_native_fbank not importable: {e}")
"#,
        ])
        .output()
        .expect("failed to run python to locate kaldi_native_fbank");
    if !out.status.success() {
        panic!(
            "Could not locate kaldi-native-fbank.\n\
             Install the Python package or set KNF_LIB_DIR / KNF_INCLUDE_DIR.\n\
             stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines().filter(|l| !l.is_empty());
    let lib = PathBuf::from(lines.next().expect("lib path"));
    let inc = PathBuf::from(lines.next().expect("include path"));
    assert!(
        lib.join("libkaldi-native-fbank-core.dylib").exists()
            || lib.join("libkaldi-native-fbank-core.so").exists()
            || lib.join("kaldi-native-fbank-core.lib").exists()
            || lib
                .read_dir()
                .map(|d| d.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .contains("kaldi-native-fbank-core")
                }))
                .unwrap_or(false),
        "knf core library not found under {}",
        lib.display()
    );
    (lib, inc)
}
