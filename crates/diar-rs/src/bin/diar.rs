//! CLI for diar-rs.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use diar_rs::io::{write_abs_timeline, write_diarization_json};
use diar_rs::{diarize_ex, validate_models, DiarizeConfig, DumpOpts, ModelPaths};

fn usage() {
    eprintln!(
        "\
diar-rs — open-source diarization (Rust)

USAGE:
  diar-rs validate [--root REPO_ROOT]
  diar-rs diarize --wav WAV --out DIR \\
      [--root REPO_ROOT] [--seg PATH] [--emb PATH] [--plda-dir PATH] [--threads N] \\
      [--dump-trace]
"
    );
}

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    let cmd = args.remove(0);
    match cmd.as_str() {
        "validate" => cmd_validate(&args),
        "diarize" => cmd_diarize(&args),
        "-h" | "--help" | "help" => {
            usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}");
            usage();
            ExitCode::from(2)
        }
    }
}

fn cmd_validate(args: &[String]) -> ExitCode {
    let mut root = PathBuf::from(".");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--root needs value");
                    return ExitCode::from(2);
                }
                root = PathBuf::from(&args[i]);
            }
            _ => {
                eprintln!("unknown arg: {}", args[i]);
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    let candidates = [
        root.clone(),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
    ];
    let mut last_err = None;
    for r in &candidates {
        let models = ModelPaths::resolve(r);
        match validate_models(&models) {
            Ok(()) => {
                println!("OK models under {}", r.display());
                println!("  seg  {}", models.segmentation.display());
                println!("  emb  {}", models.embedding.display());
                println!("  plda {}", models.plda_dir.display());
                return ExitCode::SUCCESS;
            }
            Err(e) => last_err = Some((r.clone(), e)),
        }
    }
    if let Some((r, e)) = last_err {
        eprintln!("validate failed (tried {}): {e}", r.display());
    }
    ExitCode::from(1)
}

fn cmd_diarize(args: &[String]) -> ExitCode {
    let mut wav: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut seg: Option<PathBuf> = None;
    let mut emb: Option<PathBuf> = None;
    let mut plda: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    let mut dump_trace = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--wav" => {
                i += 1;
                wav = Some(PathBuf::from(&args[i]));
            }
            "--out" | "-o" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--root" => {
                i += 1;
                root = PathBuf::from(&args[i]);
            }
            "--seg" => {
                i += 1;
                seg = Some(PathBuf::from(&args[i]));
            }
            "--emb" => {
                i += 1;
                emb = Some(PathBuf::from(&args[i]));
            }
            "--plda-dir" => {
                i += 1;
                plda = Some(PathBuf::from(&args[i]));
            }
            "--threads" => {
                i += 1;
                threads = args[i].parse().ok();
            }
            "--dump-trace" => {
                dump_trace = true;
            }
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    let Some(wav) = wav else {
        eprintln!("--wav required");
        return ExitCode::from(2);
    };
    let Some(out) = out else {
        eprintln!("--out required");
        return ExitCode::from(2);
    };
    let mut models = ModelPaths::resolve(&root);
    if let Some(p) = seg {
        models.segmentation = p;
    }
    if let Some(p) = emb {
        models.embedding = p;
    }
    if let Some(p) = plda {
        models.plda_dir = p;
    }
    let mut cfg = DiarizeConfig::default();
    if let Some(t) = threads {
        cfg.threads = t;
    }
    let dump = DumpOpts {
        dir: if dump_trace {
            Some(out.join("intermediates"))
        } else {
            None
        },
    };
    match diarize_ex(&wav, &models, &cfg, &dump) {
        Ok(result) => {
            if let Err(e) = std::fs::create_dir_all(&out) {
                eprintln!("mkdir: {e}");
                return ExitCode::from(1);
            }
            let json_path = out.join("diarization.json");
            let abs_path = out.join("diarization_abs.txt");
            if let Err(e) = write_diarization_json(&result, &json_path) {
                eprintln!("write json: {e}");
                return ExitCode::from(1);
            }
            if let Err(e) = write_abs_timeline(&result, &abs_path) {
                eprintln!("write abs: {e}");
                return ExitCode::from(1);
            }
            println!(
                "OK turns={} talk={:?} {:.1}s → {}",
                result.n_turns, result.talk_sec, result.elapsed_sec, out.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("diarize failed: {e}");
            ExitCode::from(1)
        }
    }
}
