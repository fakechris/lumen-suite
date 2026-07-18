//! JSON / abs timeline writers (Stage 3).

use std::fs;
use std::path::Path;

use crate::error::Result;
use crate::pipeline::DiarizeResult;

pub fn write_diarization_json(result: &DiarizeResult, path: impl AsRef<Path>) -> Result<()> {
    let s = serde_json::to_string_pretty(result)?;
    fs::write(path, s)?;
    Ok(())
}

pub fn fmt_abs(t: f64) -> String {
    let total = t.round().max(0.0) as i64;
    let s = total % 60;
    let m = (total / 60) % 60;
    let h = total / 3600;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

pub fn write_abs_timeline(result: &DiarizeResult, path: impl AsRef<Path>) -> Result<()> {
    let mut lines = Vec::new();
    for t in &result.timeline {
        lines.push(format!(
            "Speaker{}    {}",
            t.speaker + 1,
            fmt_abs(t.start)
        ));
    }
    fs::write(path, lines.join("\n") + "\n")?;
    Ok(())
}
