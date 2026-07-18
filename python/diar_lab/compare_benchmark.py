#!/usr/bin/env python3
"""Score a diarization timeline against human-annotated ground truth (GT).

GT-first by default: pass one hypothesis (`--hyp`) and the GT; get frame_acc /
DER / turn-quality vs the human annotation. An optional `--native` (or any
external baseline) adds a second column — it is never required.

Metrics (100ms frames):
  - frame_acc : best speaker-permutation agreement on speech-union frames
  - DER-ish    : miss + fa + conf over the union of speech frames (relative,
                 not full NIST DER)
  - turn majority (>=1s ref turns), change-point recall/precision @0.5s

GT formats (`--gt-format`):
  - md-zh : Chinese transcript, `发言人 N HH:MM:SS` (start times only)
  - md-en : English transcript, `SPEAKER N ... HH:MM:SS` (start times only)
  - rttm  : NIST RTTM (Stage 6 — not yet implemented)

Usage:
  python -m diar_lab.compare_benchmark \\
    --gt data/gt/meeting_06-29_transcript.md \\
    --hyp runs/demo/diarization.json \\
    -o runs/demo_gt
  # optional baseline column:
  #   --native /path/to/native/diarization.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_timeline_json(path: Path) -> List[dict]:
    data = json.loads(Path(path).read_text())
    return list(data["timeline"])


def _gt_turn_starts(text: str, pat: "re.Pattern") -> List[Tuple[int, int]]:
    text = unicodedata.normalize("NFKC", text)
    starts: List[Tuple[int, int]] = []
    for m in pat.finditer(text):
        spk = int(m.group(1)) - 1  # 1-based -> 0-based
        h, mi, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
        starts.append((h * 3600 + mi * 60 + s, spk))
    starts.sort(key=lambda x: x[0])
    return starts


def _build_timeline(starts: List[Tuple[int, int]], duration_hint: float) -> List[dict]:
    if not starts:
        raise RuntimeError("no GT turns parsed")
    raw: List[dict] = []
    for i, (t0, spk) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else max(duration_hint, t0 + 1.0)
        if t1 <= t0:
            t1 = t0 + 0.3
        raw.append({"start": float(t0), "end": float(t1), "speaker": spk})
    # merge adjacent same-speaker turns
    merged: List[dict] = []
    for t in raw:
        if merged and merged[-1]["speaker"] == t["speaker"] and t["start"] <= merged[-1]["end"] + 0.05:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def _rttm_timeline(text: str) -> List[dict]:
    """NIST RTTM → timeline.

    Line: `SPEAKER <file> <ch> <start> <dur> <NA> <NA> <speaker> <NA> <NA>`
    Speaker labels are string IDs; mapped to 0-based ints deterministically.
    """
    tl: List[dict] = []
    for line in text.splitlines():
        if not line.startswith("SPEAKER"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        try:
            start = float(p[3])
            dur = float(p[4])
        except ValueError:
            continue
        tl.append({"start": start, "end": start + dur, "speaker": p[7]})
    if not tl:
        raise RuntimeError("no RTTM turns")
    spks = sorted({t["speaker"] for t in tl})
    smap = {s: i for i, s in enumerate(spks)}
    for t in tl:
        t["speaker"] = smap[t["speaker"]]
    tl.sort(key=lambda t: t["start"])
    merged: List[dict] = []
    for t in tl:
        if (
            merged
            and merged[-1]["speaker"] == t["speaker"]
            and t["start"] <= merged[-1]["end"] + 0.01
        ):
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def parse_gt(path: Path, fmt: str = "md-zh", duration_hint: float = 0.0) -> List[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if fmt in ("md-zh", "md-en"):
        if fmt == "md-zh":
            pat = re.compile(r"发言人\s*(\d+)\s+(\d{1,2}):(\d{2}):(\d{2})", re.MULTILINE)
        else:
            pat = re.compile(
                r"(?:SPEAKER|speaker|spk)[_\s]*(\d+)[^\d\n]*?(\d{1,2}):(\d{2}):(\d{2})",
                re.MULTILINE,
            )
        return _build_timeline(_gt_turn_starts(text, pat), duration_hint)
    if fmt == "rttm":
        return _rttm_timeline(text)
    raise ValueError(f"unknown --gt-format {fmt!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def raster(tl: List[dict], step: float, n: int) -> np.ndarray:
    r = np.full(n, -1, dtype=np.int32)
    for t in tl:
        a = int(t["start"] / step)
        b = max(a + 1, int(np.ceil(t["end"] / step)))
        r[max(0, a): min(n, b)] = int(t["speaker"])
    return r


def best_map_acc(
    ref: np.ndarray, hyp: np.ndarray
) -> Tuple[float, Dict[int, int], float, float, float, float]:
    """Return frame_acc, map hyp→ref, miss, fa, conf, der on speech union."""
    both = (ref >= 0) & (hyp >= 0)
    speech = (ref >= 0) | (hyp >= 0)
    sp_ref = sorted(set(ref[ref >= 0].tolist()))
    sp_hyp = sorted(set(hyp[hyp >= 0].tolist()))
    best, mp = 0.0, {}
    if sp_ref and sp_hyp:
        for perm in permutations(sp_ref, min(len(sp_hyp), len(sp_ref))):
            mapping = {a: b for a, b in zip(sp_hyp[: len(perm)], perm)}
            mapped = np.array(
                [mapping.get(int(x), -9) if x >= 0 else -1 for x in hyp]
            )
            acc = float((mapped[both] == ref[both]).mean()) if both.any() else 0.0
            if acc > best:
                best, mp = acc, mapping
    if not speech.any() or not mp:
        return best, mp, 1.0, 1.0, 1.0, 1.0
    mapped = np.array([mp.get(int(x), -9) if x >= 0 else -1 for x in hyp])
    denom = float(speech.sum())
    miss = float(((ref >= 0) & (hyp < 0)).sum()) / denom
    fa = float(((ref < 0) & (hyp >= 0)).sum()) / denom
    conf = float(((ref >= 0) & (hyp >= 0) & (mapped != ref)).sum()) / denom
    return best, mp, miss, fa, conf, miss + fa + conf


def talk_sec(tl: List[dict]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for t in tl:
        out[t["speaker"]] = out.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    return out


def turn_majority(
    ref_tl: List[dict], hyp: np.ndarray, step: float, min_dur: float = 1.0
) -> Tuple[float, int, int]:
    n = len(hyp)
    ref_r = raster(ref_tl, step, n)
    _, mp, *_ = best_map_acc(ref_r, hyp)
    ok = tot = 0
    for t in ref_tl:
        if t["end"] - t["start"] < min_dur:
            continue
        tot += 1
        a = int(t["start"] / step)
        b = max(a + 1, int(np.ceil(t["end"] / step)))
        win = hyp[a:b]
        win = win[win >= 0]
        if len(win) == 0:
            continue
        maj = int(np.bincount(win).argmax())
        if mp.get(maj, -9) == int(t["speaker"]):
            ok += 1
    return (ok / tot if tot else 0.0), ok, tot


def change_points(tl: List[dict]) -> List[float]:
    if not tl:
        return []
    ordered = sorted(tl, key=lambda t: t["start"])
    pts = []
    for i in range(1, len(ordered)):
        if ordered[i]["speaker"] != ordered[i - 1]["speaker"]:
            pts.append(float(ordered[i]["start"]))
    return pts


def change_point_metrics(
    ref_pts: List[float], hyp_pts: List[float], tol: float = 0.5
) -> Tuple[float, float, int, int, int]:
    if not ref_pts:
        return 0.0, 0.0, 0, 0, len(hyp_pts)
    matched_ref = set()
    matched_hyp = set()
    for i, r in enumerate(ref_pts):
        for j, h in enumerate(hyp_pts):
            if j in matched_hyp:
                continue
            if abs(r - h) <= tol:
                matched_ref.add(i)
                matched_hyp.add(j)
                break
    rec = len(matched_ref) / len(ref_pts)
    prec = len(matched_hyp) / len(hyp_pts) if hyp_pts else 0.0
    return rec, prec, len(matched_ref), len(ref_pts), len(hyp_pts)


def per_speaker_recall(
    ref: np.ndarray, hyp: np.ndarray, mp: Dict[int, int]
) -> Dict[int, float]:
    mapped = np.array([mp.get(int(x), -9) if x >= 0 else -1 for x in hyp])
    out = {}
    for s in sorted(set(ref[ref >= 0].tolist())):
        mask = ref == s
        out[s] = float((mapped[mask] == s).mean()) if mask.any() else 0.0
    return out


def evaluate_pair(
    name_ref: str, name_hyp: str, ref_tl: List[dict], hyp_tl: List[dict], step: float = 0.1
) -> dict:
    dur = max(
        max(t["end"] for t in ref_tl),
        max(t["end"] for t in hyp_tl) if hyp_tl else 0,
    )
    n = int(dur / step) + 2
    ref = raster(ref_tl, step, n)
    hyp = raster(hyp_tl, step, n)
    acc, mp, miss, fa, conf, der = best_map_acc(ref, hyp)
    maj, ok, tot = turn_majority(ref_tl, hyp, step)
    rec, prec, m, nr, nh = change_point_metrics(
        change_points(ref_tl), change_points(hyp_tl), tol=0.5
    )
    return {
        "ref": name_ref,
        "hyp": name_hyp,
        "frame_acc": acc,
        "der": der,
        "miss": miss,
        "fa": fa,
        "conf": conf,
        "map_hyp_to_ref": {str(k): int(v) for k, v in mp.items()},
        "n_turns_ref": len(ref_tl),
        "n_turns_hyp": len(hyp_tl),
        "talk_ref": {str(k): round(v, 1) for k, v in talk_sec(ref_tl).items()},
        "talk_hyp": {str(k): round(v, 1) for k, v in talk_sec(hyp_tl).items()},
        "turn_majority_ge1s": maj,
        "turn_majority_counts": f"{ok}/{tot}",
        "change_recall@0.5s": rec,
        "change_prec@0.5s": prec,
        "change_counts": f"{m}/{nr} ref, {nh} hyp",
        "spk_recall": {str(k): round(v, 3) for k, v in per_speaker_recall(ref, hyp, mp).items()},
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def md_report(
    hyp_tl: List[dict],
    gt_tl: List[dict],
    m_hg: dict,
    native_tl: Optional[List[dict]],
    m_ng: Optional[dict],
    meta: dict,
) -> str:
    have_native = native_tl is not None and m_ng is not None
    native_col = " | Native (baseline)" if have_native else ""
    native_hdr = " | Native (baseline)" if have_native else ""
    lines = [
        "# Diarization 基准对照报告（vs 人工 GT）",
        "",
        "生成自 `python/diar_lab/compare_benchmark.py`",
        "",
        "## 输入",
        "",
        f"- hyp (system under test): `{meta.get('hyp_path')}`",
    ]
    if have_native:
        lines.append(f"- native (baseline): `{meta.get('native_path')}`")
    lines += [
        f"- GT (human): `{meta.get('gt_path')}` [`--gt-format {meta.get('gt_format')}`]",
        f"- hyp method: `{meta.get('hyp_method', '')}`",
        "",
        "## 总表",
        "",
        f"| 指标 | GT{native_col} | Hyp |",
        "|---|---|---|",
    ]
    def _talk(d):
        return {f"S{k + 1}": round(v, 1) for k, v in talk_sec(d).items()}
    gt_spk = f"{len(talk_sec(gt_tl))}"
    hyp_spk = f"{len(talk_sec(hyp_tl))}"
    nat_spk = f"{len(talk_sec(native_tl))}" if have_native else ""
    lines.append(f"| 轮次数 | {len(gt_tl)}{(' | ' + str(len(native_tl))) if have_native else ''} | {len(hyp_tl)} |")
    lines.append(f"| 说话人数 | {gt_spk}{(' | ' + nat_spk) if have_native else ''} | {hyp_spk} |")
    lines.append(
        f"| talk (s) | {_talk(gt_tl)}{(' | ' + str(_talk(native_tl))) if have_native else ''} | {_talk(hyp_tl)} |"
    )
    lines += [
        "",
        "### vs GT（金标）",
        "",
    ]
    if have_native:
        lines += [
            "| 指标 | Native | Hyp |",
            "|---|---|---|",
            f"| frame_acc @100ms | {fmt_pct(m_ng['frame_acc'])} | {fmt_pct(m_hg['frame_acc'])} |",
            f"| 近似 DER | {fmt_pct(m_ng['der'])} | {fmt_pct(m_hg['der'])} |",
            f"| Miss / FA / Conf | {fmt_pct(m_ng['miss'])} / {fmt_pct(m_ng['fa'])} / {fmt_pct(m_ng['conf'])} | "
            f"{fmt_pct(m_hg['miss'])} / {fmt_pct(m_hg['fa'])} / {fmt_pct(m_hg['conf'])} |",
            f"| 轮次多数票 ≥1s | {fmt_pct(m_ng['turn_majority_ge1s'])} ({m_ng['turn_majority_counts']}) | "
            f"{fmt_pct(m_hg['turn_majority_ge1s'])} ({m_hg['turn_majority_counts']}) |",
            f"| 换人点 recall@0.5s | {fmt_pct(m_ng['change_recall@0.5s'])} | {fmt_pct(m_hg['change_recall@0.5s'])} |",
            f"| map hyp→GT | `{m_ng['map_hyp_to_ref']}` | `{m_hg['map_hyp_to_ref']}` |",
            f"| 各说话人帧召回 | {m_ng['spk_recall']} | {m_hg['spk_recall']} |",
        ]
    else:
        lines += [
            "| 指标 | Hyp |",
            "|---|---|",
            f"| frame_acc @100ms | {fmt_pct(m_hg['frame_acc'])} |",
            f"| 近似 DER | {fmt_pct(m_hg['der'])} |",
            f"| Miss / FA / Conf | {fmt_pct(m_hg['miss'])} / {fmt_pct(m_hg['fa'])} / {fmt_pct(m_hg['conf'])} |",
            f"| 轮次多数票 ≥1s | {fmt_pct(m_hg['turn_majority_ge1s'])} ({m_hg['turn_majority_counts']}) |",
            f"| 换人点 recall@0.5s | {fmt_pct(m_hg['change_recall@0.5s'])} |",
            f"| map hyp→GT | `{m_hg['map_hyp_to_ref']}` |",
            f"| 各说话人帧召回 | {m_hg['spk_recall']} |",
        ]
    lines += [
        "",
        "## 读数说明",
        "",
        "- **frame_acc**：100ms 帧上、双方有声的帧，在最优说话人置换下的标签一致率。",
        "- **近似 DER**：miss+fa+conf（相对双方有声并集），非 NIST 官方 DER，便于相对比较。",
        "- **轮次多数票**：每个 ≥1s 的参考轮次，取 hyp 在该区间多数票标签是否映射正确。",
        "- GT 为人工转写起点时间，段末=下一轮起点；短应答常被少说话人系统合并。",
        "",
    ]
    return "\n".join(lines) + "\n"


def run_multi(gt_dir: str, hyp_dir: str, fmt: str, out: Optional[str]) -> None:
    """Score many files: GT (RTTM) in gt_dir vs diarization.json in hyp_dir,
    matched by recording id (file stem). Writes an aggregate DER report."""
    import statistics

    gt_dir, hyp_dir = Path(gt_dir), Path(hyp_dir)
    gts = sorted(gt_dir.glob("*.rttm"))
    rows = []
    for gp in gts:
        rec = gp.stem
        cands = [hyp_dir / f"{rec}.json", hyp_dir / f"{rec}" / "diarization.json"]
        cands += list(hyp_dir.glob(f"{rec}*.json")) + list(hyp_dir.glob(f"{rec}*/diarization.json"))
        hp = next((c for c in cands if c.exists()), None)
        if hp is None:
            print(f"[skip] no hyp for {rec}")
            continue
        hyp_tl = list(json.loads(hp.read_text()).get("timeline", []))
        if not hyp_tl:
            print(f"[skip] empty hyp {rec}")
            continue
        dur = max(t["end"] for t in hyp_tl)
        gt_tl = parse_gt(gp, fmt=fmt, duration_hint=dur)
        rows.append((rec, evaluate_pair("gt", "hyp", gt_tl, hyp_tl)))
    if not rows:
        raise SystemExit("no matched GT/hyp pairs")
    ders = [r[1]["der"] for r in rows]
    accs = [r[1]["frame_acc"] for r in rows]
    agg = {
        "n_files": len(rows),
        "mean_frame_acc": statistics.mean(accs),
        "mean_der": statistics.mean(ders),
        "per_file": [
            {"file": r[0], "frame_acc": round(r[1]["frame_acc"], 4), "der": round(r[1]["der"], 4)}
            for r in rows
        ],
    }
    out_dir = Path(out) if out else ROOT / "runs" / "multi_gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    md = [
        "# Multi-file diarization DER (vs RTTM GT)",
        "",
        f"- files: {len(rows)}",
        f"- mean frame_acc: {agg['mean_frame_acc'] * 100:.1f}%",
        f"- mean DER: {agg['mean_der'] * 100:.1f}%",
        "",
        "| file | frame_acc | DER |",
        "|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]['frame_acc'] * 100:.1f}% | {r[1]['der'] * 100:.1f}% |")
    (out_dir / "MULTI_DER.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_dir}/aggregate.json, MULTI_DER.md")


def main():
    ap = argparse.ArgumentParser(description="Score a diarization timeline vs human GT.")
    ap.add_argument("--hyp", default=None, help="system-under-test diarization.json (single-file mode)")
    ap.add_argument(
        "--gt",
        default=str(ROOT / "data/gt/meeting_06-29_transcript.md"),
        help="human ground-truth transcript",
    )
    ap.add_argument(
        "--gt-format", default="md-zh", choices=["md-zh", "md-en", "rttm"],
    )
    ap.add_argument(
        "--native", default=None,
        help="optional baseline diarization.json (e.g. a closed-system output)",
    )
    ap.add_argument("--gt-dir", default=None, help="RTTM GT dir (multi-file mode)")
    ap.add_argument("--hyp-dir", default=None, help="hyp diarization.json dir (multi-file mode)")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    if args.gt_dir and args.hyp_dir:
        run_multi(args.gt_dir, args.hyp_dir, args.gt_format, args.out)
        return
    if not args.hyp:
        ap.error("--hyp is required in single-file mode (or pass --gt-dir/--hyp-dir for multi-file)")

    hyp_data = json.loads(Path(args.hyp).read_text())
    hyp_tl = list(hyp_data["timeline"])
    dur = max(t["end"] for t in hyp_tl)
    gt_tl = parse_gt(Path(args.gt), fmt=args.gt_format, duration_hint=dur)

    m_hg = evaluate_pair("gt", "hyp", gt_tl, hyp_tl)
    native_tl: Optional[List[dict]] = None
    m_ng: Optional[dict] = None
    if args.native:
        native_tl = load_timeline_json(Path(args.native))
        dur = max(dur, max(t["end"] for t in native_tl))
        # re-parse GT with the longer duration hint for the last-turn end
        gt_tl = parse_gt(Path(args.gt), fmt=args.gt_format, duration_hint=dur)
        m_ng = evaluate_pair("gt", "native", gt_tl, native_tl)

    out = Path(args.out) if args.out else ROOT / "runs" / f"{Path(args.hyp).stem}_gt"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "hyp_vs_gt": m_hg,
        "native_vs_gt": m_ng,
        "meta": {
            "hyp_path": args.hyp,
            "native_path": args.native,
            "gt_path": args.gt,
            "gt_format": args.gt_format,
            "hyp_method": hyp_data.get("method", ""),
        },
    }
    (out / "benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    md = md_report(hyp_tl, gt_tl, m_hg, native_tl, m_ng, payload["meta"])
    (out / "GT_COMPARE.md").write_text(md)
    print(md)
    print(f"\nwrote {out}/benchmark.json, GT_COMPARE.md")


if __name__ == "__main__":
    main()
