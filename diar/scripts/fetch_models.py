#!/usr/bin/env python3
"""Fetch open-source diarization weights into models/.

Targets (see models/README.md):
  silero_vad.onnx  Silero VAD (MIT)                 HF silero/silero-vad
  emb.onnx         WeSpeaker ResNet34-LM, 256-d      HF Wespeaker/wespeaker-voxceleb-resnet34-LM
  seg.onnx         DiariZen diarizen-wavlm-base      HF butspeechd/diarizen-wavlm-base  (NC: --accept-nc; access-gated → needs token)

HF token (for the gated DiariZen repo): set HF_TOKEN env var or write it to
`.hf_token` (gitignored). The token is sent only to huggingface.co as a Bearer
header. The DiariZen seg filename is auto-discovered from the repo file list
(since the repo is access-gated, the exact filename isn't assumed).

Env vars DIAR_SEG_ONNX / DIAR_EMB_ONNX / DIAR_PLDA_DIR override these files at
runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TOKEN_FILE = ROOT / ".hf_token"

WEBSPEAKER_REPO = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
WEBSPEAKER_FILE = "voxceleb_resnet34_LM.onnx"
CNCELEB_REPO = "Wespeaker/wespeaker-cnceleb-resnet34-LM"
CNCELEB_FILE = "cnceleb_resnet34_LM.onnx"
DIARIZEN_REPO = "butspeechd/diarizen-wavlm-base"
SILERO_URL = "https://huggingface.co/silero/silero-vad/resolve/main/silero_vad.onnx"


def hf_token() -> str | None:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t and TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text().strip()
    return t


def _request(url: str, token: str | None = None) -> urllib.request.Request:
    req = urllib.request.Request(url)
    if token and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def hf_list(repo: str, token: str | None = None) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(_request(url, token), timeout=30) as resp:
        d = json.loads(resp.read())
    return [s["rfilename"] for s in d.get("siblings", [])]


def download(url: str, dest: Path, token: str | None = None) -> None:
    print(f"[fetch] {url}\n     -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(_request(url, token), timeout=180) as resp:
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    tmp.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_diarizen_seg(token: str | None) -> str:
    """List the DiariZen repo and return the resolve URL of the best seg ONNX."""
    files = hf_list(DIARIZEN_REPO, token)
    print(f"[diarizen] {DIARIZEN_REPO} files: {files}")
    onnx = [f for f in files if f.endswith(".onnx")]
    if not onnx:
        raise RuntimeError(
            f"{DIARIZEN_REPO} has no .onnx (files={files}). "
            "It may ship a checkpoint — export ONNX via the BUTSpeechFIT/diarizen repo."
        )
    # prefer a file whose name suggests segmentation
    pick = sorted(onnx, key=lambda f: ("seg" not in f.lower(), len(f), f))[0]
    return f"https://huggingface.co/{DIARIZEN_REPO}/resolve/main/{pick}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(MODELS))
    ap.add_argument("--accept-nc", action="store_true",
                    help="required to download the non-commercial DiariZen weights")
    ap.add_argument("--only", action="append",
                    choices=["silero_vad.onnx", "emb.onnx", "emb_cnceleb.onnx", "seg.onnx"],
                    help="limit to these targets (repeatable)")
    ap.add_argument("--list-diarizen", action="store_true",
                    help="just list the DiariZen repo files (with token) and exit")
    args = ap.parse_args()

    token = hf_token()
    out = Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {"silero_vad.onnx", "emb.onnx", "seg.onnx"}
    # emb_cnceleb.onnx is opt-in (--only emb_cnceleb.onnx): Chinese-data
    # embedding for `--v2 --cluster-space raw` on Mandarin audio.

    if args.list_diarizen:
        try:
            files = hf_list(DIARIZEN_REPO, token)
        except urllib.error.HTTPError as e:
            print(f"ERROR listing {DIARIZEN_REPO}: HTTP {e.code} "
                  f"({'access-gated — accept license on the model page and provide a valid token' if e.code == 401 else e.reason})",
                  file=sys.stderr)
            sys.exit(2)
        print(f"{DIARIZEN_REPO} ({len(files)} files):")
        for f in files:
            print("  -", f)
        return

    if "seg.onnx" in want and not args.accept_nc:
        print("ERROR: DiariZen seg requires --accept-nc (non-commercial license).", file=sys.stderr)
        sys.exit(2)
    if "seg.onnx" in want and not token:
        print("ERROR: DiariZen repo is access-gated; set HF_TOKEN or write .hf_token.", file=sys.stderr)
        sys.exit(2)

    plan: list[tuple[str, str, str | None]] = []  # (dest_name, url, nc_note)
    if "silero_vad.onnx" in want:
        plan.append(("silero_vad.onnx", SILERO_URL, None))
    if "emb.onnx" in want:
        plan.append(("emb.onnx", f"https://huggingface.co/{WEBSPEAKER_REPO}/resolve/main/{WEBSPEAKER_FILE}", None))
    if "emb_cnceleb.onnx" in want:
        plan.append(("emb_cnceleb.onnx", f"https://huggingface.co/{CNCELEB_REPO}/resolve/main/{CNCELEB_FILE}", None))
    if "seg.onnx" in want:
        try:
            plan.append(("seg.onnx", resolve_diarizen_seg(token), "DiariZen (NC)"))
        except Exception as e:  # noqa: BLE001
            print(f"[fail] seg.onnx: {e}", file=sys.stderr)

    for name, url, _nc in plan:
        dest = out / name
        if dest.exists():
            print(f"[skip] {name} exists ({dest.stat().st_size} bytes)")
        else:
            try:
                download(url, dest, token)
            except Exception as e:  # noqa: BLE001
                print(f"[fail] {name}: {e}", file=sys.stderr)
                continue
        digest = sha256(dest)
        print(f"[sha] {name}: {digest}")

    print(f"\nDone. Weights under {out}/. Pin sha256 values in models/README.md.")


if __name__ == "__main__":
    main()
