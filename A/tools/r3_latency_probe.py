"""Carrier-length and latency profile for every method (the Table IX analogue).

For each method and each of its BPW settings, a small sample of the generated
stego texts is synthesised with the same TransformerTTS backend used by the
pipeline, and we record

  * the per-sample text-generation time (taken from the recorded run wall time),
  * the TTS time for that text,
  * the duration of the resulting carrier, which is what the RTP path streams.

Everything is written to ``artifacts/r3_latency/latency_summary.json``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLS = ROOT / "legacy" / "zero-shot-GLS"
STEGO_DIR = GLS / "tmp_saves" / "results" / "r3_baselines"
OUT_DIR = ROOT / "artifacts" / "r3_latency"

TTS_PYTHON = Path(os.environ.get("ZSC_TSD_TTS_PYTHON", "python"))
TTS_WORKER = ROOT / "tools" / "zts_persistent_transformertts_worker_v1.py"
TTS_MODEL = ROOT / "artifacts" / "tts_models" / "bdf06b9_ljspeech_step_90000"
ESPEAK = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")

# text generation wall time per sample, measured from the recorded runs
TEXT_SECONDS = {
    "ours_P1": 2.0, "ours_P2": 2.0, "ours_P3": 2.0,
    "nls_mb2": 0.54, "nls_mb3": 0.54, "nls_mb4": 0.54,
    "saac": 0.57,
    "rnn_mb1": 0.26, "rnn_mb2": 0.54, "rnn_mb3": 0.36, "rnn_mb4": 0.26,
    "vae_mb1": 0.2, "vae_mb2": 0.46, "vae_mb3": 0.28, "vae_mb4": 0.2,
}

SOURCES = [
    ("ours_P1", ROOT / "artifacts/d3/aligned/P1.json", "stegotext"),
    ("ours_P2", ROOT / "artifacts/d3/aligned/P2.json", "stegotext"),
    ("ours_P3", ROOT / "artifacts/d3/aligned/P3.json", "stegotext"),
    ("nls_mb2", STEGO_DIR / "nls_maxbpw2.csv", "stegotext"),
    ("nls_mb3", STEGO_DIR / "nls_maxbpw3.csv", "stegotext"),
    ("nls_mb4", STEGO_DIR / "nls_maxbpw4.csv", "stegotext"),
    ("saac", STEGO_DIR / "saac_default.csv", "stegotext"),
    ("rnn_mb1", STEGO_DIR / "rnn_maxbpw1.csv", "stegotext"),
    ("rnn_mb2", STEGO_DIR / "rnn_maxbpw2.csv", "stegotext"),
    ("rnn_mb3", STEGO_DIR / "rnn_maxbpw3.csv", "stegotext"),
    ("rnn_mb4", STEGO_DIR / "rnn_maxbpw4.csv", "stegotext"),
    ("vae_mb1", STEGO_DIR / "vae_maxbpw1.csv", "stegotext"),
    ("vae_mb2", STEGO_DIR / "vae_maxbpw2.csv", "stegotext"),
    ("vae_mb3", STEGO_DIR / "vae_maxbpw3.csv", "stegotext"),
    ("vae_mb4", STEGO_DIR / "vae_maxbpw4.csv", "stegotext"),
]


def read_texts(path: Path, column: str, limit: int) -> list[str]:
    if not path.is_file():
        return []
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(column, [])
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            values = [row.get(column, "") for row in csv.DictReader(handle)]
    return [str(v).strip() for v in values if str(v).strip()][:limit]


def wav_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 3)
    except Exception:  # noqa: BLE001
        pass
    # `soundfile` is not installed in every conda env on node A; the stdlib
    # reader is enough for a plain PCM wav header.
    try:
        import wave

        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 3)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    if not TTS_WORKER.is_file() or not TTS_PYTHON.is_file():
        print("TTS worker or interpreter missing", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wav_dir = OUT_DIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [
            str(TTS_PYTHON), "-u", str(TTS_WORKER),
            "--model-dir", str(TTS_MODEL),
            "--espeak", str(ESPEAK),
            "--griffin-lim-iterations", "32",
            "--seed", "2023",
        ],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdin and process.stdout

    ready = json.loads(process.stdout.readline())
    print("worker ready:", ready.get("status"), flush=True)

    summary: dict = {"schema": "zsc-tsd.r3-latency.v1", "results": {}}
    for label, path, column in SOURCES:
        texts = read_texts(path, column, args.samples)
        if not texts:
            print(f"  {label}: no texts, skip", flush=True)
            continue
        rows = []
        for index, text in enumerate(texts):
            out = wav_dir / f"{label}_{index:03d}.wav"
            rep = wav_dir / f"{label}_{index:03d}.json"
            if out.exists():
                out.unlink()
            if rep.exists():
                rep.unlink()
            request = {"op": "synthesize", "text": text, "output": str(out), "report": str(rep)}
            started = time.perf_counter()
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "words": len(text.split()),
                    "tts_seconds": round(elapsed, 3),
                    "carrier_seconds": wav_seconds(out),
                    "status": response.get("status"),
                }
            )
        ok = [r for r in rows if r["carrier_seconds"]]
        if ok:
            summary["results"][label] = {
                "n": len(rows),
                "text_seconds_per_sample": TEXT_SECONDS.get(label),
                "words_per_sample": round(sum(r["words"] for r in ok) / len(ok), 2),
                "tts_seconds_per_sample": round(sum(r["tts_seconds"] for r in ok) / len(ok), 3),
                "carrier_seconds_per_sample": round(
                    sum(r["carrier_seconds"] for r in ok) / len(ok), 3
                ),
                "rows": rows,
            }
            entry = summary["results"][label]
            print(
                f"  {label:<10} words={entry['words_per_sample']:>7} "
                f"tts={entry['tts_seconds_per_sample']:>7}s "
                f"carrier={entry['carrier_seconds_per_sample']:>7}s",
                flush=True,
            )

    try:
        process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
        process.stdin.flush()
        process.stdout.readline()
    except Exception:  # noqa: BLE001
        pass
    process.wait(timeout=60)

    (OUT_DIR / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("written:", OUT_DIR / "latency_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
