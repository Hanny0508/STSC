"""Build the cover side of the D3 experiment.

For N sampled IMDb sentences this script runs the same speech path the
receiver uses -- TransformerTTS synthesis followed by Whisper transcription --
so that the cover transcripts are produced by an identical pipeline to the
stego transcripts taken from the archived runs.  Only then is the comparison
"document text vs document text" meaningful.

Both models are driven through the project's persistent workers, so each model
is loaded once instead of once per sentence.  Results are cached on disk, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Windows consoles default to GBK here; an undecodable byte in a worker error
# message must not be able to kill the batch.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _safe(text: object) -> str:
    return str(text).encode("utf-8", "replace").decode("utf-8", "replace")

ROOT_DEFAULT = Path(__file__).resolve().parents[1]

TTS_PYTHON = Path(os.environ.get("ZSC_TSD_TTS_PYTHON", "python"))
ASR_PYTHON = Path(os.environ.get("ZSC_TSD_ASR_PYTHON", "python"))
TTS_WORKER = ROOT_DEFAULT / "tools" / "zts_persistent_transformertts_worker_v1.py"
ASR_WORKER = ROOT_DEFAULT / "tools" / "zts_persistent_whisper_asr_worker_v1.py"
TTS_MODEL = ROOT_DEFAULT / "artifacts" / "tts_models" / "bdf06b9_ljspeech_step_90000"
ASR_MODEL = ROOT_DEFAULT / "artifacts" / "asr_models" / "whisper-small.en"
TTS_ESPEAK = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")
TTS_GRIFFIN_LIM = 32
TTS_SEED = 2023


class Worker:
    """Line-delimited JSON worker client."""

    def __init__(self, command: list[str], expected_ready: str) -> None:
        self.command = command
        self.expected_ready = expected_ready
        self.process: subprocess.Popen | None = None

    def start(self) -> dict:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        ready = self._read()
        if ready.get("status") != self.expected_ready:
            raise RuntimeError(f"worker not ready: {ready}")
        return ready

    def ensure(self) -> None:
        """Restart the worker if it is missing or has exited."""

        if self.process is None or self.process.poll() is not None:
            self.start()

    def _read(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("worker closed its output stream")
        return json.loads(line)

    def request(self, payload: dict, *, tries: int = 4) -> dict:
        """Send one request, restarting the worker if it dies.

        The upstream workers exit on error (for example a transient Windows
        ``WinError 32`` on their scratch file), so a failure here is retried
        after a restart rather than aborting the whole batch.
        """

        last: Exception | None = None
        for attempt in range(tries):
            self.ensure()
            assert self.process is not None and self.process.stdin is not None
            try:
                self.process.stdin.write(
                    json.dumps(payload, ensure_ascii=False) + "\n"
                )
                self.process.stdin.flush()
                response = self._read()
                if response.get("status", "").endswith("ERROR"):
                    raise RuntimeError(f"worker error: {response}")
                return response
            except Exception as exc:  # noqa: BLE001
                last = exc
                self.stop()
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"request failed after {tries} attempts: {last}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            self.process.stdin.flush()
            self._read()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.process.stdin.close()
            self.process.wait(timeout=60)


def load_sentences(csv_path: Path, count: int, *, min_words: int, max_words: int):
    chosen = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            text = str(row["plaintext"]).strip()
            words = len(text.split())
            if min_words <= words <= max_words:
                chosen.append({"row_index": index, "text": text})
                if len(chosen) >= count:
                    break
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--min-words", type=int, default=12)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Read the texts to synthesise from a JSON file with a 'texts' list "
        "instead of sampling IMDb sentences (used for the unembedded control).",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/d3/covers.json"))
    parser.add_argument("--audio-dir", type=Path, default=Path("artifacts/d3/cover_audio"))
    args = parser.parse_args()

    root = args.root.resolve()
    if args.from_json is not None:
        source = args.from_json if args.from_json.is_absolute() else (root / args.from_json)
        payload = json.loads(source.read_text(encoding="utf-8"))
        # Keep every position so the produced transcripts stay index-aligned
        # with the inventory, even where the generator produced nothing.
        covers = [
            {"row_index": i, "text": t if isinstance(t, str) and t.strip() else None}
            for i, t in enumerate(payload["texts"])
        ]
        print(f"cover texts loaded from {source}")
    else:
        dataset = root / "legacy" / "zero-shot-GLS" / "zgls_data" / "imdb.csv"
        covers = load_sentences(dataset, args.count,
                                min_words=args.min_words, max_words=args.max_words)
    print(f"cover sentences: {len(covers)}")

    audio_dir = (root / args.audio_dir) if not args.audio_dir.is_absolute() else args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    tts = Worker(
        [str(TTS_PYTHON), str(TTS_WORKER),
         "--model-dir", str(TTS_MODEL), "--espeak", str(TTS_ESPEAK),
         "--griffin-lim-iterations", str(TTS_GRIFFIN_LIM), "--seed", str(TTS_SEED)],
        "ZSC_TSD_PERSISTENT_TRANSFORMERTTS_WORKER_V1_READY",
    )
    asr = Worker(
        [str(ASR_PYTHON), str(ASR_WORKER), "--model-dir", str(ASR_MODEL)],
        "ZSC_TSD_PERSISTENT_WHISPER_ASR_WORKER_V1_READY",
    )

    print("starting TTS worker ...", flush=True)
    print("  ", tts.start(), flush=True)
    print("starting ASR worker ...", flush=True)
    print("  ", asr.start(), flush=True)

    texts: list[str] = []
    transcripts: list[str | None] = []
    wavs: list[str] = []
    started = time.time()

    try:
        for position, item in enumerate(covers):
            stem = f"cover_{position:05d}"
            wav = audio_dir / f"{stem}.wav"
            tts_report = audio_dir / f"{stem}.tts.json"
            asr_report = audio_dir / f"{stem}.asr.json"

            if item["text"] is None:
                texts.append("")
                transcripts.append(None)
                wavs.append("")
                continue

            try:
                if not wav.is_file():
                    tts.request({
                        "op": "synthesize",
                        "text": item["text"],
                        "output": str(wav),
                        "report": str(tts_report),
                    })

                transcript = None
                if not asr_report.is_file() and wav.is_file():
                    asr.request({
                        "op": "transcribe",
                        "audio": str(wav),
                        "output": str(asr_report),
                    })
            except Exception as exc:  # noqa: BLE001
                # One bad item must not abort the batch; record it and move on.
                print(f"  [{position}] skipped: {type(exc).__name__}: "
                      f"{_safe(exc)[:100]}", flush=True)
                texts.append(item["text"])
                transcripts.append(None)
                wavs.append(str(wav))
                continue

            transcript = None
            if asr_report.is_file():
                payload = json.loads(asr_report.read_text(encoding="utf-8"))
                transcript = (payload.get("transcription") or {}).get("text")

            texts.append(item["text"])
            transcripts.append(transcript)
            wavs.append(str(wav))

            if (position + 1) % 25 == 0 or position + 1 == len(covers):
                rate = (position + 1) / max(time.time() - started, 1e-9)
                remaining = (len(covers) - position - 1) / max(rate, 1e-9) / 60
                print(f"  {position + 1}/{len(covers)}  {rate:.2f} sent/s"
                      f"  eta {remaining:.1f} min", flush=True)
    finally:
        tts.stop()
        asr.stop()

    out = (root / args.out) if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "params": {
                    "count": len(texts),
                    "min_words": args.min_words,
                    "max_words": args.max_words,
                    "tts": {
                        "model_dir": str(TTS_MODEL),
                        "griffin_lim_iterations": TTS_GRIFFIN_LIM,
                        "seed": TTS_SEED,
                    },
                    "asr_model": str(ASR_MODEL),
                },
                "texts": texts,
                "transcripts": transcripts,
                "audio": wavs,
            },
            ensure_ascii=False, indent=1,
        ) + "\n",
        encoding="utf-8",
    )
    missing = sum(1 for t in transcripts if not t)
    print(f"\nwritten: {out}")
    print(f"transcripts missing: {missing} / {len(texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
