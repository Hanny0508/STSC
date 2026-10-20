#!/usr/bin/env python
"""Table V ablation: run the frozen v1.4 secure batch with **StyleTTS2** as the
speech backend instead of DAQ-TTS.  Everything else (crypto, ZTS, ASR, fragment
receiver) is the untouched v1.4 logic; only ``TransformerTTSClient`` is swapped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REFERENCE_WAV = Path(
    os.environ.get(
        "ZSC_TSD_STYLETTS_REFERENCE_WAV",
        str(ROOT / "artifacts" / "facodec_reference" / "official_audio_2.wav"),
    )
)
LOCAL_UTILS = Path(os.environ.get("ZSC_TSD_STYLETTS_LOCAL", r"D:\r3tmp\styletts_utils"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- make styletts2 loadable offline (local weights, torch 2.7 pickle default) ---
import torch  # noqa: E402

_torch_load = torch.load


def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _torch_load(*a, **k)


torch.load = _load

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "artifacts" / "hf_cache"))

import styletts2.tts as _T  # noqa: E402

_URL_MAP = {
    _T.ASR_CHECKPOINT_URL: LOCAL_UTILS / "Utils/ASR/epoch_00080.pth",
    _T.ASR_CONFIG_URL: LOCAL_UTILS / "Utils/ASR/config.yml",
    _T.F0_CHECKPOINT_URL: LOCAL_UTILS / "Utils/JDC/bst.t7",
    _T.BERT_CHECKPOINT_URL: LOCAL_UTILS / "Utils/PLBERT/step_1000000.t7",
    _T.BERT_CONFIG_URL: LOCAL_UTILS / "Utils/PLBERT/config.yml",
    _T.LIBRI_TTS_CHECKPOINT_URL: LOCAL_UTILS / "Models/LibriTTS/epochs_2nd_00020.pth",
    _T.LIBRI_TTS_CONFIG_URL: LOCAL_UTILS / "Models/LibriTTS/config.yml",
}
_cached_path = _T.cached_path


def _patched_cached_path(url, *a, **k):
    key = str(url)
    for cand in (key, key[: -len("?download=true")] if key.endswith("?download=true") else key):
        if cand in _URL_MAP:
            return str(_URL_MAP[cand])
    return _cached_path(url, *a, **k)


_T.cached_path = _patched_cached_path


class StyleTTS2Client:
    """Same call surface as ``zsc_tsd.speech.TransformerTTSClient``."""

    def __init__(self, target_voice_path: Path | None = None):
        self.target_voice_path = Path(target_voice_path or DEFAULT_REFERENCE_WAV).resolve()
        if not self.target_voice_path.is_file():
            raise FileNotFoundError(self.target_voice_path)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _T.StyleTTS2()
        return self._model

    def synthesize(self, text: str, output_path, *, report_path=None):
        out = Path(output_path).resolve()
        if out.exists():
            raise FileExistsError(f"refusing to overwrite: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.inference(
            text,
            target_voice_path=str(self.target_voice_path),
            output_wav_file=str(out),
        )
        report = {
            "status": "TEXT_TO_WAV_PASS",
            "backend": "StyleTTS2",
            "schema": "zsc-tsd.styletts2.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_text": text,
            "spoken_text": text,
            "output_path": str(out),
            "output_sha256": sha256_file(out),
            "target_voice_path": str(self.target_voice_path),
            "target_voice_sha256": sha256_file(self.target_voice_path),
            "definition": {
                "project_name": "StyleTTS2",
                "upstream": "https://github.com/yl4579/StyleTTS2",
                "checkpoint": "yl4579/StyleTTS2-LibriTTS epochs_2nd_00020",
                "role": "external TTS replacement for DAQ-TTS in the Table V ablation",
            },
        }
        if report_path is not None:
            rp = Path(report_path).resolve()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-public", required=True)
    p.add_argument("--bundle-private", required=True)
    p.add_argument("--start-row", type=int, default=111)
    p.add_argument("--max-attempts-per-shard", type=int, default=8)
    p.add_argument("--max-v7-candidates", type=int, default=4096)
    p.add_argument("--max-v6-candidates", type=int, default=32768)
    p.add_argument("--reference-wav", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.reference_wav is not None:
        os.environ["ZSC_TSD_STYLETTS_REFERENCE_WAV"] = str(args.reference_wav.resolve())

    sys.path.insert(0, str(ROOT / "tools"))
    import zts_secure_imdb_batch_runner_v1_4 as legacy

    legacy.TransformerTTSClient = StyleTTS2Client
    sys.argv = [
        str(Path(__file__).resolve()),
        "--bundle-public", str(args.bundle_public),
        "--bundle-private", str(args.bundle_private),
        "--start-row", str(args.start_row),
        "--max-attempts-per-shard", str(args.max_attempts_per_shard),
        "--max-v7-candidates", str(args.max_v7_candidates),
        "--max-v6-candidates", str(args.max_v6_candidates),
    ]
    legacy.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
