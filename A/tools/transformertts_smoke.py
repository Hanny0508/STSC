from __future__ import print_function

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERTTS_ROOT = PROJECT_ROOT / "legacy" / "TransformerTTS"

if str(TRANSFORMERTTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERTTS_ROOT))

import numpy as np
import soundfile as sf
import tensorflow as tf
from phonemizer.backend.espeak import EspeakBackend

from data.audio import Audio
from model.models import ForwardTransformer


DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tts_models"
    / "bdf06b9_ljspeech_step_90000"
    / "bdf06b9_ljspeech_step_90000"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "new_e2e"
    / "transformertts_ljspeech_step90000_smoke.wav"
)

DEFAULT_ESPEAK = Path(
    r"C:\Program Files\eSpeak NG\espeak-ng.exe"
)


def sha256_file(
    path,
    chunk_size=8 * 1024 * 1024,
):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible TransformerTTS "
            "LJSpeech smoke test."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--espeak",
        type=Path,
        default=DEFAULT_ESPEAK,
    )
    parser.add_argument(
        "--text",
        default="  Secure IoT communication.",
        help=(
            "Input text. The default intentionally contains "
            "two leading spaces."
        ),
    )
    parser.add_argument(
        "--griffin-lim-iterations",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".json")
    )

    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "model_weights.hdf5"

    for required_path in (
        args.espeak,
        config_path,
        weights_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                "Required file does not exist: {}".format(
                    required_path
                )
            )

    if args.griffin_lim_iterations <= 0:
        raise ValueError(
            "griffin-lim-iterations must be positive"
        )

    original_text = args.text
    leading_space_count = (
        len(original_text)
        - len(original_text.lstrip(" "))
    )
    spoken_text = original_text[leading_space_count:]

    if not spoken_text.strip():
        raise ValueError(
            "Input text contains no pronounceable content"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    EspeakBackend.set_espeak_path(
        str(args.espeak.resolve())
    )

    load_start = time.perf_counter()
    model = ForwardTransformer.load_model(
        model_dir.as_posix()
    )
    load_seconds = time.perf_counter() - load_start

    phonemes = model.text_pipeline.phonemizer(
        spoken_text
    )
    token_ids = model.text_pipeline.tokenizer(
        phonemes
    )

    mel_start = time.perf_counter()
    prediction = model.predict(
        token_ids,
        encode=False,
        phoneme_max_duration=None,
    )
    mel_seconds = time.perf_counter() - mel_start

    mel = prediction["mel"].numpy().T

    if not np.isfinite(mel).all():
        raise RuntimeError(
            "TransformerTTS produced non-finite Mel values"
        )

    audio = Audio.from_config(model.config)

    wav_start = time.perf_counter()
    waveform = audio.reconstruct_waveform(
        mel,
        n_iter=args.griffin_lim_iterations,
    )
    wav_seconds = time.perf_counter() - wav_start

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.size == 0:
        raise RuntimeError(
            "Waveform reconstruction returned no samples"
        )

    if not np.isfinite(waveform).all():
        raise RuntimeError(
            "Waveform contains non-finite values"
        )

    audio.save_wav(
        waveform,
        str(output_path),
    )

    wav_info = sf.info(str(output_path))
    saved_waveform, saved_sample_rate = sf.read(
        str(output_path),
        dtype="float32",
    )

    report = {
        "status": "TEXT_TO_WAV_PASS",
        "original_text": original_text,
        "spoken_text": spoken_text,
        "leading_space_count": leading_space_count,
        "phonemes": phonemes,
        "token_count": len(token_ids),
        "token_ids": [int(value) for value in token_ids],
        "model": {
            "directory": str(model_dir),
            "step": str(model.config.get("step")),
            "data_name": str(
                model.config.get("data_name")
            ),
            "checkpoint_git_hash": str(
                model.config.get("git_hash")
            ),
            "parameter_count": int(
                model.count_params()
            ),
            "config_sha256": sha256_file(
                config_path
            ),
            "weights_sha256": sha256_file(
                weights_path
            ),
        },
        "mel": {
            "shape": [
                int(value)
                for value in mel.shape
            ],
            "dtype": str(mel.dtype),
            "finite": bool(
                np.isfinite(mel).all()
            ),
            "minimum": float(mel.min()),
            "maximum": float(mel.max()),
        },
        "audio": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "sample_rate": int(
                wav_info.samplerate
            ),
            "frames": int(wav_info.frames),
            "duration_seconds": float(
                wav_info.duration
            ),
            "channels": int(
                wav_info.channels
            ),
            "format": str(wav_info.format),
            "subtype": str(wav_info.subtype),
            "file_bytes": int(
                output_path.stat().st_size
            ),
            "raw_peak": float(
                np.max(np.abs(waveform))
            ),
            "raw_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(waveform)
                    )
                )
            ),
            "saved_peak": float(
                np.max(
                    np.abs(saved_waveform)
                )
            ),
            "saved_rms": float(
                np.sqrt(
                    np.mean(
                        np.square(saved_waveform)
                    )
                )
            ),
            "saved_sample_rate": int(
                saved_sample_rate
            ),
        },
        "timing_seconds": {
            "model_load": float(
                load_seconds
            ),
            "text_to_mel": float(
                mel_seconds
            ),
            "griffin_lim": float(
                wav_seconds
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "phonemizer": package_version(
                "phonemizer"
            ),
            "librosa": package_version(
                "librosa"
            ),
            "soundfile": package_version(
                "soundfile"
            ),
            "espeak_executable": str(
                args.espeak.resolve()
            ),
            "seed": int(args.seed),
            "griffin_lim_iterations": int(
                args.griffin_lim_iterations
            ),
        },
        "notes": [
            (
                "Leading ASCII spaces are stored as metadata "
                "because they are not represented in speech."
            ),
            (
                "The checkpoint is the public TransformerTTS "
                "LJSpeech step-90000 model, not the claimed "
                "DAQ-TTS checkpoint."
            ),
            (
                "A missing local Git repository may produce a "
                "non-fatal version-check warning."
            ),
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("TRANSFORMERTTS WORKER PASS")
    print("output:", output_path)
    print("report:", report_path)
    print(
        "leading spaces:",
        leading_space_count,
    )
    print("spoken text:", spoken_text)
    print("parameters:", model.count_params())
    print("mel shape:", mel.shape)
    print(
        "duration:",
        round(wav_info.duration, 3),
    )
    print(
        "text-to-mel seconds:",
        round(mel_seconds, 3),
    )
    print(
        "griffin-lim seconds:",
        round(wav_seconds, 3),
    )
    print(
        "audio sha256:",
        report["audio"]["sha256"],
    )


if __name__ == "__main__":
    main()