from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
import unicodedata
from importlib import metadata
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "asr_models"
    / "whisper-small.en"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "new_e2e"
    / "whisper_receiver.json"
)


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def normalize_english(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.lower()
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return " ".join(value.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an offline Whisper receiver without "
            "reference text or TTS metadata."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    audio_path = args.audio.resolve()
    output_path = args.output.resolve()

    required_files = (
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "preprocessor_config.json",
        model_dir / "tokenizer.json",
        audio_path,
    )

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required ASR files are missing:\n- "
            + "\n- ".join(missing)
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this ASR worker"
        )

    waveform, original_sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=False,
    )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(
            "Input audio is empty or has invalid dimensions"
        )

    if not np.isfinite(waveform).all():
        raise RuntimeError(
            "Input audio contains non-finite samples"
        )

    target_sample_rate = 16_000

    if original_sample_rate != target_sample_rate:
        asr_waveform = librosa.resample(
            waveform,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
        )
    else:
        asr_waveform = waveform

    asr_waveform = np.asarray(
        asr_waveform,
        dtype=np.float32,
    )

    torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()

    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )

    model_load_seconds = (
        time.perf_counter() - load_start
    )

    inputs = processor(
        asr_waveform,
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_features = inputs.input_features.to(
        device="cuda",
        dtype=torch.float16,
    )

    attention_mask = inputs.attention_mask.to(
        device="cuda",
    )

    inference_start = time.perf_counter()

    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
        )

    inference_seconds = (
        time.perf_counter() - inference_start
    )

    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    normalized = normalize_english(
        transcription
    )

    audio_info = sf.info(str(audio_path))

    report = {
        "status": "WAV_TO_TEXT_PASS",
        "transcription": {
            "text": transcription,
            "normalized": normalized,
            "nonempty": bool(
                transcription.strip()
            ),
        },
        "audio": {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "original_sample_rate": int(
                original_sample_rate
            ),
            "target_sample_rate": (
                target_sample_rate
            ),
            "original_frames": int(
                audio_info.frames
            ),
            "resampled_frames": int(
                asr_waveform.size
            ),
            "duration_seconds": float(
                asr_waveform.size
                / target_sample_rate
            ),
            "channels": int(
                audio_info.channels
            ),
            "format": str(audio_info.format),
            "subtype": str(audio_info.subtype),
        },
        "model": {
            "name": "openai/whisper-small.en",
            "directory": str(model_dir),
            "parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )
            ),
            "dtype": str(
                next(model.parameters()).dtype
            ),
            "device": str(
                next(model.parameters()).device
            ),
            "weights_bytes": int(
                (
                    model_dir
                    / "model.safetensors"
                ).stat().st_size
            ),
            "weights_sha256": sha256_file(
                model_dir / "model.safetensors"
            ),
        },
        "timing_seconds": {
            "model_load": float(
                model_load_seconds
            ),
            "inference": float(
                inference_seconds
            ),
        },
        "runtime": {
            "python": (
                platform.python_version()
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": package_version(
                "transformers"
            ),
            "accelerate": package_version(
                "accelerate"
            ),
            "librosa": package_version(
                "librosa"
            ),
            "soundfile": package_version(
                "soundfile"
            ),
            "cuda_device": (
                torch.cuda.get_device_name(0)
            ),
            "peak_allocated_mib": float(
                torch.cuda.max_memory_allocated()
                / 1024
                / 1024
            ),
            "offline_mode": True,
        },
        "receiver_constraints": {
            "reference_text_available": False,
            "reference_text_used": False,
            "tts_report_available": False,
            "tts_metadata_used": False,
            "spelling_correction_applied": False,
            "leading_spaces_restored": False,
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("WHISPER RECEIVER PASS")
    print("audio:", audio_path)
    print("report:", output_path)
    print(
        "transcription:",
        repr(transcription),
    )
    print(
        "normalized:",
        repr(normalized),
    )
    print(
        "parameters:",
        report["model"]["parameter_count"],
    )
    print(
        "inference seconds:",
        round(inference_seconds, 3),
    )
    print("reference used: False")
    print("TTS metadata used: False")


if __name__ == "__main__":
    main()