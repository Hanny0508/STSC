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
from typing import Sequence, TypeVar

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

DEFAULT_AUDIO = (
    PROJECT_ROOT
    / "artifacts"
    / "new_e2e"
    / "transformertts_client_smoke.wav"
)

DEFAULT_TTS_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "new_e2e"
    / "transformertts_client_smoke.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "new_e2e"
    / "whisper_small_en_smoke.json"
)

T = TypeVar("T")


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
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def levenshtein_distance(
    reference: Sequence[T],
    hypothesis: Sequence[T],
) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))

    for ref_index, ref_value in enumerate(
        reference,
        start=1,
    ):
        current = [ref_index]

        for hyp_index, hyp_value in enumerate(
            hypothesis,
            start=1,
        ):
            insertion = current[hyp_index - 1] + 1
            deletion = previous[hyp_index] + 1
            substitution = (
                previous[hyp_index - 1]
                + int(ref_value != hyp_value)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline Whisper small.en inference "
            "and produce an ASR evidence report."
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
        default=DEFAULT_AUDIO,
    )
    parser.add_argument(
        "--tts-report",
        type=Path,
        default=DEFAULT_TTS_REPORT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help=(
            "Optional spoken-text reference. When omitted, "
            "spoken_text is read from the TTS report."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir.resolve()
    audio_path = args.audio.resolve()
    tts_report_path = args.tts_report.resolve()
    output_path = args.output.resolve()

    required_files = (
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "preprocessor_config.json",
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

    tts_report = None

    if tts_report_path.is_file():
        tts_report = json.loads(
            tts_report_path.read_text(
                encoding="utf-8"
            )
        )

    if args.reference_text is not None:
        reference_text = args.reference_text
        reference_source = "command-line"
    elif isinstance(tts_report, dict):
        reference_text = str(
            tts_report["spoken_text"]
        )
        reference_source = str(tts_report_path)
    else:
        raise ValueError(
            "No reference text was provided and the "
            "TTS report could not be read"
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

    hypothesis = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    normalized_reference = normalize_english(
        reference_text
    )
    normalized_hypothesis = normalize_english(
        hypothesis
    )

    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()

    char_distance = levenshtein_distance(
        normalized_reference,
        normalized_hypothesis,
    )
    word_distance = levenshtein_distance(
        reference_words,
        hypothesis_words,
    )

    reference_char_count = len(
        normalized_reference
    )
    reference_word_count = len(reference_words)

    cer = (
        char_distance / reference_char_count
        if reference_char_count
        else 0.0
    )
    wer = (
        word_distance / reference_word_count
        if reference_word_count
        else 0.0
    )

    audio_info = sf.info(str(audio_path))

    report = {
        "status": "WAV_TO_TEXT_PASS",
        "reference": {
            "text": reference_text,
            "normalized": normalized_reference,
            "source": reference_source,
            "used_for_decoding": False,
            "note": (
                "The reference is used only for evaluation "
                "and does not alter the ASR hypothesis."
            ),
        },
        "transcription": {
            "text": hypothesis,
            "normalized": normalized_hypothesis,
            "nonempty": bool(hypothesis.strip()),
        },
        "metrics": {
            "raw_exact_match": (
                hypothesis == reference_text
            ),
            "normalized_exact_match": (
                normalized_hypothesis
                == normalized_reference
            ),
            "character_distance": char_distance,
            "reference_characters": (
                reference_char_count
            ),
            "cer": cer,
            "word_distance": word_distance,
            "reference_words": reference_word_count,
            "wer": wer,
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
            "channels": int(audio_info.channels),
            "format": str(audio_info.format),
            "subtype": str(audio_info.subtype),
            "peak": float(
                np.max(np.abs(waveform))
            ),
            "rms": float(
                np.sqrt(
                    np.mean(
                        np.square(waveform)
                    )
                )
            ),
        },
        "model": {
            "directory": str(model_dir),
            "name": "openai/whisper-small.en",
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
            "model_load": model_load_seconds,
            "inference": inference_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
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
            "cuda_available": bool(
                torch.cuda.is_available()
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
        "tts_provenance": {
            "report_path": (
                str(tts_report_path)
                if tts_report is not None
                else None
            ),
            "original_text": (
                tts_report.get("original_text")
                if isinstance(tts_report, dict)
                else None
            ),
            "spoken_text": (
                tts_report.get("spoken_text")
                if isinstance(tts_report, dict)
                else None
            ),
            "leading_space_count": (
                tts_report.get(
                    "leading_space_count"
                )
                if isinstance(tts_report, dict)
                else None
            ),
            "metadata_used_to_repair_asr": False,
        },
        "notes": [
            (
                "The TTS report supplies ground truth only "
                "for CER and WER calculation."
            ),
            (
                "No spelling correction, language-model "
                "repair, or reference-guided replacement "
                "is applied to the ASR output."
            ),
            (
                "Leading spaces cannot be represented by "
                "speech and are not restored here."
            ),
        ],
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

    print("WHISPER ASR WORKER PASS")
    print("audio:", audio_path)
    print("report:", output_path)
    print("reference:", repr(reference_text))
    print("hypothesis:", repr(hypothesis))
    print(
        "normalized reference:",
        repr(normalized_reference),
    )
    print(
        "normalized hypothesis:",
        repr(normalized_hypothesis),
    )
    print(
        "exact match:",
        report["metrics"][
            "normalized_exact_match"
        ],
    )
    print("CER:", round(cer, 6))
    print("WER:", round(wer, 6))
    print(
        "parameters:",
        report["model"]["parameter_count"],
    )
    print(
        "inference seconds:",
        round(inference_seconds, 3),
    )


if __name__ == "__main__":
    main()