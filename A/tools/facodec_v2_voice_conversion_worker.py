#!/usr/bin/env python
"""Reference-conditioned FACodec V2 voice conversion for DAQ-TTS.

Source WAV supplies content/prosody codes. Reference WAV supplies the speaker
embedding. Source residual acoustic-detail codes are deliberately excluded,
matching the upstream zero-shot voice-conversion example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import wave
from datetime import datetime, timezone
from pathlib import Path


HF_REPO = "amphion/naturalspeech3_facodec"
HF_REVISION = "314afc3ea1455ba881a0e484ef9408b6cb996736"
ENCODER_NAME = "ns3_facodec_encoder_v2.bin"
DECODER_NAME = "ns3_facodec_decoder_v2.bin"
SOURCE_REPOSITORY = "https://github.com/lifeiteng/naturalspeech3_facodec"
SOURCE_REVISION = "aac16b7268b1b96e70b9e3526b13e216984e80d0"
SAMPLE_RATE = 16_000
HOP_SIZE = 200
CODEBOOKS = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_info(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as source:
        frames = source.getnframes()
        rate = source.getframerate()
        value = {
            "sample_rate_hz": rate,
            "channels": source.getnchannels(),
            "sample_width_bytes": source.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / rate,
        }
    value.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_weights(module: object, path: Path, torch: object) -> None:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location="cpu")
    module.load_state_dict(state)


def checkpoints(model_dir: Path | None) -> tuple[Path, Path]:
    if model_dir is None:
        _local = os.environ.get("ZSC_TSD_FACODEC_MODEL_DIR")
        if _local:
            model_dir = Path(_local)
    if model_dir is not None:
        encoder, decoder = model_dir / ENCODER_NAME, model_dir / DECODER_NAME
        missing = [str(item) for item in (encoder, decoder) if not item.is_file()]
        if missing:
            raise FileNotFoundError("FACodec V2 checkpoint missing:\n- " + "\n- ".join(missing))
        return encoder, decoder
    from huggingface_hub import hf_hub_download

    return (
        Path(hf_hub_download(repo_id=HF_REPO, revision=HF_REVISION, filename=ENCODER_NAME)),
        Path(hf_hub_download(repo_id=HF_REPO, revision=HF_REVISION, filename=DECODER_NAME)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACodec V2 two-WAV voice conversion")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    encoder_path, decoder_path = checkpoints(args.model_dir.resolve() if args.model_dir else None)
    weights = {
        "repository": HF_REPO,
        "revision": HF_REVISION,
        "encoder_v2": {"path": str(encoder_path.resolve()), "bytes": encoder_path.stat().st_size, "sha256": sha256_file(encoder_path)},
        "decoder_v2": {"path": str(decoder_path.resolve()), "bytes": decoder_path.stat().st_size, "sha256": sha256_file(decoder_path)},
    }
    if args.download_only:
        result = {
            "status": "FACODEC_V2_CHECKPOINTS_READY",
            "schema": "zsc-tsd.facodec-v2-checkpoints.v1",
            "checkpoints": weights,
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        }
        write_json(args.report.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.source is None or args.reference is None or args.output is None:
        raise SystemExit("--source, --reference, and --output are required")
    source_path, reference_path, output_path = (
        args.source.resolve(), args.reference.resolve(), args.output.resolve()
    )
    for item in (source_path, reference_path):
        if not item.is_file():
            raise FileNotFoundError(item)
    if source_path == reference_path:
        raise ValueError("source and reference WAV must be different files")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from ns3_codec import FACodecDecoderV2, FACodecEncoderV2

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    def load_audio(path: Path) -> tuple[object, int, int, int]:
        values, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = np.mean(values, axis=1, dtype=np.float32)
        tensor = torch.from_numpy(mono).unsqueeze(0)
        if rate != SAMPLE_RATE:
            tensor = torchaudio.functional.resample(tensor, rate, SAMPLE_RATE)
        frames = int(tensor.shape[-1])
        padding = (-frames) % HOP_SIZE
        if padding:
            tensor = torch.nn.functional.pad(tensor, (0, padding))
        return tensor.unsqueeze(0).to(device), rate, frames, padding

    source, source_rate, source_frames, source_padding = load_audio(source_path)
    reference, reference_rate, reference_frames, reference_padding = load_audio(reference_path)
    if source_frames < SAMPLE_RATE or reference_frames < 2 * SAMPLE_RATE:
        raise ValueError("source must be at least 1 s and reference at least 2 s")

    encoder = FACodecEncoderV2(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
    decoder = FACodecDecoderV2(
        in_channels=256, upsample_initial_channel=1024, ngf=32,
        up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1,
        vq_num_q_r=3, vq_dim=256, codebook_dim=8,
        codebook_size_prosody=10, codebook_size_content=10,
        codebook_size_residual=10, use_gr_x_timbre=True,
        use_gr_residual_f0=True, use_gr_residual_phone=True,
    )
    load_weights(encoder, encoder_path, torch)
    load_weights(decoder, decoder_path, torch)
    encoder, decoder = encoder.eval().to(device), decoder.eval().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        source_encoded = encoder(source)
        source_prosody = encoder.get_prosody_feature(source)
        reference_encoded = encoder(reference)
        reference_prosody = encoder.get_prosody_feature(reference)
        _, source_ids, _, source_quantized, source_speaker = decoder(
            source_encoded, source_prosody, eval_vq=False, vq=True
        )
        _, reference_ids, _, reference_quantized, reference_speaker = decoder(
            reference_encoded, reference_prosody, eval_vq=False, vq=True
        )
        converted_embedding = decoder.vq2emb(source_ids, use_residual=False)
        reconstructed = decoder.inference(converted_embedding, reference_speaker)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    reconstructed = reconstructed[0, 0, :source_frames].detach().float().cpu().numpy()
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("FACodec V2 generated non-finite samples")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.clip(reconstructed, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")

    result = {
        "status": "FACODEC_V2_VOICE_CONVERSION_PASS",
        "schema": "zsc-tsd.facodec-v2-voice-conversion.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "two_wav_zero_shot_voice_conversion",
        "source_audio_role": "TransformerTTS content and prosody",
        "reference_audio_role": "natural reference speaker embedding",
        "source_residual_codes_used": False,
        "path_executed": [
            "FACodecEncoderV2(source)", "FACodecEncoderV2(reference)",
            "FACodecDecoderV2.quantize(source)", "FACodecDecoderV2.quantize(reference)",
            "FACodecDecoderV2.vq2emb(source_ids,use_residual=False)",
            "FACodecDecoderV2.inference(source_codes,reference_speaker_embedding)",
        ],
        "source_audio": wav_info(source_path),
        "reference_audio": wav_info(reference_path),
        "output_audio": wav_info(output_path),
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE, "hop_size_samples": HOP_SIZE,
            "source_original_sample_rate_hz": source_rate,
            "reference_original_sample_rate_hz": reference_rate,
            "source_padding_samples": source_padding,
            "reference_padding_samples": reference_padding,
        },
        "tensors": {
            "source_encoded_shape": list(source_encoded.shape),
            "reference_encoded_shape": list(reference_encoded.shape),
            "source_vq_id_shape": list(source_ids.shape),
            "reference_vq_id_shape": list(reference_ids.shape),
            "source_prosody_code_shape": list(source_ids[:1].shape),
            "source_content_code_shape": list(source_ids[1:3].shape),
            "source_residual_code_shape": list(source_ids[3:].shape),
            "reference_speaker_embedding_shape": list(reference_speaker.shape),
            "converted_embedding_shape": list(converted_embedding.shape),
            "source_quantized_component_shapes": [list(item.shape) for item in source_quantized],
            "reference_quantized_component_shapes": [list(item.shape) for item in reference_quantized],
        },
        "checkpoints": weights,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchaudio": torchaudio.__version__, "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    write_json(args.report.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
