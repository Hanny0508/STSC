#!/usr/bin/env python
"""Reconstruct a real WAV through the official NaturalSpeech 3 FACodec.

This worker is intentionally isolated from TensorFlow/TransformerTTS.  It
performs the actual FACodec encoder -> vector quantizer -> decoder path and
writes machine-verifiable evidence for every step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path


HF_REPO = "amphion/naturalspeech3_facodec"
HF_REVISION = "5b957313db7b2ff39ed36169df1604e6c1d8add3"
ENCODER_NAME = "ns3_facodec_encoder.bin"
DECODER_NAME = "ns3_facodec_decoder.bin"
SAMPLE_RATE = 16_000
HOP_SIZE = 200
CODEBOOKS = 6
SOURCE_REPOSITORY = "https://github.com/lifeiteng/naturalspeech3_facodec"
SOURCE_REVISION = "aac16b7268b1b96e70b9e3526b13e216984e80d0"


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
        result = {
            "sample_rate_hz": rate,
            "channels": source.getnchannels(),
            "sample_width_bytes": source.getsampwidth(),
            "frames": frames,
            "duration_seconds": frames / rate,
        }
    result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_state_dict(torch_module: object, path: Path, torch: object) -> None:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location="cpu")
    torch_module.load_state_dict(state)


def resolve_checkpoints(model_dir: Path | None) -> tuple[Path, Path]:
    if model_dir is not None:
        encoder = model_dir / ENCODER_NAME
        decoder = model_dir / DECODER_NAME
        missing = [str(item) for item in (encoder, decoder) if not item.is_file()]
        if missing:
            raise FileNotFoundError("FACodec checkpoint missing:\n- " + "\n- ".join(missing))
        return encoder, decoder
    from huggingface_hub import hf_hub_download

    encoder = Path(
        hf_hub_download(repo_id=HF_REPO, revision=HF_REVISION, filename=ENCODER_NAME)
    )
    decoder = Path(
        hf_hub_download(repo_id=HF_REPO, revision=HF_REVISION, filename=DECODER_NAME)
    )
    return encoder, decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official FACodec encode/quantize/decode reconstruction worker."
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    encoder_ckpt, decoder_ckpt = resolve_checkpoints(
        args.model_dir.resolve() if args.model_dir else None
    )
    checkpoint_evidence = {
        "repository": HF_REPO,
        "revision": HF_REVISION,
        "encoder": {
            "path": str(encoder_ckpt.resolve()),
            "bytes": encoder_ckpt.stat().st_size,
            "sha256": sha256_file(encoder_ckpt),
        },
        "decoder": {
            "path": str(decoder_ckpt.resolve()),
            "bytes": decoder_ckpt.stat().st_size,
            "sha256": sha256_file(decoder_ckpt),
        },
    }
    if args.download_only:
        report = {
            "status": "FACODEC_CHECKPOINTS_READY",
            "schema": "zsc-tsd.facodec-checkpoints.v1",
            "checkpoints": checkpoint_evidence,
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        }
        atomic_json(args.report.resolve(), report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.input is None or args.output is None:
        raise SystemExit("--input and --output are required unless --download-only is used")
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from ns3_codec import FACodecDecoder, FACodecEncoder

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    audio, source_rate = sf.read(str(input_path), dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1, dtype=np.float32)
    tensor = torch.from_numpy(audio).unsqueeze(0)
    if source_rate != SAMPLE_RATE:
        tensor = torchaudio.functional.resample(tensor, source_rate, SAMPLE_RATE)
    original_frames = int(tensor.shape[-1])
    padding = (-original_frames) % HOP_SIZE
    if padding:
        tensor = torch.nn.functional.pad(tensor, (0, padding))
    tensor = tensor.unsqueeze(0).to(device)

    encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
    decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    load_state_dict(encoder, encoder_ckpt, torch)
    load_state_dict(decoder, decoder_ckpt, torch)
    encoder = encoder.eval().to(device)
    decoder = decoder.eval().to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        encoded = encoder(tensor)
        vq_post_emb, vq_ids, _, quantized, speaker_embedding = decoder(
            encoded, eval_vq=False, vq=True
        )
        reconstructed = decoder.inference(vq_post_emb, speaker_embedding)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    reconstructed = reconstructed[0, 0, :original_frames].detach().float().cpu().numpy()
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("FACodec reconstructed non-finite samples")
    reconstructed = np.clip(reconstructed, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), reconstructed, SAMPLE_RATE, subtype="PCM_16")

    report = {
        "status": "FACODEC_ROUNDTRIP_PASS",
        "schema": "zsc-tsd.facodec-roundtrip.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "FACodec is an upstream codec component. In this artifact, DAQ-TTS is "
            "the engineering combination TransformerTTS plus FACodec; it is not "
            "claimed as a newly invented TTS algorithm."
        ),
        "input": wav_info(input_path),
        "output": wav_info(output_path),
        "preprocessing": {
            "source_sample_rate_hz": source_rate,
            "facodec_sample_rate_hz": SAMPLE_RATE,
            "mono_mix": True,
            "hop_size_samples": HOP_SIZE,
            "padding_samples": padding,
        },
        "facodec": {
            "path_executed": ["FACodecEncoder", "vector_quantizer", "FACodecDecoder.inference"],
            "codebooks": CODEBOOKS,
            "encoded_shape": list(encoded.shape),
            "vq_post_embedding_shape": list(vq_post_emb.shape),
            "vq_id_shape": list(vq_ids.shape),
            "quantized_component_shapes": [list(item.shape) for item in quantized],
            "speaker_embedding_shape": list(speaker_embedding.shape),
            "reconstructed_shape": list(reconstructed.shape),
        },
        "checkpoints": checkpoint_evidence,
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
