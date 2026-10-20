from __future__ import annotations

import runpy

import pytest
from pathlib import Path

from zsc_tsd.speech.daqtts import DAQTTSConfig


ROOT = Path(__file__).resolve().parents[1]


def test_facodec_worker_is_pinned_and_executes_all_real_stages() -> None:
    values = runpy.run_path(str(ROOT / "tools" / "facodec_roundtrip_worker.py"))
    assert values["HF_REPO"] == "amphion/naturalspeech3_facodec"
    assert len(values["HF_REVISION"]) == 40
    assert len(values["SOURCE_REVISION"]) == 40
    source = (ROOT / "tools" / "facodec_roundtrip_worker.py").read_text(encoding="utf-8")
    assert "encoded = encoder(tensor)" in source
    assert "eval_vq=False, vq=True" in source
    assert "decoder.inference(vq_post_emb, speaker_embedding)" in source


def test_facodec_v2_uses_two_wavs_and_reference_speaker() -> None:
    try:
        values = runpy.run_path(
            str(ROOT / "tools" / "facodec_v2_voice_conversion_worker.py")
        )
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"FACodec runtime not installed: {exc}")
    assert values["HF_REPO"] == "amphion/naturalspeech3_facodec"
    assert values["ENCODER_NAME"] == "ns3_facodec_encoder_v2.bin"
    assert values["DECODER_NAME"] == "ns3_facodec_decoder_v2.bin"
    assert len(values["HF_REVISION"]) == 40
    source = (
        ROOT / "tools" / "facodec_v2_voice_conversion_worker.py"
    ).read_text(encoding="utf-8")
    assert "FACodecEncoderV2" in source
    assert "FACodecDecoderV2" in source
    assert "decoder.vq2emb(source_ids, use_residual=False)" in source
    assert "decoder.inference(converted_embedding, reference_speaker)" in source


def test_daqtts_definition_does_not_claim_algorithmic_novelty() -> None:
    source = (ROOT / "zsc_tsd" / "speech" / "daqtts.py").read_text(encoding="utf-8")
    assert '"novel_algorithm_claimed": False' in source
    assert "FACodecEncoderV2(source and reference)" in source
    assert "FACodecDecoderV2 vector quantizer" in source
    assert "FACodecDecoderV2.inference(source_codes,reference_speaker_embedding)" in source
    assert DAQTTSConfig().facodec_device == "auto"


def test_acl_imdb_source_is_official_and_frozen() -> None:
    values = runpy.run_path(str(ROOT / "tools" / "prepare_acl_imdb_prompts.py"))
    assert values["OFFICIAL_URL"].startswith("https://ai.stanford.edu/")
    assert values["OFFICIAL_MD5"] == "7c2ac02c03563afcf9b574c7e56c153a"
    assert values["choose_sentence"](
        "This deliberately longer review sentence contains enough ordinary English words to satisfy the frozen deterministic prompt selection rule."
    ) is not None


def test_v15_routes_public_carrier_through_daqtts() -> None:
    source = (
        ROOT / "tools" / "zts_secure_imdb_daqtts_batch_runner_v1_5.py"
    ).read_text(encoding="utf-8")
    assert "legacy.TransformerTTSClient = DAQTTSClient" in source
    assert '"facodec_included": True' in source
    assert '"historical_compatibility_dataset_used": False' in source
    assert '"source_residual_codes_used": False' in source
    assert "--reference-wav" in source
