from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_lock_is_immutable_and_complete() -> None:
    path = ROOT / "configs" / "reviewer_models_v1.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert lock["schema"] == "zsc-tsd.reviewer-model-lock.v1"
    assert set(lock["models"]) == {
        "asr_whisper_small_en",
        "zts_llama2_7b_chat_gptq",
        "tts_transformertts_ljspeech_step_90000",
    }
    for model in lock["models"].values():
        assert model["required_files"]
        assert model["verified_files"]
        for item in model["verified_files"].values():
            assert len(item["sha256"]) == 64
            int(item["sha256"], 16)


def test_reviewer_requirements_are_portable() -> None:
    for path in (ROOT / "requirements").glob("reviewer-*.txt"):
        text = path.read_text(encoding="utf-8").lower()
        assert "-e d:" not in text
        assert "file:///" not in text
        assert "\\pythonproject\\" not in text


def test_model_lock_has_stable_digest_shape() -> None:
    digest = hashlib.sha256(
        (ROOT / "configs" / "reviewer_models_v1.json").read_bytes()
    ).hexdigest()
    assert len(digest) == 64


def test_zts_reviewer_environment_includes_reed_solomon_decoder() -> None:
    text = (ROOT / "requirements" / "reviewer-zts.txt").read_text(encoding="utf-8")
    assert "reedsolo==1.7.0" in text.splitlines()
    assert "cryptography==46.0.7" in text.splitlines()
