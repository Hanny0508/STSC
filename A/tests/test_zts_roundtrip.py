from __future__ import annotations

import pytest

from zsc_tsd.zts import (
    ZTSDecoder,
    ZTSEncoder,
    ZTSModelAdapter,
)


PROMPT = (
    "[INST] Continue with one concise sentence about "
    "secure IoT communication. [/INST]"
)

PAYLOAD = b"ZTS1"


def test_zts_adapter_is_lazy() -> None:
    adapter = ZTSModelAdapter()
    assert adapter.is_loaded is False


@pytest.fixture(scope="module")
def loaded_adapter() -> ZTSModelAdapter:
    torch = pytest.importorskip("torch")
    pytest.importorskip("gptqmodel")
    pytest.importorskip("transformers")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the GPTQ ZTS integration test")

    adapter = ZTSModelAdapter()
    try:
        adapter.validate_files()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    adapter.load()

    try:
        yield adapter
    finally:
        adapter.release()


def test_zts_paper_baseline_roundtrip(
    loaded_adapter: ZTSModelAdapter,
) -> None:
    encoder = ZTSEncoder(loaded_adapter)
    decoder = ZTSDecoder(loaded_adapter)

    encoded = encoder.encode(
        payload=PAYLOAD,
        prompt=PROMPT,
    )

    decoded = decoder.decode_encoded(encoded)

    # Encoder validation.
    assert encoded.payload == PAYLOAD
    assert encoded.payload_base64 == "WlRTMQ=="
    assert encoded.source_bit_count == 32
    assert encoded.used_bit_count == 32
    assert encoded.generated_token_count == 22
    assert encoded.truncated is False
    assert encoded.retokenize_exact is True
    assert encoded.payload_fully_embedded is True
    assert encoded.perplexity == pytest.approx(
        11.69293,
        abs=1e-5,
    )

    # Preserve the exact SentencePiece boundary.
    assert encoded.stegotext.startswith("  Securing")

    # Decoder validation.
    assert decoded.decoder_succeeded is True
    assert decoded.prompt_retokenize_exact is True
    assert decoded.prompt_prefix_exact is True
    assert decoded.expected_tokens_exact is True

    assert decoded.expected_payload_size == len(PAYLOAD)
    assert decoded.expected_bit_count == 32
    assert decoded.recovered_bit_count == 33
    assert decoded.extra_bit_count == 1

    assert decoded.payload_size_exact is True
    assert decoded.payload_base64 == "WlRTMQ=="
    assert decoded.payload == PAYLOAD
