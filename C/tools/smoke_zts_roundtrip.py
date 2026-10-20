from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from gptqmodel import BACKEND, GPTQModel
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / "legacy" / "zero-shot-GLS"
LEGACY_SRC = LEGACY_ROOT / "src"
COMPAT_ROOT = ROOT / "zsc_tsd" / "compat"

# Put the Windows compatibility implementation before legacy imports.
sys.path.insert(0, str(LEGACY_SRC))
sys.path.insert(0, str(COMPAT_ROOT))

import codec  # noqa: E402
import hide_extract  # noqa: E402
import prompt_gen  # noqa: E402
from p_utils import seed_everything  # noqa: E402


MODEL_DIR = (
    LEGACY_ROOT
    / "tmp_saves"
    / "local_models"
    / "llama-2-7b-chat-gptq-4bit-32g-actorder"
)

# Paper-compatible ZTS parameters.
SEED = 2023
EGS_MODE = "huffman"
THRESHOLD = 0.002
TEMPERATURE = 1.0
TEMPERATURE_ALPHA = 1.25
MAX_BPW = 5
MAX_NEW_TOKENS = 160

PAYLOAD = b"ZTS1"
PROMPT = (
    "[INST] Continue with one concise sentence about "
    "secure IoT communication. [/INST]"
)


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    seed_everything(42, deterministic=True, warn_only=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

    print("Loading tokenizer:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
        legacy=True,
    )

    print("Loading GPTQ model...")
    start = time.time()

    wrapper = GPTQModel.load(
        str(MODEL_DIR),
        device="cuda:0",
        backend=BACKEND.TORCH,
        trust_remote_code=False,
        local_files_only=True,
    )

    model = getattr(wrapper, "model", wrapper).eval()

    # Compatibility for legacy zero-shot-GLS code.
    if not hasattr(model, "vocab_size"):
        model.vocab_size = model.config.vocab_size

    print(f"Model loaded in {time.time() - start:.2f}s")

    payload_b64 = base64.b64encode(PAYLOAD).decode("ascii")
    source_bits = codec.base642bits(payload_b64)

    # -------------------------
    # Encode bits into text
    # -------------------------
    with prompt_gen.random_state(SEED):
        prompt_ids = tokenizer(
            PROMPT,
            return_tensors="pt",
        ).input_ids.to(model.device)

        out_ids, truncated, used_bits, ppl = (
            hide_extract.hide_bits_with_prompt_ids_by_egs(
                model=model,
                prompt_ids=prompt_ids,
                bits=source_bits,
                mode=EGS_MODE,
                threshold=THRESHOLD,
                temperature=TEMPERATURE,
                temperature_alpha=TEMPERATURE_ALPHA,
                max_bpw=MAX_BPW,
                max_new_tokens=MAX_NEW_TOKENS,
                complete_sent=False,
            )
        )

    prompt_len = prompt_ids.shape[1]
    stego_ids = out_ids[0, prompt_len:]

    # Decode the complete sequence first so that SentencePiece preserves
    # the whitespace boundary between the prompt and generated continuation.
    normalized_prompt = tokenizer.decode(
        prompt_ids[0].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    normalized_full_text = tokenizer.decode(
        out_ids[0].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if not normalized_full_text.startswith(normalized_prompt):
        raise RuntimeError(
            "Decoded full text does not begin with the normalized prompt.\n"
            f"prompt={normalized_prompt!r}\n"
            f"full={normalized_full_text!r}"
        )

    # This preserves the leading space before the first generated word.
    stegotext = normalized_full_text[len(normalized_prompt):]

    # -------------------------
    # Decode text back to bits
    # -------------------------
    with prompt_gen.random_state(SEED):
        decode_prompt_ids = tokenizer(
            normalized_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(model.device)

        hide_ids = tokenizer(
            normalized_prompt + stegotext,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(model.device)

        recovered_bits, decode_succeeded = (
            hide_extract.extract_bits_with_prompt_ids_by_egs(
                model=model,
                prompt_ids=decode_prompt_ids,
                hide_ids=hide_ids,
                mode=EGS_MODE,
                threshold=THRESHOLD,
                temperature=TEMPERATURE,
                temperature_alpha=TEMPERATURE_ALPHA,
                max_bpw=MAX_BPW,
            )
        )

    # Huffman may pad the final codeword. Only used_bits are payload bits.
    trimmed_bits = recovered_bits[:used_bits]
    recovered_b64 = codec.bits2base64(trimmed_bits)
    recovered_payload = base64.b64decode(recovered_b64)

    token_roundtrip = torch.equal(hide_ids, out_ids)
    if not token_roundtrip:
        expected = out_ids[0].tolist()
        actual = hide_ids[0].tolist()

        mismatch_index = next(
            (
                i
                for i, (expected_id, actual_id) in enumerate(
                zip(expected, actual)
            )
                if expected_id != actual_id
            ),
            min(len(expected), len(actual)),
        )

        print("first mismatch idx :", mismatch_index)
        print(
            "expected token IDs :",
            expected[mismatch_index:mismatch_index + 8],
        )
        print(
            "actual token IDs   :",
            actual[mismatch_index:mismatch_index + 8],
        )
        print(
            "expected tokens    :",
            tokenizer.convert_ids_to_tokens(
                expected[mismatch_index:mismatch_index + 8]
            ),
        )
        print(
            "actual tokens      :",
            tokenizer.convert_ids_to_tokens(
                actual[mismatch_index:mismatch_index + 8]
            ),
        )
    exact_payload = recovered_payload == PAYLOAD

    print()
    print("========== ZTS ROUNDTRIP ==========")
    print("payload bytes      :", PAYLOAD)
    print("payload base64     :", payload_b64)
    print("payload bits       :", len(source_bits))
    print("used bits          :", used_bits)
    print("generated tokens   :", len(stego_ids))
    print("truncated          :", truncated)
    print("perplexity         :", round(float(ppl), 6))
    print("stegotext          :", repr(stegotext))
    print("retokenize exact   :", token_roundtrip)
    print("decoder succeeded  :", decode_succeeded)
    print("recovered base64   :", recovered_b64)
    print("recovered payload  :", recovered_payload)
    print("payload exact      :", exact_payload)
    print("===================================")

    passed = (
        not truncated
        and used_bits == len(source_bits)
        and token_roundtrip
        and decode_succeeded
        and exact_payload
    )

    if not passed:
        raise SystemExit("ZTS ROUNDTRIP FAIL")

    print("ZTS ROUNDTRIP PASS")


if __name__ == "__main__":
    main()