from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .adapter import ZTSModelAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .encoder import ZTSEncodedText


class ZTSDecodingError(RuntimeError):
    """Raised when steganographic text cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class ZTSDecodedPayload:
    """Result of recovering a payload from steganographic text."""

    payload: bytes
    payload_base64: str

    prompt: str
    normalized_prompt: str
    stegotext: str
    full_text: str

    expected_payload_size: int
    expected_bit_count: int
    recovered_bit_count: int

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    hide_token_ids: tuple[int, ...]

    decoder_succeeded: bool
    prompt_retokenize_exact: bool
    prompt_prefix_exact: bool
    expected_tokens_exact: bool | None

    @property
    def extra_bit_count(self) -> int:
        """
        Number of Huffman completion bits after the actual payload.

        A nonzero value is allowed. These bits are discarded according
        to expected_payload_size.
        """

        return self.recovered_bit_count - self.expected_bit_count

    @property
    def payload_size_exact(self) -> bool:
        return len(self.payload) == self.expected_payload_size


class ZTSDecoder:
    """
    Formal receiver for the legacy zero-shot-GLS text-to-bits algorithm.

    Sender and receiver must share:
    - the same model checkpoint;
    - the same prompt;
    - the same ZTSConfig;
    - the same Huffman implementation;
    - the expected payload byte length.
    """

    def __init__(self, adapter: ZTSModelAdapter) -> None:
        self.adapter = adapter

    def decode(
        self,
        stegotext: str,
        prompt: str,
        *,
        expected_payload_size: int,
        expected_output_token_ids: "Sequence[int] | Any | None" = None,
        strict: bool = True,
    ) -> ZTSDecodedPayload:
        """
        Recover payload bytes from a steganographic continuation.

        Args:
            stegotext:
                Exact continuation received from the text or ASR stage.
                Leading spaces and punctuation must not be stripped.
            prompt:
                Shared initial context used during encoding.
            expected_payload_size:
                Original payload length in bytes. This should be carried
                as authenticated control-channel metadata.
            expected_output_token_ids:
                Optional sender-side token sequence for diagnostic tests.
                A real receiver does not require this field.
            strict:
                Raise ZTSDecodingError when validation fails.
        """

        if not isinstance(stegotext, str):
            raise TypeError("stegotext must be a string")

        if stegotext == "":
            raise ValueError("stegotext must not be empty")

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if prompt == "":
            raise ValueError("prompt must not be empty")

        if isinstance(expected_payload_size, bool) or not isinstance(
            expected_payload_size,
            int,
        ):
            raise TypeError("expected_payload_size must be an integer")

        if expected_payload_size <= 0:
            raise ValueError(
                "expected_payload_size must be greater than zero"
            )

        import torch

        self.adapter.load()
        self.adapter.configure_determinism()

        tokenizer = self.adapter.tokenizer
        model = self.adapter.model
        config = self.adapter.config
        legacy = self.adapter.legacy

        # Recreate the same normalized prompt boundary used by encoder.py.
        original_prompt_ids_tensor = tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids

        original_prompt_token_ids = self._to_token_tuple(
            original_prompt_ids_tensor[0]
        )

        normalized_prompt = tokenizer.decode(
            list(original_prompt_token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        prompt_ids = tokenizer(
            normalized_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self.adapter.device)

        prompt_token_ids = self._to_token_tuple(prompt_ids[0])

        prompt_retokenize_exact = (
            prompt_token_ids == original_prompt_token_ids
        )

        # Deliberately concatenate without strip/lstrip/rstrip.
        full_text = normalized_prompt + stegotext

        hide_ids = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self.adapter.device)

        hide_token_ids = self._to_token_tuple(hide_ids[0])

        prompt_len = len(prompt_token_ids)

        if len(hide_token_ids) <= prompt_len:
            raise ZTSDecodingError(
                "The received text contains no tokens after the prompt."
            )

        prompt_prefix_exact = (
            hide_token_ids[:prompt_len] == prompt_token_ids
        )

        generated_token_ids = tuple(
            hide_token_ids[prompt_len:]
        )

        expected_tokens_exact: bool | None = None

        if expected_output_token_ids is not None:
            expected_tokens = self._to_token_tuple(
                expected_output_token_ids
            )
            expected_tokens_exact = (
                hide_token_ids == expected_tokens
            )

        with (
            torch.inference_mode(),
            legacy.prompt_gen.random_state(config.prompt_seed),
        ):
            (
                recovered_bits,
                decoder_succeeded,
            ) = legacy.hide_extract.extract_bits_with_prompt_ids_by_egs(
                model=model,
                prompt_ids=prompt_ids,
                hide_ids=hide_ids,
                mode=config.egs_mode,
                threshold=config.threshold,
                temperature=config.temperature,
                temperature_alpha=config.temperature_alpha,
                max_bpw=config.max_bpw,
            )

        recovered_bit_count = len(recovered_bits)
        expected_bit_count = expected_payload_size * 8

        if recovered_bit_count < expected_bit_count:
            raise ZTSDecodingError(
                "The decoder recovered fewer bits than required: "
                f"recovered={recovered_bit_count}, "
                f"required={expected_bit_count}."
            )

        # The final Huffman token may include completion bits. Only the
        # authenticated payload length determines the valid prefix.
        trimmed_bits = recovered_bits[:expected_bit_count]

        try:
            payload_base64 = legacy.codec.bits2base64(
                trimmed_bits
            )

            payload = base64.b64decode(
                payload_base64.encode("ascii"),
                validate=True,
            )
        except Exception as exc:
            raise ZTSDecodingError(
                "Recovered bits could not be converted to payload bytes."
            ) from exc

        result = ZTSDecodedPayload(
            payload=payload,
            payload_base64=payload_base64,
            prompt=prompt,
            normalized_prompt=normalized_prompt,
            stegotext=stegotext,
            full_text=full_text,
            expected_payload_size=expected_payload_size,
            expected_bit_count=expected_bit_count,
            recovered_bit_count=recovered_bit_count,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            hide_token_ids=hide_token_ids,
            decoder_succeeded=bool(decoder_succeeded),
            prompt_retokenize_exact=prompt_retokenize_exact,
            prompt_prefix_exact=prompt_prefix_exact,
            expected_tokens_exact=expected_tokens_exact,
        )

        if strict:
            self._validate_strict_result(result)

        return result

    def decode_encoded(
        self,
        encoded: "ZTSEncodedText",
        *,
        strict: bool = True,
    ) -> ZTSDecodedPayload:
        """
        Decode an encoder result with sender-side token validation.

        This helper is intended for tests and local experiments. A real
        receiver only needs decode(), stegotext, prompt, and payload size.
        """

        return self.decode(
            stegotext=encoded.stegotext,
            prompt=encoded.prompt,
            expected_payload_size=len(encoded.payload),
            expected_output_token_ids=encoded.output_token_ids,
            strict=strict,
        )

    @staticmethod
    def _to_token_tuple(
        token_ids: "Sequence[int] | Any",
    ) -> tuple[int, ...]:
        detach = getattr(token_ids, "detach", None)

        if callable(detach):
            token_ids = detach().cpu().tolist()

        return tuple(int(token_id) for token_id in token_ids)

    @staticmethod
    def _validate_strict_result(
        result: ZTSDecodedPayload,
    ) -> None:
        errors: list[str] = []

        if not result.decoder_succeeded:
            errors.append(
                "a received token was not found in the reconstructed "
                "candidate set"
            )

        if not result.prompt_retokenize_exact:
            errors.append(
                "the normalized prompt does not reproduce the sender's "
                "prompt token sequence"
            )

        if not result.prompt_prefix_exact:
            errors.append(
                "the received full text does not preserve the prompt "
                "token prefix"
            )

        if result.expected_tokens_exact is False:
            errors.append(
                "the serialized text does not match the sender-side "
                "token sequence"
            )

        if not result.payload_size_exact:
            errors.append(
                "decoded payload size differs from authenticated metadata: "
                f"decoded={len(result.payload)}, "
                f"expected={result.expected_payload_size}"
            )

        if errors:
            raise ZTSDecodingError(
                "ZTS decoding validation failed:\n- "
                + "\n- ".join(errors)
            )