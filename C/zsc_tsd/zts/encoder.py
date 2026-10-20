from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .adapter import ZTSModelAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence


class ZTSEncodingError(RuntimeError):
    """Raised when a payload cannot be encoded losslessly."""


@dataclass(frozen=True, slots=True)
class ZTSEncodedText:
    """Result of embedding a binary payload into generated text."""

    payload: bytes
    payload_base64: str

    prompt: str
    normalized_prompt: str
    stegotext: str
    full_text: str

    source_bit_count: int
    used_bit_count: int

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]

    truncated: bool
    retokenize_exact: bool
    perplexity: float

    @property
    def payload_fully_embedded(self) -> bool:
        return (
            not self.truncated
            and self.used_bit_count == self.source_bit_count
        )

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_token_ids)


class ZTSEncoder:
    """
    Formal adapter for the legacy zero-shot-GLS bits-to-text algorithm.

    The encoder preserves the exact SentencePiece text boundary so that
    the receiver can reconstruct the same token sequence and Huffman trees.
    """

    def __init__(self, adapter: ZTSModelAdapter) -> None:
        self.adapter = adapter

    def encode(
        self,
        payload: bytes | bytearray | memoryview,
        prompt: str,
        *,
        strict: bool = True,
    ) -> ZTSEncodedText:
        """
        Embed payload bytes into a generated steganographic continuation.

        Args:
            payload:
                Secret binary data to embed.
            prompt:
                Shared initial context used by sender and receiver.
                It is never stripped or otherwise modified.
            strict:
                When true, raise an exception unless every payload bit is
                embedded and the serialized text retokenizes exactly.

        Returns:
            A ZTSEncodedText object containing the text, token IDs, bit
            counts, and validation metadata.
        """

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                "payload must be bytes, bytearray, or memoryview"
            )

        payload_bytes = bytes(payload)

        if not payload_bytes:
            raise ValueError("payload must not be empty")

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        if prompt == "":
            raise ValueError("prompt must not be empty")

        import torch

        self.adapter.load()
        self.adapter.configure_determinism()

        tokenizer = self.adapter.tokenizer
        model = self.adapter.model
        config = self.adapter.config
        legacy = self.adapter.legacy

        payload_base64 = base64.b64encode(
            payload_bytes
        ).decode("ascii")

        source_bits = legacy.codec.base642bits(payload_base64)
        source_bit_count = len(source_bits)

        with (
            torch.inference_mode(),
            legacy.prompt_gen.random_state(config.prompt_seed),
        ):
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids.to(self.adapter.device)

            (
                output_ids,
                truncated,
                used_bits,
                perplexity,
            ) = legacy.hide_extract.hide_bits_with_prompt_ids_by_egs(
                model=model,
                prompt_ids=prompt_ids,
                bits=source_bits,
                mode=config.egs_mode,
                threshold=config.threshold,
                temperature=config.temperature,
                temperature_alpha=config.temperature_alpha,
                max_bpw=config.max_bpw,
                max_new_tokens=config.max_new_tokens,
                complete_sent=False,
            )

        prompt_len = int(prompt_ids.shape[1])
        used_bit_count = int(used_bits)
        truncated = bool(truncated)

        prompt_token_ids = self._to_token_tuple(prompt_ids[0])
        output_token_ids = self._to_token_tuple(output_ids[0])

        # Do not remove the last token. In the legacy CLI, prompt_len:-1
        # could discard a token that still carries secret bits.
        generated_token_ids = tuple(
            output_token_ids[prompt_len:]
        )

        normalized_prompt = tokenizer.decode(
            list(prompt_token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Decode the complete token sequence before separating the
        # continuation. This preserves SentencePiece boundary spaces.
        normalized_full_text = tokenizer.decode(
            list(output_token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if not normalized_full_text.startswith(normalized_prompt):
            raise ZTSEncodingError(
                "The decoded output does not begin with the normalized "
                "prompt. The continuation boundary cannot be recovered.\n"
                f"normalized_prompt={normalized_prompt!r}\n"
                f"normalized_full_text={normalized_full_text!r}"
            )

        stegotext = normalized_full_text[len(normalized_prompt):]

        # Verify that serializing and tokenizing the text gives exactly
        # the token IDs selected by the embedding algorithm.
        retokenized_ids = tokenizer(
            normalized_prompt + stegotext,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids[0]

        retokenized_token_ids = self._to_token_tuple(
            retokenized_ids
        )

        retokenize_exact = (
            retokenized_token_ids == output_token_ids
        )

        result = ZTSEncodedText(
            payload=payload_bytes,
            payload_base64=payload_base64,
            prompt=prompt,
            normalized_prompt=normalized_prompt,
            stegotext=stegotext,
            full_text=normalized_full_text,
            source_bit_count=source_bit_count,
            used_bit_count=used_bit_count,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            output_token_ids=output_token_ids,
            truncated=truncated,
            retokenize_exact=retokenize_exact,
            perplexity=float(perplexity),
        )

        if strict:
            self._validate_strict_result(result)

        return result

    @staticmethod
    def _to_token_tuple(
        token_ids: "Sequence[int] | object",
    ) -> tuple[int, ...]:
        """
        Convert a Torch tensor or integer sequence into an immutable tuple.
        """

        detach = getattr(token_ids, "detach", None)

        if callable(detach):
            token_ids = detach().cpu().tolist()

        return tuple(int(token_id) for token_id in token_ids)

    @staticmethod
    def _validate_strict_result(
        result: ZTSEncodedText,
    ) -> None:
        errors: list[str] = []

        if result.truncated:
            errors.append(
                "generation reached max_new_tokens before completing "
                "the payload"
            )

        if result.used_bit_count != result.source_bit_count:
            errors.append(
                "not all payload bits were embedded: "
                f"used={result.used_bit_count}, "
                f"required={result.source_bit_count}"
            )

        if not result.generated_token_ids:
            errors.append("the encoder generated no continuation tokens")

        if not result.retokenize_exact:
            errors.append(
                "the serialized stegotext does not reproduce the exact "
                "generated token sequence"
            )

        if errors:
            raise ZTSEncodingError(
                "ZTS encoding validation failed:\n- "
                + "\n- ".join(errors)
            )