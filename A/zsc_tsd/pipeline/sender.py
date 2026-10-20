"""Stable sender facade over ZTS generation and TransformerTTS."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..manifest import sha256_file, write_json_new
from ..speech import TransformerTTSClient
from ..zts import ZTSDecoder, ZTSEncoder, ZTSModelAdapter


@dataclass(frozen=True, slots=True)
class PreparedCarrier:
    prompt: str
    stegotext: str
    payload_bytes: int
    payload_sha256: str
    source_wav: Path
    source_wav_sha256: str
    sender_report: Path
    tts_report: Path
    perplexity: float
    generated_token_count: int

    def public_metadata(self) -> dict[str, Any]:
        """Metadata safe for synchronization; excludes payload and stegotext."""

        return {
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "payload_bytes": self.payload_bytes,
            "source_wav_sha256": self.source_wav_sha256,
            "perplexity_milli": round(self.perplexity * 1000),
            "generated_token_count": self.generated_token_count,
        }


class SenderPipeline:
    def __init__(
        self,
        *,
        adapter: ZTSModelAdapter | None = None,
        tts: TransformerTTSClient | None = None,
    ) -> None:
        self.adapter = adapter or ZTSModelAdapter()
        self.encoder = ZTSEncoder(self.adapter)
        self.decoder = ZTSDecoder(self.adapter)
        self.tts = tts or TransformerTTSClient()

    def prepare_carrier(
        self,
        payload: bytes,
        prompt: str,
        output_dir: str | Path,
    ) -> PreparedCarrier:
        target = Path(output_dir).resolve()
        target.mkdir(parents=True, exist_ok=False)
        encoded = self.encoder.encode(payload, prompt, strict=True)
        control_decode = self.decoder.decode_encoded(encoded, strict=True)
        if control_decode.payload != payload:
            raise RuntimeError("sender-side exact ZTS roundtrip failed")

        sender_report = target / "sender.json"
        write_json_new(
            sender_report,
            {
                "status": "ZSC_TSD_SENDER_CARRIER_PREPARED",
                "prompt": prompt,
                "stegotext": encoded.stegotext,
                "payload_hex": payload.hex(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "source_bit_count": encoded.source_bit_count,
                "used_bit_count": encoded.used_bit_count,
                "retokenize_exact": encoded.retokenize_exact,
                "sender_exact_roundtrip": True,
                "perplexity": encoded.perplexity,
                "generated_token_count": encoded.generated_token_count,
            },
        )

        source_wav = target / "carrier.wav"
        tts_report = target / "tts.json"
        self.tts.synthesize(encoded.stegotext, source_wav, report_path=tts_report)
        return PreparedCarrier(
            prompt=prompt,
            stegotext=encoded.stegotext,
            payload_bytes=len(payload),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            source_wav=source_wav,
            source_wav_sha256=sha256_file(source_wav),
            sender_report=sender_report,
            tts_report=tts_report,
            perplexity=encoded.perplexity,
            generated_token_count=encoded.generated_token_count,
        )

    def close(self) -> None:
        self.adapter.release()

    def __enter__(self) -> "SenderPipeline":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
