"""One-carrier local integration runner (explicitly not a VoIP test)."""

from __future__ import annotations

from pathlib import Path

from ..report import RunDirectory, result_document
from ..transport import LocalFileTransport
from .receiver import ReceiverPipeline
from .sender import SenderPipeline


def run_local_fragment(
    *,
    frame: bytes,
    prompt: str,
    expected_shard_index: int,
    output_base: str | Path,
) -> Path:
    run = RunDirectory.create(output_base, "local_fragment_e2e")
    with SenderPipeline() as sender:
        carrier = sender.prepare_carrier(frame, prompt, run.sender_private / "carrier")
    transfer = LocalFileTransport().transfer(
        carrier.source_wav, run.receiver / "received_carrier.wav"
    )
    recovery = ReceiverPipeline().recover_fragment(
        transfer.received,
        prompt=prompt,
        expected_shard_index=expected_shard_index,
        output_dir=run.receiver / "decode",
    )
    output = run.write_public(
        "final_result.json",
        result_document(
            status="ZSC_TSD_LOCAL_FRAGMENT_PASS" if recovery.passed else "ZSC_TSD_LOCAL_FRAGMENT_FAIL",
            experiment="local_fragment_e2e",
            checks={
                "local_transfer_byte_exact": transfer.byte_exact,
                "receiver_fragment_pass": recovery.passed,
            },
            measurements={
                "candidate_v7_count": recovery.candidate_v7_count,
                "candidate_v6_count": recovery.candidate_v6_count,
            },
            evidence={
                "transport_scope": "local-file-copy-not-voip",
                "receiver_report": str(recovery.report_path),
            },
        ),
    )
    return output
