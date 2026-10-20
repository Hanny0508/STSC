from __future__ import annotations

from zsc_tsd.report import result_document


def test_result_document_has_stable_public_schema() -> None:
    result = result_document(
        status="PASS",
        experiment="unit-test",
        checks={"exact": True},
        evidence={"secret_in_public_report": False},
    )
    assert result["schema"] == "zsc-tsd.experiment-result.v1"
    assert result["checks"]["exact"] is True
