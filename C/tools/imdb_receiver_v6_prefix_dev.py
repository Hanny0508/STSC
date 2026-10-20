#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from zsc_tsd.crypto.framing import decode_crc32_frame
from zsc_tsd.zts.adapter import ZTSModelAdapter


ROOT = Path(__file__).resolve().parents[1]
MAX_FORCED_TOKENS_PER_NODE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development Receiver v6: exact zero-shot-GLS decision-prefix "
            "cache over a frozen, payload-independent candidate list."
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expected-frame-bytes", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--continue-after-accept", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_repeat_state(
    state: dict[int, float], token_id: int
) -> dict[int, float]:
    # Exact sparse equivalent of LogitsRepeatPenaltyStrategy.update():
    # offset += 0.5; clamp_max_(0); offset[token_id] -= 4.0
    updated = {
        key: value + 0.5
        for key, value in state.items()
        if value + 0.5 < 0.0
    }
    updated[token_id] = updated.get(token_id, 0.0) - 4.0
    return updated


@dataclass
class Stats:
    model_calls: int = 0
    expanded_decision_nodes: int = 0
    forced_token_steps: int = 0
    decision_edges_followed: int = 0
    decision_node_reuses: int = 0


@dataclass
class DecisionNode:
    cur_ids: torch.Tensor
    temperature: float
    repeat_state: dict[int, float]
    bits: str
    expanded: bool = False
    decision_length: int | None = None
    choice_rows: dict[int, tuple[torch.Tensor, str]] = field(
        default_factory=dict
    )
    post_forced_temperature: float | None = None
    post_forced_repeat_state: dict[int, float] = field(default_factory=dict)
    children: dict[int, "DecisionNode"] = field(default_factory=dict)

    def expand(self, context: "Context") -> None:
        if self.expanded:
            context.stats.decision_node_reuses += 1
            return

        cur_ids = self.cur_ids
        temperature = self.temperature
        repeat_state = dict(self.repeat_state)
        forced_steps = 0

        while True:
            logits_offset = torch.zeros(
                context.model.vocab_size,
                dtype=torch.double,
                device=context.model.device,
            )
            if repeat_state:
                indices = list(repeat_state)
                values = [repeat_state[index] for index in indices]
                logits_offset[indices] = torch.tensor(
                    values,
                    dtype=torch.double,
                    device=context.model.device,
                )

            result = context.search.enhanced_greedy_search(
                context.model,
                cur_ids,
                threshold=context.config.threshold,
                temperature=temperature,
                max_bits_len=context.config.max_bpw,
                logits_offset=logits_offset,
            )
            context.stats.model_calls += 1

            new_ids = result["comb_ids"]
            sorted_probs = result["sorted_probs"]
            choice_count = int(new_ids.size(0))

            if choice_count == 1:
                token_id = int(new_ids[0, -1].item())
                cur_ids = new_ids
                temperature = context.config.temperature
                repeat_state = update_repeat_state(repeat_state, token_id)
                forced_steps += 1
                context.stats.forced_token_steps += 1
                if forced_steps > MAX_FORCED_TOKENS_PER_NODE:
                    raise RuntimeError(
                        "Exceeded forced-token safety bound before the next "
                        "multi-choice EGS decision."
                    )
                continue

            probabilities = sorted_probs.tolist()
            codes = context.huffman.from_frequencies(
                {index: value for index, value in enumerate(probabilities)}
            )

            self.decision_length = int(new_ids.size(1))
            self.post_forced_temperature = temperature
            self.post_forced_repeat_state = dict(repeat_state)
            self.choice_rows = {
                int(new_ids[index, -1].item()): (
                    new_ids[index].unsqueeze(0),
                    str(codes[index]),
                )
                for index in range(choice_count)
            }
            self.expanded = True
            context.stats.expanded_decision_nodes += 1
            return

    def child(self, token_id: int, context: "Context") -> "DecisionNode" | None:
        row = self.choice_rows.get(token_id)
        if row is None:
            return None
        if token_id in self.children:
            return self.children[token_id]

        assert self.post_forced_temperature is not None
        child_ids, code = row
        child = DecisionNode(
            cur_ids=child_ids,
            temperature=(
                self.post_forced_temperature
                * context.config.temperature_alpha
            ),
            repeat_state=update_repeat_state(
                self.post_forced_repeat_state, token_id
            ),
            bits=self.bits + code,
        )
        self.children[token_id] = child
        return child


@dataclass
class Context:
    model: Any
    config: Any
    search: Any
    huffman: Any
    stats: Stats


def main() -> None:
    args = parse_args()
    if args.expected_frame_bytes < 5:
        raise SystemExit("--expected-frame-bytes must be at least 5")

    candidate_path = resolve_path(args.candidates)
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"STOP: refusing to overwrite: {output_path}")

    report = json.loads(candidate_path.read_text(encoding="utf-8"))
    rows = report.get("candidates")
    if not isinstance(rows, list):
        raise SystemExit("Candidate JSON has no candidates list")
    if report.get("candidate_count") != len(rows):
        raise SystemExit("Candidate count does not match candidates list")
    if bool(report.get("truncated")):
        raise SystemExit("Candidate space is truncated; refusing formal replay")

    adapter = ZTSModelAdapter()
    started = time.perf_counter()
    trials: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    errors: dict[str, int] = {}
    maximum_recovered_bits = 0

    try:
        adapter.load()
        adapter.configure_determinism()
        tokenizer = adapter.tokenizer
        model = adapter.model
        config = adapter.config
        legacy = adapter.legacy

        original_prompt_ids = tokenizer(
            args.prompt, return_tensors="pt"
        ).input_ids
        original_prompt_tuple = tuple(
            int(value) for value in original_prompt_ids[0].tolist()
        )
        normalized_prompt = tokenizer.decode(
            list(original_prompt_tuple),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prompt_ids = tokenizer(
            normalized_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(adapter.device)
        prompt_tuple = tuple(int(value) for value in prompt_ids[0].tolist())
        if prompt_tuple != original_prompt_tuple:
            raise RuntimeError("Prompt retokenization is not exact")

        tokenized: list[tuple[dict[str, Any], tuple[int, ...]] | None] = []
        for row in rows:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or text == "":
                tokenized.append(None)
                continue
            hide_ids = tokenizer(
                normalized_prompt + text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids[0]
            hide_tuple = tuple(int(value) for value in hide_ids.tolist())
            if (
                len(hide_tuple) <= len(prompt_tuple)
                or hide_tuple[: len(prompt_tuple)] != prompt_tuple
            ):
                tokenized.append(None)
                continue
            tokenized.append((row, hide_tuple))

        stats = Stats()
        context = Context(
            model=model,
            config=config,
            search=legacy.hide_extract.search,
            huffman=legacy.codec.huffman,
            stats=stats,
        )
        root = DecisionNode(
            cur_ids=prompt_ids,
            temperature=config.temperature,
            repeat_state={},
            bits="",
        )
        frame_bits = args.expected_frame_bytes * 8
        payload_bytes = args.expected_frame_bytes - 4

        with (
            torch.inference_mode(),
            legacy.prompt_gen.random_state(config.prompt_seed),
        ):
            for position, item in enumerate(tokenized):
                if position == 0 or (position + 1) % 100 == 0:
                    print(
                        f"Candidate {position + 1}/{len(tokenized)}, "
                        f"model calls={stats.model_calls}",
                        flush=True,
                    )

                if item is None:
                    errors["InvalidCandidateOrPromptPrefix"] = (
                        errors.get("InvalidCandidateOrPromptPrefix", 0) + 1
                    )
                    continue

                row, hide_tuple = item
                node = root
                failed = False
                while True:
                    node.expand(context)
                    assert node.decision_length is not None
                    if node.decision_length > len(hide_tuple):
                        break
                    token_id = hide_tuple[node.decision_length - 1]
                    child = node.child(token_id, context)
                    if child is None:
                        maximum_recovered_bits = max(
                            maximum_recovered_bits, len(node.bits)
                        )
                        errors["ZTSDecodingError"] = (
                            errors.get("ZTSDecodingError", 0) + 1
                        )
                        failed = True
                        break
                    stats.decision_edges_followed += 1
                    node = child

                if failed:
                    continue

                recovered_count = len(node.bits)
                maximum_recovered_bits = max(
                    maximum_recovered_bits, recovered_count
                )
                trial = {
                    "candidate_list_index": position,
                    "candidate_rank": row.get("rank", position),
                    "candidate_id": row.get("candidate_id"),
                    "generation_stage": row.get("generation_stage"),
                    "repair_count": row.get("repair_count"),
                    "transforms": row.get("transforms"),
                    "text": row.get("text"),
                    "recovered_bits": recovered_count,
                }
                trials.append(trial)

                if recovered_count < frame_bits:
                    errors["InsufficientRecoveredBits"] = (
                        errors.get("InsufficientRecoveredBits", 0) + 1
                    )
                    continue

                bit_prefix = node.bits[:frame_bits]
                frame = int(bit_prefix, 2).to_bytes(
                    args.expected_frame_bytes, byteorder="big"
                )
                try:
                    payload = decode_crc32_frame(
                        frame, expected_payload_size=payload_bytes
                    )
                except Exception as exc:
                    name = type(exc).__name__
                    errors[name] = errors.get(name, 0) + 1
                    continue

                accepted_row = {
                    **trial,
                    "decoded_frame_hex": frame.hex(),
                    "recovered_payload_hex": payload.hex(),
                    "crc32_pass": True,
                    "expected_bits": frame_bits,
                    "extra_bits": recovered_count - frame_bits,
                    "prompt_prefix_exact": True,
                }
                accepted.append(accepted_row)
                print(f"CRC32-valid candidate accepted: {position}")
                if not args.continue_after_accept:
                    break

        distinct_payloads = sorted(
            {row["recovered_payload_hex"] for row in accepted}
        )
        elapsed = time.perf_counter() - started
        result = {
            "status": (
                "IMDB_RECEIVER_V6_PREFIX_DEV_PASS"
                if accepted
                else "IMDB_RECEIVER_V6_PREFIX_DEV_FAIL"
            ),
            "development_replay": True,
            "official_blind_result": False,
            "introduced_after_imdb_row_1_reveal": True,
            "candidate_source": str(candidate_path),
            "candidate_source_sha256": sha256_file(candidate_path),
            "candidate_count": len(rows),
            "candidate_space_truncated": False,
            "decode_attempts": (
                accepted[0]["candidate_list_index"] + 1
                if accepted and not args.continue_after_accept
                else len(rows)
            ),
            "decode_error_counts": errors,
            "accepted": accepted,
            "distinct_crc32_valid_payloads": distinct_payloads,
            "unambiguous_payload": len(distinct_payloads) == 1,
            "maximum_recovered_bits": maximum_recovered_bits,
            "elapsed_seconds": elapsed,
            "shared_state_cache": {
                "model_calls": stats.model_calls,
                "expanded_decision_nodes": stats.expanded_decision_nodes,
                "forced_token_steps": stats.forced_token_steps,
                "decision_edges_followed": stats.decision_edges_followed,
                "decision_node_reuses": stats.decision_node_reuses,
            },
            "receiver_control_metadata": {
                "prompt": args.prompt,
                "expected_frame_bytes": args.expected_frame_bytes,
                "expected_payload_bytes": payload_bytes,
            },
            "receiver_isolation": {
                "sender_stegotext_used": False,
                "sender_token_ids_used": False,
                "sender_payload_used": False,
                "reference_transcription_used": False,
                "crc32_used_only_after_zts_decode": True,
                "candidate_order_preserved": True,
            },
        }
        write_json(output_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        adapter.release()


if __name__ == "__main__":
    main()