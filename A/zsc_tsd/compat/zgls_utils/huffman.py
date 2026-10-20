from __future__ import annotations

import heapq
from typing import Dict, TypeVar, Union

T = TypeVar("T")
Frequency = Union[int, float]


def from_frequencies(
    freqs: Dict[T, Frequency],
    larger_as_zero: bool = True,
) -> Dict[T, str]:
    """
    Windows-compatible fallback for zero-shot-GLS zgls_utils.

    It follows the original C++ implementation:
    - repeatedly combine the two lowest-frequency nodes;
    - the second-lowest node becomes the left child;
    - when larger_as_zero=True, the left branch receives 0.
    """
    if not freqs:
        return {}

    keys = list(freqs.keys())

    # Each node is represented by:
    # (key_index_or_none, left_node_index, right_node_index)
    nodes: list[tuple[int | None, int, int]] = []
    heap: list[tuple[float, int, int]] = []

    for order, key in enumerate(keys):
        node_index = len(nodes)
        nodes.append((order, -1, -1))
        heapq.heappush(
            heap,
            (float(freqs[key]), order, node_index),
        )

    # Match the original C++ behavior for a one-symbol alphabet:
    # the only symbol receives code "0".
    if len(heap) == 1:
        return {keys[0]: "0"}

    next_order = len(heap)

    while len(heap) > 1:
        # Original C++ names the smallest node "right" and the
        # second-smallest node "left".
        right_frequency, _, right_index = heapq.heappop(heap)
        left_frequency, _, left_index = heapq.heappop(heap)

        parent_index = len(nodes)
        nodes.append((None, left_index, right_index))

        heapq.heappush(
            heap,
            (
                left_frequency + right_frequency,
                next_order,
                parent_index,
            ),
        )
        next_order += 1

    root_index = heap[0][2]
    index_to_code: dict[int, str] = {}

    def visit(node_index: int, prefix: str) -> None:
        key_index, left_index, right_index = nodes[node_index]

        if left_index < 0 and right_index < 0:
            if key_index is None:
                raise RuntimeError("Invalid Huffman leaf node.")
            index_to_code[key_index] = prefix
            return

        left_bit = "0" if larger_as_zero else "1"
        right_bit = "1" if larger_as_zero else "0"

        if left_index >= 0:
            visit(left_index, prefix + left_bit)

        if right_index >= 0:
            visit(right_index, prefix + right_bit)

    visit(root_index, "")

    return {
        keys[key_index]: code
        for key_index, code in index_to_code.items()
    }