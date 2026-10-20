from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


class ExactInputLogitsMemo:
    """Memoize raw last-position logits for identical full input token contexts."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._cache: dict[tuple[int, ...], torch.Tensor] = {}
        self.forward_requests = 0
        self.real_model_forwards = 0
        self.cache_hits = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # The frozen ZTS search path calls exactly model(input_ids=input_ids).
        # Delegate every other model call unchanged and do not cache it.
        if args or set(kwargs) != {"input_ids"}:
            return self._model(*args, **kwargs)

        input_ids = kwargs["input_ids"]
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dim() != 2
            or input_ids.size(0) != 1
        ):
            return self._model(*args, **kwargs)

        self.forward_requests += 1
        key = tuple(int(x) for x in input_ids[0].detach().cpu().tolist())
        cached = self._cache.get(key)

        if cached is None:
            output = self._model(input_ids=input_ids)
            last_logits = output.logits[0, -1].detach().clone()
            self._cache[key] = last_logits
            self.real_model_forwards += 1
        else:
            last_logits = cached
            self.cache_hits += 1

        # Frozen enhanced_greedy_search reads only .logits[0, -1].
        return SimpleNamespace(logits=last_logits.unsqueeze(0).unsqueeze(0))

    @property
    def unique_contexts(self) -> int:
        return len(self._cache)

    @property
    def cache_hit_rate(self) -> float:
        if self.forward_requests == 0:
            return 0.0
        return self.cache_hits / self.forward_requests

    def clear_cache(self) -> None:
        self._cache.clear()
        self.forward_requests = 0
        self.real_model_forwards = 0
        self.cache_hits = 0
