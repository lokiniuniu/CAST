"""Exact per-layer text K/V cache for Matrix cross-attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class _Entry:
    context: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class MatrixCrossAttentionKVCache:
    """Cache invariant text K/V projections without changing attention math."""

    def __init__(self) -> None:
        self._entries: dict[int, _Entry] = {}
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._entries.clear()

    def forward(
        self,
        layer_index: int,
        module: Any,
        attention: Callable[..., torch.Tensor],
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        fa_version: Any = None,
    ) -> torch.Tensor:
        if x.device.type != "cuda" or context.device != x.device:
            raise RuntimeError("Matrix cross-attention K/V cache requires one CUDA device")
        batch, heads, head_dim = x.shape[0], module.num_heads, module.head_dim
        q = module.norm_q(module.q(x.to(torch.bfloat16))).view(
            batch, -1, heads, head_dim
        )
        entry = self._entries.get(int(layer_index))
        # Keep the source tensor alive in the entry. Object identity therefore
        # cannot alias a later allocation, while repeated calls in one sample
        # hit without any content hash or synchronization.
        if entry is None or entry.context is not context:
            context_bf16 = context.to(torch.bfloat16)
            key = module.norm_k(module.k(context_bf16)).view(
                batch, -1, heads, head_dim
            )
            value = module.v(context_bf16).view(batch, -1, heads, head_dim)
            entry = _Entry(context=context, key=key, value=value)
            self._entries[int(layer_index)] = entry
            self.misses += 1
        else:
            self.hits += 1
        output = attention(
            q,
            entry.key,
            entry.value,
            k_lens=context_lens,
            version=fa_version,
        )
        return module.o(output.flatten(2))

