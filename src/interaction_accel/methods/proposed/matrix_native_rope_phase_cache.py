"""Exact bounded phase-layout cache for Matrix native indexed RoPE."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


class MatrixNativeRoPEPhaseCache:
    """Reproduce ``rope_apply_with_indices`` while caching two phase layouts."""

    def __init__(self, native: Any, maximum_entries: int = 2) -> None:
        if maximum_entries != 2:
            raise ValueError("Matrix native RoPE cache is fixed to Current+Memory")
        self.native = native
        self.maximum_entries = maximum_entries
        self._cache: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._cache.clear()

    @staticmethod
    def _indices(t_indices: Any, frames: int) -> tuple[int, ...] | None:
        if t_indices is None:
            return tuple(range(frames))
        if torch.is_tensor(t_indices):
            # The production Matrix path supplies Python lists. Avoid adding a
            # device synchronization for unsupported callers.
            if t_indices.device.type != "cpu":
                return None
            values = t_indices.reshape(-1).tolist()
        else:
            values = list(t_indices)
        return tuple(int(value) for value in values)

    def __call__(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        t_indices: Any = None,
    ) -> torch.Tensor:
        if x.device.type != "cuda" or x.shape[0] != 1:
            return self.native(x, grid_sizes, freqs, t_indices)
        grids = tuple(tuple(int(value) for value in row) for row in grid_sizes.tolist())
        if len(grids) != 1:
            return self.native(x, grid_sizes, freqs, t_indices)
        frames, height, width = grids[0]
        temporal_indices = self._indices(t_indices, frames)
        if temporal_indices is None or len(temporal_indices) < frames:
            return self.native(x, grid_sizes, freqs, t_indices)

        heads, half = x.size(2), x.size(3) // 2
        split = [half - 2 * (half // 3), half // 3, half // 3]
        key = (
            int(freqs.data_ptr()),
            tuple(freqs.shape),
            frames,
            height,
            width,
            temporal_indices[:frames],
            heads,
            half,
            str(x.device),
        )
        phase = self._cache.get(key)
        if phase is None:
            axes = freqs.split(split, dim=2 if freqs.dim() == 3 else 1)
            t_idx = torch.tensor(
                temporal_indices[:frames], device=freqs.device, dtype=torch.long
            )
            if freqs.dim() == 3:
                t_freqs = axes[0][:, t_idx, :]
                h_freqs = axes[1][:, :height, :]
                w_freqs = axes[2][:, :width, :]
                phase = torch.cat(
                    [
                        t_freqs.permute(1, 0, 2)
                        .view(frames, 1, 1, heads, -1)
                        .expand(frames, height, width, heads, -1),
                        h_freqs.permute(1, 0, 2)
                        .view(1, height, 1, heads, -1)
                        .expand(frames, height, width, heads, -1),
                        w_freqs.permute(1, 0, 2)
                        .view(1, 1, width, heads, -1)
                        .expand(frames, height, width, heads, -1),
                    ],
                    dim=-1,
                ).reshape(frames * height * width, heads, -1)
            else:
                phase = torch.cat(
                    [
                        axes[0][t_idx]
                        .view(frames, 1, 1, -1)
                        .expand(frames, height, width, -1),
                        axes[1][:height]
                        .view(1, height, 1, -1)
                        .expand(frames, height, width, -1),
                        axes[2][:width]
                        .view(1, 1, width, -1)
                        .expand(frames, height, width, -1),
                    ],
                    dim=-1,
                ).reshape(frames * height * width, 1, -1)
            phase = phase.to(torch.complex64)
            self._cache[key] = phase
            self._cache.move_to_end(key)
            while len(self._cache) > self.maximum_entries:
                self._cache.popitem(last=False)
            self.misses += 1
        else:
            self._cache.move_to_end(key)
            self.hits += 1

        sequence = frames * height * width
        x_i = torch.view_as_complex(
            x[0, :sequence].to(torch.float32).reshape(sequence, heads, -1, 2)
        )
        output = torch.view_as_real(x_i * phase).flatten(2)
        output = torch.cat([output, x[0, sequence:]])
        return output.unsqueeze(0).float()

