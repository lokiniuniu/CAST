"""Fixed-degree sparse-attention closure under a C2 return involution."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch


@dataclass(frozen=True)
class ReciprocalDegreeAttentionReport:
    query_rows: int
    edges_per_row: int
    edge_orbits_per_row: int
    replaced_edges: int
    closure_violations: int
    involution: str


class MatrixReciprocalDegreeAttentionCompiler:
    """Close each fixed-budget KV row under deterministic return reversal.

    C3's native geometry order supplies the priority.  A target block and its
    same-spatial-coordinate time-reversed target form a two-element C2 orbit.
    On the unique middle temporal slab, adjacent spatial cells form the orbit,
    removing fixed points.  The compiler chooses exactly half a row's worth of
    highest-priority orbits and emits both endpoints.  Hence every row remains
    at the native 20% degree while outward and return support cannot be pruned
    independently.  No Q/K activation, threshold, score blend, or sample
    metric enters the construction.
    """

    name = "matrix_reciprocal_degree_attention_compiler"

    @staticmethod
    def _involution(
        block: int, *, temporal_blocks: int, spatial_blocks: int
    ) -> int:
        temporal = block // spatial_blocks
        spatial = block % spatial_blocks
        reverse_temporal = temporal_blocks - 1 - temporal
        if reverse_temporal == temporal:
            if spatial_blocks % 2:
                raise RuntimeError(
                    "middle-slab reciprocal closure requires even spatial blocks"
                )
            spatial = spatial + 1 if spatial % 2 == 0 else spatial - 1
        return reverse_temporal * spatial_blocks + spatial

    def compile(
        self,
        layout: Any,
        *,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, ReciprocalDegreeAttentionReport]:
        if bool(getattr(layout, "protected_current")):
            raise ValueError("reciprocal-degree closure requires aggressive C3 layout")
        tt, th, tw = tuple(int(value) for value in layout.block_shape)
        temporal = int(layout.memory_length) + int(layout.current_length)
        temporal_blocks = math.ceil(temporal / tt)
        height_blocks = math.ceil(int(token_h) / th)
        width_blocks = math.ceil(int(token_w) / tw)
        spatial_blocks = height_blocks * width_blocks
        num_blocks = temporal_blocks * spatial_blocks
        if tuple(layout.indices.shape[:3]) != (1, 1, num_blocks):
            raise RuntimeError("layout dimensions do not match its block geometry")
        counts = layout.counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise RuntimeError("reciprocal-degree closure requires uniform row degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree % 2:
            raise RuntimeError("reciprocal-degree closure requires positive even degree")

        mate = [
            self._involution(
                block,
                temporal_blocks=temporal_blocks,
                spatial_blocks=spatial_blocks,
            )
            for block in range(num_blocks)
        ]
        if any(mate[mate[block]] != block or mate[block] == block for block in range(num_blocks)):
            raise RuntimeError("target mapping is not a fixed-point-free C2 involution")
        all_orbits = [(block, mate[block]) for block in range(num_blocks) if block < mate[block]]

        output = torch.empty_like(layout.indices)
        replaced_edges = 0
        closure_violations = 0
        for query in range(num_blocks):
            native = [
                int(value)
                for value in layout.indices[0, 0, query, :degree]
                .detach()
                .to(device="cpu", dtype=torch.int64)
                .tolist()
            ]
            native_rank = {block: rank for rank, block in enumerate(native)}

            def orbit_priority(orbit: tuple[int, int]) -> tuple[int, int, int, int, int]:
                left, right = orbit
                ranks = [native_rank[value] for value in orbit if value in native_rank]
                best_rank = min(ranks, default=degree + num_blocks)
                native_hits = len(ranks)
                q_temporal, q_spatial = divmod(query, spatial_blocks)
                distances = []
                for target in orbit:
                    t_temporal, t_spatial = divmod(target, spatial_blocks)
                    t_y, t_x = divmod(t_spatial, width_blocks)
                    q_y, q_x = divmod(q_spatial, width_blocks)
                    distances.append(
                        abs(t_temporal - q_temporal)
                        + abs(t_y - q_y)
                        + abs(t_x - q_x)
                    )
                return (
                    best_rank,
                    -native_hits,
                    min(distances),
                    left,
                    right,
                )

            selected_orbits = sorted(all_orbits, key=orbit_priority)[: degree // 2]
            row = []
            for left, right in selected_orbits:
                endpoints = sorted(
                    (left, right),
                    key=lambda value: (native_rank.get(value, degree + num_blocks), value),
                )
                row.extend(endpoints)
            if len(row) != degree or len(set(row)) != degree:
                raise RuntimeError("reciprocal-degree row violated fixed degree")
            row_set = set(row)
            closure_violations += sum(mate[value] not in row_set for value in row)
            replaced_edges += len(row_set.difference(native_rank))
            output[0, 0, query, :degree] = torch.tensor(
                row, dtype=layout.indices.dtype, device=layout.indices.device
            )
            if output.shape[-1] > degree:
                output[0, 0, query, degree:] = 0

        report = ReciprocalDegreeAttentionReport(
            query_rows=num_blocks,
            edges_per_row=degree,
            edge_orbits_per_row=degree // 2,
            replaced_edges=replaced_edges,
            closure_violations=closure_violations,
            involution="same_spatial_temporal_reversal_with_mid_slab_adjacent_pairing",
        )
        if report.closure_violations:
            raise RuntimeError("reciprocal-degree attention closure is incomplete")
        return (
            replace(
                layout,
                indices=output,
                cache_key=tuple(layout.cache_key) + ("reciprocal_degree_c2",),
            ),
            report,
        )


__all__ = [
    "MatrixReciprocalDegreeAttentionCompiler",
    "ReciprocalDegreeAttentionReport",
]
