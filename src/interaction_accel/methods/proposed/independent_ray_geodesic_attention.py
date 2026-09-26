"""Independent camera-ray geodesics for Matrix sparse attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import torch

from .reciprocal_ray_geodesic_attention import (
    MatrixReciprocalRayGeodesicAttentionCompiler,
)


@dataclass(frozen=True)
class IndependentRayGeodesicAttentionReport:
    query_rows: int
    independent_query_rows: int
    paired_query_orbits: int
    edges_per_row: int
    trajectory_scale: float
    mean_selected_geodesic: float
    max_selected_geodesic: float
    paired_row_correspondence_enforced: bool
    natural_pair_correspondence_violations: int
    row_duplicate_violations: int
    activation_residual_fill: bool
    geometry_guard_ratio: float
    qk_read_for_selection: bool
    selector_count: int
    selection_standard: str
    involution: str


class MatrixIndependentRayGeodesicAttentionCompiler:
    """Select every sparse row independently on the v42 ray manifold.

    This is the strict paired-row ablation of v42.  It reuses exactly the same
    block centers, temporal pose representatives, trajectory normalization,
    six-dimensional camera-ray states, fixed 20% budget, and deterministic
    block-id tie break.  The only removed operation is the C2 quotient mean
    and its forced return lift ``Row(gq) = g Row(q)``.  Each query instead
    selects the 30 nearest key rays by

        d(q, k)^2 = ||r_q - r_k||^2.

    No Q/K activation, prefix, fill, learned weight, or second objective is
    introduced.
    """

    name = "matrix_independent_ray_geodesic_attention_compiler"

    def compile(
        self,
        layout: Any,
        *,
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, IndependentRayGeodesicAttentionReport]:
        if bool(getattr(layout, "protected_current")):
            raise ValueError("independent camera-ray attention requires aggressive C3 layout")
        if current_c2ws.ndim != 3 or tuple(current_c2ws.shape[-2:]) != (4, 4):
            raise ValueError("current_c2ws must have shape [T,4,4]")
        tt, th, tw = tuple(int(value) for value in layout.block_shape)
        temporal = int(layout.memory_length) + int(layout.current_length)
        temporal_blocks = math.ceil(temporal / tt)
        height_blocks = math.ceil(int(token_h) / th)
        width_blocks = math.ceil(int(token_w) / tw)
        spatial_blocks = height_blocks * width_blocks
        num_blocks = temporal_blocks * spatial_blocks
        if tuple(layout.indices.shape[:3]) != (1, 1, num_blocks):
            raise RuntimeError("layout dimensions do not match camera-ray geometry")
        counts = layout.counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise RuntimeError("independent camera-ray attention requires uniform degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree > num_blocks:
            raise RuntimeError("independent camera-ray attention received invalid degree")

        block_poses = MatrixReciprocalRayGeodesicAttentionCompiler._temporal_block_poses(
            active_atoms,
            current_c2ws,
            temporal_tile=tt,
        )
        if len(block_poses) != temporal_blocks:
            raise RuntimeError("temporal pose representatives do not match block grid")
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(
            block_poses
        )
        rays = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            block_poses,
            token_h=int(token_h),
            token_w=int(token_w),
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        )
        if rays.shape != (num_blocks, 6):
            raise RuntimeError("camera-ray state count does not match attention blocks")

        squared = torch.cdist(rays, rays).square()
        # One stable top-k per query.  There is deliberately no reciprocal
        # orbit loop and no assignment of a return row from its outbound mate.
        section = torch.argsort(squared, dim=-1, stable=True)[:, :degree]
        output = torch.zeros_like(layout.indices)
        output[0, 0, :, :degree] = section.to(output.dtype)

        duplicate_violations = int(
            torch.count_nonzero(section[:, 1:] == section[:, :-1]).item()
        )
        if duplicate_violations:
            raise RuntimeError("independent camera-ray attention selected duplicate blocks")

        # Measure, but never enforce, the correspondence removed from v42.
        mate_list = MatrixReciprocalRayGeodesicAttentionCompiler._block_involution(
            num_blocks,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        mate = torch.tensor(mate_list, device=rays.device, dtype=torch.long)
        natural_violations = 0
        for query in range(num_blocks):
            expected = torch.sort(mate[section[query]]).values
            actual = torch.sort(section[mate_list[query]]).values
            natural_violations += int(torch.count_nonzero(expected != actual).item())

        selected_distances = torch.sqrt(
            squared.gather(1, section).clamp_min(0.0)
        )
        report = IndependentRayGeodesicAttentionReport(
            query_rows=num_blocks,
            independent_query_rows=num_blocks,
            paired_query_orbits=0,
            edges_per_row=degree,
            trajectory_scale=float(scale.item()),
            mean_selected_geodesic=float(selected_distances.mean().item()),
            max_selected_geodesic=float(selected_distances.max().item()),
            paired_row_correspondence_enforced=False,
            natural_pair_correspondence_violations=natural_violations,
            row_duplicate_violations=duplicate_violations,
            activation_residual_fill=False,
            geometry_guard_ratio=1.0,
            qk_read_for_selection=False,
            selector_count=1,
            selection_standard=(
                "independent_trajectory_normalized_camera_ray_geodesic"
            ),
            involution="none_independent_query_rows",
        )
        return (
            replace(
                layout,
                indices=output,
                activation_residual_fill=False,
                geometry_guard_ratio=1.0,
                cache_key=tuple(layout.cache_key)
                + ("independent_camera_ray_geodesic",),
            ),
            report,
        )


__all__ = [
    "IndependentRayGeodesicAttentionReport",
    "MatrixIndependentRayGeodesicAttentionCompiler",
]
