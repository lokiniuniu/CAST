"""Command-oriented full-ray/QK product attention for Matrix WorldMark."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .command_phase_ray_worldline_attention import (
    MatrixCommandPhaseRayWorldlineAttentionCompiler,
)
from .independent_ray_geodesic_attention import (
    MatrixIndependentRayGeodesicAttentionCompiler,
)
from .reciprocal_joint_attention import _tile_visual_tensor
from .reciprocal_ray_geodesic_attention import (
    MatrixReciprocalRayGeodesicAttentionCompiler,
)


@dataclass(frozen=True)
class CommandOrientedRayQKCompileReport:
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
    command_axis_source: str
    command_strength: float
    orientation_consistent_candidates_min: int
    selected_orientation_violations: int
    max_orientation_violations_per_row: int


@dataclass(frozen=True)
class CommandOrientedRayQKSelectionReport:
    batch_heads: int
    query_rows: int
    edges_per_row: int
    row_duplicate_violations: int
    selected_orientation_violations: int
    max_orientation_violations_per_row: int
    mean_selected_ray_cosine: float
    mean_selected_qk_cosine: float
    qk_read_for_selection: bool
    selector_count: int


class MatrixCommandOrientedRayQKProductAttentionCompiler:
    """Select once in a command-feasible full-ray/QK direct-sum geometry.

    The discrete Matrix command defines the sign-consistent action-time
    feasible set.  Inside that set, a single top-k ranks the canonical direct
    sum of the normalized six-dimensional camera ray and the live Q/K block
    feature.  There is no geometry preselection, Q/K fill, learned mixing
    weight, paired-row binding, or second selector.  The unchanged v65 Solver
    receives its original independent ray graph as a shadow metric.
    """

    name = "matrix_command_oriented_full_ray_qk_product_attention_compiler"
    selection_standard = "command_oriented_full_ray_qk_single_product_topk"
    cache_key_suffix = "command_oriented_full_ray_qk_product"

    def __init__(self) -> None:
        self._ray_features: torch.Tensor | None = None
        self._orientation_violations: torch.Tensor | None = None
        self._num_blocks = 0
        self._degree = 0
        self._block_shape = (0, 0, 0)

    def compile(
        self,
        layout: Any,
        *,
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, CommandOrientedRayQKCompileReport]:
        if bool(getattr(layout, "protected_current")):
            raise ValueError("command-oriented QK attention requires aggressive layout")
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
            raise RuntimeError("layout dimensions do not match command/ray/QK grid")
        counts = layout.counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise RuntimeError("command-oriented QK attention requires uniform degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree > num_blocks:
            raise RuntimeError("command-oriented QK attention received invalid degree")

        # Keep Module 3 byte-for-byte on the v65 independent ray graph.  The
        # Transformer ignores these indices because ``select`` replaces them
        # at the actual Q/K boundary with one joint constrained top-k.
        solver_layout, _ = MatrixIndependentRayGeodesicAttentionCompiler().compile(
            layout,
            active_atoms=active_atoms,
            current_c2ws=current_c2ws,
            token_h=int(token_h),
            token_w=int(token_w),
        )
        poses = MatrixReciprocalRayGeodesicAttentionCompiler._temporal_block_poses(
            active_atoms,
            current_c2ws,
            temporal_tile=tt,
        )
        if len(poses) != temporal_blocks:
            raise RuntimeError("temporal pose representatives do not match block grid")
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(poses)
        rays = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            poses,
            token_h=int(token_h),
            token_w=int(token_w),
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        ).float()
        if rays.shape != (num_blocks, 6):
            raise RuntimeError("camera-ray state count does not match attention blocks")
        axis, axis_source, command_strength = (
            MatrixCommandPhaseRayWorldlineAttentionCompiler._command_axis(
                current_action,
                current_c2ws,
            )
        )
        temporal_ids = torch.arange(num_blocks, device=rays.device) // spatial_blocks
        time_delta = temporal_ids[None, :] - temporal_ids[:, None]
        longitudinal = rays @ axis
        axis_delta = longitudinal[None, :] - longitudinal[:, None]
        violations = time_delta * axis_delta < 0.0
        consistent_counts = torch.count_nonzero(~violations, dim=-1)
        # A full-degree layout is the explicit dense ablation: every block is
        # selected, so command orientation can only order candidates and must
        # not make the complete graph infeasible.  Sparse layouts retain the
        # strict feasibility certificate used by the proposed selector.
        if degree < num_blocks and int(consistent_counts.min().item()) < degree:
            raise RuntimeError("command orientation leaves fewer candidates than degree")

        squared = torch.cdist(rays, rays).square()
        geometry_rows = []
        selected_distances = []
        selected_violation_counts = []
        for query in range(num_blocks):
            order = torch.argsort(squared[query], stable=True)
            order = order[
                torch.argsort(violations[query, order].to(torch.int8), stable=True)
            ]
            section = order[:degree]
            geometry_rows.append(section)
            selected_distances.append(
                torch.sqrt(squared[query, section].clamp_min(0.0))
            )
            selected_violation_counts.append(
                int(torch.count_nonzero(violations[query, section]).item())
            )
        geometry = torch.stack(geometry_rows)
        duplicate_violations = sum(
            int(len(torch.unique(row)) != degree) for row in geometry
        )
        if duplicate_violations:
            raise RuntimeError("command-oriented geometry emitted duplicate blocks")

        mate_list = MatrixReciprocalRayGeodesicAttentionCompiler._block_involution(
            num_blocks,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        mate = torch.tensor(mate_list, device=rays.device, dtype=torch.long)
        natural_violations = 0
        for query in range(num_blocks):
            expected = torch.sort(mate[geometry[query]]).values
            actual = torch.sort(geometry[mate_list[query]]).values
            natural_violations += int(torch.count_nonzero(expected != actual).item())

        self._ray_features = F.normalize(rays, dim=-1).detach()
        self._orientation_violations = violations.detach()
        self._num_blocks = num_blocks
        self._degree = degree
        self._block_shape = (tt, th, tw)
        distance_tensor = torch.cat(selected_distances)
        report = CommandOrientedRayQKCompileReport(
            query_rows=num_blocks,
            independent_query_rows=num_blocks,
            paired_query_orbits=0,
            edges_per_row=degree,
            trajectory_scale=float(scale.item()),
            mean_selected_geodesic=float(distance_tensor.mean().item()),
            max_selected_geodesic=float(distance_tensor.max().item()),
            paired_row_correspondence_enforced=False,
            natural_pair_correspondence_violations=natural_violations,
            row_duplicate_violations=duplicate_violations,
            activation_residual_fill=False,
            geometry_guard_ratio=1.0,
            qk_read_for_selection=True,
            selector_count=1,
            selection_standard=self.selection_standard,
            involution="none_independent_command_feasible_rows",
            command_axis_source=axis_source,
            command_strength=command_strength,
            orientation_consistent_candidates_min=int(
                consistent_counts.min().item()
            ),
            selected_orientation_violations=sum(selected_violation_counts),
            max_orientation_violations_per_row=max(selected_violation_counts),
        )
        return (
            replace(
                solver_layout,
                activation_residual_fill=False,
                geometry_guard_ratio=1.0,
                cache_key=tuple(layout.cache_key)
                + (self.cache_key_suffix,),
            ),
            report,
        )

    def _content_score(
        self,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        return torch.matmul(q_content, k_content.transpose(-2, -1))

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        geometry_indices: torch.Tensor,
        geometry_counts: torch.Tensor,
        latent_hw: tuple[int, int],
        block_shape: tuple[int, int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        CommandOrientedRayQKSelectionReport,
    ]:
        del geometry_indices
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("command-oriented selector requires equal B,H,L,D Q/K")
        if self._ray_features is None or self._orientation_violations is None:
            raise RuntimeError("command-oriented selector has no compiled geometry")
        if tuple(int(value) for value in block_shape) != self._block_shape:
            raise RuntimeError("runtime sparse block shape differs from compiled geometry")
        counts = geometry_counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != self._num_blocks or not torch.all(counts == self._degree):
            raise RuntimeError("runtime sparse degree differs from compiled geometry")
        height, width = (int(value) for value in latent_hw)
        tokens_per_frame = height * width
        if tokens_per_frame <= 0 or q.shape[2] % tokens_per_frame:
            raise ValueError("Q/K tokens do not define an integral temporal grid")
        temporal = q.shape[2] // tokens_per_frame
        q_blocks = _tile_visual_tensor(
            q,
            temporal=temporal,
            height=height,
            width=width,
            block_shape=self._block_shape,
        )
        k_blocks = _tile_visual_tensor(
            k,
            temporal=temporal,
            height=height,
            width=width,
            block_shape=self._block_shape,
        )
        if q_blocks.shape[2] != self._num_blocks:
            raise RuntimeError("tiled Q/K count differs from compiled command geometry")
        q_content = F.normalize(q_blocks.mean(dim=-2).float(), dim=-1)
        k_content = F.normalize(k_blocks.mean(dim=-2).float(), dim=-1)
        content_score = self._content_score(q_content, k_content)
        ray = self._ray_features.to(device=q.device)
        ray_score = torch.matmul(ray, ray.transpose(0, 1))[None, None, :, :]
        # This is one canonical direct-sum inner product.  The common factor
        # 1/2 does not affect top-k and therefore is omitted.
        score = ray_score + content_score
        violations = self._orientation_violations.to(device=q.device)
        score = score.masked_fill(violations[None, None, :, :], -torch.inf)
        selected = torch.argsort(score, dim=-1, descending=True, stable=True)[
            ..., : self._degree
        ]
        selected = torch.sort(selected, dim=-1).values
        duplicates = int(
            torch.count_nonzero(selected[..., 1:] == selected[..., :-1]).item()
        )
        if duplicates:
            raise RuntimeError("command-oriented product selector emitted duplicates")
        selected_violations = torch.gather(
            violations[None, None, :, :].expand(*selected.shape[:2], -1, -1),
            -1,
            selected,
        )
        per_row_violations = torch.count_nonzero(selected_violations, dim=-1)
        total_violations = int(per_row_violations.sum().item())
        max_violations = int(per_row_violations.max().item())
        if total_violations:
            raise RuntimeError("command-oriented product top-k violated feasibility")
        expanded_ray_score = ray_score.expand(*selected.shape[:2], -1, -1)
        selected_ray = torch.gather(expanded_ray_score, -1, selected)
        selected_content = torch.gather(content_score, -1, selected)
        batch, heads = q.shape[:2]
        selected_counts = torch.full(
            (batch, heads, self._num_blocks),
            self._degree,
            device=q.device,
            dtype=torch.int32,
        )
        report = CommandOrientedRayQKSelectionReport(
            batch_heads=batch * heads,
            query_rows=self._num_blocks,
            edges_per_row=self._degree,
            row_duplicate_violations=duplicates,
            selected_orientation_violations=total_violations,
            max_orientation_violations_per_row=max_violations,
            mean_selected_ray_cosine=float(selected_ray.mean().item()),
            mean_selected_qk_cosine=float(selected_content.mean().item()),
            qk_read_for_selection=True,
            selector_count=1,
        )
        return selected.to(torch.int32), selected_counts, report


__all__ = [
    "CommandOrientedRayQKCompileReport",
    "CommandOrientedRayQKSelectionReport",
    "MatrixCommandOrientedRayQKProductAttentionCompiler",
]
