"""Plucker-worldline sparse topology for Matrix attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .reciprocal_ray_geodesic_attention import (
    MatrixReciprocalRayGeodesicAttentionCompiler,
)


@dataclass(frozen=True)
class PluckerWorldlineCompileReport:
    query_rows: int
    edges_per_row: int
    trajectory_scale: float
    mean_selected_plucker_cosine: float
    row_duplicate_violations: int
    activation_residual_fill: bool
    geometry_guard_ratio: float
    qk_read_for_selection: bool
    selector_count: int
    selection_standard: str


@dataclass(frozen=True)
class PluckerWorldlineSelectionReport:
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
    protected_local_edges: int
    missing_local_edges: int
    mean_geometry_gate: float
    max_geometry_gate: float


class MatrixPluckerWorldlineAttentionCompiler:
    """Compile a pure Plucker kNN graph, then leave Q/K to sparse softmax.

    Each spatiotemporal block center defines an oriented world ray with unit
    direction ``d`` and trajectory-normalized moment ``m = o_bar x d``.
    The six-dimensional Plucker descriptor ``normalize([d, m])`` determines
    the complete fixed 20-percent topology by cosine Top-K.  Live Q/K values
    are never read for topology selection; the unchanged sparse attention
    kernel uses Q/K normally inside the selected block pairs.
    """

    name = "matrix_plucker_worldline_attention_compiler"
    selection_standard = "plucker_worldline_geometry_topk_20_percent"
    cache_key_suffix = "plucker_worldline_geometry_topk20"

    def __init__(self) -> None:
        self._indices: torch.Tensor | None = None
        self._num_blocks = 0
        self._degree = 0
        self._block_shape = (0, 0, 0)
        self._mean_selected_cosine = 0.0

    @staticmethod
    def _geometry_features(
        normalized_origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        moments = torch.linalg.cross(normalized_origins, directions, dim=-1)
        return F.normalize(torch.cat((directions, moments), dim=-1), dim=-1)

    def _geometry_affinity(
        self,
        normalized_origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        features = self._geometry_features(normalized_origins, directions)
        return features @ features.transpose(0, 1)

    def compile(
        self,
        layout: Any,
        *,
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, PluckerWorldlineCompileReport]:
        del current_action
        if bool(getattr(layout, "protected_current")):
            raise ValueError("PWA requires the aggressive fixed-degree layout")
        if current_c2ws.ndim != 3 or tuple(current_c2ws.shape[-2:]) != (4, 4):
            raise ValueError("current_c2ws must have shape [T,4,4]")
        tt, th, tw = tuple(int(value) for value in layout.block_shape)
        temporal = int(layout.memory_length) + int(layout.current_length)
        temporal_blocks = math.ceil(temporal / tt)
        spatial_blocks = math.ceil(int(token_h) / th) * math.ceil(int(token_w) / tw)
        num_blocks = temporal_blocks * spatial_blocks
        if tuple(layout.indices.shape[:3]) != (1, 1, num_blocks):
            raise RuntimeError("layout dimensions do not match PWA block grid")
        counts = layout.counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise RuntimeError("PWA requires a uniform sparse degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree > num_blocks:
            raise RuntimeError("PWA received an invalid sparse degree")

        poses = MatrixReciprocalRayGeodesicAttentionCompiler._temporal_block_poses(
            active_atoms, current_c2ws, temporal_tile=tt
        )
        if len(poses) != temporal_blocks:
            raise RuntimeError("PWA temporal poses do not match the block grid")
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(poses)
        ray_states = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            poses,
            token_h=int(token_h),
            token_w=int(token_w),
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        ).float()
        if ray_states.shape != (num_blocks, 6):
            raise RuntimeError("PWA ray count does not match attention blocks")

        origins = ray_states[:, :3]
        directions = F.normalize(ray_states[:, 3:], dim=-1)
        reference_origin = poses[0, :3, 3].float() / scale
        normalized_origins = origins - reference_origin[None]
        geometry_score = self._geometry_affinity(normalized_origins, directions)
        _, selected = torch.topk(geometry_score, k=degree, dim=-1)
        selected = torch.sort(selected, dim=-1).values
        duplicates = int(
            torch.count_nonzero(selected[:, 1:] == selected[:, :-1]).item()
        )
        if duplicates:
            raise RuntimeError("PWA emitted duplicate topology edges")
        chosen_score = torch.gather(geometry_score, -1, selected)
        output = torch.zeros_like(layout.indices)
        output[0, 0, :, :degree] = selected.to(output.dtype)

        self._indices = selected.detach().to(torch.int32)
        self._num_blocks = num_blocks
        self._degree = degree
        self._block_shape = (tt, th, tw)
        self._mean_selected_cosine = float(chosen_score.mean().item())
        report = PluckerWorldlineCompileReport(
            query_rows=num_blocks,
            edges_per_row=degree,
            trajectory_scale=float(scale.item()),
            mean_selected_plucker_cosine=self._mean_selected_cosine,
            row_duplicate_violations=duplicates,
            activation_residual_fill=False,
            geometry_guard_ratio=1.0,
            qk_read_for_selection=False,
            selector_count=1,
            selection_standard=self.selection_standard,
        )
        return (
            replace(
                layout,
                indices=output,
                activation_residual_fill=False,
                geometry_guard_ratio=1.0,
                cache_key=tuple(layout.cache_key) + (self.cache_key_suffix,),
            ),
            report,
        )

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        geometry_indices: torch.Tensor,
        geometry_counts: torch.Tensor,
        latent_hw: tuple[int, int],
        block_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, PluckerWorldlineSelectionReport]:
        del geometry_indices, latent_hw
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("PWA requires equal B,H,L,D Q/K tensors")
        if self._indices is None:
            raise RuntimeError("PWA topology was not compiled")
        if tuple(int(value) for value in block_shape) != self._block_shape:
            raise RuntimeError("PWA runtime block shape differs from compilation")
        runtime_counts = geometry_counts[0, 0].detach().to("cpu", torch.int64)
        if runtime_counts.numel() != self._num_blocks or not torch.all(
            runtime_counts == self._degree
        ):
            raise RuntimeError("PWA runtime sparse degree differs from compilation")
        batch, heads = q.shape[:2]
        selected = self._indices.to(device=q.device)
        selected = selected[None, None].expand(batch, heads, -1, -1)
        counts = torch.full(
            (batch, heads, self._num_blocks),
            self._degree,
            device=q.device,
            dtype=torch.int32,
        )
        report = PluckerWorldlineSelectionReport(
            batch_heads=batch * heads,
            query_rows=self._num_blocks,
            edges_per_row=self._degree,
            row_duplicate_violations=0,
            selected_orientation_violations=0,
            max_orientation_violations_per_row=0,
            mean_selected_ray_cosine=self._mean_selected_cosine,
            mean_selected_qk_cosine=0.0,
            qk_read_for_selection=False,
            selector_count=1,
            protected_local_edges=0,
            missing_local_edges=0,
            mean_geometry_gate=1.0,
            max_geometry_gate=1.0,
        )
        return selected, counts, report


class MatrixRayCompletePluckerAttentionCompiler(
    MatrixPluckerWorldlineAttentionCompiler
):
    """Restore Plucker's missing longitudinal camera-origin coordinate.

    Plucker ``[d, o_bar x d]`` identifies an oriented supporting line but is
    invariant to moving the camera center along that line.  RCPA adds
    ``a=(o_bar^T d)d`` and normalizes the symmetric 9D descriptor
    ``[d, m, a]`` before the same pure-geometry fixed-budget Top-K.
    """

    name = "matrix_ray_complete_plucker_attention_compiler"
    selection_standard = "ray_complete_plucker_geometry_topk_20_percent"
    cache_key_suffix = "ray_complete_plucker_geometry_topk20"

    @staticmethod
    def _geometry_features(
        normalized_origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        moments = torch.linalg.cross(normalized_origins, directions, dim=-1)
        longitudinal_scalar = (normalized_origins * directions).sum(
            dim=-1, keepdim=True
        )
        longitudinal = longitudinal_scalar * directions
        return F.normalize(
            torch.cat((directions, moments, longitudinal), dim=-1), dim=-1
        )


class MatrixCheiralityAwareRayCrossingAttentionCompiler(
    MatrixPluckerWorldlineAttentionCompiler
):
    """Build topology from closest points on pairs of forward camera rays.

    For every pair, CRA solves the two-variable non-negative least-squares
    problem over ray depths and adds squared view-direction disagreement.  A
    larger affinity is exactly a smaller dimensionless oriented-ray energy.
    """

    name = "matrix_cheirality_aware_ray_crossing_attention_compiler"
    selection_standard = "cheirality_ray_crossing_geometry_topk_20_percent"
    cache_key_suffix = "cheirality_ray_crossing_geometry_topk20"

    @staticmethod
    def _forward_ray_distance_squared(
        origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        # Minimize ||(o_i-o_j) + alpha*d_i - beta*d_j||^2 for alpha,beta>=0.
        delta = origins[:, None, :] - origins[None, :, :]
        cosine = directions @ directions.transpose(0, 1)
        d_term = torch.einsum("ijd,id->ij", delta, directions)
        e_term = torch.einsum("ijd,jd->ij", delta, directions)
        denominator = 1.0 - cosine.square()
        eps = 32.0 * torch.finfo(origins.dtype).eps

        alpha_line = (cosine * e_term - d_term) / denominator.clamp_min(eps)
        beta_line = (e_term - cosine * d_term) / denominator.clamp_min(eps)
        line_valid = (
            (denominator > eps) & (alpha_line >= 0.0) & (beta_line >= 0.0)
        )
        line_residual = (
            delta
            + alpha_line[..., None] * directions[:, None, :]
            - beta_line[..., None] * directions[None, :, :]
        ).square().sum(dim=-1)
        infinity = torch.full_like(line_residual, torch.inf)
        line_residual = torch.where(line_valid, line_residual, infinity)

        # Boundary alpha=0, including the corner alpha=beta=0.
        beta_boundary = e_term.clamp_min(0.0)
        alpha_zero = (
            delta - beta_boundary[..., None] * directions[None, :, :]
        ).square().sum(dim=-1)
        # Boundary beta=0, likewise including the shared ray origin.
        alpha_boundary = (-d_term).clamp_min(0.0)
        beta_zero = (
            delta + alpha_boundary[..., None] * directions[:, None, :]
        ).square().sum(dim=-1)
        return torch.minimum(line_residual, torch.minimum(alpha_zero, beta_zero))

    def _geometry_affinity(
        self,
        normalized_origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        crossing_distance = self._forward_ray_distance_squared(
            normalized_origins, directions
        )
        direction_distance = (
            directions[:, None, :] - directions[None, :, :]
        ).square().sum(dim=-1)
        return -(crossing_distance + direction_distance)


__all__ = [
    "MatrixPluckerWorldlineAttentionCompiler",
    "MatrixRayCompletePluckerAttentionCompiler",
    "MatrixCheiralityAwareRayCrossingAttentionCompiler",
    "PluckerWorldlineCompileReport",
    "PluckerWorldlineSelectionReport",
]
