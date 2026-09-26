"""C2-quotient camera-ray geodesics for Matrix sparse attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .reciprocal_degree_attention import MatrixReciprocalDegreeAttentionCompiler


@dataclass(frozen=True)
class ReciprocalRayGeodesicAttentionReport:
    query_rows: int
    query_orbits: int
    edges_per_row: int
    trajectory_scale: float
    mean_selected_geodesic: float
    max_selected_geodesic: float
    equivariance_violations: int
    row_duplicate_violations: int
    activation_residual_fill: bool
    geometry_guard_ratio: float
    qk_read_for_selection: bool
    selector_count: int
    selection_standard: str
    involution: str


class MatrixReciprocalRayGeodesicAttentionCompiler:
    """Select sparse edges by one world-model-native geometric distance.

    Matrix conditions every latent location with a six-channel camera ray:
    camera origin plus unit world direction.  This compiler assigns the same
    ray state to each attention block center, normalizes origins by the
    trajectory diameter, and measures the ordinary chordal distance on that
    single ray manifold.  On a free reciprocal query orbit ``(q, gq)``, edge
    ``q -> k`` has the C2 quotient distance

        d_C2(q, k)^2 = (||r_q-r_k||^2 + ||r_gq-r_gk||^2) / 2.

    The 30 smallest distances are selected once and lifted by the involution.
    There is no prefix, fill, Q/K score, weighted multi-objective, learned
    parameter, or sample-dependent threshold.
    """

    name = "matrix_reciprocal_ray_geodesic_attention_compiler"

    @staticmethod
    def _block_involution(
        num_blocks: int,
        *,
        temporal_blocks: int,
        spatial_blocks: int,
    ) -> list[int]:
        mate = [
            MatrixReciprocalDegreeAttentionCompiler._involution(
                block,
                temporal_blocks=temporal_blocks,
                spatial_blocks=spatial_blocks,
            )
            for block in range(num_blocks)
        ]
        if any(mate[mate[block]] != block or mate[block] == block for block in range(num_blocks)):
            raise RuntimeError("camera-ray quotient requires a free C2 involution")
        return mate

    @staticmethod
    def _trajectory_scale(poses: torch.Tensor) -> torch.Tensor:
        origins = poses[:, :3, 3].float()
        if len(origins) < 2:
            return origins.new_tensor(1.0)
        diameter = torch.cdist(origins, origins).max()
        return diameter.clamp_min(torch.finfo(torch.float32).eps)

    @staticmethod
    def _temporal_block_poses(
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
        *,
        temporal_tile: int,
    ) -> torch.Tensor:
        poses = [getattr(atom, "c2w").to(device=current_c2ws.device) for atom in active_atoms]
        poses.extend(current_c2ws[index] for index in range(len(current_c2ws)))
        if not poses:
            raise ValueError("camera-ray attention requires at least one pose")
        combined = torch.stack(poses).float()
        representatives = []
        temporal_blocks = math.ceil(len(combined) / temporal_tile)
        for block in range(temporal_blocks):
            low = block * temporal_tile
            high = min(len(combined), (block + 1) * temporal_tile)
            center = block * temporal_tile + temporal_tile // 2
            index = min(range(low, high), key=lambda value: abs(value - center))
            representatives.append(combined[index])
        return torch.stack(representatives)

    @staticmethod
    def _ray_states(
        poses: torch.Tensor,
        *,
        token_h: int,
        token_w: int,
        block_h: int,
        block_w: int,
        trajectory_scale: torch.Tensor,
    ) -> torch.Tensor:
        nh = math.ceil(token_h / block_h)
        nw = math.ceil(token_w / block_w)
        centers_y = torch.arange(nh, device=poses.device, dtype=torch.float32)
        centers_x = torch.arange(nw, device=poses.device, dtype=torch.float32)
        centers_y = torch.minimum(
            centers_y * block_h + block_h * 0.5,
            centers_y.new_tensor(token_h - 0.5),
        )
        centers_x = torch.minimum(
            centers_x * block_w + block_w * 0.5,
            centers_x.new_tensor(token_w - 0.5),
        )
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing="ij")
        fx = float(token_w) / 2.0
        fy = float(token_h) / 2.0
        cx = float(token_w) / 2.0
        cy = float(token_h) / 2.0
        local = torch.stack(
            (
                (grid_x.reshape(-1) - cx) / fx,
                (grid_y.reshape(-1) - cy) / fy,
                torch.ones(nh * nw, device=poses.device),
            ),
            dim=-1,
        )
        local = F.normalize(local, dim=-1)
        directions = torch.einsum("tij,sj->tsi", poses[:, :3, :3], local)
        directions = F.normalize(directions, dim=-1)
        origins = poses[:, None, :3, 3].expand_as(directions) / trajectory_scale
        return torch.cat((origins, directions), dim=-1).reshape(-1, 6)

    def compile(
        self,
        layout: Any,
        *,
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, ReciprocalRayGeodesicAttentionReport]:
        if bool(getattr(layout, "protected_current")):
            raise ValueError("camera-ray attention requires aggressive C3 layout")
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
            raise RuntimeError("camera-ray attention requires uniform sparse degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree > num_blocks:
            raise RuntimeError("camera-ray attention received invalid sparse degree")

        block_poses = self._temporal_block_poses(
            active_atoms,
            current_c2ws,
            temporal_tile=tt,
        )
        if len(block_poses) != temporal_blocks:
            raise RuntimeError("temporal pose representatives do not match block grid")
        scale = self._trajectory_scale(block_poses)
        rays = self._ray_states(
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
        mate_list = self._block_involution(
            num_blocks,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        mate = torch.tensor(mate_list, device=rays.device, dtype=torch.long)
        output = torch.empty_like(layout.indices)
        selected_distances = []
        query_orbits = 0
        for representative in range(num_blocks):
            return_query = mate_list[representative]
            if representative > return_query:
                continue
            query_orbits += 1
            # One scalar on the quotient ray manifold. Stable argsort uses the
            # immutable block id only when two distances are bitwise equal.
            score = 0.5 * (
                squared[representative]
                + squared[return_query].index_select(0, mate)
            )
            section = torch.argsort(score, stable=True)[:degree]
            output[0, 0, representative, :degree] = section.to(output.dtype)
            output[0, 0, return_query, :degree] = mate[section].to(output.dtype)
            if output.shape[-1] > degree:
                output[0, 0, representative, degree:] = 0
                output[0, 0, return_query, degree:] = 0
            selected_distances.append(torch.sqrt(score[section].clamp_min(0.0)))

        equivariance_violations = 0
        duplicate_violations = 0
        for query in range(num_blocks):
            row = output[0, 0, query, :degree].to(torch.long)
            expected = mate[row]
            actual = output[0, 0, mate_list[query], :degree].to(torch.long)
            equivariance_violations += int(torch.count_nonzero(expected != actual).item())
            duplicate_violations += int(len(torch.unique(row)) != degree)
        if equivariance_violations or duplicate_violations:
            raise RuntimeError("camera-ray attention certificate failed")
        distance_tensor = torch.cat(selected_distances)
        report = ReciprocalRayGeodesicAttentionReport(
            query_rows=num_blocks,
            query_orbits=query_orbits,
            edges_per_row=degree,
            trajectory_scale=float(scale.item()),
            mean_selected_geodesic=float(distance_tensor.mean().item()),
            max_selected_geodesic=float(distance_tensor.max().item()),
            equivariance_violations=equivariance_violations,
            row_duplicate_violations=duplicate_violations,
            activation_residual_fill=False,
            geometry_guard_ratio=1.0,
            qk_read_for_selection=False,
            selector_count=1,
            selection_standard="c2_quotient_trajectory_normalized_camera_ray_geodesic",
            involution="same_spatial_temporal_reversal_with_mid_slab_adjacent_pairing",
        )
        return (
            replace(
                layout,
                indices=output,
                activation_residual_fill=False,
                geometry_guard_ratio=1.0,
                cache_key=tuple(layout.cache_key)
                + ("reciprocal_camera_ray_geodesic_c2",),
            ),
            report,
        )


__all__ = [
    "MatrixReciprocalRayGeodesicAttentionCompiler",
    "ReciprocalRayGeodesicAttentionReport",
]
