"""Command-oriented action-time product geometry for Matrix attention."""

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
class CommandPhaseRayWorldlineAttentionReport:
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
    phase_step: float
    orientation_consistent_candidates_min: int
    selected_orientation_violations: int
    max_orientation_violations_per_row: int
    mean_selected_action_time_residual: float
    max_selected_action_time_residual: float


class MatrixCommandPhaseRayWorldlineAttentionCompiler:
    """Select once in a command-oriented ray/action-time product geometry."""

    name = "matrix_command_phase_ray_worldline_attention_compiler"

    @staticmethod
    def _command_vector(current_action: torch.Tensor) -> torch.Tensor:
        if not isinstance(current_action, torch.Tensor):
            raise TypeError("command-phase attention requires a tensor action")
        value = current_action.detach().float()
        if value.numel() == 0:
            raise ValueError("command-phase attention received an empty action")
        if value.ndim > 1:
            value = value.reshape(-1, value.shape[-1]).mean(dim=0)
        return value.reshape(-1)

    @classmethod
    def _command_axis(
        cls,
        current_action: torch.Tensor,
        current_c2ws: torch.Tensor,
    ) -> tuple[torch.Tensor, str, float]:
        """Return one command-oriented axis in six-dimensional ray state.

        Translational controls act on the ray-origin coordinates and camera
        rotations act on the ray-direction coordinates.  This keeps the
        action feasibility test meaningful for WorldMark's pure L/R camera
        protocols, whose camera origins are constant by construction.
        """
        command = cls._command_vector(current_action)
        eps = torch.finfo(torch.float32).eps
        if command.numel() >= 4:
            forward = float((command[2] - command[3]).item())
            if abs(forward) > eps:
                optical = current_c2ws[:, :3, 2].float().mean(dim=0)
                if float(torch.linalg.vector_norm(optical).item()) <= eps:
                    raise RuntimeError("Matrix optical command axis is degenerate")
                origin_axis = F.normalize(optical, dim=0) * (
                    1.0 if forward > 0 else -1.0
                )
                axis = torch.cat((origin_axis, torch.zeros_like(origin_axis)))
                return axis, "keyboard_forward_reverse_optical_axis", abs(forward)
        if command.numel() >= 6:
            lateral = float((command[5] - command[4]).item())
            if abs(lateral) > eps:
                right = current_c2ws[:, :3, 0].float().mean(dim=0)
                if float(torch.linalg.vector_norm(right).item()) <= eps:
                    raise RuntimeError("Matrix lateral command axis is degenerate")
                origin_axis = F.normalize(right, dim=0) * (
                    1.0 if lateral > 0 else -1.0
                )
                axis = torch.cat((origin_axis, torch.zeros_like(origin_axis)))
                return axis, "keyboard_left_right_camera_axis", abs(lateral)
        if command.numel() >= 2:
            yaw = float(command[1].item())
            if abs(yaw) > eps:
                right = current_c2ws[0, :3, 0].float()
                if float(torch.linalg.vector_norm(right).item()) <= eps:
                    raise RuntimeError("Matrix yaw command axis is degenerate")
                direction_axis = F.normalize(right, dim=0) * (
                    1.0 if yaw > 0 else -1.0
                )
                axis = torch.cat((torch.zeros_like(direction_axis), direction_axis))
                return axis, "mouse_yaw_ray_direction_axis", abs(yaw)
            pitch = float(command[0].item())
            if abs(pitch) > eps:
                up = current_c2ws[0, :3, 1].float()
                if float(torch.linalg.vector_norm(up).item()) <= eps:
                    raise RuntimeError("Matrix pitch command axis is degenerate")
                direction_axis = F.normalize(up, dim=0) * (
                    1.0 if pitch > 0 else -1.0
                )
                axis = torch.cat((torch.zeros_like(direction_axis), direction_axis))
                return axis, "mouse_pitch_ray_direction_axis", abs(pitch)
        origins = current_c2ws[:, :3, 3].float()
        tangent = origins[-1] - origins[0]
        norm = float(torch.linalg.vector_norm(tangent).item())
        if norm <= eps:
            raise RuntimeError("Matrix command has no resolvable camera axis")
        origin_axis = F.normalize(tangent, dim=0)
        return (
            torch.cat((origin_axis, torch.zeros_like(origin_axis))),
            "pose_translation_fallback",
            norm,
        )

    def compile(
        self,
        layout: Any,
        *,
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, CommandPhaseRayWorldlineAttentionReport]:
        if bool(getattr(layout, "protected_current")):
            raise ValueError("command-phase attention requires aggressive layout")
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
            raise RuntimeError("layout dimensions do not match command-phase geometry")
        counts = layout.counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise RuntimeError("command-phase attention requires uniform degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree > num_blocks:
            raise RuntimeError("command-phase attention received invalid degree")

        block_poses = MatrixReciprocalRayGeodesicAttentionCompiler._temporal_block_poses(
            active_atoms, current_c2ws, temporal_tile=tt
        )
        if len(block_poses) != temporal_blocks:
            raise RuntimeError("temporal pose representatives do not match block grid")
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(block_poses)
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

        axis, axis_source, command_strength = self._command_axis(
            current_action, current_c2ws
        )
        normalized_current_origins = current_c2ws[:, :3, 3].float() / scale
        current_optical = F.normalize(current_c2ws[:, :3, 2].float(), dim=-1)
        current_ray_state = torch.cat(
            (normalized_current_origins, current_optical), dim=-1
        )
        current_longitudinal = current_ray_state @ axis
        current_temporal_blocks = max(1, math.ceil(len(current_c2ws) / tt))
        phase_step = float(
            (
                (current_longitudinal[-1] - current_longitudinal[0]).abs()
                / max(1, current_temporal_blocks - 1)
            ).item()
        )
        if not math.isfinite(phase_step):
            raise RuntimeError("command-phase attention has non-finite phase speed")

        squared_ray = torch.cdist(rays, rays).square()
        temporal_ids = torch.arange(num_blocks, device=rays.device) // spatial_blocks
        longitudinal = rays @ axis
        output = torch.zeros_like(layout.indices)
        selected_distances = []
        selected_residuals = []
        consistent_counts = []
        selected_violation_counts = []
        for query in range(num_blocks):
            time_delta = temporal_ids - temporal_ids[query]
            axis_delta = longitudinal - longitudinal[query]
            violation = time_delta * axis_delta < 0.0
            action_time_residual = (
                axis_delta - time_delta.to(torch.float32) * phase_step
            ).abs()
            product_squared = squared_ray[query] + action_time_residual.square()
            product_order = torch.argsort(product_squared, stable=True)
            orientation_order = torch.argsort(
                violation[product_order].to(torch.int8), stable=True
            )
            section = product_order[orientation_order[:degree]]
            output[0, 0, query, :degree] = section.to(output.dtype)
            consistent_counts.append(int(torch.count_nonzero(~violation).item()))
            selected_violation_counts.append(
                int(torch.count_nonzero(violation[section]).item())
            )
            selected_distances.append(
                torch.sqrt(squared_ray[query, section].clamp_min(0.0))
            )
            selected_residuals.append(action_time_residual[section])

        duplicate_violations = sum(
            int(len(torch.unique(output[0, 0, query, :degree])) != degree)
            for query in range(num_blocks)
        )
        if duplicate_violations:
            raise RuntimeError("command-phase attention selected duplicate blocks")
        mate_list = MatrixReciprocalRayGeodesicAttentionCompiler._block_involution(
            num_blocks,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        mate = torch.tensor(mate_list, device=rays.device, dtype=torch.long)
        natural_violations = 0
        for query in range(num_blocks):
            row = output[0, 0, query, :degree].to(torch.long)
            expected = torch.sort(mate[row]).values
            actual = torch.sort(output[0, 0, mate_list[query], :degree]).values
            natural_violations += int(torch.count_nonzero(expected != actual).item())

        distance_tensor = torch.cat(selected_distances)
        residual_tensor = torch.cat(selected_residuals)
        report = CommandPhaseRayWorldlineAttentionReport(
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
            qk_read_for_selection=False,
            selector_count=1,
            selection_standard="command_oriented_ray_action_time_phase_product_geodesic",
            involution="none_independent_query_rows",
            command_axis_source=axis_source,
            command_strength=command_strength,
            phase_step=phase_step,
            orientation_consistent_candidates_min=min(consistent_counts),
            selected_orientation_violations=sum(selected_violation_counts),
            max_orientation_violations_per_row=max(selected_violation_counts),
            mean_selected_action_time_residual=float(residual_tensor.mean().item()),
            max_selected_action_time_residual=float(residual_tensor.max().item()),
        )
        return (
            replace(
                layout,
                indices=output,
                activation_residual_fill=False,
                geometry_guard_ratio=1.0,
                cache_key=tuple(layout.cache_key)
                + ("command_oriented_ray_action_time_phase_product",),
            ),
            report,
        )


__all__ = [
    "CommandPhaseRayWorldlineAttentionReport",
    "MatrixCommandPhaseRayWorldlineAttentionCompiler",
]
