"""Faithful JiT spatial-token baseline for Matrix-Game-3.0.

This ports the released CVPR 2026 JiT implementation without action, camera,
or WorldMark-aware token scoring.  JiT's spatial anchor set is shared across
video time (the smallest necessary image-to-video adaptation).  Matrix's
native CWCA block topology is retained and evaluated only on active tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import json
import os
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


@dataclass
class _CallState:
    chunk_index: int
    step_index: int
    requested_ratio: float
    active_spatial: torch.Tensor | None = None
    spatial_height: int = 0
    spatial_width: int = 0
    total_temporal: int = 0
    memory_temporal: int = 0
    full_sequence_length: int = 0
    compact_indices: torch.Tensor | None = None
    compact_coordinates: torch.Tensor | None = None
    packed_indices: torch.Tensor | None = None
    compact_to_packed: torch.Tensor | None = None
    packed_valid: torch.Tensor | None = None
    packed_block_size: int = 0


class MatrixJiTSpatialAcceleration:
    """SAG-ODE + ITA + DMF with an explicit three-phase token schedule."""

    name = "matrix_jit_official_spatial_35_62_100"

    def __init__(
        self,
        *,
        stage_ratios: tuple[float, float, float] = (0.35, 0.62, 1.0),
        gaussian_c: float = 0.4,
        microflow_relax_steps: int = 3,
        force_custom_steps: tuple[int, ...] = (),
    ) -> None:
        if len(stage_ratios) != 3 or any(
            value < 0.0 or value > 1.0 for value in stage_ratios
        ):
            raise ValueError("JiT requires three active-token ratios in [0, 1]")
        if stage_ratios[-1] != 1.0:
            raise ValueError("JiT terminal stage must be exact")
        self.stage_ratios = tuple(float(value) for value in stage_ratios)
        self.force_custom_steps = tuple(int(value) for value in force_custom_steps)
        if any(value not in (0, 1, 2) for value in self.force_custom_steps):
            raise ValueError("JiT forced custom steps must be in {0,1,2}")
        if self.force_custom_steps:
            forced = "_".join(str(value) for value in self.force_custom_steps)
            self.name = f"matrix_jit_custom_path_equivalence_steps_{forced}"
        elif self.stage_ratios == (0.35, 0.62, 1.0):
            self.name = "matrix_jit_official_spatial_35_62_100"
        else:
            tag = "_".join(str(int(round(value * 100))) for value in stage_ratios)
            self.name = f"matrix_jit_spatial_{tag}"
        self.gaussian_c = float(gaussian_c)
        self.microflow_relax_steps = int(microflow_relax_steps)
        self.model: Any = None
        self._native_block_forwards: list[tuple[Any, Any]] = []
        self._native_head_forward: Any = None
        self._state: _CallState | None = None
        self._active_spatial: torch.Tensor | None = None
        self._last_velocity: torch.Tensor | None = None
        self._last_input: torch.Tensor | None = None
        self._last_sigma: float | None = None
        self._global_noise: torch.Tensor | None = None
        self._last_lifted_patch_velocity: torch.Tensor | None = None
        self._records: list[dict[str, Any]] = []
        self._current_record: dict[str, Any] | None = None
        self._sparse_validation: dict[str, Any] = {}

    def install(self, model: Any) -> None:
        if self.model is not None:
            raise RuntimeError("JiT accelerator is already installed")
        self.model = model
        for block_index, block in enumerate(model.blocks):
            native = block.forward
            self._native_block_forwards.append((block, native))

            def sparse_forward(
                block_self: Any,
                x: torch.Tensor,
                *args: Any,
                _index: int = block_index,
                _native: Any = native,
                **kwargs: Any,
            ) -> torch.Tensor:
                return self._forward_block(
                    _index, block_self, _native, x, *args, **kwargs
                )

            block.forward = MethodType(sparse_forward, block)
        self._native_head_forward = model.head.forward

        def sparse_head(
            head_self: Any,
            x: torch.Tensor,
            e: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return self._forward_head(head_self, x, e, *args, **kwargs)

        model.head.forward = MethodType(sparse_head, model.head)

    def reset_runtime_state(self) -> None:
        self._state = None
        self._active_spatial = None
        self._last_velocity = None
        self._last_input = None
        self._last_sigma = None
        self._global_noise = None
        self._last_lifted_patch_velocity = None
        self._records.clear()
        self._current_record = None
        self._sparse_validation.clear()

    def begin_model_call(
        self,
        *,
        chunk_index: int,
        step_index: int,
        current_action: Any = None,
        memory_atoms: int = 0,
        **_: Any,
    ) -> str:
        del current_action, memory_atoms
        if step_index not in (0, 1, 2):
            raise ValueError(f"unexpected Matrix scheduler step {step_index}")
        if step_index == 0:
            self._active_spatial = None
            self._last_velocity = None
            self._last_input = None
            self._last_sigma = None
            self._global_noise = None
            self._last_lifted_patch_velocity = None
        self._state = _CallState(
            chunk_index=int(chunk_index),
            step_index=int(step_index),
            requested_ratio=self.stage_ratios[step_index],
        )
        self._current_record = {
            "chunk_index": int(chunk_index),
            "step_index": int(step_index),
            "requested_active_ratio": self.stage_ratios[step_index],
            "mode": (
                "jit_zero_active_lift"
                if self.stage_ratios[step_index] == 0.0
                else "jit_full_active_custom"
                if step_index in self.force_custom_steps
                else "jit_full_exact"
                if self.stage_ratios[step_index] >= 1.0
                else "jit_sparse_sag_ode"
            ),
        }
        return str(self._current_record["mode"])

    def zero_active_prediction(
        self, x: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Return the lifted preceding velocity without executing the DiT.

        An empty active set cannot be passed through Matrix's RoPE,
        ActionModule, or CWCA kernels.  Zero active tokens therefore has the
        SAG-ODE meaning: evaluate no new token velocities and use the complete
        lifted velocity field from the preceding phase.
        """
        if self._state is None or self._current_record is None:
            raise RuntimeError("JiT zero-active call was not begun")
        if self._state.requested_ratio != 0.0:
            raise RuntimeError("JiT zero-active prediction requires a zero ratio")
        if self._last_velocity is None:
            raise RuntimeError("JiT zero-active phase lacks a preceding velocity")
        patch_h, patch_w = (int(value) for value in self.model.patch_size[1:])
        total = (int(x.shape[-2]) // patch_h) * (int(x.shape[-1]) // patch_w)
        self._active_spatial = torch.empty(0, dtype=torch.long, device=x.device)
        self._last_input = x.detach().clone()
        self._last_sigma = self._sigma_from_timestep(timestep)
        self._current_record.update(
            {
                "active_spatial_tokens": 0,
                "total_spatial_tokens": total,
                "actual_active_ratio": 0.0,
                "newly_activated_spatial_tokens": 0,
                "packed_cwca_block_size": 0,
                "native_cwca_block_size": 128,
                "gaussian_c": self.gaussian_c,
                "exact_anchor_restore": True,
                "dmf_applied": False,
                "native_dit_skipped": True,
            }
        )
        return self._last_velocity.to(device=x.device, dtype=x.dtype).clone()

    @staticmethod
    def _sigma_from_timestep(timestep: torch.Tensor) -> float:
        value = float(timestep.detach().float().max().item())
        return value / 1000.0 if value > 1.0 else value

    def prepare_model_input(
        self, x: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Apply the released DMF target to newly activated raw latent patches.

        Matrix's pipeline aliases ``latent_model_input`` to its scheduler state,
        so the in-place copy also makes the scheduler integrate from the DMF
        state, matching JiT rather than applying a model-only perturbation.
        """
        if self._state is None:
            raise RuntimeError("JiT model call was not begun")
        sigma = self._sigma_from_timestep(timestep)
        if self._state.step_index == 0:
            self._global_noise = x.detach().clone()
            self._last_input = x.detach().clone()
            self._last_sigma = sigma
            return x
        if (
            self._active_spatial is None
            or self._last_velocity is None
            or self._last_input is None
            or self._last_sigma is None
            or self._global_noise is None
        ):
            raise RuntimeError("JiT stage transition lacks the preceding exact state")

        patch_h, patch_w = (int(value) for value in self.model.patch_size[1:])
        previous_active = self._active_spatial
        target_active = self._select_stage_anchors(
            height=x.shape[-2] // patch_h,
            width=x.shape[-1] // patch_w,
            ratio=self._state.requested_ratio,
            device=x.device,
        )
        newly_active = target_active[~torch.isin(target_active, previous_active)]
        if newly_active.numel() > 0:
            clean_previous = self._last_input - self._last_sigma * self._last_velocity
            clean_tokens = self._pack_raw_latent(clean_previous)
            noise_tokens = self._pack_raw_latent(self._global_noise)
            current_tokens = self._pack_raw_latent(x)
            if previous_active.numel() == 0:
                # A zero-active preceding phase used the already full lifted
                # q0 velocity, so its clean estimate is already dense.
                clean_lifted = clean_tokens
            else:
                clean_active = clean_tokens[:, :, previous_active, :]
                lift_height = x.shape[-2] // patch_h
                lift_width = x.shape[-1] // patch_w
                clean_lifted = self._lift_spatial(
                    clean_active,
                    previous_active,
                    lift_height,
                    lift_width,
                )
                if os.environ.get("MATRIX_JIT_SPARSE_TRACE"):
                    self._sparse_validation["dmf_coarse_to_fine"] = {
                        "previous_active_tokens": int(previous_active.numel()),
                        "target_active_tokens": int(target_active.numel()),
                        "newly_active_tokens": int(newly_active.numel()),
                        "lift_grid": [int(lift_height), int(lift_width)],
                        "latent_token_count": int(clean_tokens.shape[2]),
                        "latent_channel_dim": int(clean_tokens.shape[3]),
                        "grid_product_matches_tokens": bool(
                            lift_height * lift_width == clean_tokens.shape[2]
                        ),
                    }
            target = (
                (1.0 - sigma) * clean_lifted[:, :, newly_active, :]
                + sigma * noise_tokens[:, :, newly_active, :]
            ).to(current_tokens.dtype)
            if self.microflow_relax_steps <= 0:
                current_tokens[:, :, newly_active, :] = target
            else:
                weight = 1.0 / float(self.microflow_relax_steps)
                current_tokens[:, :, newly_active, :] = (
                    (1.0 - weight) * current_tokens[:, :, newly_active, :]
                    + weight * target
                )
            condition_frames = self._condition_frame_count(timestep, x)
            if condition_frames > 0:
                original = self._pack_raw_latent(x)
                current_tokens[:, :condition_frames] = original[:, :condition_frames]
            x.copy_(self._unpack_raw_latent(current_tokens, x.shape))
        self._active_spatial = target_active
        self._last_input = x.detach().clone()
        self._last_sigma = sigma
        if self._current_record is not None:
            self._current_record["newly_activated_spatial_tokens"] = int(
                newly_active.numel()
            )
            self._current_record["dmf_applied"] = True
            self._current_record["dmf_relax_steps"] = self.microflow_relax_steps
        return x

    def _condition_frame_count(
        self, timestep: torch.Tensor, x: torch.Tensor
    ) -> int:
        temporal = int(x.shape[2])
        spatial_tokens = (x.shape[-2] // self.model.patch_size[1]) * (
            x.shape[-1] // self.model.patch_size[2]
        )
        flat = timestep.detach().reshape(-1)
        if flat.numel() < temporal * spatial_tokens:
            return 0
        per_frame = flat[: temporal * spatial_tokens].reshape(temporal, spatial_tokens)
        zero = per_frame.abs().amax(dim=1) == 0
        count = 0
        for value in zero.tolist():
            if not value:
                break
            count += 1
        return count

    def _pack_raw_latent(self, x: torch.Tensor) -> torch.Tensor:
        ph, pw = (int(value) for value in self.model.patch_size[1:])
        batch, channels, temporal, height, width = x.shape
        return (
            x.view(batch, channels, temporal, height // ph, ph, width // pw, pw)
            .permute(0, 2, 3, 5, 1, 4, 6)
            .reshape(batch, temporal, (height // ph) * (width // pw), channels * ph * pw)
        )

    def _unpack_raw_latent(
        self, tokens: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        ph, pw = (int(value) for value in self.model.patch_size[1:])
        batch, channels, temporal, height, width = shape
        return (
            tokens.reshape(
                batch, temporal, height // ph, width // pw, channels, ph, pw
            )
            .permute(0, 4, 1, 2, 5, 3, 6)
            .reshape(batch, channels, temporal, height, width)
        )

    def _initial_sparse_grid(
        self, height: int, width: int, ratio: float, device: torch.device
    ) -> torch.Tensor:
        total = height * width
        target = int(total * ratio)
        stride = max(1, int(np.sqrt(1.0 / ratio)))
        grid_h = torch.arange(0, height, stride, device=device)
        grid_w = torch.arange(0, width, stride, device=device)
        if grid_h.numel() == 0 or int(grid_h[-1]) != height - 1:
            grid_h = torch.cat([grid_h, grid_h.new_tensor([height - 1])])
        if grid_w.numel() == 0 or int(grid_w[-1]) != width - 1:
            grid_w = torch.cat([grid_w, grid_w.new_tensor([width - 1])])
        yy, xx = torch.meshgrid(grid_h, grid_w, indexing="ij")
        indices = (yy.flatten() * width + xx.flatten()).unique(sorted=True)
        if indices.numel() < target:
            mask = torch.ones(total, dtype=torch.bool, device=device)
            mask[indices] = False
            available = torch.arange(total, device=device)[mask]
            supplement = available[
                torch.randperm(available.numel(), device=device)[: target - indices.numel()]
            ]
            indices = torch.cat([indices, supplement])
        elif indices.numel() > target:
            indices = indices[
                torch.randperm(indices.numel(), device=device)[:target]
            ]
        return indices.sort().values.long()

    def _select_stage_anchors(
        self, *, height: int, width: int, ratio: float, device: torch.device
    ) -> torch.Tensor:
        total = height * width
        target = int(total * ratio)
        if ratio >= 1.0:
            return torch.arange(total, device=device, dtype=torch.long)
        if self._active_spatial is None:
            return self._initial_sparse_grid(height, width, ratio, device)
        if self._last_lifted_patch_velocity is None:
            raise RuntimeError("JiT importance densification lacks lifted velocity")
        current = self._active_spatial
        if current.numel() == target:
            return current
        velocity = self._last_lifted_patch_velocity.float()
        # Video adaptation: local 3x3 variance is computed per frame exactly as
        # JiT, then averaged over time.  No action/camera signal is read.
        batch, temporal, spatial, dim = velocity.shape
        if spatial != height * width:
            raise RuntimeError("JiT lifted velocity does not match the spatial grid")
        velocity = velocity.reshape(batch, temporal, height, width, dim)
        image = velocity.permute(0, 1, 4, 2, 3).reshape(
            batch * temporal, dim, height, width
        )
        mean = F.avg_pool2d(image, 3, stride=1, padding=1)
        variance = F.avg_pool2d(image.square(), 3, stride=1, padding=1) - mean.square()
        importance = variance.mean(dim=1).reshape(batch, temporal, height, width)
        importance = importance.mean(dim=(0, 1)).flatten()
        if current.numel() > target:
            # The released JiT schedule is monotone, so it never needs this
            # branch.  A descending operator-requested schedule deactivates
            # the least informative current anchors using the same released
            # local-velocity-variance signal used for densification.
            selected = current[torch.topk(importance[current], k=target).indices]
            return selected.sort().values.long()
        inactive = torch.ones(total, dtype=torch.bool, device=device)
        inactive[current] = False
        candidates = torch.arange(total, device=device)[inactive]
        add = min(target - current.numel(), candidates.numel())
        selected = candidates[torch.topk(importance[candidates], k=add).indices]
        return torch.cat([current, selected]).sort().values.long()

    def _calculate_blur_params(self, ratio: float) -> tuple[int, float]:
        if ratio <= 0.0 or ratio >= 1.0:
            return 3, 1.0
        distance = 1.0 / np.sqrt(ratio)
        sigma = float(np.clip(self.gaussian_c * distance, 1.0, 10.0))
        return 2 * int(np.ceil(3.0 * sigma)) + 1, sigma

    def _lift_spatial(
        self,
        active_values: torch.Tensor,
        active_indices: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Official nearest velocity lifting + Gaussian + exact restore."""
        batch, temporal, _, dim = active_values.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=active_values.device),
            torch.arange(width, device=active_values.device),
            indexing="ij",
        )
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
        nearest = torch.cdist(coords, coords[active_indices]).argmin(dim=-1)
        full_nearest = active_values[:, :, nearest, :]
        image = full_nearest.permute(0, 1, 3, 2).reshape(
            batch * temporal, dim, height, width
        )
        ratio = float(active_indices.numel()) / float(height * width)
        kernel, sigma = self._calculate_blur_params(ratio)
        blurred = gaussian_blur(image, (kernel, kernel), (sigma, sigma))
        blurred = blurred.reshape(batch, temporal, dim, height * width).permute(
            0, 1, 3, 2
        )
        blurred[:, :, active_indices, :] = active_values
        return blurred

    def _ensure_layout(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        memory_length: int,
        orbit_sparse_layout: Any,
    ) -> _CallState:
        if self._state is None:
            raise RuntimeError("JiT sparse layout requested without a model call")
        state = self._state
        if state.compact_indices is not None:
            return state
        temporal, height, width = (
            int(value) for value in grid_sizes[0].tolist()
        )
        if self._active_spatial is None:
            self._active_spatial = self._select_stage_anchors(
                height=height,
                width=width,
                ratio=state.requested_ratio,
                device=x.device,
            )
        spatial = self._active_spatial
        time = torch.arange(temporal, device=x.device)[:, None]
        compact = (time * (height * width) + spatial[None, :]).reshape(-1)
        spatial_y = spatial // width
        spatial_x = spatial % width
        coords = torch.stack(
            [
                time.expand(-1, spatial.numel()).reshape(-1),
                spatial_y.repeat(temporal),
                spatial_x.repeat(temporal),
            ],
            dim=-1,
        )
        state.active_spatial = spatial
        state.spatial_height = height
        state.spatial_width = width
        state.total_temporal = temporal
        state.memory_temporal = int(memory_length)
        state.full_sequence_length = int(x.shape[1])
        state.compact_indices = compact.long()
        state.compact_coordinates = coords.long()

        if orbit_sparse_layout is not None:
            tt, th, tw = (
                int(value) for value in orbit_sparse_layout.block_shape
            )
            nh = math.ceil(height / th)
            nw = math.ceil(width / tw)
            memory_t = int(memory_length)
            current_t = temporal - memory_t
            t_coord, h_coord, w_coord = coords.unbind(dim=1)
            grouped = bool(getattr(orbit_sparse_layout, "protected_current", False))
            if grouped:
                memory_blocks = (
                    math.ceil(memory_t / tt) * nh * nw if memory_t else 0
                )
                in_memory = t_coord < memory_t
                block_id = torch.empty_like(t_coord)
                if memory_t:
                    block_id[in_memory] = (
                        (t_coord[in_memory] // tt) * nh * nw
                        + (h_coord[in_memory] // th) * nw
                        + w_coord[in_memory] // tw
                    )
                block_id[~in_memory] = (
                    memory_blocks
                    + ((t_coord[~in_memory] - memory_t) // tt) * nh * nw
                    + (h_coord[~in_memory] // th) * nw
                    + w_coord[~in_memory] // tw
                )
                total_blocks = (
                    memory_blocks + math.ceil(current_t / tt) * nh * nw
                )
            else:
                block_id = (
                    (t_coord // tt) * nh * nw
                    + (h_coord // th) * nw
                    + w_coord // tw
                )
                total_blocks = math.ceil(temporal / tt) * nh * nw
            if orbit_sparse_layout.indices.shape[-2] != total_blocks:
                raise RuntimeError(
                    "JiT/CWCA block mismatch: "
                    f"layout={orbit_sparse_layout.indices.shape[-2]} runtime={total_blocks}"
                )
            per_block = [torch.nonzero(block_id == index).flatten() for index in range(total_blocks)]
            maximum_active = max(int(values.numel()) for values in per_block)
            # Triton's arange/block pointers require a power-of-two tile.  This
            # padding is a Matrix kernel adapter, not an anchor-selection rule.
            packed_size = 1 << max(0, maximum_active - 1).bit_length()
            packed = torch.full(
                (total_blocks, packed_size), -1, device=x.device, dtype=torch.long
            )
            for block_index, values in enumerate(per_block):
                packed[block_index, : values.numel()] = values
            valid = packed >= 0
            compact_to_packed = torch.empty(
                compact.numel(), device=x.device, dtype=torch.long
            )
            flat_positions = torch.arange(
                total_blocks * packed_size, device=x.device
            ).reshape(total_blocks, packed_size)
            compact_to_packed[packed[valid]] = flat_positions[valid]
            state.packed_indices = packed
            state.compact_to_packed = compact_to_packed
            state.packed_valid = valid
            state.packed_block_size = packed_size
        if os.environ.get("MATRIX_JIT_SPARSE_TRACE"):
            expected = temporal * int(spatial.numel())
            layout_report: dict[str, Any] = {
                "active_sorted_unique": bool(
                    spatial.numel() == torch.unique(spatial).numel()
                    and (spatial.numel() < 2 or torch.all(spatial[1:] > spatial[:-1]))
                ),
                "compact_count": int(compact.numel()),
                "expected_compact_count": expected,
                "compact_unique": int(torch.unique(compact).numel()),
                "coordinate_count": int(coords.shape[0]),
                "compact_in_bounds": bool(
                    compact.numel() == 0
                    or (
                        int(compact.min()) >= 0
                        and int(compact.max()) < int(x.shape[1])
                    )
                ),
            }
            if state.packed_indices is not None:
                packed_valid_values = state.packed_indices[state.packed_valid]
                roundtrip = (
                    state.packed_indices.reshape(-1)[state.compact_to_packed]
                )
                layout_report.update(
                    {
                        "packed_valid_count": int(packed_valid_values.numel()),
                        "packed_valid_unique": int(
                            torch.unique(packed_valid_values).numel()
                        ),
                        "packed_roundtrip_exact": bool(
                            torch.equal(
                                roundtrip,
                                torch.arange(
                                    compact.numel(), device=compact.device
                                ),
                            )
                        ),
                    }
                )
            self._sparse_validation["layout"] = layout_report
        return state

    @staticmethod
    def _apply_sparse_rope(
        value: torch.Tensor,
        coords: torch.Tensor,
        freqs: torch.Tensor,
        memory_length: int,
        memory_latent_idx: Any,
        predict_latent_idx: Any,
    ) -> torch.Tensor:
        heads, half = value.shape[2], value.shape[3] // 2
        split = [half - 2 * (half // 3), half // 3, half // 3]
        axes = freqs.split(split, dim=2 if freqs.dim() == 3 else 1)
        temporal = coords[:, 0].clone()
        memory_count = int(memory_length)
        if memory_count:
            mem = (
                [int(value) for value in memory_latent_idx]
                if memory_latent_idx is not None
                else list(range(memory_count))
            )
            temporal[: memory_count * (coords.shape[0] // int(coords[:, 0].max().item() + 1))] = torch.tensor(
                mem, device=coords.device
            ).repeat_interleave(coords.shape[0] // int(coords[:, 0].max().item() + 1))
        current_frames = int(coords[:, 0].max().item() + 1) - memory_count
        if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
            pred = list(range(int(predict_latent_idx[0]), int(predict_latent_idx[1])))
        elif predict_latent_idx is not None:
            pred = [int(value) for value in predict_latent_idx]
        else:
            pred = list(range(current_frames))
        per_frame = coords.shape[0] // int(coords[:, 0].max().item() + 1)
        if current_frames:
            temporal[memory_count * per_frame :] = torch.tensor(
                pred[:current_frames], device=coords.device
            ).repeat_interleave(per_frame)
        indices = (temporal.long(), coords[:, 1].long(), coords[:, 2].long())
        if freqs.dim() == 3:
            phase = torch.cat(
                [axis[:, index, :].permute(1, 0, 2) for axis, index in zip(axes, indices)],
                dim=-1,
            )
        else:
            phase = torch.cat(
                [axis[index].unsqueeze(1) for axis, index in zip(axes, indices)],
                dim=-1,
            )
        complex_value = torch.view_as_complex(
            value.float().reshape(value.shape[0], value.shape[1], heads, -1, 2)
        )
        rotated = complex_value * phase.unsqueeze(0).to(complex_value.dtype)
        return torch.view_as_real(rotated).flatten(3).float()

    @staticmethod
    def _longcat_active(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
        valid: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        import triton
        from wan.modules.longcat_kernel import _attn_fwd_bsa_align

        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()
        indices = indices.to(torch.int32).contiguous()
        counts = counts.to(torch.int32).contiguous()
        batch, heads, q_len, head_dim = q.shape
        k_len = k.shape[2]
        key_valid = valid.reshape(-1).to(torch.uint8).contiguous()
        output = torch.empty_like(q)
        maximum = torch.empty(
            (batch, heads, q_len), device=q.device, dtype=torch.float32
        )
        grid = (triton.cdiv(q_len, block_size), batch * heads)
        _attn_fwd_bsa_align[grid](
            Q=q,
            K=k,
            V=v,
            K_valid=key_valid,
            sm_scale=1.0 / math.sqrt(head_dim),
            M=maximum,
            Out=output,
            block_indices=indices,
            block_indices_lens=counts,
            stride_qz=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm=q.stride(2),
            stride_qk=q.stride(3),
            stride_kz=k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_kk=k.stride(3),
            stride_vz=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            stride_vk=v.stride(3),
            stride_oz=output.stride(0),
            stride_oh=output.stride(1),
            stride_om=output.stride(2),
            stride_ok=output.stride(3),
            stride_bz=indices.stride(0),
            stride_bh=indices.stride(1),
            stride_bm=indices.stride(2),
            stride_bs=indices.stride(3),
            stride_lz=counts.stride(0),
            stride_lh=counts.stride(1),
            stride_lm=counts.stride(2),
            H=heads,
            Q_LEN=q_len,
            K_LEN=k_len,
            HEAD_DIM=head_dim,
            BLOCK_M=block_size,
            BLOCK_N=block_size,
        )
        return output

    def _sparse_self_attention(
        self,
        module: Any,
        x: torch.Tensor,
        state: _CallState,
        *,
        seq_lens: torch.Tensor,
        freqs: torch.Tensor,
        memory_length: int,
        memory_latent_idx: Any,
        predict_latent_idx: Any,
        fa_version: Any,
        orbit_sparse_layout: Any,
    ) -> torch.Tensor:
        batch, sequence = x.shape[:2]
        if (
            state.active_spatial is not None
            and int(state.active_spatial.numel())
            == int(state.spatial_height * state.spatial_width)
        ):
            # A 100%-active JiT stage is an identity token schedule.  Preserve
            # the model's native Self-Attention call (including its exact
            # autocast/kernel dispatch) instead of numerically re-creating it.
            return module(
                x,
                seq_lens,
                torch.tensor(
                    [[state.total_temporal, state.spatial_height, state.spatial_width]],
                    device=x.device,
                    dtype=torch.long,
                ),
                freqs,
                memory_length,
                memory_latent_idx=memory_latent_idx,
                predict_latent_idx=predict_latent_idx,
                fa_version=fa_version,
                orbit_sparse_layout=orbit_sparse_layout,
            )
        heads, dim = module.num_heads, module.head_dim
        q = module.norm_q(module.q(x)).view(batch, sequence, heads, dim)
        k = module.norm_k(module.k(x)).view(batch, sequence, heads, dim)
        v = module.v(x).view(batch, sequence, heads, dim)
        q = self._apply_sparse_rope(
            q, state.compact_coordinates, freqs, memory_length,
            memory_latent_idx, predict_latent_idx,
        )
        k = self._apply_sparse_rope(
            k, state.compact_coordinates, freqs, memory_length,
            memory_latent_idx, predict_latent_idx,
        )
        # Match WanSelfAttention exactly: Matrix only enables its block-sparse
        # kernel after memory tokens exist.  A CWCA layout may already be
        # supplied for the first chunk, but native attention deliberately
        # remains dense while memory_length == 0.
        if orbit_sparse_layout is None or memory_length == 0:
            from wan.modules.attention import attention

            output = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=module.window_size,
                version=fa_version,
            )
        else:
            packed = state.packed_indices.clamp_min(0).reshape(-1)
            valid = state.packed_valid
            q_packed = q[:, packed].transpose(1, 2).contiguous()
            k_packed = k[:, packed].transpose(1, 2).contiguous()
            v_packed = v[:, packed].transpose(1, 2).contiguous()
            mask = valid.reshape(-1)
            q_packed[:, :, ~mask] = 0
            k_packed[:, :, ~mask] = 0
            v_packed[:, :, ~mask] = 0
            indices = orbit_sparse_layout.indices.expand(
                batch, heads, -1, -1
            )
            counts = orbit_sparse_layout.counts.expand(batch, heads, -1)
            packed_output = self._longcat_active(
                q_packed,
                k_packed,
                v_packed,
                indices,
                counts,
                valid,
                state.packed_block_size,
            )
            packed_output = packed_output.transpose(1, 2)
            output = packed_output[:, state.compact_to_packed]
            if (
                os.environ.get("MATRIX_JIT_SPARSE_TRACE")
                and memory_length > 0
                and "self_attention_oracle" not in self._sparse_validation
            ):
                sample_count = min(8, sequence)
                sample_queries = torch.linspace(
                    0,
                    sequence - 1,
                    steps=sample_count,
                    device=x.device,
                ).round().long().unique()
                reference_rows = []
                candidate_rows = []
                query_reports = []
                q_reference = q.to(torch.bfloat16).float()
                k_reference = k.to(torch.bfloat16).float()
                v_reference = v.to(torch.bfloat16).float()
                flat_packed = state.packed_indices.reshape(-1)
                for query_index_tensor in sample_queries:
                    query_index = int(query_index_tensor)
                    packed_position = int(state.compact_to_packed[query_index])
                    query_block = packed_position // state.packed_block_size
                    allowed_count = int(orbit_sparse_layout.counts[0, 0, query_block])
                    allowed_blocks = orbit_sparse_layout.indices[
                        0, 0, query_block, :allowed_count
                    ].long()
                    key_indices = state.packed_indices[allowed_blocks].reshape(-1)
                    key_indices = key_indices[key_indices >= 0]
                    query_value = q_reference[:, query_index]
                    key_value = k_reference[:, key_indices]
                    logits = torch.einsum(
                        "bhd,bkhd->bhk", query_value, key_value
                    ) / math.sqrt(dim)
                    weights = torch.softmax(logits, dim=-1)
                    reference = torch.einsum(
                        "bhk,bkhd->bhd", weights, v_reference[:, key_indices]
                    )
                    candidate = output[:, query_index].float()
                    reference_rows.append(reference)
                    candidate_rows.append(candidate)
                    query_reports.append(
                        {
                            "query_index": query_index,
                            "query_block": query_block,
                            "allowed_blocks": allowed_count,
                            "allowed_active_keys": int(key_indices.numel()),
                            **self._equivalence_stats(reference, candidate),
                        }
                    )
                self._sparse_validation["self_attention_oracle"] = {
                    "sampled_queries": query_reports,
                    "aggregate": self._equivalence_stats(
                        torch.stack(reference_rows), torch.stack(candidate_rows)
                    ),
                    "reference": "explicit active-token softmax over CWCA-allowed blocks",
                }
        return module.o(output.flatten(2).to(x.dtype))

    @staticmethod
    def _equivalence_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
        reference_f = reference.detach().float()
        candidate_f = candidate.detach().float()
        delta = candidate_f - reference_f
        reference_flat = reference_f.reshape(-1)
        candidate_flat = candidate_f.reshape(-1)
        return {
            "relative_l2": float(
                torch.linalg.vector_norm(delta)
                / (torch.linalg.vector_norm(reference_f) + 1.0e-12)
            ),
            "cosine": float(torch.nn.functional.cosine_similarity(
                reference_flat, candidate_flat, dim=0
            )),
            "mae": float(delta.abs().mean()),
            "max_abs": float(delta.abs().max()),
        }

    def _write_equivalence_trace(
        self,
        *,
        block_index: int,
        block: Any,
        x_input: torch.Tensor,
        normalized: torch.Tensor,
        custom_attention: torch.Tensor,
        custom_attention_residual: torch.Tensor,
        custom_camera: torch.Tensor,
        custom_cross: torch.Tensor,
        custom_action: torch.Tensor,
        custom_ffn: torch.Tensor,
        custom_output: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
        state: _CallState,
    ) -> None:
        """One-shot, same-input native/custom sublayer equivalence diagnostic."""
        trace_path = os.environ.get("MATRIX_JIT_EQUIV_TRACE")
        if not trace_path or getattr(self, "_equivalence_trace_written", False):
            return
        if block_index != 0 or self._state is None or self._state.step_index != 0:
            return

        grid_sizes = kwargs["grid_sizes"]
        memory_length = int(kwargs.get("memory_length", 0))
        self_attn = block.self_attn
        batch, sequence = normalized.shape[:2]
        heads, head_dim = self_attn.num_heads, self_attn.head_dim
        q_raw = self_attn.norm_q(self_attn.q(normalized)).view(
            batch, sequence, heads, head_dim
        )
        k_raw = self_attn.norm_k(self_attn.k(normalized)).view(
            batch, sequence, heads, head_dim
        )
        v_raw = self_attn.v(normalized).view(batch, sequence, heads, head_dim)
        from wan.modules.model import rope_apply_with_indices

        if kwargs.get("predict_latent_idx") is not None:
            prediction = kwargs["predict_latent_idx"]
            if isinstance(prediction, tuple) and len(prediction) == 2:
                prediction = list(range(int(prediction[0]), int(prediction[1])))
        else:
            prediction = list(range(int(grid_sizes[0][0])))
        q_native_rope = rope_apply_with_indices(
            q_raw, grid_sizes, kwargs["freqs"], prediction
        )
        k_native_rope = rope_apply_with_indices(
            k_raw, grid_sizes, kwargs["freqs"], prediction
        )
        q_custom_rope = self._apply_sparse_rope(
            q_raw,
            state.compact_coordinates,
            kwargs["freqs"],
            memory_length,
            kwargs.get("memory_latent_idx"),
            kwargs.get("predict_latent_idx"),
        )
        k_custom_rope = self._apply_sparse_rope(
            k_raw,
            state.compact_coordinates,
            kwargs["freqs"],
            memory_length,
            kwargs.get("memory_latent_idx"),
            kwargs.get("predict_latent_idx"),
        )
        native_attention = self_attn(
            normalized,
            kwargs["seq_lens"],
            grid_sizes,
            kwargs["freqs"],
            memory_length,
            memory_latent_idx=kwargs.get("memory_latent_idx"),
            predict_latent_idx=kwargs.get("predict_latent_idx"),
            fa_version=kwargs.get("fa_version"),
            orbit_sparse_layout=kwargs.get("orbit_sparse_layout"),
        )
        native_attention_residual = x_input + native_attention * modulation[2].squeeze(2)
        native_camera = native_attention_residual
        plucker = kwargs.get("plucker_emb")
        if plucker is not None:
            camera = block.cam_injector_layer2(
                F.silu(block.cam_injector_layer1(plucker))
            ) + plucker
            native_camera = (
                (1.0 + block.cam_scale_layer(camera)) * native_camera
                + block.cam_shift_layer(camera)
            )
        native_cross = block.norm3(native_camera)
        native_cross = native_cross + block.cross_attn(
            native_cross,
            kwargs["context"],
            kwargs.get("context_lens"),
            fa_version=kwargs.get("fa_version"),
        )
        native_action = native_cross
        if block.action_model is not None:
            temporal, height, width = (int(value) for value in grid_sizes[0].tolist())
            native_action = block.action_model(
                native_action.to(block.ffn[0].weight.dtype),
                temporal,
                height,
                width,
                kwargs.get("mouse_cond"),
                kwargs.get("keyboard_cond"),
                kwargs.get("mouse_cond_memory"),
                kwargs.get("keyboard_cond_memory"),
            )
        native_ffn = block.ffn(
            (
                block.norm2(native_action).float()
                * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            ).to(block.ffn[0].weight.dtype)
        )
        native_output = native_action + native_ffn * modulation[5].squeeze(2)
        pairs = {
            "q_rope": (q_native_rope, q_custom_rope),
            "k_rope": (k_native_rope, k_custom_rope),
            "self_attention": (native_attention, custom_attention),
            "self_attention_residual": (
                native_attention_residual, custom_attention_residual
            ),
            "camera": (native_camera, custom_camera),
            "cross_attention_residual": (native_cross, custom_cross),
            "action_module": (native_action, custom_action),
            "ffn": (native_ffn, custom_ffn),
            "block_output": (native_output, custom_output),
        }
        payload = {
            "block_index": block_index,
            "step_index": self._state.step_index,
            "grid_sizes": [int(value) for value in grid_sizes[0].tolist()],
            "memory_length": memory_length,
            "active_spatial": int(state.active_spatial.numel()),
            "full_spatial": int(state.spatial_height * state.spatial_width),
            "qkv_same_source": True,
            "stages": {
                name: self._equivalence_stats(reference, candidate)
                for name, (reference, candidate) in pairs.items()
            },
        }
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self._equivalence_trace_written = True

    def _forward_block(
        self,
        block_index: int,
        block: Any,
        native_forward: Any,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._state is None or self._state.requested_ratio >= 1.0:
            return native_forward(x, *args, **kwargs)
        grid_sizes = kwargs["grid_sizes"]
        memory_length = int(kwargs.get("memory_length", 0))
        layout = kwargs.get("orbit_sparse_layout")
        state = self._ensure_layout(x, grid_sizes, memory_length, layout)
        if block_index == 0 and x.shape[1] == state.full_sequence_length:
            x = x[:, state.compact_indices]
        e_full = kwargs["e"]
        e = e_full[:, state.compact_indices]
        plucker = kwargs.get("plucker_emb")
        if plucker is not None:
            plucker = plucker[:, state.compact_indices]
        x_input = x
        with torch.amp.autocast("cuda", dtype=torch.float32):
            modulation = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            normalized = (
                block.norm1(x).float() * (1 + modulation[1].squeeze(2))
                + modulation[0].squeeze(2)
            ).to(x.dtype)
            attention_out = self._sparse_self_attention(
                block.self_attn,
                normalized,
                state,
                seq_lens=kwargs["seq_lens"],
                freqs=kwargs["freqs"],
                memory_length=memory_length,
                memory_latent_idx=kwargs.get("memory_latent_idx"),
                predict_latent_idx=kwargs.get("predict_latent_idx"),
                fa_version=kwargs.get("fa_version"),
                orbit_sparse_layout=layout,
            )
            x = x + attention_out * modulation[2].squeeze(2)
        attention_residual = x
        if plucker is not None:
            camera = block.cam_injector_layer2(
                F.silu(block.cam_injector_layer1(plucker))
            ) + plucker
            x = (1.0 + block.cam_scale_layer(camera)) * x + block.cam_shift_layer(camera)
        camera_output = x
        x = block.norm3(x)
        x = x + block.cross_attn(
            x, kwargs["context"], kwargs.get("context_lens"),
            fa_version=kwargs.get("fa_version"),
        )
        cross_output = x
        if block.action_model is not None:
            active_spatial = int(state.active_spatial.numel())
            full_spatial = int(state.spatial_height * state.spatial_width)
            action_height = state.spatial_height if active_spatial == full_spatial else 1
            action_width = state.spatial_width if active_spatial == full_spatial else active_spatial
            x = block.action_model(
                x.to(block.ffn[0].weight.dtype),
                state.total_temporal,
                action_height,
                action_width,
                kwargs.get("mouse_cond"),
                kwargs.get("keyboard_cond"),
                kwargs.get("mouse_cond_memory"),
                kwargs.get("keyboard_cond_memory"),
            )
            if (
                os.environ.get("MATRIX_JIT_SPARSE_TRACE")
                and "action_module_oracle" not in self._sparse_validation
            ):
                compact_action_input = cross_output.reshape(
                    cross_output.shape[0],
                    state.total_temporal,
                    active_spatial,
                    cross_output.shape[-1],
                )
                full_action_input = cross_output.new_zeros(
                    cross_output.shape[0],
                    state.total_temporal,
                    full_spatial,
                    cross_output.shape[-1],
                )
                full_action_input[:, :, state.active_spatial, :] = compact_action_input
                full_action_output = block.action_model(
                    full_action_input.reshape(
                        cross_output.shape[0], -1, cross_output.shape[-1]
                    ).to(block.ffn[0].weight.dtype),
                    state.total_temporal,
                    state.spatial_height,
                    state.spatial_width,
                    kwargs.get("mouse_cond"),
                    kwargs.get("keyboard_cond"),
                    kwargs.get("mouse_cond_memory"),
                    kwargs.get("keyboard_cond_memory"),
                )
                gathered_action = full_action_output.reshape(
                    cross_output.shape[0],
                    state.total_temporal,
                    full_spatial,
                    cross_output.shape[-1],
                )[:, :, state.active_spatial, :].reshape_as(x)
                self._sparse_validation["action_module_oracle"] = {
                    "block_index": block_index,
                    "compact_layout": [
                        state.total_temporal, 1, active_spatial
                    ],
                    "native_layout": [
                        state.total_temporal,
                        state.spatial_height,
                        state.spatial_width,
                    ],
                    **self._equivalence_stats(gathered_action, x),
                }
        action_output = x
        y = block.ffn(
            (
                block.norm2(x).float() * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            ).to(block.ffn[0].weight.dtype)
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * modulation[5].squeeze(2)
        self._write_equivalence_trace(
            block_index=block_index,
            block=block,
            x_input=x_input,
            normalized=normalized,
            custom_attention=attention_out,
            custom_attention_residual=attention_residual,
            custom_camera=camera_output,
            custom_cross=cross_output,
            custom_action=action_output,
            custom_ffn=y,
            custom_output=x,
            modulation=modulation,
            kwargs=kwargs,
            state=state,
        )
        return x

    def _forward_head(
        self,
        head: Any,
        x: torch.Tensor,
        e: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._state is None or self._state.requested_ratio >= 1.0:
            return self._native_head_forward(x, e, *args, **kwargs)
        state = self._state
        e_active = e[:, state.compact_indices]
        active_output = self._native_head_forward(x, e_active, *args, **kwargs)
        batch, _, dim = active_output.shape
        active = active_output.reshape(
            batch,
            state.total_temporal,
            state.active_spatial.numel(),
            dim,
        )
        lifted = self._lift_spatial(
            active,
            state.active_spatial,
            state.spatial_height,
            state.spatial_width,
        )
        self._last_lifted_patch_velocity = lifted.detach()
        if os.environ.get("MATRIX_JIT_SPARSE_TRACE"):
            restored = lifted[:, :, state.active_spatial, :]
            self._sparse_validation["anchor_restore"] = {
                "active_tokens": int(state.active_spatial.numel()),
                **self._equivalence_stats(active, restored),
            }
            trace_path = os.environ["MATRIX_JIT_SPARSE_TRACE"]
            payload = {
                "chunk_index": state.chunk_index,
                "step_index": state.step_index,
                "requested_ratio": state.requested_ratio,
                "actual_ratio": float(state.active_spatial.numel())
                / float(state.spatial_height * state.spatial_width),
                "grid_sizes": [
                    state.total_temporal,
                    state.spatial_height,
                    state.spatial_width,
                ],
                "memory_temporal": state.memory_temporal,
                "checks": self._sparse_validation,
            }
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        full_valid = lifted.reshape(batch, -1, dim)
        if state.full_sequence_length > full_valid.shape[1]:
            full = full_valid.new_zeros(
                batch, state.full_sequence_length, dim
            )
            full[:, : full_valid.shape[1]] = full_valid
        else:
            full = full_valid
        actual = float(state.active_spatial.numel()) / float(
            state.spatial_height * state.spatial_width
        )
        if self._current_record is not None:
            self._current_record.update(
                {
                    "active_spatial_tokens": int(state.active_spatial.numel()),
                    "total_spatial_tokens": int(
                        state.spatial_height * state.spatial_width
                    ),
                    "actual_active_ratio": actual,
                    "packed_cwca_block_size": int(state.packed_block_size),
                    "native_cwca_block_size": 128,
                    "gaussian_c": self.gaussian_c,
                    "exact_anchor_restore": True,
                }
            )
        return full

    def finish_model_call(self, prediction: Any) -> Any:
        if self._state is None or self._current_record is None:
            raise RuntimeError("JiT finish called without an active model call")
        tensor = prediction
        if isinstance(tensor, tuple):
            tensor = tensor[-1]
        if isinstance(tensor, list):
            tensor = torch.stack(tensor)
        if not torch.is_tensor(tensor):
            raise TypeError("JiT requires tensor-valued Matrix velocity")
        self._last_velocity = tensor.detach().clone()
        if self._state.requested_ratio >= 1.0:
            height = tensor.shape[-2] // self.model.patch_size[1]
            width = tensor.shape[-1] // self.model.patch_size[2]
            total = height * width
            self._active_spatial = torch.arange(
                total, device=tensor.device, dtype=torch.long
            )
            self._current_record.update(
                {
                    "active_spatial_tokens": total,
                    "total_spatial_tokens": total,
                    "actual_active_ratio": 1.0,
                    "newly_activated_spatial_tokens": int(
                        self._current_record.get(
                            "newly_activated_spatial_tokens", total
                        )
                    ),
                    "exact_anchor_restore": True,
                }
            )
        self._records.append(dict(self._current_record))
        self._state = None
        self._current_record = None
        return prediction

    def abort_model_call(self) -> None:
        self._state = None
        self._current_record = None

    def metadata(self) -> dict[str, Any]:
        ratios = [float(row["actual_active_ratio"]) for row in self._records]
        return {
            "name": self.name,
            "records": list(self._records),
            "actual_active_ratio_mean": float(sum(ratios) / len(ratios)) if ratios else 0.0,
            "actual_active_ratio_by_step": {
                str(step): float(
                    sum(float(row["actual_active_ratio"]) for row in self._records if row["step_index"] == step)
                    / max(1, sum(1 for row in self._records if row["step_index"] == step))
                )
                for step in range(3)
            },
            "cache_contract": {
                "li_denoise_cache_enabled": False,
                "sag_ode": True,
                "importance_guided_token_activation": True,
                "importance_guided_token_deactivation": any(
                    right < left
                    for left, right in zip(self.stage_ratios, self.stage_ratios[1:])
                ),
                "nearest_velocity_lifting": True,
                "gaussian_smoothing": True,
                "exact_anchor_restore": True,
                "deterministic_micro_flow": True,
                "coarse_to_fine_schedule": list(self.stage_ratios),
                "forced_custom_steps": list(self.force_custom_steps),
                "action_camera_worldmark_scoring": False,
                "minimal_video_adaptation": "shared spatial anchors across time",
                "minimal_matrix_adaptation": "active-token ActionModule columns and CWCA topology restriction",
            },
        }
