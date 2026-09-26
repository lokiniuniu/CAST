"""Official-semantics JiT adaptation for Matrix-Game-3.0.

The implementation follows the released JiT FLUX2-Klein pipeline at commit
``818ed8de7333f0c83e3f88a006576036c0f931e8``.  In particular, JiT's stage
boundaries are discretised before inference, activation is monotone, the
initial anchors use the released checkerboard-plus-boundary construction,
new anchors are bridged with fresh fixed Gaussian noise, and condition tokens
stay dense.

Matrix is a video world model rather than an image DiT.  The deliberately
small adaptation is therefore explicit: one spatial anchor set is shared by
all generated latent frames, while R4 memory frames and zero-timestep
conditioning frames remain dense.  Matrix's frozen CWCA topology is evaluated
on that ragged sequence; this is reported as an adapter, not claimed as part
of upstream JiT.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .matrix_jit_spatial_acceleration import (
    MatrixJiTSpatialAcceleration,
    _CallState,
)


OFFICIAL_JIT_COMMIT = "818ed8de7333f0c83e3f88a006576036c0f931e8"
OFFICIAL_JIT_STAGE_BOUNDARIES = (0.4, 0.65, 1.0)
OFFICIAL_JIT_STAGE_DENSITIES = (0.35, 0.62, 1.0)


@dataclass
class _OfficialCallState(_CallState):
    """Per-call layout for dense conditions plus sparse generated tokens."""

    condition_temporal: int = 0
    protected_temporal: int = 0
    generated_temporal: int = 0
    actual_sequence_compute_ratio: float = 1.0


class MatrixJiTOfficialSemanticsAcceleration(MatrixJiTSpatialAcceleration):
    """JiT with official three-step semantics and an explicit Matrix adapter."""

    name = "matrix_jit_official_semantics_35_100_100_cwca"
    allow_anchor_deactivation = False

    def __init__(self, *, gaussian_c: float = 0.4, microflow_relax_steps: int = 3):
        # Official default_4x has stage boundaries [0.4, 0.65, 1.0] and
        # densities [0.35, 0.62, 1.0].  For three steps the released integer
        # boundary rule gives [1, 1, 3], so stage 0.62 is never executed.
        super().__init__(
            stage_ratios=self.discretize_official_schedule(total_steps=3),
            gaussian_c=gaussian_c,
            microflow_relax_steps=microflow_relax_steps,
        )
        self.name = type(self).name

    @staticmethod
    def discretize_official_schedule(total_steps: int) -> tuple[float, ...]:
        """Return the densities selected by JiT's released stage loop."""

        if total_steps <= 0:
            raise ValueError("JiT total_steps must be positive")
        stage_steps = [
            int(total_steps * boundary)
            for boundary in OFFICIAL_JIT_STAGE_BOUNDARIES
        ]
        num_stages = len(OFFICIAL_JIT_STAGE_DENSITIES)

        def ratio_of_stage(stage: int) -> float:
            return OFFICIAL_JIT_STAGE_DENSITIES[num_stages - 1 - stage]

        current_stage = num_stages - 1
        schedule: list[float] = []
        for step in range(total_steps):
            target_stage = 0
            for boundary_index, boundary_step in enumerate(stage_steps):
                if step < boundary_step:
                    target_stage = num_stages - 1 - boundary_index
                    break
            if target_stage < current_stage:
                current_stage = target_stage
            schedule.append(float(ratio_of_stage(current_stage)))
        return tuple(schedule)

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
            self._global_noise_object_id = None
            self._last_lifted_patch_velocity = None
        ratio = self.stage_ratios[step_index]
        self._state = _OfficialCallState(
            chunk_index=int(chunk_index),
            step_index=int(step_index),
            requested_ratio=ratio,
        )
        mode = "jit_sparse_sag_ode" if ratio < 1.0 else "jit_full_exact"
        self._current_record = {
            "chunk_index": int(chunk_index),
            "step_index": int(step_index),
            "requested_active_ratio": ratio,
            "mode": mode,
            "official_stage_boundary_discretization": [1, 1, 3],
        }
        return mode

    def zero_active_prediction(
        self, x: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        del x, timestep
        raise RuntimeError("official JiT activation is monotone and never zero-active")

    def prepare_model_input(
        self, x: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Apply released DMF only to newly activated generated patches."""

        if not isinstance(self._state, _OfficialCallState):
            raise RuntimeError("official JiT model call was not begun")
        state = self._state
        condition_frames = self._condition_frame_count(timestep, x)
        state.condition_temporal = condition_frames
        sigma = self._sigma_from_timestep(timestep)

        if state.step_index == 0:
            # Upstream draws a fresh tensor once and holds it fixed throughout
            # the denoising trajectory.  It is intentionally not x(q0).
            generated_shape = list(x.shape)
            generated_shape[2] -= condition_frames
            if generated_shape[2] <= 0:
                raise RuntimeError("official Matrix JiT found no generated latent frames")
            self._global_noise = torch.randn(
                generated_shape, device=x.device, dtype=x.dtype
            )
            self._global_noise_object_id = id(self._global_noise)
            self._last_input = x.detach().clone()
            self._last_sigma = sigma
            if self._current_record is not None:
                self._current_record.update(
                    {
                        "condition_temporal": condition_frames,
                        "fixed_global_noise_initialized": True,
                        "global_noise_source": "fresh_torch_randn",
                        "dmf_applied": False,
                    }
                )
            return x

        if (
            self._active_spatial is None
            or self._last_velocity is None
            or self._last_input is None
            or self._last_sigma is None
            or self._global_noise is None
            or self._global_noise_object_id is None
        ):
            raise RuntimeError("official JiT transition lacks its preceding state")

        patch_h, patch_w = (int(value) for value in self.model.patch_size[1:])
        height = int(x.shape[-2]) // patch_h
        width = int(x.shape[-1]) // patch_w
        previous_active = self._active_spatial
        target_active = self._select_stage_anchors(
            height=height,
            width=width,
            ratio=state.requested_ratio,
            device=x.device,
        )
        if (
            target_active.numel() < previous_active.numel()
            and not self.allow_anchor_deactivation
        ):
            raise RuntimeError("official JiT cannot deactivate spatial anchors")
        newly_active = target_active[~torch.isin(target_active, previous_active)]

        trace_condition_before = None
        if newly_active.numel() > 0:
            if os.environ.get("MATRIX_JIT_SPARSE_TRACE") and condition_frames:
                trace_condition_before = x[:, :, :condition_frames].detach().clone()
            clean_previous = self._last_input - self._last_sigma * self._last_velocity
            clean_tokens = self._pack_raw_latent(clean_previous)
            noise_tokens = self._pack_raw_latent(self._global_noise)
            current_tokens = self._pack_raw_latent(x)
            generated = slice(condition_frames, None)
            if clean_tokens[:, generated].shape[1] == 0:
                raise RuntimeError("official Matrix JiT found no generated latent frames")
            clean_active = clean_tokens[:, generated, previous_active, :]
            clean_lifted = self._lift_spatial(
                clean_active, previous_active, height, width
            )
            target = (
                (1.0 - sigma) * clean_lifted[:, :, newly_active, :]
                + sigma * noise_tokens[:, :, newly_active, :]
            ).to(current_tokens.dtype)
            weight = (
                1.0
                if self.microflow_relax_steps <= 0
                else 1.0 / float(self.microflow_relax_steps)
            )
            current_tokens[:, generated, newly_active, :] = (
                (1.0 - weight)
                * current_tokens[:, generated, newly_active, :]
                + weight * target
            )
            # Dense condition frames are never reconstructed or perturbed.
            x.copy_(self._unpack_raw_latent(current_tokens, x.shape))

        self._active_spatial = target_active
        self._last_input = x.detach().clone()
        self._last_sigma = sigma
        if self._current_record is not None:
            condition_max_error = None
            if trace_condition_before is not None:
                condition_max_error = float(
                    (
                        x[:, :, :condition_frames].float()
                        - trace_condition_before.float()
                    )
                    .abs()
                    .max()
                    .item()
                )
            self._current_record.update(
                {
                    "condition_temporal": condition_frames,
                    "newly_activated_spatial_tokens": int(newly_active.numel()),
                    "dmf_applied": bool(newly_active.numel()),
                    "dmf_relax_steps": self.microflow_relax_steps,
                    "conditions_modified_by_dmf": False,
                    "condition_dmf_max_abs_error": condition_max_error,
                    "fixed_noise_storage_reused": (
                        id(self._global_noise) == self._global_noise_object_id
                    ),
                }
            )
        return x

    def _initial_sparse_grid(
        self, height: int, width: int, ratio: float, device: torch.device
    ) -> torch.Tensor:
        """Released checkerboard-plus-boundary anchor initialization."""

        total = height * width
        target = int(total * ratio)
        ii, jj = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        all_indices = torch.arange(total, device=device)
        core = (ii % 2 == 0) & (jj % 2 == 0)
        boundary = (
            (ii == 0)
            | (ii == height - 1)
            | (jj == 0)
            | (jj == width - 1)
        )
        indices = all_indices[(core | boundary).flatten()]
        candidate_count = int(indices.numel())
        if indices.numel() < target:
            available = list(set(range(total)) - set(indices.tolist()))
            if available:
                supplement_count = min(target - indices.numel(), len(available))
                supplement = torch.tensor(
                    np.random.choice(available, supplement_count, replace=False),
                    device=device,
                )
                indices = torch.cat([indices, supplement])
        elif indices.numel() > target:
            indices = indices[torch.randperm(indices.numel(), device=device)[:target]]
        if self._current_record is not None:
            selected_mask = torch.zeros(total, dtype=torch.bool, device=device)
            selected_mask[indices] = True
            self._current_record.update(
                {
                    "checkerboard_boundary_candidates": candidate_count,
                    "checkerboard_core_selected": int((selected_mask & core.flatten()).sum()),
                    "checkerboard_boundary_selected": int(
                        (selected_mask & boundary.flatten()).sum()
                    ),
                }
            )
        return indices.long()

    def _select_stage_anchors(
        self, *, height: int, width: int, ratio: float, device: torch.device
    ) -> torch.Tensor:
        """Released variance densification, with no deactivation branch."""

        total = height * width
        target = int(total * ratio)
        if self._active_spatial is None:
            return self._initial_sparse_grid(height, width, ratio, device)
        current = self._active_spatial
        if target < current.numel():
            raise RuntimeError(
                "official JiT density schedule must be monotonically non-decreasing"
            )
        if target == current.numel():
            return current
        if self._last_lifted_patch_velocity is None:
            raise RuntimeError("official JiT densification lacks lifted velocity")
        velocity = self._last_lifted_patch_velocity.float()
        batch, temporal, spatial, dim = velocity.shape
        if spatial != total:
            raise RuntimeError("JiT lifted velocity does not match the spatial grid")
        image = velocity.reshape(
            batch, temporal, height, width, dim
        ).permute(0, 1, 4, 2, 3).reshape(batch * temporal, dim, height, width)
        mean = F.avg_pool2d(image, 3, stride=1, padding=1)
        variance = F.avg_pool2d(image.square(), 3, stride=1, padding=1) - mean.square()
        importance = variance.mean(dim=1).reshape(
            batch, temporal, height, width
        ).mean(dim=(0, 1)).flatten()
        importance = (importance - importance.min()) / (
            importance.max() - importance.min() + 1.0e-8
        )
        inactive = torch.ones(total, dtype=torch.bool, device=device)
        inactive[current] = False
        candidates = torch.arange(total, device=device)[inactive]
        add = target - int(current.numel())
        if add >= candidates.numel():
            selected = candidates
        else:
            probabilities = importance[candidates] / (
                importance[candidates].sum() + 1.0e-8
            )
            selected = candidates[torch.topk(probabilities, add).indices]
        return torch.cat([current, selected]).long()

    def _ensure_layout(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        memory_length: int,
        orbit_sparse_layout: Any,
    ) -> _OfficialCallState:
        """Compact only generated tokens; preserve Memory/conditions densely."""

        if not isinstance(self._state, _OfficialCallState):
            raise RuntimeError("official JiT sparse layout lacks a model call")
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
        active = self._active_spatial
        spatial = height * width
        memory_t = int(memory_length)
        protected_t = memory_t + int(state.condition_temporal)
        if protected_t > temporal:
            raise RuntimeError("dense JiT prefix exceeds Matrix temporal grid")
        generated_t = temporal - protected_t
        if generated_t <= 0:
            raise RuntimeError("official Matrix JiT found no generated tokens")

        dense_times = torch.arange(protected_t, device=x.device)
        dense_space = torch.arange(spatial, device=x.device)
        if protected_t:
            dense_compact = (
                dense_times[:, None] * spatial + dense_space[None, :]
            ).reshape(-1)
            dense_y = (dense_space // width).repeat(protected_t)
            dense_x = (dense_space % width).repeat(protected_t)
            dense_coords = torch.stack(
                [dense_times.repeat_interleave(spatial), dense_y, dense_x], dim=-1
            )
        else:
            dense_compact = torch.empty(0, device=x.device, dtype=torch.long)
            dense_coords = torch.empty((0, 3), device=x.device, dtype=torch.long)

        generated_times = torch.arange(protected_t, temporal, device=x.device)
        generated_compact = (
            generated_times[:, None] * spatial + active[None, :]
        ).reshape(-1)
        generated_coords = torch.stack(
            [
                generated_times.repeat_interleave(active.numel()),
                (active // width).repeat(generated_t),
                (active % width).repeat(generated_t),
            ],
            dim=-1,
        )
        compact = torch.cat([dense_compact, generated_compact]).long()
        coords = torch.cat([dense_coords, generated_coords]).long()

        state.active_spatial = active
        state.spatial_height = height
        state.spatial_width = width
        state.total_temporal = temporal
        state.memory_temporal = memory_t
        state.protected_temporal = protected_t
        state.generated_temporal = generated_t
        state.full_sequence_length = int(x.shape[1])
        state.compact_indices = compact
        state.compact_coordinates = coords
        state.actual_sequence_compute_ratio = float(compact.numel()) / float(
            temporal * spatial
        )

        if orbit_sparse_layout is not None:
            tt, th, tw = (
                int(value) for value in orbit_sparse_layout.block_shape
            )
            nh = math.ceil(height / th)
            nw = math.ceil(width / tw)
            current_t = temporal - memory_t
            t_coord, h_coord, w_coord = coords.unbind(dim=1)
            grouped = bool(getattr(orbit_sparse_layout, "protected_current", False))
            if grouped:
                memory_blocks = math.ceil(memory_t / tt) * nh * nw if memory_t else 0
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
                total_blocks = memory_blocks + math.ceil(current_t / tt) * nh * nw
            else:
                block_id = (
                    (t_coord // tt) * nh * nw
                    + (h_coord // th) * nw
                    + w_coord // tw
                )
                total_blocks = math.ceil(temporal / tt) * nh * nw
            if orbit_sparse_layout.indices.shape[-2] != total_blocks:
                raise RuntimeError(
                    "official JiT/CWCA block mismatch: "
                    f"layout={orbit_sparse_layout.indices.shape[-2]} "
                    f"runtime={total_blocks}"
                )
            per_block = [
                torch.nonzero(block_id == index).flatten()
                for index in range(total_blocks)
            ]
            maximum_active = max(int(values.numel()) for values in per_block)
            packed_size = 1 << max(0, maximum_active - 1).bit_length()
            packed = torch.full(
                (total_blocks, packed_size),
                -1,
                device=x.device,
                dtype=torch.long,
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

        if self._current_record is not None:
            self._current_record.update(
                {
                    "memory_temporal": memory_t,
                    "condition_temporal": int(state.condition_temporal),
                    "protected_temporal": protected_t,
                    "generated_temporal": generated_t,
                    "dense_protected_tokens": protected_t * spatial,
                    "sparse_generated_tokens": generated_t * int(active.numel()),
                    "actual_sequence_compute_ratio": state.actual_sequence_compute_ratio,
                    "compact_token_sequence_ratio": state.actual_sequence_compute_ratio,
                }
            )
        if os.environ.get("MATRIX_JIT_SPARSE_TRACE"):
            self._sparse_validation["layout"] = {
                "compact_count": int(compact.numel()),
                "expected_compact_count": int(
                    protected_t * spatial + generated_t * active.numel()
                ),
                "compact_unique": int(torch.unique(compact).numel()),
                "memory_dense": memory_t == 0
                or bool(torch.all(torch.isin(
                    torch.arange(memory_t * spatial, device=x.device), compact
                ))),
                "condition_dense": state.condition_temporal == 0
                or bool(torch.all(torch.isin(
                    torch.arange(
                        memory_t * spatial,
                        protected_t * spatial,
                        device=x.device,
                    ),
                    compact,
                ))),
                "generated_only_sparse": True,
            }
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
        """Apply Matrix RoPE using each ragged token's explicit coordinates."""

        heads, half = value.shape[2], value.shape[3] // 2
        split = [half - 2 * (half // 3), half // 3, half // 3]
        axes = freqs.split(split, dim=2 if freqs.dim() == 3 else 1)
        raw_t = coords[:, 0].long()
        temporal_count = int(raw_t.max().item()) + 1
        memory_count = int(memory_length)
        mapped_t = torch.empty_like(raw_t)
        if memory_count:
            memory_indices = (
                [int(item) for item in memory_latent_idx]
                if memory_latent_idx is not None
                else list(range(memory_count))
            )
            if len(memory_indices) < memory_count:
                raise RuntimeError("Matrix JiT memory RoPE indices are incomplete")
            memory_values = torch.tensor(memory_indices, device=coords.device)
            memory_mask = raw_t < memory_count
            mapped_t[memory_mask] = memory_values[raw_t[memory_mask]]
        else:
            memory_mask = torch.zeros_like(raw_t, dtype=torch.bool)
        current_count = temporal_count - memory_count
        if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
            predict_indices = list(
                range(int(predict_latent_idx[0]), int(predict_latent_idx[1]))
            )
        elif predict_latent_idx is not None:
            predict_indices = [int(item) for item in predict_latent_idx]
        else:
            predict_indices = list(range(current_count))
        if len(predict_indices) < current_count:
            raise RuntimeError("Matrix JiT prediction RoPE indices are incomplete")
        predict_values = torch.tensor(predict_indices, device=coords.device)
        mapped_t[~memory_mask] = predict_values[raw_t[~memory_mask] - memory_count]
        indices = (mapped_t, coords[:, 1].long(), coords[:, 2].long())
        if freqs.dim() == 3:
            phase = torch.cat(
                [
                    axis[:, index, :].permute(1, 0, 2)
                    for axis, index in zip(axes, indices)
                ],
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

    def _sparse_self_attention(
        self,
        module: Any,
        x: torch.Tensor,
        state: _CallState,
        *,
        seq_lens: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        # The compact sequence is the complete legal key set.  Passing the
        # pre-compaction length can let dense attention read beyond it.
        compact_lens = torch.full_like(seq_lens, int(x.shape[1]))
        return super()._sparse_self_attention(
            module, x, state, seq_lens=compact_lens, **kwargs
        )

    def _expand_compact_hidden(
        self, x: torch.Tensor, state: _OfficialCallState
    ) -> torch.Tensor:
        """Lift generated hidden tokens while copying dense prefixes exactly."""

        batch, _, dim = x.shape
        spatial = state.spatial_height * state.spatial_width
        dense_count = state.protected_temporal * spatial
        active_count = int(state.active_spatial.numel())
        expected = dense_count + state.generated_temporal * active_count
        if x.shape[1] != expected:
            raise RuntimeError(
                f"ragged Matrix JiT sequence has {x.shape[1]} tokens, expected {expected}"
            )
        full = x.new_zeros(batch, state.total_temporal, spatial, dim)
        if state.protected_temporal:
            full[:, : state.protected_temporal] = x[:, :dense_count].reshape(
                batch, state.protected_temporal, spatial, dim
            )
        generated_active = x[:, dense_count:].reshape(
            batch, state.generated_temporal, active_count, dim
        )
        full[:, state.protected_temporal :] = self._lift_spatial(
            generated_active,
            state.active_spatial,
            state.spatial_height,
            state.spatial_width,
        )
        return full.reshape(batch, state.total_temporal * spatial, dim)

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
        state = self._ensure_layout(
            x,
            kwargs["grid_sizes"],
            int(kwargs.get("memory_length", 0)),
            kwargs.get("orbit_sparse_layout"),
        )
        if block_index == 0 and x.shape[1] == state.full_sequence_length:
            x = x[:, state.compact_indices]
        e = kwargs["e"][:, state.compact_indices]
        plucker = kwargs.get("plucker_emb")
        if plucker is not None:
            plucker = plucker[:, state.compact_indices]
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
                memory_length=int(kwargs.get("memory_length", 0)),
                memory_latent_idx=kwargs.get("memory_latent_idx"),
                predict_latent_idx=kwargs.get("predict_latent_idx"),
                fa_version=kwargs.get("fa_version"),
                orbit_sparse_layout=kwargs.get("orbit_sparse_layout"),
            )
            x = x + attention_out * modulation[2].squeeze(2)
        if plucker is not None:
            camera = block.cam_injector_layer2(
                F.silu(block.cam_injector_layer1(plucker))
            ) + plucker
            x = (1.0 + block.cam_scale_layer(camera)) * x + block.cam_shift_layer(camera)
        x = block.norm3(x)
        x = x + block.cross_attn(
            x,
            kwargs["context"],
            kwargs.get("context_lens"),
            fa_version=kwargs.get("fa_version"),
        )
        if block.action_model is not None:
            # Matrix's ActionModule assumes a rectangular T×H×W grid.  Run it
            # on a reconstructed grid, then restore only active/dense tokens.
            full_action_input = self._expand_compact_hidden(x, state)
            full_action_output = block.action_model(
                full_action_input.to(block.ffn[0].weight.dtype),
                state.total_temporal,
                state.spatial_height,
                state.spatial_width,
                kwargs.get("mouse_cond"),
                kwargs.get("keyboard_cond"),
                kwargs.get("mouse_cond_memory"),
                kwargs.get("keyboard_cond_memory"),
            )
            x = full_action_output[:, state.compact_indices]
        y = block.ffn(
            (
                block.norm2(x).float() * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            ).to(block.ffn[0].weight.dtype)
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * modulation[5].squeeze(2)
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
        if not isinstance(self._state, _OfficialCallState):
            raise RuntimeError("official JiT head received a legacy layout")
        state = self._state
        active_output = self._native_head_forward(
            x, e[:, state.compact_indices], *args, **kwargs
        )
        batch, _, dim = active_output.shape
        spatial = state.spatial_height * state.spatial_width
        dense_count = state.protected_temporal * spatial
        full = active_output.new_zeros(
            batch, state.total_temporal, spatial, dim
        )
        if state.protected_temporal:
            full[:, : state.protected_temporal] = active_output[:, :dense_count].reshape(
                batch, state.protected_temporal, spatial, dim
            )
        generated_active = active_output[:, dense_count:].reshape(
            batch,
            state.generated_temporal,
            state.active_spatial.numel(),
            dim,
        )
        generated_lifted = self._lift_spatial(
            generated_active,
            state.active_spatial,
            state.spatial_height,
            state.spatial_width,
        )
        full[:, state.protected_temporal :] = generated_lifted
        self._last_lifted_patch_velocity = generated_lifted.detach()

        if os.environ.get("MATRIX_JIT_SPARSE_TRACE"):
            restored = generated_lifted[:, :, state.active_spatial, :]
            self._sparse_validation["anchor_restore"] = {
                "active_tokens": int(state.active_spatial.numel()),
                **self._equivalence_stats(generated_active, restored),
            }
            trace_path = os.environ["MATRIX_JIT_SPARSE_TRACE"]
            payload = {
                "implementation": self.name,
                "official_commit": OFFICIAL_JIT_COMMIT,
                "chunk_index": state.chunk_index,
                "step_index": state.step_index,
                "requested_ratio": state.requested_ratio,
                "grid_sizes": [
                    state.total_temporal,
                    state.spatial_height,
                    state.spatial_width,
                ],
                "checks": self._sparse_validation,
            }
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

        full_valid = full.reshape(batch, -1, dim)
        if state.full_sequence_length > full_valid.shape[1]:
            padded = full_valid.new_zeros(
                batch, state.full_sequence_length, dim
            )
            padded[:, : full_valid.shape[1]] = full_valid
            full_valid = padded
        actual = float(state.active_spatial.numel()) / float(spatial)
        if self._current_record is not None:
            self._current_record.update(
                {
                    "active_spatial_tokens": int(state.active_spatial.numel()),
                    "total_spatial_tokens": spatial,
                    "actual_active_ratio": actual,
                    "actual_sequence_compute_ratio": state.actual_sequence_compute_ratio,
                    "compact_token_sequence_ratio": state.actual_sequence_compute_ratio,
                    "packed_cwca_block_size": int(state.packed_block_size),
                    "native_cwca_block_size": 128,
                    "gaussian_c": self.gaussian_c,
                    "exact_anchor_restore": True,
                    "memory_dense": True,
                    "condition_dense": True,
                    "generated_only_sparse": True,
                    "action_module_layout_adapter": "dense_lift_then_exact_anchor_gather",
                }
            )
        return full_valid

    def finish_model_call(self, prediction: Any) -> Any:
        if self._current_record is not None:
            self._current_record.setdefault("actual_sequence_compute_ratio", 1.0)
            self._current_record.setdefault("compact_token_sequence_ratio", 1.0)
            self._current_record.setdefault("memory_dense", True)
            self._current_record.setdefault("condition_dense", True)
            self._current_record.setdefault("generated_only_sparse", True)
        return super().finish_model_call(prediction)

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["name"] = self.name
        sequence_ratios = [
            float(row.get("actual_sequence_compute_ratio", 1.0))
            for row in self._records
        ]
        payload["actual_sequence_compute_ratio_mean"] = (
            float(sum(sequence_ratios) / len(sequence_ratios))
            if sequence_ratios
            else 0.0
        )
        payload["actual_sequence_compute_ratio_by_step"] = {
            str(step): float(
                sum(
                    float(row.get("actual_sequence_compute_ratio", 1.0))
                    for row in self._records
                    if row["step_index"] == step
                )
                / max(
                    1,
                    sum(1 for row in self._records if row["step_index"] == step),
                )
            )
            for step in range(3)
        }
        # Retain the historical key above for runner compatibility, while the
        # qualified key makes clear that this is a token-count ratio rather
        # than a wall-clock or FLOP ratio (the Matrix ActionModule bridge is
        # deliberately dense).
        payload["compact_token_sequence_ratio_mean"] = payload[
            "actual_sequence_compute_ratio_mean"
        ]
        payload["compact_token_sequence_ratio_by_step"] = payload[
            "actual_sequence_compute_ratio_by_step"
        ]
        payload["cache_contract"].update(
            {
                "official_reference_repository": "Wenhao-Sun77/Just-in-Time",
                "official_reference_commit": OFFICIAL_JIT_COMMIT,
                "official_stage_boundaries": list(OFFICIAL_JIT_STAGE_BOUNDARIES),
                "official_stage_densities": list(OFFICIAL_JIT_STAGE_DENSITIES),
                "official_three_step_schedule": list(self.stage_ratios),
                "checkerboard_boundary_initialization": True,
                "monotonic_activation_only": True,
                "importance_guided_token_activation_configured": True,
                # In a three-step run q0 -> q1 jumps from 35% to 100%, so all
                # inactive anchors are added and the computed importance cannot
                # change which anchors are selected.
                "importance_guided_token_activation_effective": False,
                "full_densification_selects_all_inactive": True,
                "importance_guided_token_deactivation": False,
                "fresh_fixed_global_noise": True,
                "memory_tokens_dense": True,
                "zero_timestep_condition_tokens_dense": True,
                "generated_tokens_only_sparse": True,
                "shared_spatial_anchors_across_generated_time": True,
                "cwca_matrix_adapter": True,
                "jit_anchor_selector_uses_action_camera": False,
                "frozen_cwca_uses_world_trajectory": True,
                "action_module_matrix_adapter": "dense_lift_then_exact_anchor_gather",
                "minimal_video_adaptation": (
                    "shared spatial anchors across generated latent frames"
                ),
                "minimal_matrix_adaptation": (
                    "dense Memory/condition prefix plus CWCA and rectangular "
                    "ActionModule bridges"
                ),
            }
        )
        return payload


class MatrixJiTOfficialLayoutNonMonotone9010100(
    MatrixJiTOfficialSemanticsAcceleration
):
    """90/10/100 schedule on the audited Matrix JiT layout.

    This is deliberately labelled a non-monotone diagnostic rather than
    official JiT: the released coarse-to-fine schedule never deactivates
    anchors.  At q1 we retain the 10% q0 anchors with the largest released
    local velocity-variance importance; q2 restores the full grid.  All Matrix
    layout, lifting, DMF, condition, RoPE, CWCA, and ActionModule adapters are
    otherwise inherited unchanged from the audited implementation above.
    """

    name = "matrix_jit_official_layout_nonmonotone_90_10_100_cwca"
    allow_anchor_deactivation = True

    def __init__(self, *, gaussian_c: float = 0.4, microflow_relax_steps: int = 3):
        super().__init__(
            gaussian_c=gaussian_c,
            microflow_relax_steps=microflow_relax_steps,
        )
        self.stage_ratios = (0.90, 0.10, 1.00)
        self.name = type(self).name

    def _select_stage_anchors(
        self, *, height: int, width: int, ratio: float, device: torch.device
    ) -> torch.Tensor:
        total = height * width
        target = int(total * ratio)
        if self._active_spatial is None:
            return self._initial_sparse_grid(height, width, ratio, device)
        current = self._active_spatial
        if target >= current.numel():
            # Reuse the audited official densification path.  At q2 all
            # inactive anchors are restored, so importance cannot change the
            # selected set.
            return super()._select_stage_anchors(
                height=height, width=width, ratio=ratio, device=device
            )
        if self._last_lifted_patch_velocity is None:
            raise RuntimeError("90/10/100 deactivation lacks lifted velocity")
        velocity = self._last_lifted_patch_velocity.float()
        batch, temporal, spatial, dim = velocity.shape
        if spatial != total:
            raise RuntimeError("90/10/100 velocity does not match spatial grid")
        image = velocity.reshape(
            batch, temporal, height, width, dim
        ).permute(0, 1, 4, 2, 3).reshape(batch * temporal, dim, height, width)
        mean = F.avg_pool2d(image, 3, stride=1, padding=1)
        variance = F.avg_pool2d(image.square(), 3, stride=1, padding=1) - mean.square()
        importance = variance.mean(dim=1).reshape(
            batch, temporal, height, width
        ).mean(dim=(0, 1)).flatten()
        selected = current[torch.topk(importance[current], k=target).indices]
        if self._current_record is not None:
            self._current_record.update(
                {
                    "deactivation_policy": "released_local_velocity_variance_topk",
                    "deactivated_spatial_tokens": int(current.numel() - target),
                }
            )
        return selected.long()

    def begin_model_call(self, **kwargs: Any) -> str:
        mode = super().begin_model_call(**kwargs)
        if self._current_record is not None:
            self._current_record.update(
                {
                    "schedule_semantics": "operator_requested_nonmonotone_diagnostic",
                    "official_stage_boundary_discretization": None,
                }
            )
        return mode

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["name"] = self.name
        contract = payload["cache_contract"]
        contract.update(
            {
                "official_three_step_schedule": [0.35, 1.0, 1.0],
                "operator_requested_schedule": list(self.stage_ratios),
                "schedule_is_official": False,
                "monotonic_activation_only": False,
                "importance_guided_token_activation_effective": True,
                "importance_guided_token_deactivation": True,
                "full_densification_selects_all_inactive": True,
                "nonmonotone_deactivation_policy": (
                    "released_local_velocity_variance_topk_within_q0_anchors"
                ),
            }
        )
        return payload
