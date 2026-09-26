"""Closed-loop control-response budget variants of Matrix CWCA.

Only the per-query integer budget changes here.  The inherited CWCA selector
continues to rank keys with its native pooled Q/K score, protects the released
LI local stencil, and calls the existing sparse attention implementation.
"""

from __future__ import annotations

from bisect import bisect_left
import math
import os
from typing import Any

import torch

from .qk_uncertainty_tangent_attention import (
    MatrixChronologicalWorldlineCurvatureAttentionCompiler,
    MatrixCurvatureTemperedUncertaintyAttentionCompiler,
    _tile_visual_tensor,
)
from .control_response_tile_kernel import (
    fused_action_response_tiles,
    fused_camera_response_tiles,
)


class MatrixControlResponseCWCAAttentionCompiler(
    MatrixChronologicalWorldlineCurvatureAttentionCompiler
):
    """Allocate CWCA's fixed worldline edge budget from previous-layer response."""

    MODES = frozenset({"response_only", "closed_loop"})

    def __init__(
        self,
        mode: str,
        *,
        response_weight: float = 1.0,
        feedback_form: str = "additive",
        curvature_temporal_pooling: str = "mean",
        curvature_softmax_temperature: float = 1.0,
        fixed_query_budget: bool = False,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown control-response CWCA mode {mode!r}")
        super().__init__(
            curvature_temporal_pooling=curvature_temporal_pooling,
            curvature_softmax_temperature=curvature_softmax_temperature,
        )
        self.mode = mode
        self.response_weight = float(response_weight)
        self.feedback_form = str(feedback_form)
        self.fixed_query_budget = bool(fixed_query_budget)
        if self.feedback_form not in {
            "additive", "centered_multiplicative", "centered_inverse"
        }:
            raise ValueError(f"unknown control-response feedback {feedback_form!r}")
        if self.response_weight < 0.0:
            raise ValueError("control-response weight must be non-negative")
        if mode == "response_only" and self.response_weight != 1.0:
            raise ValueError("response-only CWCA has no curvature mixing weight")
        suffix = "response_only" if mode == "response_only" else "closed_loop"
        self.name = f"matrix_{suffix}_cwca_attention_compiler"
        self.selection_standard = f"{suffix}_fixed_worldline_budget_native_qk"
        self.cache_key_suffix = f"{suffix}_cwca_native_qk"
        self.trajectory_complexity_kind = suffix
        if self.curvature_temporal_pooling != "mean":
            pooling = self.curvature_temporal_pooling
            self.name = f"{self.name}_{pooling}_curvature_pool"
            self.selection_standard = (
                f"{self.selection_standard}_{pooling}_curvature_pool"
            )
            self.cache_key_suffix = (
                f"{self.cache_key_suffix}_{pooling}_curvature_pool_"
                f"temperature_{self.curvature_softmax_temperature:g}"
            )
        self._bootstrap_degrees: torch.Tensor | None = None
        self._curvature_matrix: torch.Tensor | None = None
        self._current_temporal_offset = 0
        self._responses: dict[int, torch.Tensor] = {}
        self._raw_responses: dict[int, torch.Tensor] = {}
        self._frame_responses: dict[int, torch.Tensor] = {}
        self._response_has_action: dict[int, bool] = {}
        self._frame_selection_energy_history: dict[int, list[torch.Tensor]] = {}
        self._frame_selection_payload_cache: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        self._lagged_frame_selection_payloads: dict[int, dict[str, Any]] = {}
        self._lagged_call_period_payload: dict[str, Any] | None = None
        self._call_period_energy_history: list[torch.Tensor] = []
        self._lagged_batch_finalize_records: list[dict[str, Any]] = []
        self._lagged_eligible_call_index = -1
        self.fine_frame_response_enabled = False
        self.curvature_only_response_gating_enabled = False
        self.fine_only_response_reduction_enabled = False
        self.fine_frame_response_capture_active = True
        self._response_observation_captured_layers = 0
        self._response_observation_skipped_layers = 0
        self.runtime_optimized = False
        self.temporal_history_response_enabled = False
        self.temporal_history_response_gated = False
        self.temporal_history_response_lagged_batch = False
        self.lagged_call_period_router_enabled = False
        self.temporal_history_capture_active = False
        self._layer_index = 0
        self._compact_layer_index: int | None = None
        self._call_index = -1
        self._budget_records: list[dict[str, Any]] = []
        self._compact_budget_records: list[dict[str, Any]] = []
        self._pending_budget_record: dict[str, Any] | None = None
        self._response_records: list[dict[str, Any]] = []
        self._compact_active_response_calls = 0
        self._diagnostic_calls = int(
            os.environ.pop("WORLDMARK_CONTROL_RESPONSE_DIAGNOSTIC_CALLS", "0")
        )
        self._sensitivity_records: list[dict[str, Any]] = []
        self._response_reduction = {
            "mode": "pytorch",
            "camera_fused_calls": 0,
            "action_fused_calls": 0,
        }

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if self._row_degrees is None or self._profile_current_block_curvature is None:
            raise RuntimeError("control-response CWCA did not receive CWCA bootstrap")
        self._bootstrap_degrees = self._row_degrees.clone()
        if self.fixed_query_budget:
            # Released Light Interaction uses one discrete visual-key degree
            # for every query.  Retain the CWCA response provider and native
            # Q/K ranking, while disabling only per-query redistribution.
            self._bootstrap_degrees.fill_(int(self._degree))
            self._row_degrees = self._bootstrap_degrees.clone()
        current_blocks = self._num_blocks - self._profile_memory_blocks
        if self._spatial_blocks is None or current_blocks % self._spatial_blocks:
            raise RuntimeError("control-response CWCA current layout is not rectangular")
        self._curvature_matrix = self._profile_current_block_curvature.reshape(
            current_blocks // self._spatial_blocks, self._spatial_blocks
        ).clone()
        tt = int(self._block_shape[0])
        self._current_temporal_offset = (
            math.ceil(int(layout.memory_length) / tt) * tt
            - int(layout.memory_length)
        )
        self._responses.clear()
        self._raw_responses.clear()
        self._frame_responses.clear()
        self._response_has_action.clear()
        self._layer_index = 0
        self._compact_layer_index = None
        return compiled, report

    def begin_compact_layer(self, layer_index: int) -> None:
        """Enter FrameWeave's direct-QKV path without invoking self_attn hooks."""

        layer_index = int(layer_index)
        if self._compact_layer_index is not None:
            raise RuntimeError("nested compact control-response layer")
        self._compact_layer_index = layer_index
        if layer_index == 0:
            self._call_index += 1
            self._responses.clear()
            self._raw_responses.clear()
            self._frame_responses.clear()
            self._response_has_action.clear()
        self._layer_index = layer_index

    def end_compact_layer(self, layer_index: int) -> None:
        layer_index = int(layer_index)
        if self._compact_layer_index != layer_index:
            raise RuntimeError("compact control-response layer chronology mismatch")
        self._compact_layer_index = None

    def observing_compact_layer(self, layer_index: int) -> bool:
        return self._compact_layer_index == int(layer_index)

    def enable_fine_frame_response(self) -> None:
        """Enable the expensive per-frame observer only for its owning variant."""

        self.fine_frame_response_enabled = True

    def enable_curvature_only_response_gating(self) -> None:
        """Allow the owning solver to disable provably unused observers."""

        if self.response_weight != 0.0:
            raise RuntimeError(
                "response gating requires curvature-only attention budgeting"
            )
        if not self.fine_frame_response_enabled:
            raise RuntimeError("response gating requires fine frame response")
        self.curvature_only_response_gating_enabled = True
        self.fine_frame_response_capture_active = False

    def enable_fine_only_response_reduction(self) -> None:
        """Skip coarse response reductions that no downstream path consumes."""

        if not self.curvature_only_response_gating_enabled:
            raise RuntimeError(
                "fine-only response reduction requires curvature-only gating"
            )
        if self.response_weight != 0.0:
            raise RuntimeError(
                "fine-only response reduction requires response weight zero"
            )
        self.fine_only_response_reduction_enabled = True

    def set_fine_frame_response_capture(self, active: bool) -> None:
        if not self.curvature_only_response_gating_enabled:
            raise RuntimeError("fine response capture gating is not enabled")
        self.fine_frame_response_capture_active = bool(active)

    def response_observation_active(self) -> bool:
        return bool(
            not self.curvature_only_response_gating_enabled
            or self.fine_frame_response_capture_active
        )

    def enable_runtime_optimization(self) -> None:
        """Defer valid-response failures without synchronizing every layer."""

        self.runtime_optimized = True

    def enable_temporal_history_response(self) -> None:
        """Retain raw native-tile responses for the owning dynamic variant."""

        self.temporal_history_response_enabled = True
        self.temporal_history_response_gated = False
        self.temporal_history_response_lagged_batch = False
        self.temporal_history_capture_active = True

    def enable_gated_temporal_history_response(self) -> None:
        """Retain raw responses only while an eligible q0 weave is active."""

        self.temporal_history_response_enabled = True
        self.temporal_history_response_gated = True
        self.temporal_history_response_lagged_batch = False
        self.temporal_history_capture_active = False

    def enable_lagged_batched_temporal_history_response(self) -> None:
        """Build next-q0 selection payloads with one batched D2H transfer.

        Immediate same-layer thinning materializes a scalar decision in every
        Transformer layer.  This mode instead retains the same raw
        camera/action response, batches every source layer after an eligible
        q0 call, and applies the resulting payload to the next eligible q0
        call.  The one-call lag keeps the rule causal while removing the
        per-layer device synchronization from the model critical path.
        """

        self.temporal_history_response_enabled = True
        self.temporal_history_response_gated = True
        self.temporal_history_response_lagged_batch = True
        self.temporal_history_capture_active = False

    def enable_lagged_call_period_router(self) -> None:
        """Enable one-call-lagged routing between nested temporal lattices."""

        self.enable_lagged_batched_temporal_history_response()
        self.lagged_call_period_router_enabled = True

    def set_temporal_history_capture(self, active: bool) -> None:
        if not self.temporal_history_response_gated:
            raise RuntimeError("temporal-history capture gating is not enabled")
        self.temporal_history_capture_active = bool(active)

    def previous_layer_frame_selection_signal(
        self,
        *,
        layer_index: int,
        current_temporal: int,
    ) -> dict[str, Any]:
        """Expose the causal Closed-Loop signal for complete-frame selection.

        FrameWeave selects the frames for layer ``l`` before entering that
        layer's compact attention path.  At that point Closed-Loop CWCA has
        already measured the camera/action response of layer ``l-1``.  This
        method exposes the *same* curvature-plus-response allocation signal
        used by CWCA, reduced only over spatial worldlines.  It does not rank
        frames, change an attention budget, or inspect a future layer.

        Matrix's native whole-sequence tiling classifies the partial
        Memory|Current block as Memory.  Consequently the returned cells begin
        at ``current_temporal_offset`` (three for M5/tt4); callers must retain
        that offset rather than pretending Current starts on a tile boundary.
        """

        layer_index = int(layer_index)
        current_temporal = int(current_temporal)
        if self.mode != "closed_loop":
            raise RuntimeError(
                "response-guided frame selection requires Closed-Loop CWCA"
            )
        if layer_index <= 0:
            raise RuntimeError("layer 0 has no causal previous-layer response")
        if current_temporal <= 0:
            raise RuntimeError("frame selection requires Current latent frames")
        if self._curvature_matrix is None:
            raise RuntimeError("Closed-Loop CWCA has no Current curvature layout")
        response = self._responses.get(layer_index - 1)
        if response is None:
            raise RuntimeError(
                f"layer {layer_index} has no response from layer {layer_index - 1}"
            )
        curvature = self._curvature_matrix.to(response.device, response.dtype)
        if tuple(response.shape) != tuple(curvature.shape):
            raise RuntimeError("Closed-Loop response/curvature layouts differ")
        eps = torch.finfo(curvature.dtype).eps
        normalized_curvature = curvature / curvature.mean(
            dim=0, keepdim=True
        ).clamp_min(eps)
        if self.feedback_form == "additive":
            allocation_signal = (
                normalized_curvature + self.response_weight * response
            )
        else:
            sign = -1.0 if self.feedback_form == "centered_inverse" else 1.0
            allocation_signal = normalized_curvature * torch.exp(
                sign * self.response_weight * (response - 1.0)
            )
        # This is exactly the monotonic weighting transform used by the CWCA
        # largest-remainder allocator.  Only spatial reduction is new.
        cell_scores = (1.0 + torch.log1p(allocation_signal)).mean(dim=1)
        tt = int(self._block_shape[0])
        capacity = self._current_temporal_offset + int(cell_scores.numel()) * tt
        if current_temporal > capacity:
            raise RuntimeError(
                "Current latent count exceeds Closed-Loop response capacity"
            )
        if not bool(torch.isfinite(cell_scores).all()) or bool(
            torch.any(cell_scores <= 0)
        ):
            raise RuntimeError("Closed-Loop frame-selection signal is invalid")
        return {
            "cell_scores": cell_scores.detach(),
            "previous_layer": layer_index - 1,
            "source": (
                f"closed_loop_{self.feedback_form}_curvature_control_response_"
                f"lambda_{self.response_weight:g}"
            ),
            "current_temporal_offset": int(self._current_temporal_offset),
            "source_temporal_group_size": tt,
            "current_temporal": current_temporal,
        }

    def previous_layer_fair_frame_signal(
        self,
        *,
        layer_index: int,
        current_temporal: int,
    ) -> dict[str, Any]:
        """Expose the previous layer's per-frame control response.

        This signal is intentionally separate from CWCA's attention-budget
        allocation.  The owning FrameWeave policy uses it only to choose which
        Current frames receive an exact block update.  Scores are normalized
        by their temporal mean so the fair-selection credit has a stable unit
        across layers, chunks, and control magnitudes.
        """

        layer_index = int(layer_index)
        current_temporal = int(current_temporal)
        if self.mode != "closed_loop":
            raise RuntimeError(
                "response-credit frame selection requires a Closed-Loop observer"
            )
        if layer_index <= 0:
            raise RuntimeError("layer 0 has no causal previous-layer response")
        if current_temporal <= 0:
            raise RuntimeError("frame selection requires Current latent frames")
        previous_layer = layer_index - 1
        response = self._frame_responses.get(previous_layer)
        if response is None:
            raise RuntimeError(
                f"layer {layer_index} has no fine response from layer "
                f"{previous_layer}"
            )
        response = response.detach().float()
        if tuple(response.shape) != (current_temporal,):
            raise RuntimeError(
                "fine control-response length differs from Current latent frames"
            )
        if self.runtime_optimized:
            torch._assert_async(
                torch.all(torch.isfinite(response) & (response >= 0)),
                "fine control response contains invalid values",
            )
        elif not bool(torch.isfinite(response).all()) or bool(
            torch.any(response < 0)
        ):
            raise RuntimeError("fine control response contains invalid values")
        eps = torch.finfo(response.dtype).eps
        normalized = response / response.mean().clamp_min(eps)
        return {
            "frame_response": response,
            "normalized_frame_response": normalized,
            "previous_layer": previous_layer,
            "source": "previous_layer_camera_action_relative_response_per_current_frame",
            "normalization": "current_temporal_mean",
            "current_temporal": current_temporal,
            "previous_has_action_module": bool(
                self._response_has_action.get(previous_layer, False)
            ),
        }

    def previous_layer_frame_thinning_signal(
        self,
        *,
        layer_index: int,
        current_temporal: int,
        current_has_action_module: bool,
        causal_quantile: float = 0.25,
        minimum_history: int = 2,
    ) -> dict[str, Any]:
        """Return a causal fine-frame response signal for conservative thinning.

        This path deliberately does not replace the canonical mod schedule.
        It only says whether one canonical non-structural exact frame may be
        removed, and supplies the previous layer's per-Current-frame response
        to choose that frame.  The decision compares the previous layer's raw
        camera/action response energy with earlier *camera-only* layers in the
        same model call.  ActionModule layers and transitions out of them fail
        closed to the canonical schedule.
        """

        layer_index = int(layer_index)
        current_temporal = int(current_temporal)
        minimum_history = int(minimum_history)
        causal_quantile = float(causal_quantile)
        if self.mode != "closed_loop":
            raise RuntimeError("response thinning requires Closed-Loop CWCA")
        if layer_index <= 0:
            raise RuntimeError("layer 0 has no causal previous-layer response")
        if current_temporal <= 0:
            raise RuntimeError("frame thinning requires Current latent frames")
        if minimum_history < 1:
            raise ValueError("frame thinning requires positive causal history")
        if not 0.0 < causal_quantile < 0.5:
            raise ValueError("frame-thinning quantile must lie in (0, 0.5)")
        previous_layer = layer_index - 1
        previous_has_action = self._response_has_action.get(previous_layer)
        if previous_has_action is None:
            raise RuntimeError("fine control response lacks ActionModule provenance")
        action_safe = not bool(current_has_action_module) and not previous_has_action
        if not action_safe:
            return {
                "frame_scores": torch.empty(0),
                "previous_layer": previous_layer,
                "source": "raw_camera_action_relative_response_per_current_frame",
                "previous_energy": None,
                "causal_history_layers": [],
                "causal_history_count": 0,
                "causal_history_quantile": causal_quantile,
                "causal_threshold": None,
                "minimum_history": minimum_history,
                "previous_has_action_module": previous_has_action,
                "current_has_action_module": bool(current_has_action_module),
                "action_safe": False,
                "should_thin": False,
                "current_temporal": current_temporal,
                "history_axis": "same_call_across_prior_source_layers",
            }
        frame_response = self._frame_responses.get(previous_layer)
        if frame_response is None:
            raise RuntimeError(
                f"layer {layer_index} has no fine response from layer "
                f"{previous_layer}"
            )
        if frame_response.ndim != 1 or int(frame_response.numel()) != current_temporal:
            raise RuntimeError("fine control response does not match Current frames")
        history_layers = sorted(
            layer
            for layer in self._frame_responses
            if layer < previous_layer
            and self._response_has_action.get(layer) is False
        )
        history_energies = torch.stack(
            [self._frame_responses[layer].float().mean() for layer in history_layers]
        ) if history_layers else frame_response.new_empty((0,), dtype=torch.float32)
        previous_energy = frame_response.float().mean()
        threshold = (
            torch.quantile(history_energies, causal_quantile)
            if int(history_energies.numel()) >= minimum_history
            else None
        )
        should_thin = bool(
            threshold is not None
            and bool(previous_energy < threshold)
        )
        eps = torch.finfo(frame_response.float().dtype).eps
        frame_scores = (
            frame_response.float() / previous_energy.clamp_min(eps)
            if should_thin
            else frame_response.new_empty((0,), dtype=torch.float32)
        )
        if not bool(torch.isfinite(frame_scores).all()) or bool(
            torch.any(frame_scores < 0)
        ):
            raise RuntimeError("fine control-response frame scores are invalid")
        return {
            "frame_scores": frame_scores.detach(),
            "previous_layer": previous_layer,
            "source": "raw_camera_action_relative_response_per_current_frame",
            "previous_energy": previous_energy.detach(),
            "causal_history_layers": history_layers,
            "causal_history_count": len(history_layers),
            "causal_history_quantile": causal_quantile,
            "causal_threshold": None if threshold is None else threshold.detach(),
            "minimum_history": minimum_history,
            "previous_has_action_module": previous_has_action,
            "current_has_action_module": bool(current_has_action_module),
            "action_safe": action_safe,
            "should_thin": should_thin,
            "current_temporal": current_temporal,
            "history_axis": "same_call_across_prior_source_layers",
        }

    def previous_layer_temporal_history_signal(
        self,
        *,
        layer_index: int,
        current_temporal: int,
        current_has_action_module: bool,
        causal_quantile: float = 0.25,
        minimum_history: int = 2,
    ) -> dict[str, Any]:
        """Compare one source layer only against its earlier eligible q0 calls.

        The previous candidate compared raw response energies across Transformer
        depth and therefore mostly learned a fixed late-layer schedule.  This
        signal keeps the source layer fixed and builds a causal history across
        earlier FrameWeave calls.  It uses native response tiles, so it adds no
        fine-frame observer or benchmark-dependent signal.
        """

        layer_index = int(layer_index)
        current_temporal = int(current_temporal)
        minimum_history = int(minimum_history)
        causal_quantile = float(causal_quantile)
        if self.mode != "closed_loop":
            raise RuntimeError("response thinning requires Closed-Loop CWCA")
        if layer_index <= 0:
            raise RuntimeError("layer 0 has no causal previous-layer response")
        if current_temporal <= 0:
            raise RuntimeError("frame thinning requires Current latent frames")
        if minimum_history < 1:
            raise ValueError("frame thinning requires positive causal history")
        if not 0.0 < causal_quantile < 0.5:
            raise ValueError("frame-thinning quantile must lie in (0, 0.5)")

        cache_key = (int(self._call_index), layer_index)
        cached = self._frame_selection_payload_cache.get(cache_key)
        if cached is not None:
            return cached

        previous_layer = layer_index - 1
        previous_has_action = self._response_has_action.get(previous_layer)
        if previous_has_action is None:
            raise RuntimeError("control response lacks ActionModule provenance")
        action_safe = not bool(current_has_action_module) and not previous_has_action
        if not action_safe:
            payload = {
                "frame_scores": torch.empty(0),
                "previous_layer": previous_layer,
                "source": "raw_camera_action_relative_response_native_time_tiles",
                "previous_energy": None,
                "causal_history_layers": [],
                "causal_history_count": 0,
                "causal_history_quantile": causal_quantile,
                "causal_threshold": None,
                "minimum_history": minimum_history,
                "previous_has_action_module": previous_has_action,
                "current_has_action_module": bool(current_has_action_module),
                "action_safe": False,
                "should_thin": False,
                "current_temporal": current_temporal,
                "history_axis": "same_source_layer_across_prior_q0_woven_calls",
                "current_temporal_offset": int(self._current_temporal_offset),
                "source_temporal_group_size": int(self._block_shape[0]),
            }
            self._frame_selection_payload_cache[cache_key] = payload
            return payload

        raw_response = self._raw_responses.get(previous_layer)
        if raw_response is None:
            raise RuntimeError(
                f"layer {layer_index} has no raw response from layer {previous_layer}"
            )
        if raw_response.ndim != 2 or not raw_response.numel():
            raise RuntimeError("raw control response has invalid native tile layout")
        raw_response = raw_response.detach().float()
        if not bool(torch.isfinite(raw_response).all()) or bool(
            torch.any(raw_response < 0)
        ):
            raise RuntimeError("raw control response contains invalid values")

        history = self._frame_selection_energy_history.setdefault(previous_layer, [])
        previous_energy = raw_response.mean()
        history_count = len(history)
        history_energies = (
            torch.stack([value.to(previous_energy.device) for value in history])
            if history
            else previous_energy.new_empty((0,))
        )
        threshold = (
            torch.quantile(history_energies, causal_quantile)
            if history_count >= minimum_history
            else None
        )
        should_thin = bool(
            threshold is not None and bool(previous_energy < threshold)
        )

        frame_scores = previous_energy.new_empty((0,))
        if should_thin:
            temporal_cells = raw_response.mean(dim=1)
            offset = int(self._current_temporal_offset)
            group = int(self._block_shape[0])
            capacity = offset + int(temporal_cells.numel()) * group
            if current_temporal > capacity:
                raise RuntimeError(
                    "Current latent count exceeds native response-tile capacity"
                )
            # The Current prefix inside Matrix's Memory|Current partial tile has
            # no independent native response row.  Give it the largest finite
            # score so it can never be selected as the lowest-response drop.
            normalized_cells = temporal_cells / previous_energy.clamp_min(
                torch.finfo(torch.float32).eps
            )
            frame_scores = torch.full(
                (current_temporal,),
                torch.finfo(torch.float32).max,
                device=temporal_cells.device,
                dtype=torch.float32,
            )
            for frame in range(offset, current_temporal):
                cell = (frame - offset) // group
                frame_scores[frame] = normalized_cells[cell]
            if not bool(torch.isfinite(frame_scores).all()) or bool(
                torch.any(frame_scores < 0)
            ):
                raise RuntimeError("native-tile frame scores are invalid")

        payload = {
            "frame_scores": frame_scores.detach(),
            "previous_layer": previous_layer,
            "source": "raw_camera_action_relative_response_native_time_tiles",
            "previous_energy": previous_energy.detach(),
            "causal_history_layers": [],
            "causal_history_count": history_count,
            "causal_history_quantile": causal_quantile,
            "causal_threshold": None if threshold is None else threshold.detach(),
            "minimum_history": minimum_history,
            "previous_has_action_module": previous_has_action,
            "current_has_action_module": bool(current_has_action_module),
            "action_safe": True,
            "should_thin": should_thin,
            "current_temporal": current_temporal,
            "history_axis": "same_source_layer_across_prior_q0_woven_calls",
            "current_temporal_offset": int(self._current_temporal_offset),
            "source_temporal_group_size": int(self._block_shape[0]),
        }
        history.append(previous_energy.detach().cpu())
        self._frame_selection_payload_cache[cache_key] = payload
        return payload

    def previous_layer_lagged_temporal_history_signal(
        self,
        *,
        layer_index: int,
        current_temporal: int,
        current_has_action_module: bool,
        causal_quantile: float = 0.25,
        minimum_history: int = 2,
    ) -> dict[str, Any]:
        """Return the decision prepared by the previous eligible q0 call."""

        layer_index = int(layer_index)
        current_temporal = int(current_temporal)
        minimum_history = int(minimum_history)
        causal_quantile = float(causal_quantile)
        if not self.temporal_history_response_lagged_batch:
            raise RuntimeError("lagged batched temporal response is not enabled")
        if self.mode != "closed_loop":
            raise RuntimeError("response thinning requires Closed-Loop CWCA")
        if layer_index <= 0:
            raise RuntimeError("layer 0 has no causal previous-layer response")
        if current_temporal <= 0:
            raise RuntimeError("frame thinning requires Current latent frames")
        if minimum_history < 1 or not 0.0 < causal_quantile < 0.5:
            raise ValueError("invalid lagged temporal-history rule")

        previous_layer = layer_index - 1
        payload = self._lagged_frame_selection_payloads.get(layer_index)
        if payload is None:
            previous_has_action = self._response_has_action.get(previous_layer)
            if previous_has_action is None:
                raise RuntimeError("control response lacks ActionModule provenance")
            action_safe = (
                not bool(current_has_action_module) and not previous_has_action
            )
            return {
                "frame_scores": torch.empty(0, dtype=torch.float32),
                "previous_layer": previous_layer,
                "source": (
                    "batched_raw_camera_action_relative_response_native_time_tiles"
                ),
                "previous_energy": None,
                "causal_history_layers": [],
                "causal_history_count": 0,
                "causal_history_quantile": causal_quantile,
                "causal_threshold": None,
                "minimum_history": minimum_history,
                "previous_has_action_module": previous_has_action,
                "current_has_action_module": bool(current_has_action_module),
                "action_safe": action_safe,
                "should_thin": False,
                "current_temporal": current_temporal,
                "history_axis": (
                    "same_source_layer_across_prior_q0_woven_calls_lag1"
                ),
                "current_temporal_offset": int(self._current_temporal_offset),
                "source_temporal_group_size": int(self._block_shape[0]),
                "source_eligible_call_index": None,
                "selection_eligible_call_index": (
                    self._lagged_eligible_call_index + 1
                ),
                "selection_lag_eligible_calls": 1,
                "payload_ready": False,
                "batched_host_decision": True,
            }

        if int(payload["previous_layer"]) != previous_layer:
            raise RuntimeError("lagged payload source-layer mismatch")
        if int(payload["current_temporal"]) != current_temporal:
            raise RuntimeError("lagged payload Current-frame count mismatch")
        if bool(payload["current_has_action_module"]) != bool(
            current_has_action_module
        ):
            raise RuntimeError("lagged payload ActionModule topology changed")
        expected_selection_call = self._lagged_eligible_call_index + 1
        if int(payload["selection_eligible_call_index"]) != expected_selection_call:
            raise RuntimeError("lagged payload was not produced by the prior eligible q0")
        if int(payload["selection_lag_eligible_calls"]) != 1:
            raise RuntimeError("lagged payload has an invalid causal offset")
        scores = payload.get("frame_scores")
        if not isinstance(scores, torch.Tensor) or scores.device.type != "cpu":
            raise RuntimeError("lagged payload must remain a CPU decision")
        return payload

    def previous_call_lagged_period_signal(
        self,
        *,
        current_temporal: int,
        current_action_signature: tuple[float, ...] | None,
        causal_quantile: float = 0.50,
        minimum_history: int = 2,
        base_period: int = 5,
        low_response_period: int = 10,
    ) -> dict[str, Any]:
        """Return a shared nested-lattice route from the prior eligible q0.

        Unlike per-layer thinning, this decision is shared by the whole q0
        Transformer stack.  The response statistic is produced by the prior
        eligible call, compared only with earlier call-level statistics, and
        accepted only when the current control signature matches its source.
        """

        current_temporal = int(current_temporal)
        minimum_history = int(minimum_history)
        causal_quantile = float(causal_quantile)
        base_period = int(base_period)
        low_response_period = int(low_response_period)
        if not self.lagged_call_period_router_enabled:
            raise RuntimeError("lagged call-period routing is not enabled")
        if self.mode != "closed_loop":
            raise RuntimeError("call-period routing requires Closed-Loop CWCA")
        if current_temporal <= 0:
            raise ValueError("call-period routing requires Current latent frames")
        if minimum_history < 1 or not 0.0 < causal_quantile < 1.0:
            raise ValueError("invalid call-period response rule")
        if base_period < 2 or low_response_period <= base_period:
            raise ValueError("invalid nested temporal periods")
        if low_response_period % base_period:
            raise ValueError("low-response period must be a base-period multiple")
        current_signature = (
            None
            if current_action_signature is None
            else tuple(float(value) for value in current_action_signature)
        )
        payload = self._lagged_call_period_payload
        if payload is None:
            return {
                "source": "batched_closed_loop_call_response",
                "call_response_energy": None,
                "causal_history_count": 0,
                "causal_history_quantile": causal_quantile,
                "causal_threshold": None,
                "minimum_history": minimum_history,
                "response_below_threshold": False,
                "source_action_signature": None,
                "current_action_signature": current_signature,
                "action_signature_match": False,
                "should_route": False,
                "current_temporal": current_temporal,
                "history_axis": "aggregate_closed_loop_response_across_prior_q0_calls_lag1",
                "source_eligible_call_index": None,
                "selection_eligible_call_index": self._lagged_eligible_call_index + 1,
                "selection_lag_eligible_calls": 1,
                "payload_ready": False,
                "batched_host_decision": True,
                "base_period": base_period,
                "low_response_period": low_response_period,
            }

        expected_selection_call = self._lagged_eligible_call_index + 1
        if int(payload["selection_eligible_call_index"]) != expected_selection_call:
            raise RuntimeError("call-period payload was not produced by the prior q0")
        if int(payload["current_temporal"]) != current_temporal:
            raise RuntimeError("call-period payload Current-frame count changed")
        if int(payload["base_period"]) != base_period or int(
            payload["low_response_period"]
        ) != low_response_period:
            raise RuntimeError("call-period payload lattice changed")
        source_signature = payload.get("source_action_signature")
        action_match = bool(
            source_signature is not None
            and current_signature is not None
            and tuple(float(value) for value in source_signature)
            == current_signature
        )
        return {
            **payload,
            "current_action_signature": current_signature,
            "action_signature_match": action_match,
            "should_route": bool(
                payload["response_below_threshold"] and action_match
            ),
        }

    def finalize_lagged_call_period_route(
        self,
        *,
        current_temporal: int,
        layer_count: int,
        source_action_signature: tuple[float, ...] | None,
        causal_quantile: float = 0.50,
        minimum_history: int = 2,
        base_period: int = 5,
        low_response_period: int = 10,
    ) -> dict[str, Any]:
        """Create one next-call nested-lattice route from all q0 responses."""

        current_temporal = int(current_temporal)
        layer_count = int(layer_count)
        causal_quantile = float(causal_quantile)
        minimum_history = int(minimum_history)
        base_period = int(base_period)
        low_response_period = int(low_response_period)
        if not self.lagged_call_period_router_enabled:
            raise RuntimeError("lagged call-period routing is not enabled")
        if not self.temporal_history_capture_active:
            raise RuntimeError("cannot finalize an inactive call-period capture")
        if current_temporal <= 0 or layer_count < 3:
            raise ValueError("invalid call-period response geometry")
        if minimum_history < 1 or not 0.0 < causal_quantile < 1.0:
            raise ValueError("invalid call-period response rule")
        if low_response_period <= base_period or low_response_period % base_period:
            raise ValueError("call-period lattices must be strictly nested")
        signature = (
            None
            if source_action_signature is None
            else tuple(float(value) for value in source_action_signature)
        )
        if not signature:
            raise RuntimeError("call-period route lacks its control signature")

        source_layers = list(range(0, layer_count - 2))
        missing = [layer for layer in source_layers if layer not in self._raw_responses]
        if missing:
            raise RuntimeError(
                f"call-period response batch is missing source layers {missing}"
            )
        shapes = {tuple(self._raw_responses[layer].shape) for layer in source_layers}
        devices = {self._raw_responses[layer].device for layer in source_layers}
        if len(shapes) != 1 or len(devices) != 1:
            raise RuntimeError("call-period response batch has inconsistent layouts")
        batch = torch.stack(
            [self._raw_responses[layer].detach().float() for layer in source_layers]
        ).cpu()
        if not bool(torch.isfinite(batch).all()) or bool(torch.any(batch < 0)):
            raise RuntimeError("call-period response batch contains invalid values")
        call_energy = float(batch.mean().item())
        history_count = len(self._call_period_energy_history)
        threshold = (
            float(
                torch.quantile(
                    torch.stack(self._call_period_energy_history).float().cpu(),
                    causal_quantile,
                ).item()
            )
            if history_count >= minimum_history
            else None
        )
        response_below = bool(
            threshold is not None and call_energy < float(threshold)
        )
        source_call = self._lagged_eligible_call_index + 1
        selection_call = source_call + 1
        payload = {
            "source": "batched_closed_loop_call_response",
            "call_response_energy": call_energy,
            "causal_history_count": history_count,
            "causal_history_quantile": causal_quantile,
            "causal_threshold": threshold,
            "minimum_history": minimum_history,
            "response_below_threshold": response_below,
            "source_action_signature": signature,
            "current_action_signature": None,
            "action_signature_match": None,
            "should_route": False,
            "current_temporal": current_temporal,
            "history_axis": "aggregate_closed_loop_response_across_prior_q0_calls_lag1",
            "source_eligible_call_index": source_call,
            "selection_eligible_call_index": selection_call,
            "selection_lag_eligible_calls": 1,
            "payload_ready": True,
            "batched_host_decision": True,
            "base_period": base_period,
            "low_response_period": low_response_period,
        }
        self._call_period_energy_history.append(
            torch.tensor(call_energy, dtype=torch.float32)
        )
        self._lagged_eligible_call_index = source_call
        self._lagged_call_period_payload = payload
        self._lagged_frame_selection_payloads.clear()
        record = {
            "source_eligible_call_index": source_call,
            "selection_eligible_call_index": selection_call,
            "selection_lag_eligible_calls": 1,
            "source_layers": source_layers,
            "target_layers": list(range(1, layer_count - 1)),
            "batch_shape": [int(value) for value in batch.shape],
            "single_device_to_host_batch": True,
            "payloads_written": 1,
            "route_payload_kind": "shared_nested_period",
            "call_response_energy": call_energy,
            "causal_history_count": history_count,
            "causal_threshold": threshold,
            "response_below_threshold": response_below,
            "source_action_signature": list(signature),
            "base_period": base_period,
            "low_response_period": low_response_period,
            "all_finite_nonnegative": True,
        }
        self._lagged_batch_finalize_records.append(record)
        return record

    def finalize_lagged_temporal_history_call(
        self,
        *,
        current_temporal: int,
        layer_count: int,
        causal_quantile: float = 0.25,
        minimum_history: int = 2,
    ) -> dict[str, Any]:
        """Batch one eligible q0's layer responses into next-q0 payloads."""

        current_temporal = int(current_temporal)
        layer_count = int(layer_count)
        causal_quantile = float(causal_quantile)
        minimum_history = int(minimum_history)
        if not self.temporal_history_response_lagged_batch:
            raise RuntimeError("lagged batched temporal response is not enabled")
        if not self.temporal_history_capture_active:
            raise RuntimeError("cannot finalize an inactive lagged response capture")
        if current_temporal <= 0 or layer_count < 3:
            raise ValueError("invalid lagged response geometry")
        if minimum_history < 1 or not 0.0 < causal_quantile < 0.5:
            raise ValueError("invalid lagged temporal-history rule")

        target_layers = list(range(1, layer_count - 1))
        source_layers = [layer - 1 for layer in target_layers]
        missing = [layer for layer in source_layers if layer not in self._raw_responses]
        if missing:
            raise RuntimeError(
                f"lagged response batch is missing source layers {missing}"
            )
        missing_provenance = [
            layer
            for layer in sorted(set(source_layers + target_layers))
            if layer not in self._response_has_action
        ]
        if missing_provenance:
            raise RuntimeError(
                "lagged response batch lacks ActionModule provenance for "
                f"layers {missing_provenance}"
            )

        shapes = {tuple(self._raw_responses[layer].shape) for layer in source_layers}
        devices = {self._raw_responses[layer].device for layer in source_layers}
        if len(shapes) != 1 or len(devices) != 1:
            raise RuntimeError("lagged response batch has inconsistent tensor layouts")
        # This is the only device-to-host materialization in an eligible q0
        # call.  Every quantile, scalar comparison, and frame expansion below
        # runs on the copied CPU tensor.
        batch = torch.stack(
            [self._raw_responses[layer].detach().float() for layer in source_layers]
        ).cpu()
        if not bool(torch.isfinite(batch).all()) or bool(torch.any(batch < 0)):
            raise RuntimeError("lagged response batch contains invalid values")

        source_to_row = {layer: row for row, layer in enumerate(source_layers)}
        source_call = self._lagged_eligible_call_index + 1
        selection_call = source_call + 1
        payloads: dict[int, dict[str, Any]] = {}
        thinning_payloads = 0
        offset = int(self._current_temporal_offset)
        group = int(self._block_shape[0])
        eps = torch.finfo(torch.float32).eps
        for target_layer in target_layers:
            previous_layer = target_layer - 1
            previous_has_action = bool(
                self._response_has_action[previous_layer]
            )
            current_has_action = bool(self._response_has_action[target_layer])
            action_safe = not previous_has_action and not current_has_action
            raw_response = batch[source_to_row[previous_layer]]
            previous_energy = float(raw_response.mean().item())
            history = self._frame_selection_energy_history.setdefault(
                previous_layer, []
            )
            history_count = len(history)
            threshold = (
                float(
                    torch.quantile(
                        torch.stack(
                            [value.detach().float().cpu() for value in history]
                        ),
                        causal_quantile,
                    ).item()
                )
                if action_safe and history_count >= minimum_history
                else None
            )
            should_thin = bool(
                action_safe
                and threshold is not None
                and previous_energy < threshold
            )
            frame_scores = torch.empty(0, dtype=torch.float32)
            if should_thin:
                temporal_cells = raw_response.mean(dim=1)
                capacity = offset + int(temporal_cells.numel()) * group
                if current_temporal > capacity:
                    raise RuntimeError(
                        "Current latent count exceeds native response-tile capacity"
                    )
                normalized_cells = temporal_cells / max(previous_energy, eps)
                frame_scores = torch.full(
                    (current_temporal,),
                    torch.finfo(torch.float32).max,
                    dtype=torch.float32,
                )
                for frame in range(offset, current_temporal):
                    frame_scores[frame] = normalized_cells[(frame - offset) // group]
                if not bool(torch.isfinite(frame_scores).all()) or bool(
                    torch.any(frame_scores < 0)
                ):
                    raise RuntimeError("lagged native-tile frame scores are invalid")
                thinning_payloads += 1

            payloads[target_layer] = {
                "frame_scores": frame_scores,
                "previous_layer": previous_layer,
                "source": (
                    "batched_raw_camera_action_relative_response_native_time_tiles"
                ),
                "previous_energy": previous_energy,
                "causal_history_layers": [],
                "causal_history_count": history_count,
                "causal_history_quantile": causal_quantile,
                "causal_threshold": threshold,
                "minimum_history": minimum_history,
                "previous_has_action_module": previous_has_action,
                "current_has_action_module": current_has_action,
                "action_safe": action_safe,
                "should_thin": should_thin,
                "current_temporal": current_temporal,
                "history_axis": (
                    "same_source_layer_across_prior_q0_woven_calls_lag1"
                ),
                "current_temporal_offset": offset,
                "source_temporal_group_size": group,
                "source_eligible_call_index": source_call,
                "selection_eligible_call_index": selection_call,
                "selection_lag_eligible_calls": 1,
                "payload_ready": True,
                "batched_host_decision": True,
            }
            if action_safe:
                history.append(torch.tensor(previous_energy, dtype=torch.float32))

        self._lagged_eligible_call_index = source_call
        self._lagged_frame_selection_payloads = payloads
        record = {
            "source_eligible_call_index": source_call,
            "selection_eligible_call_index": selection_call,
            "selection_lag_eligible_calls": 1,
            "source_layers": source_layers,
            "target_layers": target_layers,
            "batch_shape": [int(value) for value in batch.shape],
            "single_device_to_host_batch": True,
            "payloads_written": len(payloads),
            "thinning_payloads": thinning_payloads,
            "all_finite_nonnegative": True,
        }
        self._lagged_batch_finalize_records.append(record)
        return record

    @staticmethod
    def _compact_time_cells(
        values: torch.Tensor,
        active_current: torch.Tensor,
        compact_temporal_group_size: int,
        source_temporal_group_size: int,
        current_temporal_offset: int,
    ) -> torch.Tensor:
        # The native whole-sequence layout classifies the partial block at the
        # Memory|Current boundary as a Memory row.  Consequently CWCA's first
        # Current profile row begins at ``current_temporal_offset`` (three for
        # Matrix M5/tt4), exactly as the parent compiler and response observer
        # aggregate it.  Prefix Current frames retain that native uniform-row
        # signal (zero allocation signal); the final partial tile remains a
        # valid profile row.
        local = active_current.to(values.device)
        offset = int(current_temporal_offset)
        profiled = local >= offset
        source_cells = torch.div(
            (local - offset).clamp_min(0),
            int(source_temporal_group_size),
            rounding_mode="floor",
        )
        if bool(torch.any(profiled & (source_cells >= int(values.shape[0])))):
            raise RuntimeError("compact Current frame lies outside CWCA tiles")
        selected = torch.zeros(
            (int(local.numel()), int(values.shape[1])),
            device=values.device,
            dtype=values.dtype,
        )
        if bool(profiled.any()):
            selected[profiled] = values.index_select(0, source_cells[profiled])
        rows = []
        for low in range(
            0, int(selected.shape[0]), int(compact_temporal_group_size)
        ):
            rows.append(
                selected[low : low + int(compact_temporal_group_size)].mean(dim=0)
            )
        if not rows:
            raise RuntimeError("compact Closed-Loop CWCA has no Current cells")
        return torch.stack(rows)

    @staticmethod
    def _redistribute_compact_capacity(
        degrees: torch.Tensor,
        *,
        capacity: int,
        target_per_worldline: int,
    ) -> torch.Tensor:
        """Saturate rows while retaining each spatial-worldline edge budget."""

        degrees = degrees.clamp(max=capacity)
        for spatial_index in range(int(degrees.shape[1])):
            missing = target_per_worldline - int(degrees[:, spatial_index].sum().item())
            while missing > 0:
                candidates = torch.nonzero(
                    degrees[:, spatial_index] < capacity
                ).flatten()
                if not int(candidates.numel()):
                    raise RuntimeError("compact Closed-Loop CWCA exhausted key capacity")
                take = min(missing, int(candidates.numel()))
                degrees[candidates[:take], spatial_index] += 1
                missing -= take
        return degrees

    def compact_closed_loop_degrees(
        self,
        *,
        layer_index: int,
        active_current: torch.Tensor,
        current_frame_origin: int,
        temporal_group_size: int,
        spatial_blocks: int,
        mean_degree: int,
        base_degree: int,
        total_blocks: int,
        memory_blocks: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Allocate compact Current rows from the same Closed-Loop signal.

        The returned matrix is ``[compact_time, spatial_worldline]``.  Its
        per-worldline total is fixed; this method never selects keys.
        """

        layer_index = int(layer_index)
        if self.mode != "closed_loop":
            raise RuntimeError("compact feedback is defined for Closed-Loop CWCA")
        if self._compact_layer_index != layer_index or self._layer_index != layer_index:
            raise RuntimeError("compact Closed-Loop layer was not entered")
        if self._curvature_matrix is None:
            raise RuntimeError("compact Closed-Loop CWCA has no curvature prior")
        if int(self._curvature_matrix.shape[1]) != int(spatial_blocks):
            raise RuntimeError("compact Closed-Loop spatial worldlines differ from CWCA")
        current_frame_origin = int(current_frame_origin)
        if not int(active_current.numel()):
            raise RuntimeError("compact Closed-Loop has no active Current frames")
        if bool(torch.any(active_current < current_frame_origin)):
            raise RuntimeError("compact Closed-Loop Current set contains Memory frames")
        active_current_local = active_current - current_frame_origin
        current_capacity = self._current_temporal_offset + int(
            self._curvature_matrix.shape[0]
        ) * int(self._block_shape[0])
        if bool(torch.any(active_current_local >= current_capacity)):
            raise RuntimeError("compact Current frame lies outside CWCA frame capacity")
        curvature = self._curvature_matrix
        eps = torch.finfo(curvature.dtype).eps
        normalized_curvature = curvature / curvature.mean(
            dim=0, keepdim=True
        ).clamp_min(eps)
        compact_curvature = self._compact_time_cells(
            normalized_curvature,
            active_current_local,
            int(temporal_group_size),
            int(self._block_shape[0]),
            self._current_temporal_offset,
        )
        if self.fixed_query_budget:
            time_cells = int(compact_curvature.shape[0])
            degrees = torch.full(
                (time_cells, int(spatial_blocks)),
                int(mean_degree),
                device=compact_curvature.device,
                dtype=torch.long,
            )
            expected_current_edges = (
                time_cells * int(spatial_blocks) * int(mean_degree)
            )
            metadata = {
                "call": self._call_index,
                "layer": layer_index,
                "source": "compact_light_interaction_fixed_query_budget",
                "previous_layer": None,
                "active_current_global_frames": [
                    int(v) for v in active_current.tolist()
                ],
                "active_current_local_frames": [
                    int(v) for v in active_current_local.tolist()
                ],
                "current_frame_origin": current_frame_origin,
                "native_current_temporal_offset": self._current_temporal_offset,
                "compact_time_cells": time_cells,
                "spatial_worldlines": int(spatial_blocks),
                "response": self._materialize(self._stats(None)),
                "curvature": self._materialize(self._stats(compact_curvature)),
                "current_worldline_budget": time_cells * int(mean_degree),
                "current_edge_budget": expected_current_edges,
                "global_edge_budget": int(memory_blocks) * int(mean_degree)
                + expected_current_edges,
                "reference_edge_budget": int(total_blocks) * int(mean_degree),
                "uniform_query_budget": True,
            }
            return degrees, metadata
        if layer_index == 0:
            allocation_signal = compact_curvature
            source = "compact_cwca_curvature_bootstrap"
            previous_layer = None
            response_stats = self._stats(None)
        elif self.response_weight == 0.0:
            # Keep the response observer live for FrameWeave's exact-frame
            # policy, but make the attention allocation mathematically and
            # operationally curvature-only.
            allocation_signal = compact_curvature
            source = "compact_curvature_only_response_weight_0"
            previous_layer = None
            response_stats = self._stats(None)
        else:
            response = self._responses.get(layer_index - 1)
            if response is None:
                raise RuntimeError(
                    f"compact layer {layer_index} has no response from layer "
                    f"{layer_index - 1}"
                )
            compact_response = self._compact_time_cells(
                response.to(curvature.device, curvature.dtype),
                active_current_local,
                int(temporal_group_size),
                int(self._block_shape[0]),
                self._current_temporal_offset,
            )
            if self.feedback_form == "additive":
                allocation_signal = (
                    compact_curvature + self.response_weight * compact_response
                )
            else:
                sign = -1.0 if self.feedback_form == "centered_inverse" else 1.0
                allocation_signal = compact_curvature * torch.exp(
                    sign * self.response_weight * (compact_response - 1.0)
                )
            source = (
                f"compact_{self.feedback_form}_curvature_control_response_"
                f"lambda_{self.response_weight:g}"
            )
            previous_layer = layer_index - 1
            response_stats = self._stats(compact_response)
        weights = 1.0 + torch.log1p(allocation_signal)
        time_cells = int(weights.shape[0])
        residual_per_worldline = time_cells * (int(mean_degree) - int(base_degree))
        bonuses = (
            MatrixCurvatureTemperedUncertaintyAttentionCompiler
            ._apportion_section_matrix(
                weights.transpose(0, 1), residual_per_worldline
            )
            .transpose(0, 1)
        )
        degrees = bonuses + int(base_degree)
        expected_worldline = time_cells * int(mean_degree)
        degrees = self._redistribute_compact_capacity(
            degrees,
            capacity=int(total_blocks),
            target_per_worldline=expected_worldline,
        )
        worldline_totals = degrees.sum(dim=0)
        if not bool(torch.all(worldline_totals == expected_worldline)):
            raise RuntimeError("compact Closed-Loop lost a worldline edge budget")
        expected_current_edges = time_cells * int(spatial_blocks) * int(mean_degree)
        if int(degrees.sum().item()) != expected_current_edges:
            raise RuntimeError("compact Closed-Loop lost its Current edge budget")
        metadata = {
            "call": self._call_index,
            "layer": layer_index,
            "source": source,
            "previous_layer": previous_layer,
            "active_current_global_frames": [
                int(v) for v in active_current.tolist()
            ],
            "active_current_local_frames": [
                int(v) for v in active_current_local.tolist()
            ],
            "current_frame_origin": current_frame_origin,
            "native_current_temporal_offset": self._current_temporal_offset,
            "compact_time_cells": time_cells,
            "spatial_worldlines": int(spatial_blocks),
            "response": self._materialize(response_stats),
            "curvature": self._materialize(self._stats(compact_curvature)),
            "current_worldline_budget": expected_worldline,
            "current_edge_budget": expected_current_edges,
            "global_edge_budget": int(memory_blocks) * int(mean_degree)
            + expected_current_edges,
            "reference_edge_budget": int(total_blocks) * int(mean_degree),
        }
        return degrees, metadata

    def record_compact_budget(
        self,
        metadata: dict[str, Any],
        degrees: torch.Tensor,
        *,
        local_support_preserved: bool,
        native_qk_topk: bool,
    ) -> None:
        record = dict(metadata)
        record.update(
            {
                "k_min": int(degrees.min().item()),
                "k_max": int(degrees.max().item()),
                "k_mean": float(degrees.float().mean().item()),
                "budget_exact": metadata["global_edge_budget"]
                == metadata["reference_edge_budget"],
                "local_support_preserved": bool(local_support_preserved),
                "native_qk_topk": bool(native_qk_topk),
            }
        )
        if not record["budget_exact"]:
            raise RuntimeError("compact Closed-Loop global edge certificate failed")
        self._compact_budget_records.append(record)

    def set_layer_index(self, layer_index: int) -> None:
        layer_index = int(layer_index)
        if layer_index == 0:
            self._call_index += 1
            self._responses.clear()
            self._raw_responses.clear()
            self._frame_responses.clear()
            self._response_has_action.clear()
        self._layer_index = layer_index
        self._install_layer_budget(layer_index)

    def _install_layer_budget(self, layer_index: int) -> None:
        if self._bootstrap_degrees is None or self._spatial_blocks is None:
            return
        if self.fixed_query_budget:
            self._row_degrees = torch.full_like(
                self._bootstrap_degrees, int(self._degree)
            )
            current_rows = self._row_degrees[self._profile_memory_blocks :]
            current_temporal = (
                int(current_rows.numel()) // int(self._spatial_blocks)
            )
            self._current_worldline_budgets = torch.full(
                (int(self._spatial_blocks),),
                current_temporal * int(self._degree),
                device=self._row_degrees.device,
                dtype=self._row_degrees.dtype,
            )
            source = "light_interaction_fixed_query_budget"
        elif layer_index == 0:
            self._row_degrees = self._bootstrap_degrees.clone()
            source = "cwca_curvature_bootstrap"
        elif self.mode == "closed_loop" and self.response_weight == 0.0:
            if self._curvature_matrix is None:
                raise RuntimeError("curvature-only CWCA has no curvature prior")
            curvature = self._curvature_matrix
            eps = torch.finfo(curvature.dtype).eps
            normalized_curvature = curvature / curvature.mean(
                dim=0, keepdim=True
            ).clamp_min(eps)
            weights = 1.0 + torch.log1p(normalized_curvature)
            source = "curvature_only_response_weight_0"

            temporal, _spatial = normalized_curvature.shape
            base = math.ceil(4 * self._degree / 5)
            residual = temporal * (self._degree - base)
            bonuses = (
                MatrixCurvatureTemperedUncertaintyAttentionCompiler
                ._apportion_section_matrix(weights.transpose(0, 1), residual)
                .transpose(0, 1)
            )
            current_degrees = bonuses + base
            degrees = torch.full_like(self._bootstrap_degrees, self._degree)
            start = self._profile_memory_blocks
            degrees[start : start + current_degrees.numel()] = (
                current_degrees.reshape(-1)
            )
            global_total = degrees.sum()
            torch._assert_async(
                global_total == self._num_blocks * self._degree,
                "curvature-only CWCA lost the global edge budget",
            )
            expected = temporal * self._degree
            worldline_totals = current_degrees.sum(dim=0)
            torch._assert_async(
                torch.all(worldline_totals == expected),
                "curvature-only CWCA violated a worldline budget",
            )
            self._row_degrees = degrees.detach()
            self._current_worldline_budgets = worldline_totals.detach()
        else:
            response = self._responses.get(layer_index - 1)
            if response is None:
                raise RuntimeError(
                    f"layer {layer_index} has no response from layer {layer_index - 1}"
                )
            if self.mode == "response_only":
                weights = 1.0 + torch.log1p(response)
                source = "previous_layer_control_response"
            else:
                if self._curvature_matrix is None:
                    raise RuntimeError("closed-loop CWCA has no curvature prior")
                curvature = self._curvature_matrix.to(response.device, response.dtype)
                eps = torch.finfo(curvature.dtype).eps
                normalized_curvature = curvature / curvature.mean(
                    dim=0, keepdim=True
                ).clamp_min(eps)
                if self.feedback_form == "additive":
                    allocation_signal = (
                        normalized_curvature + self.response_weight * response
                    )
                else:
                    sign = -1.0 if self.feedback_form == "centered_inverse" else 1.0
                    allocation_signal = normalized_curvature * torch.exp(
                        sign * self.response_weight * (response - 1.0)
                    )
                weights = 1.0 + torch.log1p(allocation_signal)
                source = (
                    f"{self.feedback_form}_curvature_control_response_"
                    f"lambda_{self.response_weight:g}"
                )

            temporal, spatial = response.shape
            base = math.ceil(4 * self._degree / 5)
            residual = temporal * (self._degree - base)
            bonuses = (
                MatrixCurvatureTemperedUncertaintyAttentionCompiler
                ._apportion_section_matrix(weights.transpose(0, 1), residual)
                .transpose(0, 1)
            )
            current_degrees = bonuses + base
            degrees = torch.full_like(self._bootstrap_degrees, self._degree)
            start = self._profile_memory_blocks
            degrees[start : start + current_degrees.numel()] = current_degrees.reshape(-1)
            global_total = degrees.sum()
            torch._assert_async(
                global_total == self._num_blocks * self._degree,
                "control-response CWCA lost the global edge budget",
            )
            expected = temporal * self._degree
            worldline_totals = current_degrees.sum(dim=0)
            torch._assert_async(
                torch.all(worldline_totals == expected),
                "control-response CWCA violated a worldline budget",
            )
            self._row_degrees = degrees.detach()
            self._current_worldline_budgets = worldline_totals.detach()

        current = self._row_degrees[self._profile_memory_blocks :]
        self._pending_budget_record = {
                "call": self._call_index,
                "layer": layer_index,
                "source": source,
                "k_min": current.min() if current.numel() else 0,
                "k_max": current.max() if current.numel() else 0,
                "k_mean": current.float().mean() if current.numel() else 0.0,
                "global_edge_budget": self._row_degrees.sum(),
                "reference_edge_budget": int(self._num_blocks * self._degree),
                "budget_exact": self._row_degrees.sum()
                == int(self._num_blocks * self._degree),
            }

    @staticmethod
    def _stats(value: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
        if value is None or not value.numel():
            return {"min": None, "max": None, "mean": None, "std": None}
        value = value.float()
        return {
            "min": value.min(),
            "max": value.max(),
            "mean": value.mean(),
            "std": value.std(unbiased=False),
        }

    @classmethod
    def _materialize(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalar = value.item()
            if value.dtype == torch.bool:
                return bool(scalar)
            if not value.dtype.is_floating_point:
                return int(scalar)
            return float(scalar)
        if isinstance(value, dict):
            return {key: cls._materialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._materialize(item) for item in value]
        return value

    def record_layer_response(
        self,
        layer_index: int,
        camera_response: torch.Tensor,
        action_response: torch.Tensor | None,
        *,
        execution_path: str = "full_native",
        camera_frame_response: torch.Tensor | None = None,
        action_frame_response: torch.Tensor | None = None,
        fine_only: bool = False,
    ) -> None:
        if self._curvature_matrix is None:
            return
        expected_shape = tuple(self._curvature_matrix.shape)
        if tuple(camera_response.shape) != expected_shape:
            raise RuntimeError(
                f"camera response shape {tuple(camera_response.shape)} != {expected_shape}"
            )
        if fine_only:
            if (
                not self.curvature_only_response_gating_enabled
                or not self.fine_only_response_reduction_enabled
                or self.response_weight != 0.0
            ):
                raise RuntimeError(
                    "fine-only response requires gated curvature-only attention"
                )
            response = torch.zeros_like(camera_response)
        else:
            eps = torch.finfo(camera_response.dtype).eps
            cam_norm = camera_response / camera_response.mean(
                dim=0, keepdim=True
            ).clamp_min(eps)
            if action_response is None:
                response = cam_norm
            else:
                if tuple(action_response.shape) != expected_shape:
                    raise RuntimeError(
                        "action response layout differs from camera response"
                    )
                action_norm = action_response / action_response.mean(
                    dim=0, keepdim=True
                ).clamp_min(eps)
                response = 0.5 * (cam_norm + action_norm)
            self._responses[int(layer_index)] = response.detach()
        raw_response = None
        if (
            self.temporal_history_response_enabled
            and self.temporal_history_capture_active
        ):
            raw_response = (
                camera_response.float()
                if action_response is None
                else 0.5 * (camera_response.float() + action_response.float())
            )
            if (
                not self.temporal_history_response_lagged_batch
                and (
                    not bool(torch.isfinite(raw_response).all())
                    or bool(torch.any(raw_response < 0))
                )
            ):
                raise RuntimeError("raw control response contains invalid values")
            self._raw_responses[int(layer_index)] = raw_response.detach()
        self._response_has_action[int(layer_index)] = action_response is not None
        if camera_frame_response is None and action_frame_response is not None:
            raise RuntimeError("action frame response lacks a camera profile")
        if (
            camera_frame_response is not None
            and action_response is not None
            and action_frame_response is None
        ):
            raise RuntimeError("fine camera/action response pair is incomplete")
        frame_response = None
        if camera_frame_response is not None:
            if camera_frame_response.ndim != 1 or not camera_frame_response.numel():
                raise RuntimeError("camera frame response must be a non-empty vector")
            if action_response is None:
                if action_frame_response is not None:
                    raise RuntimeError("camera-only layer received action frame response")
                frame_response = camera_frame_response.float()
            else:
                if (
                    action_frame_response is None
                    or tuple(action_frame_response.shape)
                    != tuple(camera_frame_response.shape)
                ):
                    raise RuntimeError("action frame response layout differs from camera")
                frame_response = 0.5 * (
                    camera_frame_response.float() + action_frame_response.float()
                )
            if self.runtime_optimized:
                torch._assert_async(
                    torch.all(
                        torch.isfinite(frame_response) & (frame_response >= 0)
                    ),
                    "fine control response contains invalid values",
                )
            elif not bool(torch.isfinite(frame_response).all()) or bool(
                torch.any(frame_response < 0)
            ):
                raise RuntimeError("fine control response contains invalid values")
            self._frame_responses[int(layer_index)] = frame_response.detach()
        self._response_records.append(
            {
                "call": self._call_index,
                "layer": int(layer_index),
                "has_action_module": action_response is not None,
                "execution_path": str(execution_path),
                "camera": self._stats(camera_response),
                "action": self._stats(action_response),
                "d": self._stats(response),
                "raw_d": self._stats(raw_response),
                "frame_response": self._stats(frame_response),
                "fine_frame_response_available": frame_response is not None,
                "fine_only": bool(fine_only),
            }
        )

    @staticmethod
    def relative_token_response(
        after: torch.Tensor, before: torch.Tensor
    ) -> torch.Tensor:
        after = after.float()
        before = before.float()
        eps = torch.finfo(before.dtype).eps
        return torch.linalg.vector_norm(after - before, dim=-1) / (
            torch.linalg.vector_norm(before, dim=-1) + eps
        )

    def _aggregate_dense_response(
        self,
        tokens: torch.Tensor,
        *,
        grid: tuple[int, int, int],
        memory_length: int,
    ) -> torch.Tensor:
        tokens = tokens.mean(dim=0) if tokens.ndim == 2 else tokens
        total_t, height, width = (int(value) for value in grid)
        values = tokens[: total_t * height * width].reshape(
            total_t, height, width
        )[int(memory_length) :]
        tt, th, tw = self._block_shape
        spatial_h, spatial_w = math.ceil(height / th), math.ceil(width / tw)
        memory_temporal = math.ceil(int(memory_length) / tt)
        current_offset = memory_temporal * tt - int(memory_length)
        if self._curvature_matrix is None:
            raise RuntimeError("control-response CWCA has no response layout")
        rows = []
        for time_block in range(int(self._curvature_matrix.shape[0])):
            low = current_offset + time_block * tt
            high = min(low + tt, int(values.shape[0]))
            spatial_rows = []
            for ih in range(spatial_h):
                for iw in range(spatial_w):
                    tile = values[
                        low:high,
                        ih * th : min((ih + 1) * th, height),
                        iw * tw : min((iw + 1) * tw, width),
                    ]
                    spatial_rows.append(
                        tile.mean() if tile.numel() else values.new_zeros(())
                    )
            rows.append(torch.stack(spatial_rows))
        return torch.stack(rows)

    @staticmethod
    def _aggregate_frame_response(
        tokens: torch.Tensor,
        *,
        grid: tuple[int, int, int],
        memory_length: int,
    ) -> torch.Tensor:
        tokens = tokens.mean(dim=0) if tokens.ndim == 2 else tokens
        total_t, height, width = (int(value) for value in grid)
        values = tokens[: total_t * height * width].reshape(
            total_t, height, width
        )[int(memory_length) :]
        if not values.numel():
            raise RuntimeError("fine control response has no Current frames")
        return values.float().mean(dim=(1, 2))

    def record_compact_layer_response(
        self,
        layer_index: int,
        *,
        camera_before: torch.Tensor,
        camera_after: torch.Tensor,
        action_before: torch.Tensor | None,
        action_after: torch.Tensor | None,
        grid: tuple[int, int, int],
        memory_length: int,
    ) -> None:
        if self._compact_layer_index != int(layer_index):
            raise RuntimeError("compact response recorded outside its layer")
        camera_tokens = self.relative_token_response(camera_after, camera_before)
        fine_only = bool(self.fine_only_response_reduction_enabled)
        camera = (
            torch.zeros_like(self._curvature_matrix)
            if fine_only
            else self._aggregate_dense_response(
                camera_tokens,
                grid=grid,
                memory_length=memory_length,
            )
        )
        if (action_before is None) != (action_after is None):
            raise RuntimeError("compact action response pair is incomplete")
        action = None
        action_frames = None
        if action_before is not None and action_after is not None:
            action_tokens = self.relative_token_response(action_after, action_before)
            action = (
                torch.zeros_like(self._curvature_matrix)
                if fine_only
                else self._aggregate_dense_response(
                    action_tokens,
                    grid=grid,
                    memory_length=memory_length,
                )
            )
            if self.fine_frame_response_enabled:
                action_frames = self._aggregate_frame_response(
                    action_tokens, grid=grid, memory_length=memory_length
                )
        camera_frames = (
            self._aggregate_frame_response(
                camera_tokens, grid=grid, memory_length=memory_length
            )
            if self.fine_frame_response_enabled
            else None
        )
        self.record_layer_response(
            int(layer_index),
            camera,
            action,
            execution_path="compact_frame_weave",
            camera_frame_response=camera_frames,
            action_frame_response=action_frames,
            fine_only=fine_only,
        )

    def record_compact_active_layer_response(
        self,
        layer_index: int,
        *,
        camera_before: torch.Tensor,
        camera_after: torch.Tensor,
        action_before: torch.Tensor | None,
        action_after: torch.Tensor | None,
        active_frame_values: tuple[int, ...],
        grid: tuple[int, int, int],
        memory_length: int,
    ) -> None:
        """Record fine control response from exact frames only.

        This path is valid only when attention budgeting is curvature-only
        (response weight zero).  It preserves the real camera/action response
        on every exact frame and reconstructs only the scalar per-frame
        scheduling signal for skipped Current frames with local bracketing
        anchors.  No hidden feature or attention output is reconstructed here.
        """

        if self._compact_layer_index != int(layer_index):
            raise RuntimeError("compact active response recorded outside its layer")
        if self.response_weight != 0.0:
            raise RuntimeError(
                "active-only compact response requires curvature-only attention"
            )
        total_t, height, width = (int(value) for value in grid)
        spatial = height * width
        if tuple(sorted(set(active_frame_values))) != active_frame_values:
            raise RuntimeError("active response frames must be chronological")
        if len(active_frame_values) < 2:
            raise RuntimeError("active response requires bracketing frames")

        def frame_profile(
            after: torch.Tensor, before: torch.Tensor
        ) -> torch.Tensor:
            tokens = self.relative_token_response(after, before)
            if tuple(tokens.shape) != (
                int(before.shape[0]),
                len(active_frame_values) * spatial,
            ):
                raise RuntimeError("compact response/token layout mismatch")
            exact = tokens.reshape(
                int(before.shape[0]), len(active_frame_values), spatial
            ).mean(dim=(0, 2))
            current_positions = [
                index
                for index, frame in enumerate(active_frame_values)
                if frame >= int(memory_length)
            ]
            current_frames = [
                active_frame_values[index] - int(memory_length)
                for index in current_positions
            ]
            current_count = total_t - int(memory_length)
            if (
                not current_frames
                or current_frames[0] != 0
                or current_frames[-1] != current_count - 1
            ):
                raise RuntimeError(
                    "active response Current frames lack structural endpoints"
                )
            exact_current = exact.index_select(
                0,
                torch.tensor(
                    current_positions,
                    device=exact.device,
                    dtype=torch.long,
                ),
            )
            rows = []
            for frame in range(current_count):
                right = bisect_left(current_frames, frame)
                if right < len(current_frames) and current_frames[right] == frame:
                    rows.append(exact_current[right])
                    continue
                if right == 0 or right == len(current_frames):
                    raise RuntimeError(
                        "active response target lacks bracketing exact frames"
                    )
                left = right - 1
                alpha = float(frame - current_frames[left]) / float(
                    current_frames[right] - current_frames[left]
                )
                rows.append(
                    (1.0 - alpha) * exact_current[left]
                    + alpha * exact_current[right]
                )
            return torch.stack(rows)

        camera_frames = frame_profile(camera_after, camera_before)
        if (action_before is None) != (action_after is None):
            raise RuntimeError("compact active action response pair is incomplete")
        action_frames = (
            frame_profile(action_after, action_before)
            if action_before is not None and action_after is not None
            else None
        )
        coarse_zero = torch.zeros_like(self._curvature_matrix)
        self.record_layer_response(
            int(layer_index),
            coarse_zero,
            coarse_zero if action_frames is not None else None,
            execution_path="compact_frame_weave",
            camera_frame_response=camera_frames,
            action_frame_response=action_frames,
        )
        self._compact_active_response_calls += 1

    def summary(self) -> dict[str, Any]:
        return self._materialize({
            "mode": self.mode,
            "response_weight": self.response_weight,
            "feedback_form": self.feedback_form,
            "fixed_query_budget": self.fixed_query_budget,
            "normalization": "per-spatial-worldline temporal mean",
            "curvature_temporal_pooling": self.curvature_temporal_pooling,
            "curvature_softmax_temperature": self.curvature_softmax_temperature,
            "curvature_pooling_profile": self._profile_curvature_pooling,
            "budget_records": self._budget_records,
            "compact_budget_records": self._compact_budget_records,
            "response_records": self._response_records,
            "compact_active_response_calls": self._compact_active_response_calls,
            "response_observation_gating": {
                "enabled": self.curvature_only_response_gating_enabled,
                "fine_only_response_reduction": (
                    self.fine_only_response_reduction_enabled
                ),
                "capture_active_at_summary": self.fine_frame_response_capture_active,
                "captured_layers": self._response_observation_captured_layers,
                "skipped_layers": self._response_observation_skipped_layers,
                "attention_budget_response_independent": self.response_weight == 0.0,
            },
            "temporal_history_response": {
                "enabled": self.temporal_history_response_enabled,
                "gated": self.temporal_history_response_gated,
                "lagged_batch": self.temporal_history_response_lagged_batch,
                "call_period_router": self.lagged_call_period_router_enabled,
                "capture_active_at_summary": self.temporal_history_capture_active,
                "raw_response_records": sum(
                    isinstance(record.get("raw_d"), dict)
                    and record["raw_d"].get("mean") is not None
                    for record in self._response_records
                ),
                "eligible_calls_finalized": len(
                    self._lagged_batch_finalize_records
                ),
                "eligible_call_index": self._lagged_eligible_call_index,
                "batch_finalize_records": self._lagged_batch_finalize_records,
                "call_period_history_count": len(
                    self._call_period_energy_history
                ),
                "single_device_to_host_batch_per_eligible_call": bool(
                    self.temporal_history_response_lagged_batch
                ) and all(
                    record.get("single_device_to_host_batch") is True
                    for record in self._lagged_batch_finalize_records
                ),
            },
            "sensitivity_diagnostic": {
                "kind": "qk_omitted_probability_mass_proxy",
                "records": self._sensitivity_records,
            },
            "all_global_budgets_exact": bool(self._budget_records)
            and torch.stack([
                torch.as_tensor(record["budget_exact"])
                for record in self._budget_records
            ]).all(),
            "all_compact_global_budgets_exact": bool(self._compact_budget_records)
            and all(
                bool(record["budget_exact"])
                for record in self._compact_budget_records
            ),
            "response_reduction": self._response_reduction,
            "runtime_optimization": {
                "enabled": self.runtime_optimized,
                "response_validation": (
                    "cuda_async_assert"
                    if self.runtime_optimized
                    else "per_layer_host_boolean"
                ),
                "response_math_unchanged": True,
            },
        })

    def reset_runtime_state(self) -> None:
        self._budget_records.clear()
        self._compact_budget_records.clear()
        self._response_records.clear()
        self._compact_active_response_calls = 0
        self._response_observation_captured_layers = 0
        self._response_observation_skipped_layers = 0
        self.fine_frame_response_capture_active = bool(
            not self.curvature_only_response_gating_enabled
        )
        self._responses.clear()
        self._raw_responses.clear()
        self._frame_responses.clear()
        self._response_has_action.clear()
        self._frame_selection_energy_history.clear()
        self._frame_selection_payload_cache.clear()
        self._lagged_frame_selection_payloads.clear()
        self._lagged_call_period_payload = None
        self._call_period_energy_history.clear()
        self._lagged_batch_finalize_records.clear()
        self._lagged_eligible_call_index = -1
        self._sensitivity_records.clear()
        self._pending_budget_record = None
        self._compact_layer_index = None
        self._call_index = -1
        self.temporal_history_capture_active = bool(
            self.temporal_history_response_enabled
            and not self.temporal_history_response_gated
        )

    @staticmethod
    def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.float().flatten()
        right = right.float().flatten()
        left_rank = torch.argsort(torch.argsort(left, stable=True), stable=True).float()
        right_rank = torch.argsort(torch.argsort(right, stable=True), stable=True).float()
        left_rank -= left_rank.mean()
        right_rank -= right_rank.mean()
        denominator = torch.linalg.vector_norm(left_rank) * torch.linalg.vector_norm(right_rank)
        if not bool(denominator > 0):
            return float("nan")
        return float((left_rank * right_rank).sum().div(denominator).item())

    def select(self, q: torch.Tensor, k: torch.Tensor, **kwargs):
        selected, counts, report = super().select(q, k, **kwargs)
        if self._pending_budget_record is not None:
            self._budget_records.append(self._pending_budget_record)
            self._pending_budget_record = None
        if (
            self._diagnostic_calls > self._call_index
            and self._layer_index > 0
            and self._layer_index - 1 in self._responses
        ):
            height, width = (int(value) for value in kwargs["latent_hw"])
            temporal = q.shape[2] // (height * width)
            q_blocks = _tile_visual_tensor(
                q, temporal=temporal, height=height, width=width,
                block_shape=self._block_shape,
            )
            k_blocks = _tile_visual_tensor(
                k, temporal=temporal, height=height, width=width,
                block_shape=self._block_shape,
            )
            q_content = torch.nn.functional.normalize(
                q_blocks.float().mean(dim=-2), dim=-1
            )
            k_content = torch.nn.functional.normalize(
                k_blocks.float().mean(dim=-2), dim=-1
            )
            probability = torch.softmax(
                torch.matmul(q_content, k_content.transpose(-2, -1)), dim=-1
            )
            retained = torch.gather(probability, -1, selected)
            valid = torch.arange(selected.shape[-1], device=q.device)
            valid = valid[None, None, None, :] < counts[..., None]
            sensitivity = 1.0 - (retained * valid).sum(dim=-1)
            sensitivity = sensitivity.mean(dim=(0, 1))[self._profile_memory_blocks :]
            response = self._responses[self._layer_index - 1]
            sensitivity = sensitivity.reshape_as(response)
            self._sensitivity_records.append(
                {
                    "call": self._call_index,
                    "previous_layer": self._layer_index - 1,
                    "next_layer": self._layer_index,
                    "spearman": self._spearman(response, sensitivity),
                    "sensitivity": self._stats(sensitivity),
                }
            )
        return selected, counts, report


class MatrixResponseOnlyCWCAAttentionCompiler(MatrixControlResponseCWCAAttentionCompiler):
    def __init__(self) -> None:
        super().__init__("response_only")


class MatrixClosedLoopCWCAAttentionCompiler(MatrixControlResponseCWCAAttentionCompiler):
    def __init__(
        self,
        response_weight: float = 1.0,
        feedback_form: str = "additive",
        curvature_temporal_pooling: str = "mean",
        curvature_softmax_temperature: float = 1.0,
        fixed_query_budget: bool = False,
    ) -> None:
        super().__init__(
            "closed_loop",
            response_weight=response_weight,
            feedback_form=feedback_form,
            curvature_temporal_pooling=curvature_temporal_pooling,
            curvature_softmax_temperature=curvature_softmax_temperature,
            fixed_query_budget=fixed_query_budget,
        )


class MatrixControlResponseObserver:
    """Observe Camera Injection and ActionModule residuals without model edits."""

    def __init__(self, model: torch.nn.Module, compiler: MatrixControlResponseCWCAAttentionCompiler):
        self.compiler = compiler
        self._use_fused = os.environ.pop(
            "WORLDMARK_CONTROL_RESPONSE_FUSED", "0"
        ) == "1"
        if self._use_fused:
            self.compiler._response_reduction["mode"] = "fused_triton"
        self._states: dict[int, dict[str, Any]] = {}
        self._handles: list[Any] = []
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise RuntimeError("control-response observer requires model.blocks")
        for layer_index, block in enumerate(blocks):
            self._handles.append(block.register_forward_pre_hook(
                self._block_pre_hook(layer_index, block), with_kwargs=True
            ))
            self._handles.append(block.self_attn.register_forward_hook(
                self._self_attention_hook(layer_index)
            ))
            self._handles.append(block.self_attn.register_forward_pre_hook(
                self._self_attention_pre_hook(layer_index)
            ))
            self._handles.append(block.norm3.register_forward_pre_hook(
                self._camera_hook(layer_index)
            ))
            if block.action_model is not None:
                self._handles.append(block.action_model.register_forward_pre_hook(
                    self._action_pre_hook(layer_index), with_kwargs=True
                ))
                self._handles.append(block.action_model.register_forward_hook(
                    self._action_post_hook(layer_index)
                ))
            self._handles.append(block.register_forward_hook(
                self._block_post_hook(layer_index)
            ))

    def _block_pre_hook(self, layer: int, block: torch.nn.Module):
        def hook(_module, args, kwargs):
            if self.compiler._curvature_matrix is None:
                return
            if not self.compiler.response_observation_active():
                self._states.pop(layer, None)
                self.compiler._response_observation_skipped_layers += 1
                return
            self.compiler._response_observation_captured_layers += 1
            x = args[0] if args else kwargs["x"]
            e = args[1] if len(args) > 1 else kwargs["e"]
            grid_sizes = args[3] if len(args) > 3 else kwargs["grid_sizes"]
            memory_length = kwargs.get("memory_length", args[13] if len(args) > 13 else 0)
            plucker = kwargs.get("plucker_emb", args[10] if len(args) > 10 else None)
            self._states[layer] = {
                "x": x,
                "e": e,
                "grid": tuple(int(v) for v in grid_sizes[0].tolist()),
                "memory_length": int(memory_length),
                "camera_enabled": plucker is not None,
                "block": block,
            }
        return hook

    def _self_attention_hook(self, layer: int):
        def hook(_module, _args, output):
            state = self._states.get(layer)
            if state is not None:
                state["self_attention_output"] = output
        return hook

    def _self_attention_pre_hook(self, layer: int):
        def hook(_module, _args):
            if self.compiler._curvature_matrix is not None:
                self.compiler.set_layer_index(layer)
        return hook

    @staticmethod
    def _relative_response(after: torch.Tensor, before: torch.Tensor) -> torch.Tensor:
        return MatrixControlResponseCWCAAttentionCompiler.relative_token_response(
            after, before
        )

    def _camera_hook(self, layer: int):
        def hook(_module, args):
            state = self._states.get(layer)
            if self.compiler.observing_compact_layer(layer):
                return
            if state is None or not state.get("camera_enabled") or not args:
                return
            y = state.get("self_attention_output")
            if y is None:
                raise RuntimeError("camera observer did not see self-attention output")
            block = state["block"]
            if self._use_fused:
                state["camera_tiles"] = fused_camera_response_tiles(
                    args[0], state["x"], y, state["e"], block.modulation,
                    grid=state["grid"], memory=state["memory_length"],
                    block_shape=self.compiler._block_shape,
                    temporal_blocks=self.compiler._curvature_matrix.shape[0],
                ).detach()
                self.compiler._response_reduction["camera_fused_calls"] += 1
            else:
                e2 = state["e"][:, :, 2, :] + block.modulation[0, 2, :]
                before = state["x"].float() + y.float() * e2.float()
                state["camera_tokens"] = self._relative_response(args[0], before).detach()
        return hook

    def _action_pre_hook(self, layer: int):
        def hook(_module, args, _kwargs):
            state = self._states.get(layer)
            if self.compiler.observing_compact_layer(layer):
                return
            if state is not None and args:
                # ActionModule returns a new tensor and does not mutate its input;
                # retain the inference tensor instead of issuing a full clone.
                state["action_before"] = args[0].detach()
        return hook

    def _action_post_hook(self, layer: int):
        def hook(_module, _args, output):
            state = self._states.get(layer)
            if self.compiler.observing_compact_layer(layer):
                return
            if state is not None and "action_before" in state:
                before = state.pop("action_before")
                if self._use_fused:
                    state["action_tiles"] = fused_action_response_tiles(
                        output, before, grid=state["grid"],
                        memory=state["memory_length"],
                        block_shape=self.compiler._block_shape,
                        temporal_blocks=self.compiler._curvature_matrix.shape[0],
                    ).detach()
                    self.compiler._response_reduction["action_fused_calls"] += 1
                else:
                    state["action_tokens"] = self._relative_response(
                        output, before
                    ).detach()
        return hook

    def _aggregate(self, tokens: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        tokens = tokens.mean(dim=0) if tokens.ndim == 2 else tokens
        total_t, height, width = state["grid"]
        memory = state["memory_length"]
        valid = total_t * height * width
        values = tokens[:valid].reshape(total_t, height, width)[memory:]
        tt, th, tw = self.compiler._block_shape
        spatial_h = math.ceil(height / th)
        spatial_w = math.ceil(width / tw)
        memory_temporal = math.ceil(memory / tt)
        current_offset = memory_temporal * tt - memory
        temporal_blocks = self.compiler._curvature_matrix.shape[0]
        rows = []
        for time_block in range(temporal_blocks):
            low = current_offset + time_block * tt
            high = min(low + tt, values.shape[0])
            spatial_rows = []
            for ih in range(spatial_h):
                for iw in range(spatial_w):
                    tile = values[low:high, ih * th : min((ih + 1) * th, height), iw * tw : min((iw + 1) * tw, width)]
                    spatial_rows.append(tile.mean() if tile.numel() else values.new_zeros(()))
            rows.append(torch.stack(spatial_rows))
        return torch.stack(rows)

    def _block_post_hook(self, layer: int):
        def hook(_module, _args, _output):
            state = self._states.pop(layer, None)
            if state is None or not (
                "camera_tokens" in state or "camera_tiles" in state
            ):
                return
            fine_only = bool(
                self.compiler.fine_only_response_reduction_enabled
            )
            camera = state.get("camera_tiles")
            camera_frames = None
            camera_tokens = None
            if camera is None:
                camera_tokens = state["camera_tokens"]
                camera = (
                    torch.zeros_like(self.compiler._curvature_matrix)
                    if fine_only
                    else self._aggregate(camera_tokens, state)
                )
            action_frames = None
            action = (
                state["action_tiles"]
                if "action_tiles" in state
                else torch.zeros_like(self.compiler._curvature_matrix)
                if fine_only and "action_tokens" in state
                else self._aggregate(state["action_tokens"], state)
                if "action_tokens" in state
                else None
            )
            action_tokens = state.get("action_tokens")
            if self.compiler.fine_frame_response_enabled and camera_tokens is not None:
                camera_frames = self.compiler._aggregate_frame_response(
                    camera_tokens,
                    grid=state["grid"],
                    memory_length=state["memory_length"],
                )
                if action is not None:
                    if action_tokens is None:
                        raise RuntimeError(
                            "fine frame response is incompatible with fused response tiles"
                        )
                    action_frames = self.compiler._aggregate_frame_response(
                        action_tokens,
                        grid=state["grid"],
                        memory_length=state["memory_length"],
                    )
            self.compiler.record_layer_response(
                layer,
                camera,
                action,
                camera_frame_response=camera_frames,
                action_frame_response=action_frames,
                fine_only=fine_only,
            )
        return hook
