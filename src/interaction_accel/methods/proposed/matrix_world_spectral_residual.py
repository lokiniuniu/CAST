"""World-aligned spectral correction for Matrix frame-weave residuals.

This module deliberately operates *after* the existing Module-3 residual
reconstruction.  It never interpolates hidden states and it never uses the
relative Pluecker embedding as an absolute camera pose.  The only state it
changes is the residual of inactive Current frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


class _NoOpCudaEvent:
    """Drop-in timing marker used when production profiling is disabled."""

    def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None

from .matrix_fc_pasm_kernel import (
    fc_extract_overlap_tiles,
    fc_phat_peak_subpixel,
    fc_ramp_confidence,
    fc_phase_mix,
    fc_phase_mix_batched,
    fc_unphase_overlap_tiles,
)


@dataclass(frozen=True)
class MatrixWorldAlignmentGeometry:
    """Absolute camera metadata for one Matrix chunk."""

    current_abs_c2ws: torch.Tensor
    memory_abs_c2ws: torch.Tensor
    base_K: torch.Tensor
    target_h: int
    target_w: int
    current_latent_indices: tuple[float, ...]
    memory_latent_indices: tuple[int, ...]

    def validate(self, *, memory_length: int, current_length: int) -> None:
        if self.current_abs_c2ws.shape != (current_length, 4, 4):
            raise RuntimeError(
                "Current absolute-pose count does not match Current latents: "
                f"{tuple(self.current_abs_c2ws.shape)} versus {current_length}"
            )
        if self.memory_abs_c2ws.shape != (memory_length, 4, 4):
            raise RuntimeError(
                "Memory absolute-pose count does not match Memory latents: "
                f"{tuple(self.memory_abs_c2ws.shape)} versus {memory_length}"
            )
        if self.base_K.numel() != 4:
            raise RuntimeError("Matrix base_K must contain fx, fy, cx, cy")
        if self.target_h <= 0 or self.target_w <= 0:
            raise RuntimeError("Matrix target image geometry must be positive")
        if len(self.current_latent_indices) != current_length:
            raise RuntimeError("Current latent-index certificate is incomplete")
        if len(self.memory_latent_indices) != memory_length:
            raise RuntimeError("Memory latent-index certificate is incomplete")

    def to(self, device: torch.device) -> "MatrixWorldAlignmentGeometry":
        return MatrixWorldAlignmentGeometry(
            current_abs_c2ws=self.current_abs_c2ws.to(
                device=device, dtype=torch.float32
            ),
            memory_abs_c2ws=self.memory_abs_c2ws.to(
                device=device, dtype=torch.float32
            ),
            base_K=self.base_K.to(device=device, dtype=torch.float32).reshape(4),
            target_h=self.target_h,
            target_w=self.target_w,
            current_latent_indices=self.current_latent_indices,
            memory_latent_indices=self.memory_latent_indices,
        )


class MatrixWorldGeometryRuntime:
    """Capture absolute geometry at the interactive-pipeline boundary.

    The upstream tree remains read-only.  The adapter wraps the pipeline's
    imported ``build_plucker_from_c2ws`` only to retain its *pre-relative*
    intrinsics/index arguments, and records the exact ORBIT/native atoms that
    are subsequently assembled as Memory.  No generated tensor is modified.
    """

    def __init__(self) -> None:
        self.reset()
        self._installed_globals: MutableMapping[str, Any] | None = None
        self._native_build_plucker: Any = None

    def reset(self) -> None:
        self._base_K: torch.Tensor | None = None
        self._target_h: int | None = None
        self._target_w: int | None = None
        self._current_latent_indices: tuple[float, ...] = ()
        self._current_abs_c2ws: torch.Tensor | None = None
        self._memory_abs_c2ws: torch.Tensor | None = None
        self._memory_latent_indices: tuple[int, ...] = ()
        self._capture_count = 0

    def install_pipeline_capture(
        self, generate_globals: MutableMapping[str, Any]
    ) -> None:
        if self._installed_globals is not None:
            raise RuntimeError("absolute-pose pipeline capture was installed twice")
        native = generate_globals.get("build_plucker_from_c2ws")
        if not callable(native):
            raise RuntimeError("Matrix pipeline lost build_plucker_from_c2ws")
        self._installed_globals = generate_globals
        self._native_build_plucker = native

        def captured_build_plucker(*args: Any, **kwargs: Any) -> torch.Tensor:
            result = native(*args, **kwargs)
            base_K = kwargs.get("base_K")
            target_h = kwargs.get("target_h")
            target_w = kwargs.get("target_w")
            tgt_indices = kwargs.get("tgt_indices")
            if base_K is None and len(args) >= 5:
                base_K = args[4]
            if target_h is None and len(args) >= 6:
                target_h = args[5]
            if target_w is None and len(args) >= 7:
                target_w = args[6]
            if tgt_indices is None and len(args) >= 3:
                tgt_indices = args[2]
            if not isinstance(base_K, torch.Tensor):
                raise RuntimeError("Matrix pipeline did not expose base_K")
            indices = np.asarray(tgt_indices, dtype=np.float64).reshape(-1)
            self._base_K = base_K.detach().float().cpu().clone()
            self._target_h = int(target_h)
            self._target_w = int(target_w)
            self._current_latent_indices = tuple(float(v) for v in indices.tolist())
            # A new Current Pluecker build starts a new chunk.  The exact
            # Current poses arrive at sparse-layout construction below.
            self._current_abs_c2ws = None
            self._memory_abs_c2ws = None
            self._memory_latent_indices = ()
            return result

        generate_globals["build_plucker_from_c2ws"] = captured_build_plucker

    def capture_selected_world_poses(
        self,
        *,
        active_atoms: Sequence[Any],
        current_c2ws: torch.Tensor,
    ) -> None:
        if not isinstance(current_c2ws, torch.Tensor):
            raise RuntimeError("Matrix current absolute c2w tensor is missing")
        if current_c2ws.ndim != 3 or current_c2ws.shape[-2:] != (4, 4):
            raise RuntimeError("Matrix current absolute c2ws must be [T,4,4]")
        memory_poses: list[torch.Tensor] = []
        memory_indices: list[int] = []
        for atom in active_atoms:
            pose = getattr(atom, "c2w", None)
            index = getattr(atom, "original_time_index", None)
            if not isinstance(pose, torch.Tensor) or index is None:
                raise RuntimeError("selected Matrix Memory atom lacks c2w/index")
            memory_poses.append(pose.detach().float().cpu().clone())
            memory_indices.append(int(index))
        self._current_abs_c2ws = current_c2ws.detach().float().cpu().clone()
        self._memory_abs_c2ws = (
            torch.stack(memory_poses)
            if memory_poses
            else torch.empty((0, 4, 4), dtype=torch.float32)
        )
        self._memory_latent_indices = tuple(memory_indices)
        self._capture_count += 1

    def snapshot(self, *, memory_length: int) -> MatrixWorldAlignmentGeometry | None:
        if memory_length <= 0:
            return None
        if (
            self._base_K is None
            or self._target_h is None
            or self._target_w is None
            or self._current_abs_c2ws is None
            or self._memory_abs_c2ws is None
        ):
            raise RuntimeError("world alignment requested before pipeline geometry capture")
        geometry = MatrixWorldAlignmentGeometry(
            current_abs_c2ws=self._current_abs_c2ws,
            memory_abs_c2ws=self._memory_abs_c2ws,
            base_K=self._base_K,
            target_h=self._target_h,
            target_w=self._target_w,
            current_latent_indices=self._current_latent_indices,
            memory_latent_indices=self._memory_latent_indices,
        )
        geometry.validate(
            memory_length=memory_length,
            current_length=int(self._current_abs_c2ws.shape[0]),
        )
        return geometry

    def certificate(self) -> dict[str, Any]:
        return {
            "capture_count": self._capture_count,
            "absolute_current_before_relative_pluecker": True,
            "absolute_memory_from_selected_atoms": True,
            "relative_pluecker_used_for_alignment": False,
            "selected_index_latent_mapping_preserved": True,
            "base_intrinsics_captured": self._base_K is not None,
        }


class WorldAlignedSpectralResidualCorrector:
    """Correct inactive Current residuals from absolute world-ray matches."""

    MODES = frozenset(
        {
            "fixed_lowpass",
            "world_aligned",
            "self_calibrated",
            "current_anchored_self_calibrated",
            "current_anchored_phase_transport",
            "current_anchored_polar_mixing",
            "phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_precision_regime",
            "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
            "overlap_local_linear_phase_ramp_cv_gate",
            "frequency_confidence_phase_aligned_spectral_mixing",
            "coarse_probe_lowfreq_self_calibrated",
        }
    )

    def __init__(
        self,
        *,
        mode: str,
        align_depth_samples: int = 10,
        align_top_l: int = 4,
        spectral_num_bands: int = 4,
        gamma_max: float = 1.5,
        ridge: float = 1e-4,
        eta: float = 0.25,
        gamma_ema: float = 0.9,
        depth_near: float = 0.1,
        depth_far: float = 30.0,
        block_shape: tuple[int, int, int] = (4, 4, 8),
        current_anchor_boundary: int = 4,
        current_anchor_match_tile: tuple[int, int] = (2, 4),
        current_anchor_descriptor_groups: int = 96,
        current_anchor_consensus_mix: float = 0.5,
        regime_gain_margin: float = 0.05,
        regime_win_rate: float = 0.75,
        regime_coherence: float = 0.5,
        regime_min_calibration_anchors: int = 2,
        fc_tau_low: float = 0.3,
        fc_tau_high: float = 0.6,
        fc_freq_power: float = 1.0,
        fc_temperature: float = 0.05,
        fc_ramp_confidence: bool = False,
        fc_temporal_consistency: bool = False,
        fc_trust_eta: float = 0.0,
        fc_layer_gate_threshold: float = -1.0,
        fc_layer_gate_period: int = 0,
        fc_pair_gate_threshold: float = -1.0,
        fc_transport_tile_topk: int = 0,
        fc_fused_complex_weights: bool = False,
        fc_lean_runtime: bool = False,
        fc_reference_numerics: bool = False,
        fc_reference_layers: tuple[int, ...] = (),
        fc_zero_transport_fastpath: bool = False,
        fc_elide_scalar_readback: bool = False,
        fc_v21_bypass: bool = False,
        fc_v21_blend: float = 0.0,
        fc_parallel_streams: int = 0,
        fc_prealloc_targets: bool = False,
        fc_prune_unused_pairs: bool = False,
        fc_batched_endpoint_stats: bool = False,
        fc_active_layers: tuple[int, ...] = (),
        fc_coarse_transport: bool = False,
        fc_triton_phase_mix: bool = False,
        fc_triton_accurate_phase_mix: bool = False,
        fc_triton_batched_mix: bool = False,
        fc_triton_tile_extract: bool = False,
        fc_triton_ola: bool = False,
        fc_triton_phat_peak: bool = False,
        fc_triton_ramp_confidence: bool = False,
        fc_triton_reference_lowfreq: bool = False,
        fc_triton_lowfreq_radius: float = 0.35,
        fc_legacy_lowfreq_index_bug: bool = False,
        fc_profile_timing: bool = True,
        fc_batched_reference_mix: bool = False,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown world-spectral mode {mode!r}")
        if not 8 <= int(align_depth_samples) <= 12:
            raise ValueError("align_depth_samples must be in [8,12]")
        if int(align_top_l) < 1:
            raise ValueError("align_top_l must be positive")
        if int(spectral_num_bands) != 4:
            raise ValueError("the validated radial partition uses exactly four bands")
        if not 0.0 <= float(regime_gain_margin) <= 1.0:
            raise ValueError("regime gain margin must be in [0,1]")
        if not 0.0 <= float(regime_win_rate) <= 1.0:
            raise ValueError("regime win rate must be in [0,1]")
        if not 0.0 <= float(regime_coherence) <= 1.0:
            raise ValueError("regime coherence must be in [0,1]")
        if int(regime_min_calibration_anchors) < 1:
            raise ValueError("regime minimum calibration anchors must be positive")
        if not 0.0 <= float(fc_tau_low) <= float(fc_tau_high) <= 1.0:
            raise ValueError("FC-PASM thresholds must satisfy 0 <= low <= high <= 1")
        if not math.isfinite(fc_freq_power) or float(fc_freq_power) <= 0.0:
            raise ValueError("FC-PASM frequency power must be finite and positive")
        if not math.isfinite(fc_temperature) or float(fc_temperature) <= 0.0:
            raise ValueError("FC-PASM temperature must be finite and positive")
        if not math.isfinite(fc_triton_lowfreq_radius) or not (
            0.0 < float(fc_triton_lowfreq_radius) <= 1.0
        ):
            raise ValueError("FC Triton low-frequency radius must lie in (0,1]")
        if bool(fc_temporal_consistency) and not bool(fc_ramp_confidence):
            raise ValueError("FC temporal consistency requires ramp confidence")
        if not math.isfinite(fc_trust_eta) or not 0.0 <= float(fc_trust_eta) <= 1.0:
            raise ValueError("FC trust eta must be finite and lie in [0,1]")
        # The residual trust region is an independent safety bound.  It may be
        # used with endpoint-only confidence (the default) or with the optional
        # temporal/ramp confidence products; tying it to the latter would make
        # a trust-only ablation impossible and can silently zero the affinity.
        if not math.isfinite(fc_layer_gate_threshold) or not (
            float(fc_layer_gate_threshold) == -1.0
            or 0.0 <= float(fc_layer_gate_threshold) <= 1.0
        ):
            raise ValueError("FC layer gate threshold must be -1 or lie in [0,1]")
        if int(fc_layer_gate_period) < 0:
            raise ValueError("FC layer gate period must be non-negative")
        if float(fc_layer_gate_threshold) >= 0.0 and int(fc_layer_gate_period) < 2:
            raise ValueError("FC layer gate period must be >=2 when enabled")
        if not math.isfinite(fc_pair_gate_threshold) or not (
            float(fc_pair_gate_threshold) == -1.0
            or 0.0 <= float(fc_pair_gate_threshold) <= 1.0
        ):
            raise ValueError("FC pair gate threshold must be -1 or lie in [0,1]")
        if not 0 <= int(fc_transport_tile_topk) <= 18:
            raise ValueError("FC transport tile top-k must lie in [0,18]")
        normalized_fc_layers = tuple(sorted({int(layer) for layer in fc_active_layers}))
        if any(layer < 0 or layer >= 30 for layer in normalized_fc_layers):
            raise ValueError("FC active layers must lie in [0,29]")
        if len(normalized_fc_layers) != len(tuple(fc_active_layers)):
            raise ValueError("FC active layers must be unique")
        normalized_fc_reference_layers = tuple(
            sorted({int(layer) for layer in fc_reference_layers})
        )
        if any(layer < 0 or layer >= 30 for layer in normalized_fc_reference_layers):
            raise ValueError("FC reference layers must lie in [0,29]")
        if len(normalized_fc_reference_layers) != len(tuple(fc_reference_layers)):
            raise ValueError("FC reference layers must be unique")
        if int(fc_parallel_streams) not in {0, 2, 4, 8}:
            raise ValueError("FC parallel streams must be one of 0, 2, 4, or 8")
        if not math.isfinite(fc_v21_blend) or not 0.0 <= float(fc_v21_blend) <= 0.25:
            raise ValueError("FC V21 blend must be finite and lie in [0,0.25]")
        if not math.isfinite(gamma_max) or gamma_max < 0.0:
            raise ValueError("gamma_max must be finite and non-negative")
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        if not math.isfinite(eta) or eta < 0.0:
            raise ValueError("eta must be finite and non-negative")
        if not 0.0 <= gamma_ema < 1.0:
            raise ValueError("gamma_ema must lie in [0,1)")
        if not 0.0 < depth_near < depth_far:
            raise ValueError("invalid world-alignment depth interval")
        if block_shape != (4, 4, 8):
            raise ValueError("Matrix world correction must share CWCA's (4,4,8) tile")
        if not 1 <= int(current_anchor_boundary) <= 13:
            raise ValueError("current-anchor calibration boundary must lie in (0,14)")
        if tuple(current_anchor_match_tile) != (2, 4):
            raise ValueError("validated current-anchor matching tile is (2,4)")
        if int(current_anchor_descriptor_groups) <= 0:
            raise ValueError("current-anchor descriptor groups must be positive")
        if not 0.0 <= float(current_anchor_consensus_mix) <= 1.0:
            raise ValueError("current-anchor consensus mix must lie in [0,1]")
        self.mode = mode
        self.align_depth_samples = int(align_depth_samples)
        self.align_top_l = int(align_top_l)
        self.spectral_num_bands = int(spectral_num_bands)
        self.gamma_max = float(gamma_max)
        self.ridge = float(ridge)
        self.eta = float(eta)
        self.gamma_ema = float(gamma_ema)
        self.depth_near = float(depth_near)
        self.depth_far = float(depth_far)
        self.block_shape = block_shape
        self.current_anchor_boundary = int(current_anchor_boundary)
        self.current_anchor_match_tile = tuple(current_anchor_match_tile)
        self.current_anchor_descriptor_groups = int(
            current_anchor_descriptor_groups
        )
        self.current_anchor_consensus_mix = float(current_anchor_consensus_mix)
        self.regime_gain_margin = float(regime_gain_margin)
        self.regime_win_rate = float(regime_win_rate)
        self.regime_coherence = float(regime_coherence)
        self.regime_min_calibration_anchors = int(
            regime_min_calibration_anchors
        )
        self.fc_tau_low = float(fc_tau_low)
        self.fc_tau_high = float(fc_tau_high)
        self.fc_freq_power = float(fc_freq_power)
        self.fc_temperature = float(fc_temperature)
        self.fc_ramp_confidence = bool(fc_ramp_confidence)
        self.fc_temporal_consistency = bool(fc_temporal_consistency)
        self.fc_trust_eta = float(fc_trust_eta)
        self.fc_layer_gate_threshold = float(fc_layer_gate_threshold)
        self.fc_layer_gate_period = int(fc_layer_gate_period)
        self.fc_pair_gate_threshold = float(fc_pair_gate_threshold)
        self.fc_transport_tile_topk = int(fc_transport_tile_topk)
        self.fc_fused_complex_weights = bool(fc_fused_complex_weights)
        self.fc_lean_runtime = bool(fc_lean_runtime)
        self.fc_reference_numerics = bool(fc_reference_numerics)
        self.fc_reference_layers = normalized_fc_reference_layers
        self.fc_zero_transport_fastpath = bool(fc_zero_transport_fastpath)
        # Runtime-only optimization: this scalar is diagnostic-only unless
        # the zero-transport fast path is enabled.  Avoid a host readback in
        # the canonical lean FC-R route while leaving all tensor arithmetic,
        # routing inputs, and residual values unchanged.
        self.fc_elide_scalar_readback = bool(fc_elide_scalar_readback)
        self.fc_v21_bypass = bool(fc_v21_bypass)
        # A small, explicit quality safety blend is kept separate from the
        # forbidden global V21 bypass.  FC-PASM and ROCSA-A still execute for
        # every woven site; this only contracts the resulting residual toward
        # the already-computed V21 interpolation by a bounded amount.
        self.fc_v21_blend = float(fc_v21_blend)
        self.fc_parallel_streams = int(fc_parallel_streams)
        self.fc_prealloc_targets = bool(fc_prealloc_targets)
        self.fc_prune_unused_pairs = bool(fc_prune_unused_pairs)
        # Runtime-only launch fusion.  The batched helper preserves the
        # endpoint equations and reference numerics; it only folds the tiny
        # pair axis into one reduction launch.  Keep it opt-in until an
        # exact-output smoke has passed against the stream0 reference.
        self.fc_batched_endpoint_stats = bool(fc_batched_endpoint_stats)
        self.fc_active_layers = normalized_fc_layers
        self.fc_coarse_transport = bool(fc_coarse_transport)
        self.fc_triton_phase_mix = bool(fc_triton_phase_mix)
        self.fc_triton_accurate_phase_mix = bool(
            fc_triton_accurate_phase_mix
        )
        if self.fc_triton_accurate_phase_mix and not self.fc_triton_phase_mix:
            raise ValueError("accurate phase mix requires the Triton phase kernel")
        self.fc_triton_batched_mix = bool(fc_triton_batched_mix)
        self.fc_triton_tile_extract = bool(fc_triton_tile_extract)
        self.fc_triton_ola = bool(fc_triton_ola)
        self.fc_triton_phat_peak = bool(fc_triton_phat_peak)
        self.fc_triton_ramp_confidence = bool(fc_triton_ramp_confidence)
        self.fc_triton_reference_lowfreq = bool(fc_triton_reference_lowfreq)
        self.fc_triton_lowfreq_radius = float(fc_triton_lowfreq_radius)
        self.fc_legacy_lowfreq_index_bug = bool(fc_legacy_lowfreq_index_bug)
        self.fc_profile_timing = bool(fc_profile_timing)
        self.fc_batched_reference_mix = bool(fc_batched_reference_mix)
        if self.fc_batched_reference_mix and self.fc_triton_phase_mix:
            raise ValueError(
                "batched reference mix and Triton phase mix are exclusive"
            )
        if self.fc_batched_reference_mix and self.fc_fused_complex_weights:
            raise ValueError(
                "batched reference mix and fused complex weights are exclusive"
            )
        if self.fc_triton_reference_lowfreq and not self.fc_triton_phase_mix:
            raise ValueError(
                "reference low-frequency splice requires the Triton phase mix"
            )
        if self.fc_triton_batched_mix and not self.fc_triton_phase_mix:
            raise ValueError("batched FC Triton mix requires the Triton phase kernel")
        if self.fc_triton_batched_mix and self.fc_coarse_transport:
            raise ValueError("batched FC Triton mix and coarse transport are exclusive")
        self.reset_runtime_state()

    def _timing_event(self) -> torch.cuda.Event | _NoOpCudaEvent:
        if self.fc_profile_timing:
            return torch.cuda.Event(enable_timing=True)
        return _NoOpCudaEvent()

    def reset_runtime_state(self) -> None:
        self._geometry: MatrixWorldAlignmentGeometry | None = None
        self._chunk = -1
        self._step = -1
        self._ema_by_layer: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        self._polar_ema_by_layer: dict[
            Any, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._last_ema: torch.Tensor | None = None
        self._fc_previous_mean_affinity: torch.Tensor | None = None
        # Tiny detached transport sketches consumed only by the optional
        # sparse-routing refinement in the following Transformer layer.  The
        # generated residual path never reads this cache.
        self._fc_routing_transport: dict[tuple[int, int], dict[str, Any]] = {}
        self._fc_layer_gate_skips = 0
        self._fc_layer_schedule_bypasses = 0
        self._fc_pair_gate_skipped_targets = 0
        self._fc_pair_gate_skipped_pairs = 0
        self._fc_pair_gate_runtime_active = True
        self._fc_pair_gate_active_calls = 0
        self._fc_pair_gate_inactive_calls = 0
        self._fc_runtime_active = True
        self._fc_causal_v21_bypass_calls = 0
        self._fc_stream_pool: dict[str, tuple[torch.cuda.Stream, ...]] = {}
        self._current_projection_cache: dict[
            tuple[Any, ...], tuple[list[torch.Tensor], list[torch.Tensor]]
        ] = {}
        self._radial_mask_cache: dict[
            tuple[int, int, int, str], torch.Tensor
        ] = {}

        # These tensors depend only on the tiny spatial tile geometry and the
        # immutable FC hyperparameters.  Keeping them across model calls
        # removes repeated fftfreq/linspace/exp launches from every woven
        # layer while preserving the exact equations.
        self._fc_frequency_geometry_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self._overlap_window_cache: dict[
            tuple[int, int, str, str], torch.Tensor
        ] = {}
        self._overlap_normalization_cache: dict[
            tuple[int, int, int, int, int, int, int, str, str], torch.Tensor
        ] = {}
        self._records: list[dict[str, Any]] = []
        self._fc_target_spectrum_workspaces: dict[
            tuple[str, str, tuple[int, ...]], torch.Tensor
        ] = {}

    def should_publish_fc_routing_transport(self, layer_index: int) -> bool:
        """Return whether a later active router can consume this layer.

        ``None`` preserves the reference behavior. Runtime-only FrameWeave
        candidates may install the exact predecessor set of their sparse
        router allowlist; omitted entries have no consumer and cannot affect
        attention selection or reconstructed residuals.
        """

        publish_layers = getattr(self, "_fc_routing_publish_layers", None)
        return publish_layers is None or int(layer_index) in publish_layers

    def begin_model_call(
        self,
        *,
        geometry: MatrixWorldAlignmentGeometry | None,
        chunk_index: int,
        step_index: int,
        device: torch.device,
    ) -> None:
        self._chunk = int(chunk_index)
        self._step = int(step_index)
        self._geometry = None if geometry is None else geometry.to(device)
        self._current_projection_cache.clear()
        self._fc_routing_transport.clear()

    def set_fc_pair_gate_runtime_active(self, active: bool) -> None:
        """Select the endpoint pair gate for the current model call.

        This switch never disables FC-R itself.  When false, the frozen FC-R
        equation runs unchanged; when true, only endpoint pairs below the
        configured confidence threshold use the existing local V21 fallback.
        """
        self._fc_pair_gate_runtime_active = bool(active)

    def set_fc_runtime_active(self, active: bool) -> None:
        """Causally select FC-R or its existing V21 residual baseline.

        The caller may disable FC-R only from state already observed before
        the current woven block.  This switch does not alter Exact anchors,
        attention routing, or the residual interpolation used as the V21
        fallback.
        """
        self._fc_runtime_active = bool(active)

    def _publish_fc_routing_transport(
        self,
        *,
        layer_index: int,
        exact_frames: list[int],
        target_anchor_pairs: list[list[int]],
        theta: torch.Tensor,
        affinity: torch.Tensor,
        tile_grid: tuple[int, int],
        output_shape: tuple[int, int],
        transport_shape: tuple[int, int],
    ) -> None:
        """Cache a detached endpoint transport sketch for layer ``l+1``.

        Keep the detached, per-frequency endpoint operator for layer ``l+1``.
        Routing must apply the same spatial-frequency operator as FC-PASM; a
        scalar phase per tile is not sufficient because it mixes unrelated
        Fourier bins and turns channel-sketch coordinates into fake complex
        axes.  No target or held-out Exact residual enters this cache.
        """

        if theta.shape != affinity.shape or theta.ndim != 4:
            raise RuntimeError("FC routing transport expects target/tile/H/Wf tensors")
        if int(theta.shape[1]) != int(tile_grid[0] * tile_grid[1]):
            raise RuntimeError("FC routing transport tile grid is inconsistent")
        if len(target_anchor_pairs) != int(theta.shape[0]):
            raise RuntimeError("FC routing transport pair count is inconsistent")
        pairs: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for index, values in enumerate(target_anchor_pairs):
            pair = (int(values[0]), int(values[1]))
            if pair in pairs:
                continue
            pair_affinity = affinity[index]
            if 0 < self.fc_transport_tile_topk < int(tile_grid[0] * tile_grid[1]):
                # ``fc_transport_tile_topk`` deliberately sets every
                # non-selected spatial window to exact g=0.  Including those
                # structural zeros in the scalar admission statistic would
                # dilute a top-1 window by 18x and make ROCSA report weak
                # transport even when the selected endpoint-only window has
                # near-unit affinity.  Aggregate only over FC-R's actual
                # support; the full per-frequency tensor below remains
                # unchanged and is still what propagates reconstruction
                # error through the routing objective.
                support = pair_affinity > 0
                support_count = support.sum()
                # Top-k is positive here and sigmoid affinities are strictly
                # positive on every selected window, so the support cannot be
                # empty.  Avoid a Python bool conversion, which would add a
                # device synchronization to every anchor pair.
                support_mean = pair_affinity.sum() / support_count.to(
                    pair_affinity.dtype
                )
                mean_reduction = "active_transport_support"
            else:
                support_count = pair_affinity.new_tensor(
                    pair_affinity.numel(), dtype=torch.long
                )
                support_mean = pair_affinity.mean()
                mean_reduction = "all_tile_frequencies"
            pairs[pair] = {
                "theta": theta[index].float().detach(),
                "affinity": pair_affinity.float().detach(),
                "mean_affinity": support_mean.detach(),
                "mean_affinity_reduction": mean_reduction,
                "affinity_support_count": support_count.detach(),
                "affinity_total_count": int(pair_affinity.numel()),
            }
        self._fc_routing_transport[(int(self._step), int(layer_index))] = {
            "source_layer": int(layer_index),
            "source_step": int(self._step),
            "exact_frames": tuple(int(value) for value in exact_frames),
            "tile_grid": tuple(int(value) for value in tile_grid),
            "output_shape": tuple(int(value) for value in output_shape),
            "transport_shape": tuple(int(value) for value in transport_shape),
            "window_shape": (11, 10),
            "stride": (8, 8),
            "padding": (4, 5),
            "pairs": pairs,
            "detached": True,
            "endpoint_only": True,
            "transport_formula": "per_frequency_exp(i*g*theta)",
        }

    def fc_pasm_routing_transport(
        self, *, layer_index: int
    ) -> dict[str, Any] | None:
        """Return only the previous layer's causal detached FC-PASM sketch."""

        source_layer = int(layer_index) - 1
        if source_layer < 0:
            return None
        return self._fc_routing_transport.get((int(self._step), source_layer))

    @staticmethod
    def _pool_tiles(
        value: torch.Tensor,
        *,
        temporal: int,
        height: int,
        width: int,
        tile_h: int,
        tile_w: int,
    ) -> torch.Tensor:
        batch, value_temporal, spatial, channels = value.shape
        if value_temporal != temporal or spatial != height * width:
            raise RuntimeError("world correction received an invalid visual layout")
        flat = value.reshape(batch * temporal, height, width, channels).permute(
            0, 3, 1, 2
        )
        pooled = F.avg_pool2d(
            flat.float(),
            kernel_size=(tile_h, tile_w),
            stride=(tile_h, tile_w),
            ceil_mode=True,
            count_include_pad=False,
        )
        return pooled.reshape(batch, temporal, channels, pooled.shape[-2], pooled.shape[-1])

    def _align_memory(
        self,
        *,
        current_hidden: torch.Tensor,
        memory_hidden: torch.Tensor,
        memory_residual: torch.Tensor,
        geometry: MatrixWorldAlignmentGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Production Matrix is B=1.  Fail closed instead of silently mixing
        # batch-specific camera systems.
        if current_hidden.shape[0] != 1 or memory_hidden.shape[0] != 1:
            raise RuntimeError("world alignment currently requires Matrix batch size one")
        _, current_t, channels, hb, wb = current_hidden.shape
        memory_t = int(memory_hidden.shape[1])
        geometry.validate(memory_length=memory_t, current_length=current_t)
        device = current_hidden.device
        poses_cur = geometry.current_abs_c2ws
        poses_mem = geometry.memory_abs_c2ws
        fx, fy, cx, cy = geometry.base_K

        yy, xx = torch.meshgrid(
            torch.arange(hb, device=device, dtype=torch.float32),
            torch.arange(wb, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u = (xx + 0.5) * float(geometry.target_w) / float(wb)
        v = (yy + 0.5) * float(geometry.target_h) / float(hb)
        direction_camera = torch.stack(
            ((u - cx) / fx, (v - cy) / fy, torch.ones_like(u)), dim=-1
        )
        direction_camera = F.normalize(direction_camera, dim=-1)
        direction_world = torch.einsum(
            "tij,hwj->thwi", poses_cur[:, :3, :3], direction_camera
        )
        direction_world = F.normalize(direction_world, dim=-1)
        origin_world = poses_cur[:, None, None, :3, 3]

        inverse_depth = torch.linspace(
            1.0 / self.depth_far,
            1.0 / self.depth_near,
            self.align_depth_samples,
            device=device,
            dtype=torch.float32,
        )
        depth = inverse_depth.reciprocal()
        points_world = (
            origin_world[..., None, :]
            + direction_world[..., None, :] * depth[None, None, None, :, None]
        )
        queries = current_t * hb * wb
        points = points_world.reshape(queries, self.align_depth_samples, 3)
        current_vectors = current_hidden[0].permute(0, 2, 3, 1).reshape(
            queries, channels
        )

        sampled_hidden: list[torch.Tensor] = []
        sampled_residual: list[torch.Tensor] = []
        valid_candidates: list[torch.Tensor] = []
        for memory_index in range(memory_t):
            rotation = poses_mem[memory_index, :3, :3]
            origin = poses_mem[memory_index, :3, 3]
            # Row-vector form of R^T @ (X-o).
            camera_points = torch.matmul(points - origin, rotation)
            camera_z = camera_points[..., 2]
            projected_u = fx * camera_points[..., 0] / camera_z.clamp_min(1e-8) + cx
            projected_v = fy * camera_points[..., 1] / camera_z.clamp_min(1e-8) + cy
            valid = (
                (camera_z > 0.0)
                & (projected_u >= 0.0)
                & (projected_u < float(geometry.target_w))
                & (projected_v >= 0.0)
                & (projected_v < float(geometry.target_h))
            )
            grid = torch.stack(
                (
                    2.0 * projected_u / float(geometry.target_w) - 1.0,
                    2.0 * projected_v / float(geometry.target_h) - 1.0,
                ),
                dim=-1,
            ).unsqueeze(0)
            hidden_sample = F.grid_sample(
                memory_hidden[:, memory_index].float(),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0].permute(1, 2, 0)
            residual_sample = F.grid_sample(
                memory_residual[:, memory_index].float(),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0].permute(1, 2, 0)
            sampled_hidden.append(hidden_sample)
            sampled_residual.append(residual_sample)
            valid_candidates.append(valid)

        candidate_hidden = torch.cat(sampled_hidden, dim=1)
        candidate_residual = torch.cat(sampled_residual, dim=1)
        valid = torch.cat(valid_candidates, dim=1)
        similarity = F.cosine_similarity(
            current_vectors[:, None, :], candidate_hidden, dim=-1, eps=1e-8
        )
        similarity = similarity.masked_fill(~valid, -torch.inf)
        top_l = min(self.align_top_l, int(similarity.shape[1]))
        top_values, top_indices = torch.topk(similarity, k=top_l, dim=1)
        top_valid = torch.isfinite(top_values)
        safe_values = torch.where(top_valid, top_values, torch.full_like(top_values, -1e9))
        weights = torch.softmax(safe_values, dim=1) * top_valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        chosen_residual = torch.gather(
            candidate_residual,
            1,
            top_indices[..., None].expand(-1, -1, channels),
        )
        aligned = (weights[..., None] * chosen_residual).sum(dim=1)

        valid_count = top_valid.sum(dim=1)
        entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
        entropy_denominator = valid_count.clamp_min(2).float().log()
        normalized_entropy = torch.where(
            valid_count > 1,
            entropy / entropy_denominator,
            torch.zeros_like(entropy),
        ).clamp(0.0, 1.0)
        valid_ratio = valid.float().mean(dim=1)
        confidence = valid_ratio * (1.0 - normalized_entropy)
        any_valid = valid.any(dim=1)
        aligned = torch.where(any_valid[:, None], aligned, torch.zeros_like(aligned))
        confidence = torch.where(any_valid, confidence, torch.zeros_like(confidence))
        aligned = aligned.reshape(current_t, hb, wb, channels).permute(0, 3, 1, 2)
        confidence = confidence.reshape(current_t, hb, wb)
        return aligned, confidence, {
            "valid_projection_ratio": valid_ratio.mean().detach(),
            "matching_entropy": normalized_entropy[any_valid].mean().detach()
            if bool(any_valid.any())
            else torch.zeros((), device=device),
            "valid_query_ratio": any_valid.float().mean().detach(),
        }

    def _current_anchor_descriptors(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return grouped 11x10 descriptors without learned projection."""

        if hidden.ndim != 4:
            raise RuntimeError("Current-anchor hidden grid must be [T,C,H,W]")
        tile_h, tile_w = self.current_anchor_match_tile
        hidden_float = hidden.float()
        # Per-channel standardization prevents high-magnitude channels from
        # dominating the hand-built grouped descriptor.
        channel_mean = hidden_float.mean(dim=(-2, -1), keepdim=True)
        channel_scale = hidden_float.std(
            dim=(-2, -1), keepdim=True, unbiased=False
        ).clamp_min(1e-6)
        pooled = F.avg_pool2d(
            (hidden_float - channel_mean) / channel_scale,
            kernel_size=(tile_h, tile_w),
            stride=(tile_h, tile_w),
            ceil_mode=True,
            count_include_pad=False,
        )
        temporal, channels, match_h, match_w = pooled.shape
        groups = self.current_anchor_descriptor_groups
        if channels % groups:
            raise RuntimeError(
                "Current-anchor channels are not divisible by descriptor groups"
            )
        descriptor = pooled.reshape(
            temporal, groups, channels // groups, match_h, match_w
        ).mean(dim=2)
        local_mean = F.avg_pool2d(
            descriptor,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        # Retain both standardized appearance and its local high-pass
        # contrast.  The latter alone discards the coarse feature identity
        # needed to disambiguate geometrically valid depth candidates.
        descriptor = torch.cat((descriptor, descriptor - local_mean), dim=1)
        return F.normalize(descriptor, dim=1, eps=1e-8)

    def _current_projection_grids(
        self,
        *,
        geometry: MatrixWorldAlignmentGeometry,
        exact_local_frames: list[int],
        match_h: int,
        match_w: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Project every Current descriptor ray into each Exact Current camera."""

        device = geometry.current_abs_c2ws.device
        key = (
            tuple(exact_local_frames),
            int(match_h),
            int(match_w),
            self.align_depth_samples,
            str(device),
        )
        cached = self._current_projection_cache.get(key)
        if cached is not None:
            return cached
        poses = geometry.current_abs_c2ws
        current_t = int(poses.shape[0])
        fx, fy, cx, cy = geometry.base_K
        yy, xx = torch.meshgrid(
            torch.arange(match_h, device=device, dtype=torch.float32),
            torch.arange(match_w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u = (xx + 0.5) * float(geometry.target_w) / float(match_w)
        v = (yy + 0.5) * float(geometry.target_h) / float(match_h)
        direction_camera = F.normalize(
            torch.stack(
                ((u - cx) / fx, (v - cy) / fy, torch.ones_like(u)), dim=-1
            ),
            dim=-1,
        )
        direction_world = F.normalize(
            torch.einsum(
                "tij,hwj->thwi", poses[:, :3, :3], direction_camera
            ),
            dim=-1,
        )
        inverse_depth = torch.linspace(
            1.0 / self.depth_far,
            1.0 / self.depth_near,
            self.align_depth_samples,
            device=device,
            dtype=torch.float32,
        )
        depth = inverse_depth.reciprocal()
        points = (
            poses[:, :3, 3][:, None, None, None, :]
            + direction_world[..., None, :] * depth[None, None, None, :, None]
        ).reshape(current_t * match_h * match_w, self.align_depth_samples, 3)
        grids: list[torch.Tensor] = []
        validity: list[torch.Tensor] = []
        for local_frame in exact_local_frames:
            source_pose = poses[int(local_frame)]
            camera_points = torch.matmul(
                points - source_pose[:3, 3], source_pose[:3, :3]
            )
            z = camera_points[..., 2]
            projected_u = fx * camera_points[..., 0] / z.clamp_min(1e-8) + cx
            projected_v = fy * camera_points[..., 1] / z.clamp_min(1e-8) + cy
            valid = (
                (z > 0.0)
                & (projected_u >= 0.0)
                & (projected_u < float(geometry.target_w))
                & (projected_v >= 0.0)
                & (projected_v < float(geometry.target_h))
            )
            grids.append(
                torch.stack(
                    (
                        2.0 * projected_u / float(geometry.target_w) - 1.0,
                        2.0 * projected_v / float(geometry.target_h) - 1.0,
                    ),
                    dim=-1,
                )
            )
            validity.append(valid)
        self._current_projection_cache[key] = (grids, validity)
        return grids, validity

    def _align_current_anchors(
        self,
        *,
        current_hidden: torch.Tensor,
        exact_hidden: torch.Tensor,
        exact_residual: torch.Tensor,
        exact_local_frames: list[int],
        geometry: MatrixWorldAlignmentGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Align Current queries only to disjoint active Exact Current sources."""

        if current_hidden.ndim != 4 or exact_hidden.ndim != 4:
            raise RuntimeError("Current-anchor alignment expects channel-first grids")
        current_t, channels = current_hidden.shape[:2]
        if exact_hidden.shape[0] != len(exact_local_frames):
            raise RuntimeError("Current-anchor hidden/frame count mismatch")
        if exact_residual.shape[:2] != (len(exact_local_frames), channels):
            raise RuntimeError("Current-anchor residual layout mismatch")
        if not exact_local_frames or len(set(exact_local_frames)) != len(
            exact_local_frames
        ):
            raise RuntimeError("Current-anchor frames must be unique and non-empty")
        if exact_local_frames != sorted(exact_local_frames):
            raise RuntimeError("Current-anchor frames must be chronological")
        query_descriptor = self._current_anchor_descriptors(current_hidden)
        anchor_descriptor = self._current_anchor_descriptors(exact_hidden)
        _, groups, match_h, match_w = query_descriptor.shape
        queries_per_frame = match_h * match_w
        query = query_descriptor.permute(0, 2, 3, 1).reshape(-1, groups)
        query_consensus = F.normalize(
            query_descriptor.mean(dim=(-2, -1)), dim=-1, eps=1e-8
        )
        anchor_consensus = F.normalize(
            anchor_descriptor.mean(dim=(-2, -1)), dim=-1, eps=1e-8
        )
        grids, validity = self._current_projection_grids(
            geometry=geometry,
            exact_local_frames=exact_local_frames,
            match_h=match_h,
            match_w=match_w,
        )
        score_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        target_frames = torch.arange(
            current_t, device=current_hidden.device
        ).repeat_interleave(queries_per_frame)
        for source_index, source_frame in enumerate(exact_local_frames):
            sampled = F.grid_sample(
                anchor_descriptor[source_index : source_index + 1],
                grids[source_index].unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0].permute(1, 2, 0)
            local_score = F.cosine_similarity(
                query[:, None, :], sampled, dim=-1, eps=1e-8
            )
            consensus_score = torch.matmul(
                query_consensus, anchor_consensus[source_index]
            ).repeat_interleave(queries_per_frame)[:, None]
            score = (
                (1.0 - self.current_anchor_consensus_mix) * local_score
                + self.current_anchor_consensus_mix * consensus_score
            )
            same_segment = (
                (target_frames < self.current_anchor_boundary)
                == (int(source_frame) < self.current_anchor_boundary)
            )
            disjoint = target_frames != int(source_frame)
            allowed = validity[source_index] & same_segment[:, None] & disjoint[:, None]
            score_rows.append(score.masked_fill(~allowed, -torch.inf))
            valid_rows.append(allowed)
        scores = torch.cat(score_rows, dim=1)
        valid = torch.cat(valid_rows, dim=1)
        top_l = min(self.align_top_l, int(scores.shape[1]))
        top_values, top_indices = torch.topk(scores, k=top_l, dim=1)
        top_valid = torch.isfinite(top_values)
        safe = torch.where(top_valid, top_values, torch.full_like(top_values, -1e9))
        weights = torch.softmax(safe, dim=1) * top_valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        aligned = torch.zeros(
            (int(query.shape[0]), channels),
            device=current_hidden.device,
            dtype=torch.float32,
        )
        depth_count = self.align_depth_samples
        selected_source = torch.div(top_indices, depth_count, rounding_mode="floor")
        selected_depth = top_indices.remainder(depth_count)
        for source_index in range(len(exact_local_frames)):
            selected = top_valid & (selected_source == source_index)
            if not bool(selected.any()):
                continue
            query_index, rank_index = torch.nonzero(selected, as_tuple=True)
            depth_index = selected_depth[query_index, rank_index]
            selected_grid = grids[source_index][query_index, depth_index]
            sampled_residual = F.grid_sample(
                exact_residual[source_index : source_index + 1],
                selected_grid.reshape(1, 1, -1, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, :, 0].transpose(0, 1)
            aligned.index_add_(
                0,
                query_index,
                sampled_residual
                * weights[query_index, rank_index, None],
            )
        valid_count = top_valid.sum(dim=1)
        entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1)
        normalized_entropy = torch.where(
            valid_count > 1,
            entropy / valid_count.clamp_min(2).float().log(),
            torch.zeros_like(entropy),
        ).clamp(0.0, 1.0)
        valid_ratio = valid.float().mean(dim=1)
        any_valid = valid.any(dim=1)
        # Entropy is diagnostic only in V2.  The old entropy confidence was
        # almost zero for valid multi-depth matches and suppressed the very
        # correction that held-out Exact-frame validation is meant to judge.
        # Reliability is supplied by per-band held-out CV below; here we only
        # retain a geometric-validity mask.
        confidence = any_valid.float()
        aligned = aligned.reshape(
            current_t, match_h, match_w, channels
        ).permute(0, 3, 1, 2)
        confidence = confidence.reshape(current_t, match_h, match_w)
        correction_h, correction_w = exact_residual.shape[-2:]
        aligned = F.interpolate(
            aligned,
            size=(correction_h, correction_w),
            mode="bilinear",
            align_corners=False,
        )
        confidence = F.interpolate(
            confidence[:, None],
            size=(correction_h, correction_w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return aligned, confidence, {
            "valid_projection_ratio": valid_ratio.mean().detach(),
            "matching_entropy": normalized_entropy[any_valid].mean().detach()
            if bool(any_valid.any())
            else torch.zeros((), device=current_hidden.device),
            "valid_query_ratio": any_valid.float().mean().detach(),
        }

    @staticmethod
    def _radial_masks(
        height: int, width: int, bands: int, *, device: torch.device
    ) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, device=device).abs()
        fx = torch.fft.rfftfreq(width, device=device).abs()
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        radius = radius / radius.max().clamp_min(1e-8)
        centers = torch.linspace(0.0, 1.0, bands, device=device)
        spacing = 1.0 / float(bands - 1)
        # Overlapping Gaussian windows are smooth; normalization makes their
        # pointwise sum exactly one up to floating-point roundoff.
        masks = torch.exp(
            -0.5 * ((radius[None] - centers[:, None, None]) / spacing).square()
        )
        return masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-8)

    def _cached_radial_masks(
        self,
        height: int,
        width: int,
        bands: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Cache the fixed radial partition for a tile/device pair."""
        key = (int(height), int(width), int(bands), str(device))
        cached = self._radial_mask_cache.get(key)
        if cached is None:
            cached = self._radial_masks(
                int(height), int(width), int(bands), device=device
            )
            self._radial_mask_cache[key] = cached
        return cached

    def _cached_fc_frequency_geometry(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return cached radius, FC threshold and radial bands.

        The cache key includes every scalar that enters the tensors.  This is
        deliberately separate from the numerical implementation so changing a
        FC candidate's parameters cannot accidentally reuse another variant's
        threshold.
        """
        key = (
            int(height),
            int(width),
            str(device),
            float(self.fc_tau_low),
            float(self.fc_tau_high),
            float(self.fc_freq_power),
            int(self.spectral_num_bands),
        )
        cached = self._fc_frequency_geometry_cache.get(key)
        if cached is not None:
            return cached
        radius = self._normalized_rfft_radius(
            int(height), int(width), device=device
        )
        threshold = self.fc_tau_low + (
            self.fc_tau_high - self.fc_tau_low
        ) * radius.pow(self.fc_freq_power)
        masks = self._cached_radial_masks(
            int(height), int(width), self.spectral_num_bands, device=device
        )
        cached = (radius, threshold, masks)
        self._fc_frequency_geometry_cache[key] = cached
        return cached

    @staticmethod
    def _linear_loo(
        exact_residual: torch.Tensor,
        exact_local_frames: list[int],
        target_index: int,
    ) -> torch.Tensor | None:
        target_frame = exact_local_frames[target_index]
        others = [
            (index, frame)
            for index, frame in enumerate(exact_local_frames)
            if index != target_index
        ]
        left = [(index, frame) for index, frame in others if frame < target_frame]
        right = [(index, frame) for index, frame in others if frame > target_frame]
        if not left or not right:
            return None
        left_index, left_frame = left[-1]
        right_index, right_frame = right[0]
        weight = float(target_frame - left_frame) / float(right_frame - left_frame)
        return (
            (1.0 - weight) * exact_residual[left_index]
            + weight * exact_residual[right_index]
        )

    def _segmented_linear_loo(
        self,
        exact_residual: torch.Tensor,
        exact_local_frames: list[int],
        target_index: int,
    ) -> torch.Tensor | None:
        """LOO interpolation that can never cross overlap/new-generation."""

        target_frame = exact_local_frames[target_index]
        target_segment = target_frame < self.current_anchor_boundary
        others = [
            (index, frame)
            for index, frame in enumerate(exact_local_frames)
            if index != target_index
            and (frame < self.current_anchor_boundary) == target_segment
        ]
        left = [(index, frame) for index, frame in others if frame < target_frame]
        right = [(index, frame) for index, frame in others if frame > target_frame]
        if not left or not right:
            return None
        left_index, left_frame = left[-1]
        right_index, right_frame = right[0]
        weight = float(target_frame - left_frame) / float(right_frame - left_frame)
        return (
            (1.0 - weight) * exact_residual[left_index]
            + weight * exact_residual[right_index]
        )

    @staticmethod
    def _project_bands(value: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        spectrum = torch.fft.rfft2(value.float(), dim=(-2, -1))
        return [
            torch.fft.irfft2(
                spectrum * mask[None, None],
                s=value.shape[-2:],
                dim=(-2, -1),
            )
            for mask in masks
        ]

    @staticmethod
    def _band_energy_share(
        value: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Return a Parseval-consistent spatial error-energy partition.

        ``rfft2`` stores only the non-negative horizontal frequencies.  The
        interior columns therefore need their conjugate partner counted twice.
        The smooth radial masks sum to one, so the returned band shares also
        sum to one (up to floating-point roundoff).  This is diagnostic only;
        it is never consumed by reconstruction.
        """

        spectrum = torch.fft.rfft2(value.float(), dim=(-2, -1))
        power = spectrum.real.square() + spectrum.imag.square()
        width = int(value.shape[-1])
        conjugate_weight = torch.ones(
            spectrum.shape[-1], device=value.device, dtype=power.dtype
        )
        if width % 2 == 0:
            conjugate_weight[1:-1] = 2.0
        else:
            conjugate_weight[1:] = 2.0
        band_energy = (
            power[None]
            * masks[:, None, None]
            * conjugate_weight[None, None, None, None, :]
        ).sum(dim=(1, 2, 3, 4))
        return band_energy / band_energy.sum().clamp_min(1e-20)

    def _calibrate_gamma(
        self,
        *,
        layer_index: int,
        exact_residual: torch.Tensor,
        exact_local_frames: list[int],
        aligned: torch.Tensor,
        confidence: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, int, str, dict[str, torch.Tensor]]:
        errors: list[torch.Tensor] = []
        contrasts: list[torch.Tensor] = []
        baseline_errors: list[torch.Tensor] = []
        aligned_errors: list[torch.Tensor] = []
        loo_values: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        calibration_frames: list[int] = []
        for index, local_frame in enumerate(exact_local_frames):
            loo = self._linear_loo(exact_residual, exact_local_frames, index)
            if loo is None:
                continue
            target = exact_residual[index]
            errors.append(target - loo)
            contrasts.append(aligned[local_frame] - loo)
            loo_values.append(loo)
            targets.append(target)
            calibration_frames.append(local_frame)
            denominator = target.norm().clamp_min(1e-8)
            baseline_errors.append((target - loo).norm() / denominator)
            aligned_errors.append((target - aligned[local_frame]).norm() / denominator)
        calibration_count = len(errors)
        device = exact_residual.device
        diagnostics = {
            "loo_baseline_relative_error": torch.stack(baseline_errors).mean().detach()
            if baseline_errors
            else torch.full((), float("nan"), device=device),
            "loo_world_aligned_relative_error": torch.stack(aligned_errors).mean().detach()
            if aligned_errors
            else torch.full((), float("nan"), device=device),
        }
        if calibration_count >= 2:
            error = torch.stack(errors)
            contrast = torch.stack(contrasts)
            diagnostics["loo_error_band_energy_share"] = (
                self._band_energy_share(error, masks).detach()
            )
            projected_error = self._project_bands(error, masks)
            projected_contrast = self._project_bands(contrast, masks)
            gains = []
            for source, target in zip(projected_contrast, projected_error):
                numerator = (source * target).sum()
                denominator = source.square().sum() + self.ridge
                gains.append((numerator / denominator).clamp(0.0, self.gamma_max))
            gamma = torch.stack(gains).detach()
            loo_stack = torch.stack(loo_values)
            target_stack = torch.stack(targets)
            q = confidence[
                torch.tensor(calibration_frames, device=device, dtype=torch.long)
            ][:, None]

            def trusted_error(correction: torch.Tensor) -> torch.Tensor:
                correction = q * correction
                correction_norm = correction.flatten(1).norm(dim=1)
                reference_norm = loo_stack.flatten(1).norm(dim=1)
                scale = torch.minimum(
                    torch.ones_like(correction_norm),
                    self.eta
                    * reference_norm
                    / correction_norm.clamp_min(1e-8),
                )
                prediction = loo_stack + correction * scale[:, None, None, None]
                denominator = target_stack.flatten(1).norm(dim=1).clamp_min(1e-8)
                return (
                    (target_stack - prediction).flatten(1).norm(dim=1)
                    / denominator
                ).mean()

            diagnostics["loo_fixed_lowpass_relative_error"] = trusted_error(
                projected_contrast[0]
            ).detach()
            diagnostics["loo_world_aligned_relative_error"] = trusted_error(
                contrast
            ).detach()
            diagnostics[
                "loo_self_calibrated_relative_error"
            ] = trusted_error(
                sum(
                    gamma[band] * projected_contrast[band]
                    for band in range(self.spectral_num_bands)
                )
            ).detach()
            previous = self._ema_by_layer.get(layer_index)
            ema = (
                gamma
                if previous is None
                else self.gamma_ema * previous + (1.0 - self.gamma_ema) * gamma
            )
            self._ema_by_layer[layer_index] = ema.detach()
            self._last_ema = ema.detach()
            return gamma, calibration_count, "current_exact_loo", diagnostics
        fallback = self._ema_by_layer.get(layer_index)
        source = "previous_chunk_same_layer_ema"
        if fallback is None:
            fallback = self._last_ema
            source = "previous_layer_ema"
        if fallback is None:
            fallback = torch.zeros(self.spectral_num_bands, device=device)
            source = "zero_no_calibration"
        return fallback.to(device), calibration_count, source, diagnostics

    def _calibrate_current_anchor_gamma(
        self,
        *,
        layer_index: int,
        exact_residual: torch.Tensor,
        exact_local_frames: list[int],
        aligned: torch.Tensor,
        confidence: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, str, dict[str, torch.Tensor]]:
        """Fit spectral gain on segmented LOO anchors and certify by held-out CV."""

        errors: list[torch.Tensor] = []
        contrasts: list[torch.Tensor] = []
        loo_values: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        calibration_frames: list[int] = []
        segment_counts = [0, 0]
        for index, local_frame in enumerate(exact_local_frames):
            loo = self._segmented_linear_loo(
                exact_residual, exact_local_frames, index
            )
            if loo is None:
                continue
            target = exact_residual[index]
            errors.append(target - loo)
            contrasts.append(aligned[local_frame] - loo)
            loo_values.append(loo)
            targets.append(target)
            calibration_frames.append(local_frame)
            segment_counts[int(local_frame >= self.current_anchor_boundary)] += 1
        count = len(errors)
        device = exact_residual.device
        nan = torch.full((), float("nan"), device=device)
        diagnostics: dict[str, torch.Tensor] = {
            "calibration_segment_counts": torch.tensor(
                segment_counts, device=device, dtype=torch.float32
            ),
            "heldout_cv_baseline_relative_error": nan,
            "heldout_cv_corrected_relative_error": nan,
            "heldout_cv_reliability_rho": torch.zeros((), device=device),
            "heldout_cv_reliability_rho_b": torch.zeros(
                self.spectral_num_bands, device=device
            ),
        }
        ema_key = (int(self._step), int(layer_index))
        previous = self._ema_by_layer.get(ema_key)
        if count < 2:
            if previous is None:
                return (
                    torch.zeros(self.spectral_num_bands, device=device),
                    torch.zeros(self.spectral_num_bands, device=device),
                    count,
                    "zero_no_segmented_loo_calibration",
                    diagnostics,
                )
            return (
                previous[0].to(device),
                previous[1].to(device),
                count,
                "previous_chunk_same_step_layer_ema",
                diagnostics,
            )

        error = torch.stack(errors)
        contrast = torch.stack(contrasts)
        diagnostics["heldout_baseline_error_band_energy_share"] = (
            self._band_energy_share(error, masks).detach()
        )
        loo_stack = torch.stack(loo_values)
        target_stack = torch.stack(targets)
        projected_error = self._project_bands(error, masks)
        projected_contrast = self._project_bands(contrast, masks)
        q = confidence[
            torch.tensor(calibration_frames, device=device, dtype=torch.long)
        ][:, None]

        def fit(indices: torch.Tensor) -> torch.Tensor:
            gains: list[torch.Tensor] = []
            for source, target in zip(projected_contrast, projected_error):
                source_subset = source.index_select(0, indices)
                target_subset = target.index_select(0, indices)
                gains.append(
                    (
                        (source_subset * target_subset).sum()
                        / (source_subset.square().sum() + self.ridge)
                    ).clamp(0.0, self.gamma_max)
                )
            return torch.stack(gains)

        all_indices = torch.arange(count, device=device)
        gamma_raw = fit(all_indices).detach()
        baseline_cv: list[torch.Tensor] = []
        corrected_cv_by_band: list[list[torch.Tensor]] = [
            [] for _ in range(self.spectral_num_bands)
        ]
        for held_out in range(count):
            train = all_indices[all_indices != held_out]
            gamma_fold = fit(train)
            denominator = target_stack[held_out].norm().clamp_min(1e-8)
            baseline_cv.append(
                (target_stack[held_out] - loo_stack[held_out]).norm() / denominator
            )
            reference_norm = loo_stack[held_out].norm()
            for band in range(self.spectral_num_bands):
                correction = (
                    q[held_out]
                    * gamma_fold[band]
                    * projected_contrast[band][held_out]
                )
                correction_norm = correction.norm()
                trust = torch.minimum(
                    torch.ones((), device=device),
                    self.eta
                    * reference_norm
                    / correction_norm.clamp_min(1e-8),
                )
                prediction = loo_stack[held_out] + trust * correction
                corrected_cv_by_band[band].append(
                    (target_stack[held_out] - prediction).norm() / denominator
                )
        baseline_mean = torch.stack(baseline_cv).mean()
        corrected_mean_b = torch.stack(
            [torch.stack(values).mean() for values in corrected_cv_by_band]
        )
        rho_raw = (
            1.0 - corrected_mean_b / baseline_mean.clamp_min(1e-8)
        ).clamp(0.0, 1.0).detach()
        if previous is None:
            gamma = gamma_raw
            rho = rho_raw
        else:
            gamma = (
                self.gamma_ema * previous[0].to(device)
                + (1.0 - self.gamma_ema) * gamma_raw
            )
            rho = (
                self.gamma_ema * previous[1].to(device)
                + (1.0 - self.gamma_ema) * rho_raw
            )
        self._ema_by_layer[ema_key] = (gamma.detach(), rho.detach())
        diagnostics.update(
            {
                "heldout_cv_baseline_relative_error": baseline_mean.detach(),
                "heldout_cv_corrected_relative_error": corrected_mean_b.mean().detach(),
                "heldout_cv_corrected_relative_error_b": corrected_mean_b.detach(),
                "heldout_cv_reliability_rho": rho.mean().detach(),
                "heldout_cv_reliability_rho_b": rho.detach(),
            }
        )
        return gamma, rho, count, "current_exact_segmented_loo_cv_ema", diagnostics

    @staticmethod
    def _polar_band_prediction(
        base_spectrum: torch.Tensor,
        source_spectrum: torch.Tensor,
        mask: torch.Tensor,
        phase_gain: torch.Tensor,
        amplitude_gain: torch.Tensor,
        *,
        mix_amplitude: bool,
    ) -> torch.Tensor:
        """Apply one smooth polar band while retaining the Approx complement."""
        eps = 1e-8
        base_amplitude = base_spectrum.abs().clamp_min(eps)
        source_amplitude = source_spectrum.abs().clamp_min(eps)
        phase_delta = torch.angle(source_spectrum * base_spectrum.conj())
        log_amplitude_delta = (
            source_amplitude.log() - base_amplitude.log()
        ).clamp(-4.0, 4.0)
        transported_amplitude = base_amplitude
        if mix_amplitude:
            transported_amplitude = base_amplitude * torch.exp(
                (amplitude_gain * log_amplitude_delta).clamp(-4.0, 4.0)
            )
        unit_base = base_spectrum / base_amplitude
        transported = transported_amplitude * unit_base * torch.exp(
            1j * phase_gain * phase_delta
        )
        return base_spectrum + mask * (transported - base_spectrum)

    def _calibrate_current_anchor_polar(
        self,
        *,
        layer_index: int,
        exact_residual: torch.Tensor,
        exact_local_frames: list[int],
        aligned: torch.Tensor,
        confidence: torch.Tensor,
        masks: torch.Tensor,
        mix_amplitude: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        str,
        dict[str, torch.Tensor],
    ]:
        """Fit low-band phase/log-amplitude gains with segmented held-out CV."""
        bases: list[torch.Tensor] = []
        sources: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        calibration_frames: list[int] = []
        segment_counts = [0, 0]
        for index, local_frame in enumerate(exact_local_frames):
            loo = self._segmented_linear_loo(
                exact_residual, exact_local_frames, index
            )
            if loo is None:
                continue
            bases.append(loo)
            sources.append(aligned[local_frame])
            targets.append(exact_residual[index])
            calibration_frames.append(local_frame)
            segment_counts[int(local_frame >= self.current_anchor_boundary)] += 1
        count = len(bases)
        device = exact_residual.device
        zeros = torch.zeros(self.spectral_num_bands, device=device)
        diagnostics: dict[str, torch.Tensor] = {
            "calibration_segment_counts": torch.tensor(
                segment_counts, device=device, dtype=torch.float32
            ),
            "heldout_cv_reliability_rho_b": zeros.clone(),
            "heldout_cv_reliability_rho": torch.zeros((), device=device),
        }
        ema_key = (int(self._step), int(layer_index), self.mode)
        previous = self._polar_ema_by_layer.get(ema_key)
        if count < 2:
            if previous is None:
                return (
                    zeros.clone(), zeros.clone(), zeros.clone(), count,
                    "zero_no_segmented_loo_polar_calibration", diagnostics,
                )
            return (
                previous[0].to(device), previous[1].to(device),
                previous[2].to(device), count,
                "previous_chunk_same_step_layer_polar_ema", diagnostics,
            )

        base = torch.stack(bases).float()
        source = torch.stack(sources).float()
        target = torch.stack(targets).float()
        fb = torch.fft.rfft2(base, dim=(-2, -1))
        fs = torch.fft.rfft2(source, dim=(-2, -1))
        ft = torch.fft.rfft2(target, dim=(-2, -1))
        source_phase = torch.angle(fs * fb.conj())
        target_phase = torch.angle(ft * fb.conj())
        source_log_amp = (
            fs.abs().clamp_min(1e-8).log()
            - fb.abs().clamp_min(1e-8).log()
        ).clamp(-4.0, 4.0)
        target_log_amp = (
            ft.abs().clamp_min(1e-8).log()
            - fb.abs().clamp_min(1e-8).log()
        ).clamp(-4.0, 4.0)
        frame_weight = confidence[
            torch.tensor(calibration_frames, device=device, dtype=torch.long)
        ].mean(dim=(-2, -1)).clamp(0.0, 1.0)

        def fit(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            phase_values: list[torch.Tensor] = []
            amplitude_values: list[torch.Tensor] = []
            for band in range(self.spectral_num_bands):
                weight = (
                    masks[band][None, None]
                    * fb.index_select(0, indices).abs().square()
                    * frame_weight.index_select(0, indices)[:, None, None, None]
                )
                phase_src = source_phase.index_select(0, indices)
                phase_tgt = target_phase.index_select(0, indices)
                amp_src = source_log_amp.index_select(0, indices)
                amp_tgt = target_log_amp.index_select(0, indices)
                phase_values.append(
                    ((weight * phase_src * phase_tgt).sum()
                     / ((weight * phase_src.square()).sum() + self.ridge))
                    .clamp(0.0, self.gamma_max)
                )
                amplitude_values.append(
                    ((weight * amp_src * amp_tgt).sum()
                     / ((weight * amp_src.square()).sum() + self.ridge))
                    .clamp(0.0, self.gamma_max)
                )
            return torch.stack(phase_values), torch.stack(amplitude_values)

        all_indices = torch.arange(count, device=device)
        phase_raw, amplitude_raw = fit(all_indices)
        baseline_errors: list[torch.Tensor] = []
        corrected_by_band: list[list[torch.Tensor]] = [
            [] for _ in range(self.spectral_num_bands)
        ]
        for held_out in range(count):
            train = all_indices[all_indices != held_out]
            phase_fold, amplitude_fold = fit(train)
            baseline_error = (target[held_out] - base[held_out]).norm()
            baseline_errors.append(baseline_error)
            for band in range(self.spectral_num_bands):
                predicted_spectrum = self._polar_band_prediction(
                    fb[held_out], fs[held_out], masks[band][None],
                    phase_fold[band], amplitude_fold[band],
                    mix_amplitude=mix_amplitude,
                )
                predicted = torch.fft.irfft2(
                    predicted_spectrum,
                    s=base.shape[-2:],
                    dim=(-2, -1),
                )
                corrected_by_band[band].append(
                    (target[held_out] - predicted).norm()
                )
        baseline_mean = torch.stack(baseline_errors).mean().clamp_min(1e-8)
        corrected_mean = torch.stack(
            [torch.stack(values).mean() for values in corrected_by_band]
        )
        rho_raw = (1.0 - corrected_mean / baseline_mean).clamp(0.0, 1.0)
        # The RGB Oracle identifies only the first two bands as the target.
        phase_raw = phase_raw.clone()
        amplitude_raw = amplitude_raw.clone()
        rho_raw = rho_raw.clone()
        phase_raw[2:] = 0.0
        amplitude_raw[2:] = 0.0
        rho_raw[2:] = 0.0
        if previous is None:
            phase_gain, amplitude_gain, rho = phase_raw, amplitude_raw, rho_raw
        else:
            phase_gain = self.gamma_ema * previous[0].to(device) + (
                1.0 - self.gamma_ema
            ) * phase_raw
            amplitude_gain = self.gamma_ema * previous[1].to(device) + (
                1.0 - self.gamma_ema
            ) * amplitude_raw
            rho = self.gamma_ema * previous[2].to(device) + (
                1.0 - self.gamma_ema
            ) * rho_raw
        self._polar_ema_by_layer[ema_key] = (
            phase_gain.detach(), amplitude_gain.detach(), rho.detach()
        )
        diagnostics.update(
            {
                "heldout_cv_baseline_residual_l2": baseline_mean.detach(),
                "heldout_cv_corrected_residual_l2_b": corrected_mean.detach(),
                "heldout_cv_reliability_rho_b": rho.detach(),
                "heldout_cv_reliability_rho": rho.mean().detach(),
            }
        )
        return (
            phase_gain, amplitude_gain, rho, count,
            "current_exact_segmented_loo_polar_cv_ema", diagnostics,
        )

    @staticmethod
    def _shared_cross_phase_unit(
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the channel-voted unit-circle motion and its coherence."""
        channel_cross = right_spectrum * left_spectrum.conj()
        shared = channel_cross.sum(dim=0)
        magnitude = shared.abs()
        unit = torch.where(
            magnitude > 1e-8,
            shared / magnitude.clamp_min(1e-8),
            torch.ones_like(shared),
        )
        coherence = magnitude / channel_cross.abs().sum(dim=0).clamp_min(1e-8)
        return unit, coherence.clamp(0.0, 1.0)

    @classmethod
    def _phase_aligned_pair_spectrum(
        cls,
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transport both Exact spectra on S1 before barycentric mixing."""
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("phase-aligned interpolation alpha must be in [0,1]")
        unit, coherence = cls._shared_cross_phase_unit(
            left_spectrum, right_spectrum
        )
        shared_angle = torch.angle(unit)
        left_transport = torch.polar(
            torch.ones_like(shared_angle), float(alpha) * shared_angle
        )
        right_transport = torch.polar(
            torch.ones_like(shared_angle),
            -float(1.0 - alpha) * shared_angle,
        )
        mixed = (
            float(1.0 - alpha) * left_spectrum * left_transport[None]
            + float(alpha) * right_spectrum * right_transport[None]
        )
        return mixed, coherence

    @staticmethod
    def _phase_tiles(
        value: torch.Tensor,
        *,
        grid: tuple[int, int] = (2, 4),
    ) -> torch.Tensor:
        """Partition [...,C,H,W] into a deterministic row-major tile axis."""
        grid_h, grid_w = grid
        height, width = value.shape[-2:]
        if height % grid_h or width % grid_w:
            raise RuntimeError("local PASM grid must exactly divide the residual grid")
        tile_h, tile_w = height // grid_h, width // grid_w
        leading = value.shape[:-3]
        channels = value.shape[-3]
        return (
            value.reshape(*leading, channels, grid_h, tile_h, grid_w, tile_w)
            .permute(
                *range(len(leading)),
                len(leading) + 1,
                len(leading) + 3,
                len(leading),
                len(leading) + 2,
                len(leading) + 4,
            )
            .reshape(*leading, grid_h * grid_w, channels, tile_h, tile_w)
        )

    @staticmethod
    def _unphase_tiles(
        value: torch.Tensor,
        *,
        grid: tuple[int, int] = (2, 4),
    ) -> torch.Tensor:
        """Inverse of _phase_tiles for [...,P,C,h,w]."""
        grid_h, grid_w = grid
        leading = value.shape[:-4]
        patches, channels, tile_h, tile_w = value.shape[-4:]
        if patches != grid_h * grid_w:
            raise RuntimeError("local PASM received an invalid tile count")
        return (
            value.reshape(*leading, grid_h, grid_w, channels, tile_h, tile_w)
            .permute(
                *range(len(leading)),
                len(leading) + 2,
                len(leading),
                len(leading) + 3,
                len(leading) + 1,
                len(leading) + 4,
            )
            .reshape(*leading, channels, grid_h * tile_h, grid_w * tile_w)
        )

    @staticmethod
    def _overlap_phase_window(
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Separable sqrt-Hann used by both analysis and synthesis."""
        window_h = torch.hann_window(
            height, periodic=False, device=device, dtype=torch.float32
        )
        window_w = torch.hann_window(
            width, periodic=False, device=device, dtype=torch.float32
        )
        return (window_h[:, None] * window_w[None, :]).clamp_min(0.0).sqrt().to(dtype)

    def _cached_overlap_phase_window(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Cache the tiny overlap analysis/synthesis window for FC-PASM."""
        key = (int(height), int(width), str(device), str(dtype))
        cached = self._overlap_window_cache.get(key)
        if cached is None:
            cached = self._overlap_phase_window(
                height=int(height),
                width=int(width),
                device=device,
                dtype=dtype,
            )
            self._overlap_window_cache[key] = cached
        return cached

    def _fc_parallel_stream_pool(
        self, device: torch.device
    ) -> tuple[torch.cuda.Stream, ...]:
        if self.fc_parallel_streams <= 0:
            return ()
        key = str(device)
        cached = self._fc_stream_pool.get(key)
        if cached is None:
            cached = tuple(
                torch.cuda.Stream(device=device)
                for _ in range(self.fc_parallel_streams)
            )
            self._fc_stream_pool[key] = cached
        return cached

    def _cached_overlap_phase_normalization(
        self,
        *,
        output_shape: tuple[int, int],
        window_shape: tuple[int, int] = (11, 10),
        stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (4, 5),
        synthesis_window: torch.Tensor,
    ) -> torch.Tensor:
        """Cache the OLA partition-of-unity denominator.

        The denominator depends only on geometry and the fixed Hann window,
        never on a layer's residual values.  Recomputing a full ``fold`` for
        every woven layer was pure overhead and accounted for a measurable
        fraction of the FC-PASM OLA time.
        """
        height, width = (int(output_shape[0]), int(output_shape[1]))
        window_h, window_w = (int(window_shape[0]), int(window_shape[1]))
        stride_h, stride_w = (int(stride[0]), int(stride[1]))
        pad_h, pad_w = (int(padding[0]), int(padding[1]))
        key = (
            height,
            width,
            window_h,
            window_w,
            stride_h,
            stride_w,
            pad_h * 1000 + pad_w,
            str(synthesis_window.device),
            str(synthesis_window.dtype),
        )
        cached = self._overlap_normalization_cache.get(key)
        if cached is None:
            grid_h = (height + 2 * pad_h - window_h) // stride_h + 1
            grid_w = (width + 2 * pad_w - window_w) // stride_w + 1
            patches = grid_h * grid_w
            weight_patch = (synthesis_window * synthesis_window).reshape(
                1, -1, 1
            ).expand(1, window_h * window_w, patches)
            padded_shape = (height + 2 * pad_h, width + 2 * pad_w)
            cached = F.fold(
                weight_patch,
                output_size=padded_shape,
                kernel_size=window_shape,
                stride=stride,
            )[:, :, pad_h : pad_h + height, pad_w : pad_w + width]
            self._overlap_normalization_cache[key] = cached
        if cached.device != synthesis_window.device:
            raise RuntimeError("cached OLA normalization is on the wrong device")
        return cached

    @classmethod
    def _overlap_phase_tiles(
        cls,
        value: torch.Tensor,
        *,
        window_shape: tuple[int, int] = (11, 10),
        stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (4, 5),
        analysis_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Extract windowed row-major patches from [...,C,H,W]."""
        leading = value.shape[:-3]
        channels, height, width = value.shape[-3:]
        window_h, window_w = window_shape
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        flat = value.reshape(-1, channels, height, width)
        padded = F.pad(flat, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        columns = F.unfold(
            padded,
            kernel_size=window_shape,
            stride=stride,
        )
        grid_h = (height + 2 * pad_h - window_h) // stride_h + 1
        grid_w = (width + 2 * pad_w - window_w) // stride_w + 1
        patches = columns.transpose(1, 2).reshape(
            *leading,
            grid_h * grid_w,
            channels,
            window_h,
            window_w,
        )
        analysis = (
            cls._overlap_phase_window(
                height=window_h,
                width=window_w,
                device=value.device,
                dtype=value.dtype,
            )
            if analysis_window is None
            else analysis_window
        )
        if analysis.shape != (window_h, window_w) or analysis.device != value.device:
            raise RuntimeError("overlap PASM analysis window has an invalid layout")
        return patches * analysis, (grid_h, grid_w)

    @classmethod
    def _unphase_overlap_tiles(
        cls,
        value: torch.Tensor,
        *,
        output_shape: tuple[int, int],
        window_shape: tuple[int, int] = (11, 10),
        stride: tuple[int, int] = (8, 8),
        padding: tuple[int, int] = (4, 5),
        synthesis_window: torch.Tensor | None = None,
        synthesis_normalization: torch.Tensor | None = None,
        use_fc_triton: bool = False,
    ) -> torch.Tensor:
        """Normalized overlap-add inverse for analysis-windowed patches."""
        leading = value.shape[:-4]
        patches, channels, window_h, window_w = value.shape[-4:]
        height, width = output_shape
        pad_h, pad_w = padding
        grid_h = (height + 2 * pad_h - window_h) // stride[0] + 1
        grid_w = (width + 2 * pad_w - window_w) // stride[1] + 1
        if patches != grid_h * grid_w or (window_h, window_w) != window_shape:
            raise RuntimeError("overlap PASM received an invalid patch layout")
        synthesis = (
            cls._overlap_phase_window(
                height=window_h,
                width=window_w,
                device=value.device,
                dtype=value.dtype,
            )
            if synthesis_window is None
            else synthesis_window
        )
        if synthesis.shape != (window_h, window_w) or synthesis.device != value.device:
            raise RuntimeError("overlap PASM synthesis window has an invalid layout")
        padded_shape = (height + 2 * pad_h, width + 2 * pad_w)
        weight_patch = (synthesis * synthesis).reshape(1, -1, 1).expand(
            1, window_h * window_w, patches
        )
        normalization = (
            F.fold(
                weight_patch,
                output_size=padded_shape,
                kernel_size=window_shape,
                stride=stride,
            )
            if synthesis_normalization is None
            else synthesis_normalization
        )
        cropped_norm = (
            normalization[
                :,
                :,
                pad_h : pad_h + height,
                pad_w : pad_w + width,
            ]
            if synthesis_normalization is None
            else synthesis_normalization
        )
        if cropped_norm.shape[-2:] != (height, width):
            raise RuntimeError("cached OLA normalization has an invalid shape")
        if bool((cropped_norm <= 1e-8).any()):
            raise RuntimeError("overlap PASM window does not cover the output grid")
        if use_fc_triton:
            # The Triton path consumes the same windowed tiles and cached
            # denominator.  Keep the reference fold implementation above as
            # the default and fail closed if a non-production layout is ever
            # routed to the specialised kernel.
            return fc_unphase_overlap_tiles(
                value.reshape(-1, patches, channels, window_h, window_w),
                synthesis.float(),
                cropped_norm,
                output_shape=(height, width),
                window_shape=window_shape,
                stride=stride,
                padding=padding,
            ).reshape(*leading, channels, height, width)
        flat = value.reshape(-1, patches, channels, window_h, window_w)
        weighted = flat * synthesis
        columns = weighted.reshape(
            -1, patches, channels * window_h * window_w
        ).transpose(1, 2)
        reconstructed = F.fold(
            columns,
            output_size=padded_shape,
            kernel_size=window_shape,
            stride=stride,
        )
        cropped = reconstructed[
            :,
            :,
            pad_h : pad_h + height,
            pad_w : pad_w + width,
        ]
        return (cropped / cropped_norm).reshape(
            *leading, channels, height, width
        )

    @staticmethod
    def _local_shared_cross_phase_unit(
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel-voted unit motion independently for every spatial tile."""
        if left_spectrum.ndim != 4 or right_spectrum.shape != left_spectrum.shape:
            raise RuntimeError("local PASM spectra must have shape [P,C,h,wf]")
        channel_cross = right_spectrum * left_spectrum.conj()
        shared = channel_cross.sum(dim=1)
        magnitude = shared.abs()
        unit = torch.where(
            magnitude > 1e-8,
            shared / magnitude.clamp_min(1e-8),
            torch.ones_like(shared),
        )
        coherence = magnitude / channel_cross.abs().sum(dim=1).clamp_min(1e-8)
        return unit, coherence.clamp(0.0, 1.0)

    @classmethod
    def _local_phase_aligned_pair_spectrum(
        cls,
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("local phase-aligned alpha must be in [0,1]")
        unit, coherence = cls._local_shared_cross_phase_unit(
            left_spectrum, right_spectrum
        )
        shared_angle = torch.angle(unit)
        left_transport = torch.polar(
            torch.ones_like(shared_angle), float(alpha) * shared_angle
        )
        right_transport = torch.polar(
            torch.ones_like(shared_angle),
            -float(1.0 - alpha) * shared_angle,
        )
        mixed = (
            float(1.0 - alpha) * left_spectrum * left_transport[:, None]
            + float(alpha) * right_spectrum * right_transport[:, None]
        )
        return mixed, coherence

    @staticmethod
    def _normalized_rfft_radius(
        height: int, width: int, *, device: torch.device
    ) -> torch.Tensor:
        fy = torch.fft.fftfreq(height, device=device).abs()
        fx = torch.fft.rfftfreq(width, device=device).abs()
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        return radius / radius.max().clamp_min(1e-8)

    def _frequency_confidence_pair_spectrum(
        self,
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
        alpha: float,
        *,
        force_affinity: float | None = None,
        endpoint_stats: dict[str, torch.Tensor] | None = None,
        temporal_consistency: torch.Tensor | None = None,
        frequency_threshold: torch.Tensor | None = None,
        precomputed_tile_confidence: torch.Tensor | None = None,
        precomputed_affinity: torch.Tensor | None = None,
        control_only: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Unified V21-linear/PASM spectrum with endpoint-only confidence.

        Confidence is the cross-energy-weighted channel phase coherence of
        the two Exact endpoints.  No target or held-out Exact frame enters
        either the confidence or the transport affinity.
        """
        if left_spectrum.ndim != 4 or right_spectrum.shape != left_spectrum.shape:
            raise RuntimeError("FC-PASM spectra must have shape [P,C,h,wf]")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("FC-PASM alpha must lie in [0,1]")
        stats = endpoint_stats or self._frequency_confidence_endpoint_stats(
            left_spectrum, right_spectrum
        )
        theta = stats["theta"]
        base_confidence = stats["base_confidence"]
        ramp_confidence = stats["ramp_confidence"]
        if temporal_consistency is None:
            temporal_consistency = torch.ones_like(base_confidence)
        if temporal_consistency.shape != base_confidence.shape:
            raise RuntimeError("FC temporal consistency must have one value per tile")
        if precomputed_tile_confidence is not None:
            if precomputed_affinity is None:
                raise RuntimeError("precomputed FC confidence requires affinity")
            tile_confidence = precomputed_tile_confidence
            affinity = precomputed_affinity
        else:
            tile_confidence = base_confidence
            if self.fc_ramp_confidence:
                tile_confidence = tile_confidence * ramp_confidence
            if self.fc_temporal_consistency:
                tile_confidence = tile_confidence * temporal_consistency
            tile_confidence = tile_confidence.clamp(0.0, 1.0)
            if force_affinity is None:
                threshold = frequency_threshold
                if threshold is None:
                    radius = self._normalized_rfft_radius(
                        left_spectrum.shape[-2],
                        (left_spectrum.shape[-1] - 1) * 2,
                        device=left_spectrum.device,
                    )
                    threshold = self.fc_tau_low + (
                        self.fc_tau_high - self.fc_tau_low
                    ) * radius.pow(self.fc_freq_power)
                affinity = torch.sigmoid(
                    (tile_confidence[:, None, None] - threshold[None])
                    / self.fc_temperature
                )
            else:
                if float(force_affinity) not in {0.0, 1.0}:
                    raise ValueError("FC-PASM forced affinity is only for endpoint tests")
                affinity = torch.full_like(theta, float(force_affinity))
        if (
            force_affinity is None
            and
            self.fc_transport_tile_topk > 0
            and self.fc_transport_tile_topk < int(tile_confidence.numel())
        ):
            # Endpoint-only high-precision transport regime.  Rank windows
            # solely from the two Exact anchors; skipped/held-out targets do
            # not participate.  Frequencies in non-selected windows receive
            # g=0 and therefore reduce exactly to the V21 spectrum.
            selected_tiles = torch.topk(
                tile_confidence,
                k=self.fc_transport_tile_topk,
                largest=True,
                sorted=False,
            ).indices
            tile_mask = torch.zeros_like(tile_confidence, dtype=torch.bool)
            tile_mask.scatter_(0, selected_tiles, True)
            affinity = affinity * tile_mask[:, None, None]
        if control_only:
            mixed = left_spectrum
        else:
            left_phase = torch.polar(
                torch.ones_like(theta), affinity * float(alpha) * theta
            )
            right_phase = torch.polar(
                torch.ones_like(theta),
                -affinity * float(1.0 - alpha) * theta,
            )
            mixed = (
                float(1.0 - alpha) * left_spectrum * left_phase[:, None]
                + float(alpha) * right_spectrum * right_phase[:, None]
            )
        diagnostics = {
            "base_confidence": base_confidence,
            "ramp_confidence": ramp_confidence,
            "temporal_consistency": temporal_consistency,
            "displacement": stats["displacement"],
        }
        return mixed, tile_confidence, affinity, diagnostics

    def _fc_pair_gate_controls(
        self,
        pair_stats: dict[tuple[int, int], dict[str, torch.Tensor]],
    ) -> tuple[dict[tuple[int, int], bool], dict[tuple[int, int], torch.Tensor]]:
        """Build an endpoint-only pair gate without touching FC-PASM math.

        The scalar is deliberately formed only from the endpoint phase
        consensus already computed by ``_frequency_confidence_endpoint_stats``.
        No target/held-out frame is consulted.  A disabled threshold returns
        empty maps, preserving the frozen path byte-for-byte.
        """
        if self.fc_pair_gate_threshold < 0.0:
            return {}, {}
        if not self._fc_pair_gate_runtime_active:
            self._fc_pair_gate_inactive_calls += 1
            return {}, {}
        self._fc_pair_gate_active_calls += 1
        gates: dict[tuple[int, int], bool] = {}
        scores: dict[tuple[int, int], torch.Tensor] = {}
        ordered_keys: list[tuple[int, int]] = []
        ordered_scores: list[torch.Tensor] = []
        for pair_key, stats in pair_stats.items():
            score_tensor = stats["base_confidence"]
            if self.fc_ramp_confidence:
                score_tensor = score_tensor * stats["ramp_confidence"]
            if self.fc_temporal_consistency:
                score_tensor = score_tensor * stats.get(
                    "temporal_consistency", torch.ones_like(score_tensor)
                )
            score = score_tensor.float().mean()
            scores[pair_key] = score.detach()
            ordered_keys.append(pair_key)
            ordered_scores.append(score.detach())
        # Materialize all tiny endpoint decisions with one device-to-host
        # synchronization per woven layer.  The score arithmetic and strict
        # comparison are identical to the former per-pair ``.item()`` path.
        if ordered_scores:
            decisions = (
                torch.stack(ordered_scores)
                .lt(self.fc_pair_gate_threshold)
                .to(device="cpu")
                .tolist()
            )
            gates.update(
                (pair_key, bool(decision))
                for pair_key, decision in zip(ordered_keys, decisions, strict=True)
            )
        self._fc_pair_gate_skipped_pairs += sum(gates.values())
        return gates, scores

    def _frequency_confidence_endpoint_stats(
        self,
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Endpoint-only coherence and rigid-ramp explainability per tile."""
        if left_spectrum.ndim != 4 or right_spectrum.shape != left_spectrum.shape:
            raise RuntimeError("FC endpoint spectra must have shape [P,C,h,wf]")
        channel_cross = right_spectrum * left_spectrum.conj()
        shared = channel_cross.sum(dim=1)
        shared_magnitude = shared.abs()
        cross_energy = channel_cross.abs().sum(dim=1)
        coherence = shared_magnitude / cross_energy.clamp_min(1e-8)
        confidence_weight = cross_energy.clone()
        confidence_weight[:, 0, 0] = 0.0
        denominator = confidence_weight.sum(dim=(-2, -1)).clamp_min(1e-8)
        base_confidence = (
            (coherence * confidence_weight).sum(dim=(-2, -1)) / denominator
        ).clamp(0.0, 1.0)
        unit = torch.where(
            shared_magnitude > 1e-8,
            shared / shared_magnitude.clamp_min(1e-8),
            torch.ones_like(shared),
        )
        theta = torch.atan2(unit.imag, unit.real).clone()
        theta[:, 0, 0] = 0.0
        displacement, _coherence, _peak = self._local_shared_displacement(
            left_spectrum,
            right_spectrum,
            spatial_shape=(
                left_spectrum.shape[-2],
                (left_spectrum.shape[-1] - 1) * 2,
            ),
            phase_unit=unit,
            phase_coherence=coherence,
            use_triton_peak=self.fc_triton_phat_peak,
        )
        if self.fc_triton_ramp_confidence:
            ramp_confidence = fc_ramp_confidence(
                unit.contiguous(),
                displacement,
                confidence_weight.contiguous(),
                denominator,
                spatial_width=(left_spectrum.shape[-1] - 1) * 2,
            )
        else:
            predicted_unit = self._linear_phase_ramp(
                displacement,
                spatial_shape=(
                    left_spectrum.shape[-2],
                    (left_spectrum.shape[-1] - 1) * 2,
                ),
                scale=1.0,
                dtype=unit.dtype,
            )
            circular_agreement = (unit * predicted_unit.conj()).real.clamp(0.0, 1.0)
            ramp_confidence = (
                (circular_agreement * confidence_weight).sum(dim=(-2, -1))
                / denominator
            ).clamp(0.0, 1.0)
        return {
            "theta": theta,
            "base_confidence": base_confidence,
            "ramp_confidence": ramp_confidence,
            "displacement": displacement,
        }

    def _frequency_confidence_endpoint_stats_batched(
        self,
        left_spectra: torch.Tensor,
        right_spectra: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate endpoint confidence for several anchor pairs at once.

        ``_frequency_confidence_endpoint_stats`` operates on ``[P,C,h,wf]``.
        Pair batching only folds the pair and tile axes together; all channel
        reductions and the PHAT displacement estimator retain their original
        order for each pair/tile.  The helper is used by FC-R/FC-RT/FC-RTB to
        remove a large number of tiny CUDA launches from the woven-layer hot
        path without changing the confidence or phase equations.
        """
        if (
            left_spectra.ndim != 5
            or right_spectra.shape != left_spectra.shape
        ):
            raise RuntimeError(
                "batched FC endpoint spectra must have shape [Q,P,C,h,wf]"
            )
        pair_count, tile_count, channels, height, width_freq = left_spectra.shape
        flat = self._frequency_confidence_endpoint_stats(
            left_spectra.reshape(
                pair_count * tile_count, channels, height, width_freq
            ),
            right_spectra.reshape(
                pair_count * tile_count, channels, height, width_freq
            ),
        )
        return {
            key: value.reshape(pair_count, tile_count, *value.shape[1:])
            for key, value in flat.items()
        }

    @staticmethod
    def _frequency_confidence_temporal_score(
        displacement: torch.Tensor,
        gap: int,
        neighbors: list[tuple[torch.Tensor, int]],
    ) -> torch.Tensor:
        """Causal endpoint-pair velocity agreement on the complex-motion path."""
        if not neighbors:
            return torch.ones(displacement.shape[0], device=displacement.device)
        velocity = displacement / float(gap)
        scores = []
        for neighbor_displacement, neighbor_gap in neighbors:
            neighbor_velocity = neighbor_displacement / float(neighbor_gap)
            difference = (velocity - neighbor_velocity).norm(dim=-1)
            scale = (
                velocity.norm(dim=-1)
                + neighbor_velocity.norm(dim=-1)
                + 0.25
            )
            scores.append(torch.exp(-2.0 * difference / scale))
        return torch.stack(scores).mean(dim=0).clamp(0.0, 1.0)

    @classmethod
    def _local_shared_displacement(
        cls,
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        phase_unit: torch.Tensor | None = None,
        phase_coherence: torch.Tensor | None = None,
        use_triton_peak: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate one subpixel (dx,dy) per tile by channel-voted PHAT."""
        if phase_unit is None or phase_coherence is None:
            unit, coherence = cls._local_shared_cross_phase_unit(
                left_spectrum, right_spectrum
            )
        else:
            expected_shape = (
                left_spectrum.shape[0],
                left_spectrum.shape[-2],
                left_spectrum.shape[-1],
            )
            if (
                phase_unit.shape != expected_shape
                or phase_coherence.shape != expected_shape
            ):
                raise RuntimeError(
                    "cached local phase statistics have an invalid layout"
                )
            unit, coherence = phase_unit, phase_coherence
        height, width = spatial_shape
        correlation = torch.fft.irfft2(
            unit * coherence,
            s=spatial_shape,
            dim=(-2, -1),
        )
        patches = correlation.shape[0]
        if use_triton_peak:
            displacement, peak_confidence = fc_phat_peak_subpixel(
                correlation.contiguous()
            )
            return displacement, coherence, peak_confidence
        flat_index = correlation.reshape(patches, -1).argmax(dim=1)
        peak_y = torch.div(flat_index, width, rounding_mode="floor")
        peak_x = flat_index.remainder(width)
        patch_index = torch.arange(patches, device=correlation.device)
        center = correlation[patch_index, peak_y, peak_x]
        x_left = correlation[patch_index, peak_y, (peak_x - 1).remainder(width)]
        x_right = correlation[patch_index, peak_y, (peak_x + 1).remainder(width)]
        y_up = correlation[patch_index, (peak_y - 1).remainder(height), peak_x]
        y_down = correlation[patch_index, (peak_y + 1).remainder(height), peak_x]

        def subpixel(before: torch.Tensor, middle: torch.Tensor, after: torch.Tensor):
            denominator = before - 2.0 * middle + after
            return torch.where(
                denominator.abs() > 1e-8,
                (0.5 * (before - after) / denominator).clamp(-0.5, 0.5),
                torch.zeros_like(middle),
            )

        offset_x = subpixel(x_left, center, x_right)
        offset_y = subpixel(y_up, center, y_down)
        signed_x = torch.where(
            peak_x >= (width + 1) // 2,
            peak_x - width,
            peak_x,
        ).to(torch.float32) + offset_x
        signed_y = torch.where(
            peak_y >= (height + 1) // 2,
            peak_y - height,
            peak_y,
        ).to(torch.float32) + offset_y
        displacement = torch.stack((signed_x, signed_y), dim=-1)
        peak_confidence = center / correlation.abs().mean(dim=(-2, -1)).clamp_min(1e-8)
        return displacement, coherence, peak_confidence

    @staticmethod
    def _linear_phase_ramp(
        displacement: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        scale: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate exp(-i scale*(wx*dx+wy*dy)) for every tile."""
        height, width = spatial_shape
        device = displacement.device
        frequency_y = 2.0 * math.pi * torch.fft.fftfreq(
            height, device=device, dtype=torch.float32
        )
        frequency_x = 2.0 * math.pi * torch.fft.rfftfreq(
            width, device=device, dtype=torch.float32
        )
        dx = displacement[:, 0, None, None]
        dy = displacement[:, 1, None, None]
        angle = -float(scale) * (
            dy * frequency_y[None, :, None]
            + dx * frequency_x[None, None, :]
        )
        return torch.polar(torch.ones_like(angle), angle).to(dtype)

    @classmethod
    def _local_linear_phase_ramp_pair_spectrum(
        cls,
        left_spectrum: torch.Tensor,
        right_spectrum: torch.Tensor,
        alpha: float,
        *,
        spatial_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transport Exact spectra with a two-parameter rigid phase ramp."""
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("linear phase-ramp alpha must be in [0,1]")
        displacement, coherence, peak_confidence = cls._local_shared_displacement(
            left_spectrum,
            right_spectrum,
            spatial_shape=spatial_shape,
        )
        left_ramp = cls._linear_phase_ramp(
            displacement,
            spatial_shape=spatial_shape,
            scale=float(alpha),
            dtype=left_spectrum.dtype,
        )
        right_ramp = cls._linear_phase_ramp(
            displacement,
            spatial_shape=spatial_shape,
            scale=-float(1.0 - alpha),
            dtype=right_spectrum.dtype,
        )
        mixed = (
            float(1.0 - alpha) * left_spectrum * left_ramp[:, None]
            + float(alpha) * right_spectrum * right_ramp[:, None]
        )
        return mixed, coherence, displacement, peak_confidence

    def _correct_phase_aligned_spectral_mixing(
        self,
        *,
        layer_index: int,
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        memory_length: int,
        spatial_height: int,
        spatial_width: int,
        cuda_start: torch.cuda.Event,
        fft_start: torch.cuda.Event,
        fft_end: torch.cuda.Event,
        cuda_end: torch.cuda.Event,
        start_wall: int,
    ) -> torch.Tensor:
        if full_residual.shape[0] != 1 or active_residual.shape[0] != 1:
            raise RuntimeError("phase-aligned spectral mixing requires batch size one")
        current_length = int(full_residual.shape[1]) - int(memory_length)
        active_positions = torch.nonzero(
            active_frames >= memory_length, as_tuple=False
        ).flatten()
        active_global = active_frames.index_select(0, active_positions)
        order = torch.argsort(active_global)
        active_positions = active_positions.index_select(0, order)
        exact_local = [
            int(value) - memory_length
            for value in active_global.index_select(0, order).detach().cpu().tolist()
        ]
        if not exact_local or len(exact_local) != len(set(exact_local)):
            raise RuntimeError("PASM requires unique Exact Current anchors")
        exact_residual = active_residual.index_select(
            1, active_positions.to(active_residual.device)
        )[0].reshape(
            len(exact_local), spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        current_interp = full_residual[0, memory_length:].reshape(
            current_length, spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        masks = self._radial_masks(
            spatial_height,
            spatial_width,
            self.spectral_num_bands,
            device=full_residual.device,
        )
        low_mask = masks[:2].sum(dim=0)
        fft_start.record()
        anchor_spectra = torch.fft.rfft2(exact_residual, dim=(-2, -1))

        def bounds(target: int, *, exclude: int | None = None):
            same_segment = target < self.current_anchor_boundary
            candidates = [
                (index, frame)
                for index, frame in enumerate(exact_local)
                if frame != exclude
                and (frame < self.current_anchor_boundary) == same_segment
            ]
            left = [(index, frame) for index, frame in candidates if frame < target]
            right = [(index, frame) for index, frame in candidates if frame > target]
            return (left[-1], right[0]) if left and right else None

        inactive = [frame for frame in range(current_length) if frame not in exact_local]
        eligible: list[int] = []
        corrections: list[torch.Tensor] = []
        coherences: list[torch.Tensor] = []
        alphas: list[float] = []
        for target in inactive:
            pair = bounds(target)
            if pair is None:
                continue
            (left_index, left_frame), (right_index, right_frame) = pair
            alpha = float(target - left_frame) / float(right_frame - left_frame)
            transported, coherence = self._phase_aligned_pair_spectrum(
                anchor_spectra[left_index], anchor_spectra[right_index], alpha
            )
            base_spectrum = torch.fft.rfft2(
                current_interp[target], dim=(-2, -1)
            )
            mixed_spectrum = base_spectrum + low_mask[None] * (
                transported - base_spectrum
            )
            predicted = torch.fft.irfft2(
                mixed_spectrum,
                s=(spatial_height, spatial_width),
                dim=(-2, -1),
            )
            corrections.append(predicted - current_interp[target])
            eligible.append(target)
            coherences.append((coherence * low_mask).sum() / low_mask.sum())
            alphas.append(alpha)

        # Leave-one-out only diagnoses whether shared phase alignment beats
        # unaligned spectral mixing.  It does not inspect benchmark metrics and
        # does not mutate or gate the generated path.
        loo_baseline: list[torch.Tensor] = []
        loo_pasm: list[torch.Tensor] = []
        for held_out, target in enumerate(exact_local):
            pair = bounds(target, exclude=target)
            if pair is None:
                continue
            (left_index, left_frame), (right_index, right_frame) = pair
            alpha = float(target - left_frame) / float(right_frame - left_frame)
            unaligned = (
                float(1.0 - alpha) * anchor_spectra[left_index]
                + float(alpha) * anchor_spectra[right_index]
            )
            transported, _ = self._phase_aligned_pair_spectrum(
                anchor_spectra[left_index], anchor_spectra[right_index], alpha
            )
            pasm = unaligned + low_mask[None] * (transported - unaligned)
            baseline_value = torch.fft.irfft2(
                unaligned,
                s=(spatial_height, spatial_width),
                dim=(-2, -1),
            )
            pasm_value = torch.fft.irfft2(
                pasm,
                s=(spatial_height, spatial_width),
                dim=(-2, -1),
            )
            loo_baseline.append((exact_residual[held_out] - baseline_value).norm())
            loo_pasm.append((exact_residual[held_out] - pasm_value).norm())
        fft_end.record()

        corrected = full_residual.clone()
        trust_scales: list[torch.Tensor] = []
        if corrections:
            correction_stack = torch.stack(corrections)
            target_indices = torch.tensor(
                eligible, device=full_residual.device, dtype=torch.long
            )
            reference = current_interp.index_select(0, target_indices)
            correction_norm = correction_stack.flatten(1).norm(dim=1)
            reference_norm = reference.flatten(1).norm(dim=1)
            trust = torch.minimum(
                torch.ones_like(correction_norm),
                self.eta * reference_norm / correction_norm.clamp_min(1e-8),
            )
            correction_stack.mul_(trust[:, None, None, None])
            target_view = corrected[:, memory_length:].reshape(
                1,
                current_length,
                spatial_height,
                spatial_width,
                corrected.shape[-1],
            )
            # Advanced indexing would return a temporary and silently drop
            # the correction.  index_add_ is the actual in-place scatter into
            # the chronological Current-frame axis.
            target_view[0].index_add_(
                0,
                target_indices,
                correction_stack.permute(0, 2, 3, 1),
            )
            trust_scales = list(trust.unbind())
        corrected[:, active_frames] = active_residual
        cuda_end.record()
        baseline_mean = (
            torch.stack(loo_baseline).mean()
            if loo_baseline
            else torch.full((), float("nan"), device=full_residual.device)
        )
        pasm_mean = (
            torch.stack(loo_pasm).mean()
            if loo_pasm
            else torch.full((), float("nan"), device=full_residual.device)
        )
        self._records.append(
            {
                "chunk_index": self._chunk,
                "step_index": self._step,
                "layer_index": int(layer_index),
                "mode": self.mode,
                "status": "corrected" if corrections else "temporal_fallback",
                "fallback_reason": None if corrections else "no_segment_local_bracket",
                "exact_current_frames": exact_local,
                "approximate_current_frames": inactive,
                "target_frames": eligible,
                "target_alphas": alphas,
                "shared_phase_across_channels": True,
                "unit_circle_transport": True,
                "phase_align_before_mix": True,
                "wrapped_angle_regression": False,
                "memory_used_as_correction_source": False,
                "no_cross_boundary_matching": True,
                "inactive_current_only": True,
                "approximate_high_frequency_preserved": True,
                "low_frequency_bands_corrected": [0, 1],
                "mean_shared_cross_spectrum_coherence": (
                    torch.stack(coherences).mean()
                    if coherences
                    else torch.zeros((), device=full_residual.device)
                ),
                "trust_scale_min": (
                    torch.stack(trust_scales).min()
                    if trust_scales
                    else torch.ones((), device=full_residual.device)
                ),
                "loo_calibration_frames": len(loo_baseline),
                "loo_unaligned_residual_l2": baseline_mean.detach(),
                "loo_phase_aligned_residual_l2": pasm_mean.detach(),
                "loo_phase_aligned_improvement": (
                    1.0 - pasm_mean / baseline_mean.clamp_min(1e-8)
                ).detach(),
                "_cuda_start": cuda_start,
                "_cuda_end": cuda_end,
                "_fft_start": fft_start,
                "_fft_end": fft_end,
                "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
            }
        )
        return corrected

    def _correct_frequency_confidence_pasm_reference(
        self,
        *,
        layer_index: int,
        force_reference_numerics: bool = False,
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        active_frame_values: tuple[int, ...] | None,
        memory_length: int,
        spatial_height: int,
        spatial_width: int,
        cuda_start: torch.cuda.Event,
        fft_start: torch.cuda.Event,
        fft_end: torch.cuda.Event,
        cuda_end: torch.cuda.Event,
        start_wall: int,
    ) -> torch.Tensor:
        """Frozen FC-R arithmetic reconstructed from its validated bytecode.

        Keep pair statistics and target spectra as separate launches and stack
        only after every target has been evaluated.  Folding pair/tile axes or
        reordering these operations is mathematically equivalent, but was not
        numerically equivalent enough for the frozen WorldMark artifact.
        """
        # Keep the lean tile/OLA runtime while allowing a small, explicit set
        # of layers to use the validated reference phase arithmetic.  This is
        # still FC-R reconstruction on those layers (never a V21 bypass).
        use_triton_phase_mix = bool(
            self.fc_triton_phase_mix
            and (
                not force_reference_numerics
                or self.fc_triton_accurate_phase_mix
            )
        )
        if full_residual.shape[0] != 1 or active_residual.shape[0] != 1:
            raise RuntimeError("FC-PASM requires Matrix batch size one")
        current_length = int(full_residual.shape[1]) - int(memory_length)
        if self.fc_lean_runtime and active_frame_values is not None:
            if tuple(sorted(set(active_frame_values))) != tuple(active_frame_values):
                raise RuntimeError("FC-R active-frame values must be sorted and unique")
            if len(active_frame_values) != int(active_frames.numel()):
                raise RuntimeError("FC-R active-frame value count differs from tensor")
            current_positions = [
                index
                for index, frame in enumerate(active_frame_values)
                if int(frame) >= memory_length
            ]
            active_positions = torch.tensor(
                current_positions, device=active_frames.device, dtype=torch.long
            )
            exact_local = [
                int(active_frame_values[index]) - memory_length
                for index in current_positions
            ]
        else:
            active_positions = torch.nonzero(
                active_frames >= memory_length, as_tuple=False
            ).flatten()
            active_global = active_frames.index_select(0, active_positions)
            order = torch.argsort(active_global)
            active_positions = active_positions.index_select(0, order)
            exact_local = [
                int(value) - memory_length
                for value in active_global.index_select(0, order).detach().cpu().tolist()
            ]
        if not exact_local or len(exact_local) != len(set(exact_local)):
            raise RuntimeError("FC-PASM requires unique Exact Current anchors")
        gate_skip = False
        if (
            self.fc_layer_gate_threshold >= 0.0
            and self.fc_layer_gate_period > 0
            and int(layer_index) % self.fc_layer_gate_period != 0
        ):
            # The frozen/reference arithmetic path must obey the same
            # explicit threshold=1 periodic schedule as the lean path.  The
            # previous implementation applied the gate only in the lean
            # branch, so reference-numerics candidates silently ran FC in
            # every layer despite carrying gate metadata.
            if self.fc_layer_gate_threshold >= 1.0:
                gate_skip = True
            elif self._fc_previous_mean_affinity is not None:
                gate_skip = bool(
                    (
                        self._fc_previous_mean_affinity
                        < self.fc_layer_gate_threshold
                    ).detach().cpu().item()
                )
        if gate_skip:
            self._fc_layer_gate_skips += 1
            fft_start.record()
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "low_previous_layer_affinity",
                    "exact_current_frames": exact_local,
                    "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
                    "fc_layer_gate_period": self.fc_layer_gate_period,
                    "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
                    "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
                    "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall)
                    / 1_000_000.0,
                }
            )
            return full_residual
        if self.fc_v21_bypass:
            # Isolated control: preserve the ROSCA-A routing decision and all
            # frame/attention budgets, but use the already materialized V21
            # residual interpolation instead of running FC-PASM.  This is
            # intentionally a diagnostic control, never the paper variant.
            fft_start.record()
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "fc_v21_bypass",
                    "exact_current_frames": exact_local,
                    "fc_v21_bypass": True,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall)
                    / 1_000_000.0,
                }
            )
            return full_residual
        exact_residual = active_residual.index_select(
            1, active_positions.to(active_residual.device)
        )[0].reshape(
            len(exact_local), spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        current_interp = full_residual[0, memory_length:].reshape(
            current_length, spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        overlap_window = (
            self._cached_overlap_phase_window(
                height=11,
                width=10,
                device=full_residual.device,
                dtype=exact_residual.dtype,
            )
            if self.fc_lean_runtime
            else None
        )
        if self.fc_triton_tile_extract:
            if overlap_window is None:
                overlap_window = self._overlap_phase_window(
                    height=11,
                    width=10,
                    device=full_residual.device,
                    dtype=exact_residual.dtype,
                )
            exact_tiles = fc_extract_overlap_tiles(
                exact_residual,
                overlap_window.float(),
            )
            tile_grid = (3, 6)
        else:
            exact_tiles, tile_grid = self._overlap_phase_tiles(
                exact_residual, analysis_window=overlap_window
            )
        tile_height, tile_width = exact_tiles.shape[-2:]
        transport_tiles = exact_tiles
        if self.fc_coarse_transport:
            exact_count, tile_count, channels = exact_tiles.shape[:3]
            transport_tiles = F.avg_pool2d(
                exact_tiles.reshape(
                    exact_count * tile_count, channels, tile_height, tile_width
                ),
                kernel_size=(2, 2),
                stride=(2, 2),
                ceil_mode=True,
                count_include_pad=False,
            ).reshape(exact_count, tile_count, channels, 6, 5)
        transport_height, transport_width = transport_tiles.shape[-2:]
        radial_masks = self._radial_masks(
            transport_height, transport_width, 4, device=full_residual.device
        )
        fft_start.record()
        anchor_spectra = torch.fft.rfft2(transport_tiles, dim=(-2, -1))
        if use_triton_phase_mix:
            # cuFFT may retain a batch-major stride that makes an individual
            # anchor slice non-contiguous.  Materialize the complete anchor
            # bank once so every target reuses coalesced P/C/H/Wf reads.
            anchor_spectra = anchor_spectra.contiguous()
        fft_forward_end = self._timing_event()
        fft_forward_end.record()
        phase_start = self._timing_event()
        phase_end = self._timing_event()
        ola_start = self._timing_event()
        ola_end = self._timing_event()
        phase_start.record()

        def bounds(target: int):
            same_segment = target < self.current_anchor_boundary
            candidates = [
                (index, frame)
                for index, frame in enumerate(exact_local)
                if (frame < self.current_anchor_boundary) == same_segment
            ]
            left = [(index, frame) for index, frame in candidates if frame < target]
            right = [(index, frame) for index, frame in candidates if frame > target]
            return (left[-1], right[0]) if left and right else None

        pair_stats: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        consecutive_pairs: list[tuple[int, int, int, int]] = []
        target_pair_keys = {
            (bounds(target)[0][1], bounds(target)[1][1])
            for target in range(current_length)
            if target not in exact_local and bounds(target) is not None
        }
        prune_unused_pairs = bool(
            self.fc_prune_unused_pairs and not self.fc_temporal_consistency
        )
        parallel_streams = (
            self._fc_parallel_stream_pool(full_residual.device)
            if self.fc_lean_runtime and self.fc_parallel_streams > 0
            else ()
        )
        default_stream = torch.cuda.current_stream(full_residual.device)
        pair_streams: dict[tuple[int, int], torch.cuda.Stream] = {}
        batched_pair_specs: list[tuple[tuple[int, int], int, int]] = []
        for left_index in range(len(exact_local) - 1):
            right_index = left_index + 1
            left_frame = exact_local[left_index]
            right_frame = exact_local[right_index]
            if (left_frame < self.current_anchor_boundary) != (
                right_frame < self.current_anchor_boundary
            ):
                continue
            pair_key = (left_frame, right_frame)
            if prune_unused_pairs and pair_key not in target_pair_keys:
                continue
            if parallel_streams:
                stream = parallel_streams[len(consecutive_pairs) % len(parallel_streams)]
                stream.wait_stream(default_stream)
                pair_streams[pair_key] = stream
                with torch.cuda.stream(stream):
                    pair_stats[pair_key] = self._frequency_confidence_endpoint_stats(
                        anchor_spectra[left_index], anchor_spectra[right_index]
                    )
            elif self.fc_batched_endpoint_stats:
                # Collect the same chronological pairs first, then perform
                # one [pair,tile,channel,h,wf] endpoint reduction below.
                # This changes launch granularity only; each pair's channel
                # and spatial reduction order remains in the reference
                # helper and the resulting per-pair tensors are unpacked in
                # exactly the original order.
                batched_pair_specs.append((pair_key, left_index, right_index))
            else:
                pair_stats[pair_key] = self._frequency_confidence_endpoint_stats(
                    anchor_spectra[left_index], anchor_spectra[right_index]
                )
            consecutive_pairs.append(
                (left_index, right_index, left_frame, right_frame)
            )

        if batched_pair_specs:
            left_indices = torch.tensor(
                [item[1] for item in batched_pair_specs],
                device=anchor_spectra.device,
                dtype=torch.long,
            )
            right_indices = torch.tensor(
                [item[2] for item in batched_pair_specs],
                device=anchor_spectra.device,
                dtype=torch.long,
            )
            batched_stats = self._frequency_confidence_endpoint_stats_batched(
                anchor_spectra.index_select(0, left_indices),
                anchor_spectra.index_select(0, right_indices),
            )
            for pair_position, (pair_key, _, _) in enumerate(batched_pair_specs):
                pair_stats[pair_key] = {
                    key: value[pair_position]
                    for key, value in batched_stats.items()
                }

        pair_gate, pair_gate_scores = self._fc_pair_gate_controls(pair_stats)

        eligible: list[int] = []
        target_alphas: list[float] = []
        target_anchor_pairs: list[list[int]] = []
        mixed_spectra: list[torch.Tensor] = []
        target_left_indices: list[int] = []
        target_right_indices: list[int] = []
        target_thetas: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        base_confidences: list[torch.Tensor] = []
        ramp_confidences: list[torch.Tensor] = []
        temporal_confidences: list[torch.Tensor] = []
        affinities: list[torch.Tensor] = []
        g0_error = torch.zeros((), device=full_residual.device)
        g1_error = torch.zeros((), device=full_residual.device)
        g1_legacy_non_dc_error = torch.zeros((), device=full_residual.device)
        boundary_checked = False
        pair_controls: dict[
            tuple[int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
        ] = {}
        frequency_threshold = None
        if self.fc_lean_runtime:
            radius = self._normalized_rfft_radius(
                transport_height, transport_width, device=full_residual.device
            )
            frequency_threshold = self.fc_tau_low + (
                self.fc_tau_high - self.fc_tau_low
            ) * radius.pow(self.fc_freq_power)
        preallocated_mixed_spectra = None
        preallocated_target_count = 0
        preallocated_workspace_reused = False
        if self.fc_prealloc_targets:
            # The target order below is chronological and already fully
            # determined by the segmented bracket rule.  Allocate the final
            # stack once, then copy each target's reference-arithmetic result
            # into its slot.  This removes the later torch.stack copy while
            # keeping each target's phase/polar operations and launch order
            # unchanged.
            preallocated_targets = [
                target
                for target in range(current_length)
                if target not in exact_local
                and bounds(target) is not None
                and not pair_gate.get(
                    (bounds(target)[0][1], bounds(target)[1][1]), False
                )
            ]
            preallocated_target_count = len(preallocated_targets)
            if preallocated_target_count:
                device_key = str(anchor_spectra.device)
                workspace_key = (
                    device_key,
                    str(anchor_spectra.dtype),
                    tuple(int(value) for value in anchor_spectra.shape[1:]),
                )
                workspace = self._fc_target_spectrum_workspaces.get(workspace_key)
                if workspace is not None and workspace.shape[0] >= preallocated_target_count:
                    preallocated_workspace_reused = True
                else:
                    workspace = torch.empty(
                        (
                            preallocated_target_count,
                            *anchor_spectra.shape[1:],
                        ),
                        device=anchor_spectra.device,
                        dtype=anchor_spectra.dtype,
                    )
                    self._fc_target_spectrum_workspaces[workspace_key] = workspace
                preallocated_mixed_spectra = workspace[:preallocated_target_count]
        preallocated_slot = 0
        reference_lowfreq_indices = None
        if self.fc_triton_reference_lowfreq:
            reference_radius = self._normalized_rfft_radius(
                transport_height, transport_width, device=full_residual.device
            )
            reference_lowfreq_indices = torch.nonzero(
                (
                    reference_radius <= self.fc_triton_lowfreq_radius
                    if self.fc_legacy_lowfreq_index_bug
                    else (
                        reference_radius <= self.fc_triton_lowfreq_radius
                    ).flatten()
                ),
                as_tuple=False,
            ).flatten()

        def splice_reference_lowfreq(
            mixed: torch.Tensor,
            left: torch.Tensor,
            right: torch.Tensor,
            theta: torch.Tensor,
            affinity: torch.Tensor,
            alpha_value: float,
        ) -> torch.Tensor:
            """Restore only low-frequency bins with reference arithmetic.

            The Triton phase mix remains responsible for the full spectrum;
            this narrow splice avoids changing the high-frequency runtime path
            while preserving the numerically sensitive low-frequency bins that
            dominate the FC-R quality delta.  It is an independent runtime
            candidate and is disabled by default.
            """
            if reference_lowfreq_indices is None or not int(
                reference_lowfreq_indices.numel()
            ):
                return mixed
            indices = reference_lowfreq_indices
            left_low = left.flatten(-2).index_select(-1, indices)
            right_low = right.flatten(-2).index_select(-1, indices)
            theta_low = theta.flatten(-2).index_select(-1, indices)
            affinity_low = affinity.flatten(-2).index_select(-1, indices)
            left_phase = torch.polar(
                torch.ones_like(theta_low),
                affinity_low * float(alpha_value) * theta_low,
            )
            right_phase = torch.polar(
                torch.ones_like(theta_low),
                -affinity_low * (1.0 - float(alpha_value)) * theta_low,
            )
            reference_low = (
                (1.0 - float(alpha_value)) * left_low * left_phase[:, None]
                + float(alpha_value) * right_low * right_phase[:, None]
            )
            mixed_flat = mixed.flatten(-2).clone()
            mixed_flat.index_copy_(-1, indices, reference_low)
            return mixed_flat.reshape_as(mixed)

        def fused_reference_mix(
            left: torch.Tensor,
            right: torch.Tensor,
            theta: torch.Tensor,
            affinity: torch.Tensor,
            alpha_value: float,
        ) -> torch.Tensor:
            """Reference-path runtime fold of the barycentric coefficients.

            This is deliberately opt-in.  The frozen reference path keeps its
            original arithmetic when ``fc_fused_complex_weights`` is false.
            When enabled, the real temporal coefficients are folded into the
            complex polar weights before the channel broadcast, removing two
            full-C elementwise passes while preserving the same FC-R/ROCSA-A
            equation and all routing/budget semantics.
            """
            alpha_tensor = torch.as_tensor(
                float(alpha_value), device=theta.device, dtype=theta.real.dtype
            )
            left_phase = torch.polar(
                (1.0 - alpha_tensor).expand_as(theta),
                affinity * alpha_tensor * theta,
            )
            right_phase = torch.polar(
                alpha_tensor.expand_as(theta),
                -affinity * (1.0 - alpha_tensor) * theta,
            )
            mixed_value = left * left_phase[:, None]
            mixed_value.add_(right * right_phase[:, None])
            return mixed_value

        for target in range(current_length):
            if target in exact_local:
                continue
            pair = bounds(target)
            if pair is None:
                continue
            (left_index, left_frame), (right_index, right_frame) = pair
            alpha = float(target - left_frame) / float(right_frame - left_frame)
            left_spectrum = anchor_spectra[left_index]
            right_spectrum = anchor_spectra[right_index]
            pair_key = (left_frame, right_frame)
            if pair_gate.get(pair_key, False):
                self._fc_pair_gate_skipped_targets += 1
                continue
            stats = pair_stats[pair_key]
            cached_control = pair_controls.get(pair_key)
            if cached_control is not None:
                confidence, affinity, temporal_score, diagnostics = cached_control
            elif self.fc_lean_runtime and not self.fc_temporal_consistency:
                temporal_score = torch.ones_like(stats["base_confidence"])
            else:
                temporal_neighbors: list[tuple[torch.Tensor, int]] = []
                for _, _, neighbor_left, neighbor_right in consecutive_pairs:
                    if (neighbor_left, neighbor_right) == (left_frame, right_frame):
                        continue
                    if neighbor_right == left_frame or neighbor_left == right_frame:
                        neighbor_stats = pair_stats[(neighbor_left, neighbor_right)]
                        temporal_neighbors.append(
                            (neighbor_stats["displacement"], neighbor_right - neighbor_left)
                        )
                temporal_score = self._frequency_confidence_temporal_score(
                    stats["displacement"],
                    right_frame - left_frame,
                    temporal_neighbors,
                )
            if cached_control is None:
                if parallel_streams:
                    pair_streams[pair_key].wait_stream(default_stream)
                    with torch.cuda.stream(pair_streams[pair_key]):
                        mixed, confidence, affinity, diagnostics = (
                            self._frequency_confidence_pair_spectrum(
                                left_spectrum,
                                right_spectrum,
                                alpha,
                                endpoint_stats=stats,
                                temporal_consistency=temporal_score,
                                frequency_threshold=frequency_threshold,
                                control_only=(
                                    use_triton_phase_mix
                                    or self.fc_triton_batched_mix
                                    or self.fc_batched_reference_mix
                                ),
                            )
                        )
                        if self.fc_fused_complex_weights and not use_triton_phase_mix:
                            mixed = fused_reference_mix(
                                left_spectrum,
                                right_spectrum,
                                stats["theta"],
                                affinity,
                                alpha,
                            )
                else:
                    mixed, confidence, affinity, diagnostics = (
                        self._frequency_confidence_pair_spectrum(
                            left_spectrum,
                            right_spectrum,
                            alpha,
                            endpoint_stats=stats,
                            temporal_consistency=temporal_score,
                            frequency_threshold=frequency_threshold,
                            control_only=(
                                use_triton_phase_mix
                                or self.fc_triton_batched_mix
                                or self.fc_batched_reference_mix
                            ),
                        )
                    )
                    if self.fc_fused_complex_weights and not use_triton_phase_mix:
                        mixed = fused_reference_mix(
                            left_spectrum,
                            right_spectrum,
                            stats["theta"],
                            affinity,
                            alpha,
                        )
                if use_triton_phase_mix and not self.fc_triton_batched_mix:
                    mixed = fc_phase_mix(
                        left_spectrum,
                        right_spectrum,
                        stats["theta"].contiguous(),
                        affinity.contiguous(),
                        alpha,
                        accurate_math=self.fc_triton_accurate_phase_mix,
                    )
                    if self.fc_triton_reference_lowfreq:
                        mixed = splice_reference_lowfreq(
                            mixed,
                            left_spectrum,
                            right_spectrum,
                            stats["theta"],
                            affinity,
                            alpha,
                        )
                if self.fc_lean_runtime:
                    pair_controls[pair_key] = (
                        confidence, affinity, temporal_score, diagnostics
                    )
            else:
                if self.fc_batched_reference_mix:
                    mixed = left_spectrum
                elif self.fc_triton_batched_mix:
                    # The final batched kernel below owns this target's
                    # spectrum.  The old reference mix was dead work because
                    # it was unconditionally overwritten after the loop.
                    # Endpoint confidence/affinity remains fully evaluated.
                    mixed = left_spectrum
                elif use_triton_phase_mix:
                    mixed = fc_phase_mix(
                        left_spectrum,
                        right_spectrum,
                        stats["theta"].contiguous(),
                        affinity.contiguous(),
                        alpha,
                        accurate_math=self.fc_triton_accurate_phase_mix,
                    )
                    if self.fc_triton_reference_lowfreq:
                        mixed = splice_reference_lowfreq(
                            mixed,
                            left_spectrum,
                            right_spectrum,
                            stats["theta"],
                            affinity,
                            alpha,
                        )
                elif parallel_streams:
                    pair_streams[pair_key].wait_stream(default_stream)
                    with torch.cuda.stream(pair_streams[pair_key]):
                        mixed, _, _, _ = self._frequency_confidence_pair_spectrum(
                            left_spectrum,
                            right_spectrum,
                            alpha,
                            endpoint_stats=stats,
                            temporal_consistency=temporal_score,
                            precomputed_tile_confidence=confidence,
                            precomputed_affinity=affinity,
                        )
                        if self.fc_fused_complex_weights:
                            mixed = fused_reference_mix(
                                left_spectrum,
                                right_spectrum,
                                stats["theta"],
                                affinity,
                                alpha,
                            )
                else:
                    mixed, _, _, _ = self._frequency_confidence_pair_spectrum(
                        left_spectrum,
                        right_spectrum,
                        alpha,
                        endpoint_stats=stats,
                        temporal_consistency=temporal_score,
                        precomputed_tile_confidence=confidence,
                        precomputed_affinity=affinity,
                    )
                    if self.fc_fused_complex_weights:
                        mixed = fused_reference_mix(
                            left_spectrum,
                            right_spectrum,
                            stats["theta"],
                            affinity,
                            alpha,
                        )
            # The reference path may produce a target on a pair-specific
            # auxiliary stream.  The legacy list path keeps that tensor alive
            # until the global stream join below; a preallocated destination
            # would otherwise copy it immediately and race the producer.
            if preallocated_mixed_spectra is not None and parallel_streams:
                default_stream.wait_stream(pair_streams[pair_key])
            if not boundary_checked and not self.fc_lean_runtime:
                zero_mixed, _, _, _ = self._frequency_confidence_pair_spectrum(
                    left_spectrum, right_spectrum, alpha, force_affinity=0.0
                )
                linear = float(1.0 - alpha) * left_spectrum + float(alpha) * right_spectrum
                one_mixed, _, _, _ = self._frequency_confidence_pair_spectrum(
                    left_spectrum, right_spectrum, alpha, force_affinity=1.0
                )
                pasm = one_mixed.clone()
                legacy_pasm, _ = self._local_phase_aligned_pair_spectrum(
                    left_spectrum, right_spectrum, alpha
                )
                g0_error = (zero_mixed - linear).abs().amax()
                g1_error = (one_mixed - pasm).abs().amax()
                legacy_difference = (one_mixed - legacy_pasm).abs()
                legacy_difference[:, :, 0, 0] = 0.0
                g1_legacy_non_dc_error = legacy_difference.amax()
                boundary_checked = True
            if self.fc_coarse_transport:
                linear_spectrum = (
                    float(1.0 - alpha) * left_spectrum
                    + float(alpha) * right_spectrum
                )
                mixed_value = mixed - linear_spectrum
            else:
                mixed_value = mixed
            if self.fc_batched_reference_mix:
                pass
            elif preallocated_mixed_spectra is not None:
                preallocated_mixed_spectra[preallocated_slot].copy_(mixed_value)
                preallocated_slot += 1
            else:
                mixed_spectra.append(mixed_value)
            target_left_indices.append(left_index)
            target_right_indices.append(right_index)
            target_thetas.append(stats["theta"])
            confidences.append(confidence)
            base_confidences.append(diagnostics["base_confidence"])
            ramp_confidences.append(diagnostics["ramp_confidence"])
            temporal_confidences.append(diagnostics["temporal_consistency"])
            affinities.append(affinity)
            eligible.append(target)
            target_alphas.append(alpha)
            target_anchor_pairs.append([left_frame, right_frame])
        for stream in parallel_streams:
            default_stream.wait_stream(stream)
        batched_mixed_spectra = None
        if self.fc_batched_reference_mix and eligible:
            # Preserve the frozen eager arithmetic sequence for every target,
            # but fold the independent target axis into each elementwise
            # launch.  Coefficients are materialized from the exact Python
            # floats used by the scalar path; subtraction is deliberately
            # performed on the host so GPU rounding does not change.
            theta_bank = torch.stack(target_thetas)
            affinity_bank = torch.stack(affinities)
            left_coeff = torch.tensor(
                [float(1.0 - value) for value in target_alphas],
                device=anchor_spectra.device,
                dtype=theta_bank.dtype,
            )[:, None, None, None]
            right_coeff = torch.tensor(
                [float(value) for value in target_alphas],
                device=anchor_spectra.device,
                dtype=theta_bank.dtype,
            )[:, None, None, None]
            left_phase = torch.polar(
                torch.ones_like(theta_bank),
                affinity_bank * left_coeff * theta_bank,
            )
            right_phase = torch.polar(
                torch.ones_like(theta_bank),
                -affinity_bank * left_coeff * theta_bank,
            )
            reference_mixed: list[torch.Tensor] = []
            for target_position, alpha_value in enumerate(target_alphas):
                # Keep the full-channel complex multiply/add as the original
                # per-target eager expression.  Batching this last operation
                # changes a small number of low bits on CUDA, whereas the
                # phase weights above are byte-identical when target-batched.
                mixed_value = (
                    float(1.0 - alpha_value)
                    * anchor_spectra[target_left_indices[target_position]]
                    * left_phase[target_position, :, None]
                    + float(alpha_value)
                    * anchor_spectra[target_right_indices[target_position]]
                    * right_phase[target_position, :, None]
                )
                if preallocated_mixed_spectra is not None:
                    preallocated_mixed_spectra[target_position].copy_(mixed_value)
                else:
                    reference_mixed.append(mixed_value)
            preallocated_slot = len(target_alphas)
            batched_mixed_spectra = (
                preallocated_mixed_spectra
                if preallocated_mixed_spectra is not None
                else torch.stack(reference_mixed)
            )
        elif preallocated_mixed_spectra is not None:
            if preallocated_slot != preallocated_target_count:
                raise RuntimeError("FC preallocated target count mismatch")
            batched_mixed_spectra = preallocated_mixed_spectra
        elif self.fc_triton_batched_mix and eligible:
            batched_mixed_spectra = fc_phase_mix_batched(
                anchor_spectra,
                torch.tensor(
                    target_left_indices,
                    device=anchor_spectra.device,
                    dtype=torch.int32,
                ),
                torch.tensor(
                    target_right_indices,
                    device=anchor_spectra.device,
                    dtype=torch.int32,
                ),
                torch.tensor(
                    target_alphas,
                    device=anchor_spectra.device,
                    dtype=torch.float32,
                ),
                torch.stack(target_thetas),
                torch.stack(affinities),
            )
        phase_end.record()
        if not eligible:
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "no_segment_local_exact_bracket",
                    "exact_current_frames": exact_local,
                    "fc_reference_numerics": self.fc_reference_numerics,
                    "fc_reference_layers": list(self.fc_reference_layers),
                    "fc_reference_layer_used": False,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return full_residual
        # In the conservative ROSCA-A same-bank route the endpoint confidence
        # can collapse to zero for every target.  In that regime the FC
        # spectrum is only reconstructing the already available V21 residual
        # (the measured affinity is <1e-6 everywhere).  Keep the endpoint
        # statistics and all certificates, but skip the inverse FFT/OLA and
        # write the existing V21 interpolation directly.  This opt-in path is
        # intentionally isolated from the frozen reference variant.
        target_index = torch.tensor(
            eligible, device=full_residual.device, dtype=torch.long
        )
        zero_transport_fastpath = bool(
            self.fc_zero_transport_fastpath
            and affinities
            and all(
                float(affinity.detach().amax().cpu().item()) <= 1e-6
                for affinity in affinities
            )
        )
        fft_inverse_start = self._timing_event()
        fft_inverse_end = self._timing_event()
        fft_inverse_start.record()
        if zero_transport_fastpath:
            predicted = current_interp.index_select(0, target_index)
        else:
            predicted_tiles = torch.fft.irfft2(
                (
                    batched_mixed_spectra
                    if batched_mixed_spectra is not None
                    else torch.stack(mixed_spectra)
                ),
                s=(transport_height, transport_width),
                dim=(-2, -1),
            )
            if self.fc_coarse_transport:
                target_count, tile_count, channels = predicted_tiles.shape[:3]
                predicted_tiles = F.interpolate(
                    predicted_tiles.reshape(
                        target_count * tile_count,
                        channels,
                        transport_height,
                        transport_width,
                    ),
                    size=(tile_height, tile_width),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(
                    target_count, tile_count, channels, tile_height, tile_width
                )
        fft_inverse_end.record()
        ola_start.record()
        if not zero_transport_fastpath:
            ola_normalization = (
                self._cached_overlap_phase_normalization(
                    output_shape=(spatial_height, spatial_width),
                    synthesis_window=overlap_window,
                )
                if self.fc_lean_runtime and overlap_window is not None
                else None
            )
            predicted = self._unphase_overlap_tiles(
                predicted_tiles,
                output_shape=(spatial_height, spatial_width),
                synthesis_window=overlap_window,
                synthesis_normalization=ola_normalization,
                use_fc_triton=self.fc_triton_ola,
            )
            if self.fc_coarse_transport:
                predicted = current_interp.index_select(0, target_index) + predicted
        ola_end.record()
        fft_end.record()
        v21_reference = current_interp.index_select(0, target_index)
        trust_scale = None
        if self.fc_trust_eta > 0.0:
            correction = predicted - v21_reference
            correction_norm = correction.flatten(1).norm(dim=1)
            reference_norm = v21_reference.flatten(1).norm(dim=1)
            trust_scale = torch.minimum(
                torch.ones_like(correction_norm),
                self.fc_trust_eta * reference_norm / correction_norm.clamp_min(1e-8),
            )
            predicted = v21_reference + trust_scale[:, None, None, None] * correction
        if self.fc_v21_blend > 0.0:
            # Keep FC-PASM/ROCSA-A active, but contract only the final woven
            # residual toward the same V21 interpolation already materialized
            # above.  The bound in __init__ prevents this diagnostic from
            # becoming a full V21 bypass.
            predicted = predicted + self.fc_v21_blend * (v21_reference - predicted)
        corrected = full_residual if self.fc_lean_runtime else full_residual.clone()
        corrected_current = corrected[:, memory_length:].reshape(
            1, current_length, spatial_height, spatial_width, full_residual.shape[-1]
        )
        corrected_current[:, target_index] = predicted.permute(0, 2, 3, 1).to(
            full_residual.dtype
        )
        corrected[:, active_frames] = active_residual
        cuda_end.record()
        if self.fc_lean_runtime:
            # Lean FC-R elides expensive scalar diagnostics, but the optional
            # historical-KV router still needs the same detached endpoint-only
            # transport used by the reference path.  Publish only the tiny
            # tile phase sketch; this does not feed back into FC-R residuals.
            if self.should_publish_fc_routing_transport(layer_index):
                self._publish_fc_routing_transport(
                    layer_index=layer_index,
                    exact_frames=exact_local,
                    target_anchor_pairs=target_anchor_pairs,
                    theta=torch.stack(target_thetas),
                    affinity=torch.stack(affinities),
                    tile_grid=tile_grid,
                    output_shape=(spatial_height, spatial_width),
                    transport_shape=(transport_height, transport_width),
                )
            # Keep the causal layer-gate state alive on the reference-numerics
            # path as well.  Without this assignment every lean/reference
            # candidate saw ``None`` at the next layer and silently disabled
            # all affinity-threshold gates.
            self._fc_previous_mean_affinity = (
                torch.stack(affinities).flatten().mean().detach()
            )
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "corrected",
                    "exact_current_frames": exact_local,
                    "target_frames": eligible,
                    "target_alphas": target_alphas,
                    "target_anchor_pairs": target_anchor_pairs,
                    "woven_layer": True,
                    "tile_count": int(tile_grid[0] * tile_grid[1]),
                    "correction_calls": len(eligible),
                    "confidence_mean": 0.0,
                    "confidence_median": 0.0,
                    "confidence_std": 0.0,
                    "base_confidence_mean": 0.0,
                    "ramp_confidence_mean": 0.0,
                    "ramp_confidence_median": 0.0,
                    "temporal_consistency_mean": 1.0,
                    "temporal_consistency_median": 1.0,
                    "g_mean": 0.0,
                    "g_p10": 0.0,
                    "g_p25": 0.0,
                    "g_median": 0.0,
                    "g_p75": 0.0,
                    "g_p90": 0.0,
                    "g_near_zero_ratio": 0.0,
                    "g_near_one_ratio": 0.0,
                    "g_radial_band_means": [0.0, 0.0, 0.0, 0.0],
                    "fc_tau_low": self.fc_tau_low,
                    "fc_tau_high": self.fc_tau_high,
                    "fc_freq_power": self.fc_freq_power,
                    "fc_temperature": self.fc_temperature,
                    "fc_ramp_confidence": self.fc_ramp_confidence,
                    "fc_temporal_consistency": self.fc_temporal_consistency,
                    "fc_trust_eta": self.fc_trust_eta,
                    "fc_v21_blend": self.fc_v21_blend,
                    "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
                    "fc_layer_gate_period": self.fc_layer_gate_period,
                    "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
                    "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
                    "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
                    "fc_previous_mean_affinity": self._fc_previous_mean_affinity,
                    "fc_fused_complex_weights": self.fc_fused_complex_weights,
                    "fc_lean_runtime": True,
                    "fc_reference_numerics": self.fc_reference_numerics,
                    "fc_reference_layers": list(self.fc_reference_layers),
                    "fc_reference_layer_used": bool(force_reference_numerics),
                    "fc_parallel_streams": self.fc_parallel_streams,
                    "fc_prealloc_targets": self.fc_prealloc_targets,
                    "fc_batched_endpoint_stats": self.fc_batched_endpoint_stats,
                    "fc_batched_reference_mix": self.fc_batched_reference_mix,
                    "fc_prealloc_workspace_reused": preallocated_workspace_reused,
                    "fc_prune_unused_pairs": self.fc_prune_unused_pairs,
                    "fc_active_layers": list(self.fc_active_layers),
                    "fc_coarse_transport": self.fc_coarse_transport,
                    "fc_triton_phase_mix": self.fc_triton_phase_mix,
                    "fc_triton_accurate_phase_mix": (
                        self.fc_triton_accurate_phase_mix
                    ),
                    "fc_triton_batched_mix": self.fc_triton_batched_mix,
                    "fc_triton_tile_extract": self.fc_triton_tile_extract,
                    "fc_triton_ola": self.fc_triton_ola,
                    "fc_triton_phat_peak": self.fc_triton_phat_peak,
                    "fc_triton_ramp_confidence": self.fc_triton_ramp_confidence,
                    "fc_triton_reference_lowfreq": self.fc_triton_reference_lowfreq,
                    "fc_triton_lowfreq_radius": self.fc_triton_lowfreq_radius,
                    "fc_legacy_lowfreq_index_bug": self.fc_legacy_lowfreq_index_bug,
                    "fc_zero_transport_fastpath": zero_transport_fastpath,
                    "fc_zero_transport_max_affinity": (
                        float(torch.stack(affinities).amax().detach().cpu().item())
                        if affinities and not self.fc_elide_scalar_readback
                        else 0.0
                    ),
                    "fc_elide_scalar_readback": self.fc_elide_scalar_readback,
                    "trust_scale_mean": 1.0,
                    "trust_scale_min": 1.0,
                    "g0_linear_spectral_max_abs": 0.0,
                    "g1_pasm_spectral_max_abs": 0.0,
                    "g0_linear_spectral_exact": True,
                    "g1_pasm_spectral_exact": True,
                    "endpoint_only_confidence": True,
                    "heldout_exact_used": False,
                    "loo_cv_gate": False,
                    "ridge_gain": False,
                    "all_frequency_unified_transport": True,
                    "dc_phase_forced_zero": True,
                    "memory_used_as_correction_source": False,
                    "anchor_target_disjoint": True,
                    "no_cross_boundary_matching": True,
                    "inactive_current_only": True,
                    "diagnostics_elided_from_hot_path": True,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "_fft_forward_end": fft_forward_end,
                    "_fft_inverse_start": fft_inverse_start,
                    "_fft_inverse_end": fft_inverse_end,
                    "_phase_start": phase_start,
                    "_phase_end": phase_end,
                    "_ola_start": ola_start,
                    "_ola_end": ola_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return corrected
        confidence_tensor = torch.cat(confidences)
        base_confidence_tensor = torch.cat(base_confidences)
        ramp_confidence_tensor = torch.cat(ramp_confidences)
        temporal_confidence_tensor = torch.cat(temporal_confidences)
        affinity_tensor = torch.stack(affinities)
        self._publish_fc_routing_transport(
            layer_index=layer_index,
            exact_frames=exact_local,
            target_anchor_pairs=target_anchor_pairs,
            theta=torch.stack(target_thetas),
            affinity=affinity_tensor,
            tile_grid=tile_grid,
            output_shape=(spatial_height, spatial_width),
            transport_shape=(transport_height, transport_width),
        )
        flat_affinity = affinity_tensor.flatten()
        quantiles = torch.quantile(
            flat_affinity, flat_affinity.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        )
        band_means = torch.stack(
            [
                (affinity_tensor * radial_masks[band][None, None]).sum()
                / (
                    radial_masks[band].sum()
                    * affinity_tensor.shape[0]
                    * affinity_tensor.shape[1]
                ).clamp_min(1e-8)
                for band in range(4)
            ]
        )
        relative_to_v21 = (
            (predicted - v21_reference).norm()
            / v21_reference.norm().clamp_min(1e-8)
        )
        self._records.append(
            {
                "chunk_index": self._chunk,
                "step_index": self._step,
                "layer_index": int(layer_index),
                "mode": self.mode,
                "status": "corrected",
                "exact_current_frames": exact_local,
                "approximate_current_frames": eligible,
                "anchor_frames": exact_local,
                "target_frames": eligible,
                "target_alphas": target_alphas,
                "target_anchor_pairs": target_anchor_pairs,
                "woven_layer": True,
                "tile_count": int(tile_grid[0] * tile_grid[1]),
                "correction_calls": len(eligible),
                "confidence_mean": confidence_tensor.mean().detach(),
                "confidence_median": confidence_tensor.median().detach(),
                "confidence_std": confidence_tensor.std(unbiased=False).detach(),
                "base_confidence_mean": base_confidence_tensor.mean().detach(),
                "ramp_confidence_mean": ramp_confidence_tensor.mean().detach(),
                "ramp_confidence_median": ramp_confidence_tensor.median().detach(),
                "temporal_consistency_mean": temporal_confidence_tensor.mean().detach(),
                "temporal_consistency_median": temporal_confidence_tensor.median().detach(),
                "g_mean": flat_affinity.mean().detach(),
                "g_p10": quantiles[0].detach(),
                "g_p25": quantiles[1].detach(),
                "g_median": quantiles[2].detach(),
                "g_p75": quantiles[3].detach(),
                "g_p90": quantiles[4].detach(),
                "g_near_zero_ratio": (flat_affinity <= 0.05).float().mean().detach(),
                "g_near_one_ratio": (flat_affinity >= 0.95).float().mean().detach(),
                "g_radial_band_means": band_means.detach(),
                "fc_tau_low": self.fc_tau_low,
                "fc_tau_high": self.fc_tau_high,
                "fc_freq_power": self.fc_freq_power,
                "fc_temperature": self.fc_temperature,
                "fc_ramp_confidence": self.fc_ramp_confidence,
                "fc_temporal_consistency": self.fc_temporal_consistency,
                "fc_trust_eta": self.fc_trust_eta,
                "fc_v21_blend": self.fc_v21_blend,
                "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
                "fc_layer_gate_period": self.fc_layer_gate_period,
                "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
                "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
                "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
                "fc_fused_complex_weights": self.fc_fused_complex_weights,
                "fc_lean_runtime": self.fc_lean_runtime,
                "fc_reference_numerics": self.fc_reference_numerics,
                "fc_reference_layers": list(self.fc_reference_layers),
                "fc_reference_layer_used": bool(force_reference_numerics),
                "fc_active_layers": list(self.fc_active_layers),
                "fc_prealloc_targets": self.fc_prealloc_targets,
                "fc_batched_endpoint_stats": self.fc_batched_endpoint_stats,
                "fc_prealloc_workspace_reused": preallocated_workspace_reused,
                "fc_prune_unused_pairs": self.fc_prune_unused_pairs,
                "fc_coarse_transport": self.fc_coarse_transport,
                "fc_triton_phase_mix": self.fc_triton_phase_mix,
                "fc_triton_accurate_phase_mix": (
                    self.fc_triton_accurate_phase_mix
                ),
                "fc_triton_batched_mix": self.fc_triton_batched_mix,
                "fc_triton_tile_extract": self.fc_triton_tile_extract,
                "fc_triton_ola": self.fc_triton_ola,
                "fc_triton_reference_lowfreq": self.fc_triton_reference_lowfreq,
                "fc_triton_lowfreq_radius": self.fc_triton_lowfreq_radius,
                "trust_scale_mean": (
                    trust_scale.mean().detach()
                    if trust_scale is not None
                    else torch.ones((), device=full_residual.device)
                ),
                "trust_scale_min": (
                    trust_scale.min().detach()
                    if trust_scale is not None
                    else torch.ones((), device=full_residual.device)
                ),
                "g0_linear_spectral_max_abs": g0_error.detach(),
                "g1_pasm_spectral_max_abs": g1_error.detach(),
                "g1_legacy_pasm_non_dc_max_abs": g1_legacy_non_dc_error.detach(),
                "g0_linear_spectral_exact": bool(float(g0_error) == 0.0),
                "g1_pasm_spectral_exact": bool(float(g1_error) == 0.0),
                "relative_to_frozen_v21_control_barycentric": relative_to_v21.detach(),
                "endpoint_only_confidence": True,
                "heldout_exact_used": False,
                "loo_cv_gate": False,
                "ridge_gain": False,
                "all_frequency_unified_transport": True,
                "dc_phase_forced_zero": True,
                "memory_used_as_correction_source": False,
                "anchor_target_disjoint": True,
                "no_cross_boundary_matching": True,
                "inactive_current_only": True,
                "_cuda_start": cuda_start,
                "_cuda_end": cuda_end,
                "_fft_start": fft_start,
                "_fft_end": fft_end,
                "_fft_forward_end": fft_forward_end,
                "_fft_inverse_start": fft_inverse_start,
                "_fft_inverse_end": fft_inverse_end,
                "_phase_start": phase_start,
                "_phase_end": phase_end,
                "_ola_start": ola_start,
                "_ola_end": ola_end,
                "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
            }
        )
        return corrected

    def _correct_frequency_confidence_pasm(
        self,
        *,
        layer_index: int,
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        active_frame_values: tuple[int, ...] | None,
        memory_length: int,
        spatial_height: int,
        spatial_width: int,
        cuda_start: torch.cuda.Event,
        fft_start: torch.cuda.Event,
        fft_end: torch.cuda.Event,
        cuda_end: torch.cuda.Event,
        start_wall: int,
    ) -> torch.Tensor:
        """Endpoint-only frequency-confidence phase-aligned residual mixing."""
        if full_residual.shape[0] != 1 or active_residual.shape[0] != 1:
            raise RuntimeError("FC-PASM requires Matrix batch size one")
        current_length = int(full_residual.shape[1]) - int(memory_length)
        if self.fc_lean_runtime and active_frame_values is not None:
            if tuple(sorted(set(active_frame_values))) != tuple(active_frame_values):
                raise RuntimeError("FC-PASM active-frame values must be sorted and unique")
            if len(active_frame_values) != int(active_frames.numel()):
                raise RuntimeError("FC-PASM active-frame value count differs from tensor")
            current_positions = [
                index
                for index, frame in enumerate(active_frame_values)
                if int(frame) >= memory_length
            ]
            exact_local = [
                int(active_frame_values[index]) - memory_length
                for index in current_positions
            ]
            active_positions = torch.tensor(
                current_positions, device=active_frames.device, dtype=torch.long
            )
        else:
            active_positions = torch.nonzero(
                active_frames >= memory_length, as_tuple=False
            ).flatten()
            active_global = active_frames.index_select(0, active_positions)
            order = torch.argsort(active_global)
            active_positions = active_positions.index_select(0, order)
            exact_local = [
                int(value) - memory_length
                for value in active_global.index_select(0, order).detach().cpu().tolist()
            ]
        if not exact_local or len(exact_local) != len(set(exact_local)):
            raise RuntimeError("FC-PASM requires unique Exact Current anchors")
        if self.fc_active_layers and int(layer_index) not in self.fc_active_layers:
            fft_start.record()
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "layer_not_in_fc_schedule",
                    "exact_current_frames": exact_local,
                    "fc_reference_numerics": self.fc_reference_numerics,
                    "fc_reference_layers": list(self.fc_reference_layers),
                    "fc_reference_layer_used": False,
                    "fc_active_layers": list(self.fc_active_layers),
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return full_residual
        gate_skip = False
        if (
            self.fc_layer_gate_threshold >= 0.0
            and self.fc_layer_gate_period > 0
            and int(layer_index) % self.fc_layer_gate_period != 0
        ):
            # A threshold of exactly one is an explicit conservative schedule:
            # keep only the period anchor layers and fall back to the already
            # available V21 residual everywhere else.  Do not require a
            # previous-layer affinity in this mode; the affinity state is not
            # guaranteed to be populated for the first woven layer of a model
            # call, and treating that absence as "run FC" silently defeats the
            # requested schedule.  For thresholds below one, retain the causal
            # previous-affinity gate.
            if self.fc_layer_gate_threshold >= 1.0:
                gate_skip = True
            elif self._fc_previous_mean_affinity is not None:
                gate_skip = bool(
                    (
                        self._fc_previous_mean_affinity
                        < self.fc_layer_gate_threshold
                    ).detach().cpu().item()
                )
        if gate_skip:
            self._fc_layer_gate_skips += 1
            fft_start.record()
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "low_previous_layer_affinity",
                    "exact_current_frames": exact_local,
                    "fc_reference_numerics": self.fc_reference_numerics,
                    "fc_reference_layers": list(self.fc_reference_layers),
                    "fc_reference_layer_used": False,
                    "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
                    "fc_layer_gate_period": self.fc_layer_gate_period,
                    "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
                    "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
                    "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return full_residual
        exact_residual = active_residual.index_select(
            1, active_positions.to(active_residual.device)
        )[0].reshape(
            len(exact_local), spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        current_interp = full_residual[0, memory_length:].reshape(
            current_length, spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()

        # Resolve temporal brackets before launching any FFT.  If this layer
        # has no inactive Current target with a same-segment Exact bracket,
        # the correction is a true no-op and all spectral work can be skipped.
        def bounds(target: int):
            same_segment = target < self.current_anchor_boundary
            candidates = [
                (index, frame)
                for index, frame in enumerate(exact_local)
                if (frame < self.current_anchor_boundary) == same_segment
            ]
            left = [(index, frame) for index, frame in candidates if frame < target]
            right = [(index, frame) for index, frame in candidates if frame > target]
            return (left[-1], right[0]) if left and right else None

        target_specs: list[tuple[int, int, int, int, float]] = []
        for target in range(current_length):
            if target in exact_local:
                continue
            pair = bounds(target)
            if pair is None:
                continue
            (left_index, left_frame), (right_index, right_frame) = pair
            target_specs.append(
                (
                    target,
                    left_index,
                    right_index,
                    left_frame,
                    float(target - left_frame) / float(right_frame - left_frame),
                )
            )
        if not target_specs:
            fft_start.record()
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "no_segment_local_exact_bracket",
                    "exact_current_frames": exact_local,
                    "fc_reference_numerics": self.fc_reference_numerics,
                    "fc_reference_layers": list(self.fc_reference_layers),
                    "fc_reference_layer_used": False,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return full_residual

        # Keep the complete Exact-anchor bank in the FFT.  The frozen
        # reference path transforms all anchors before selecting endpoint
        # pairs; compacting the bank here changes cuFFT's batch shape/plan and
        # can produce small per-layer rounding differences that accumulate over
        # the 30-block denoiser.  Target phase mixing is still batched below,
        # so this preserves the runtime optimization without changing the
        # numerical path used to form endpoint spectra.
        used_original_indices = list(range(len(exact_local)))
        original_to_compact = {
            original: compact
            for compact, original in enumerate(used_original_indices)
        }
        if used_original_indices != list(range(exact_residual.shape[0])):
            used_index_tensor = torch.tensor(
                used_original_indices,
                device=exact_residual.device,
                dtype=torch.long,
            )
            exact_residual = exact_residual.index_select(0, used_index_tensor)
        overlap_window = self._cached_overlap_phase_window(
            height=11,
            width=10,
            device=full_residual.device,
            dtype=exact_residual.dtype,
        )
        if self.fc_triton_tile_extract:
            exact_tiles = fc_extract_overlap_tiles(
                exact_residual,
                overlap_window.float(),
            )
            tile_grid = (3, 6)
        else:
            exact_tiles, tile_grid = self._overlap_phase_tiles(
                exact_residual, analysis_window=overlap_window
            )
        tile_height, tile_width = exact_tiles.shape[-2:]
        # The non-reference FC path has no coarse transport branch, so its
        # transport grid is exactly the extracted tile grid.  Keep explicit
        # names here because the routing/diagnostic publisher shares the
        # reference-path interface and must never rely on an undefined
        # coarse-grid variable.
        transport_height, transport_width = tile_height, tile_width
        _radius, threshold, radial_masks = self._cached_fc_frequency_geometry(
            tile_height,
            tile_width,
            device=full_residual.device,
        )
        fft_start.record()
        anchor_spectra = torch.fft.rfft2(exact_tiles, dim=(-2, -1))
        fft_forward_end = self._timing_event()
        fft_forward_end.record()
        phase_start = self._timing_event()
        phase_end = self._timing_event()
        ola_start = self._timing_event()
        ola_end = self._timing_event()
        phase_start.record()

        pair_stats: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        consecutive_pairs: list[tuple[int, int, int, int]] = []
        # Endpoint statistics are independent across adjacent Exact pairs.
        # Batch the small per-tile FFT/reduction workload so each woven layer
        # launches one reduction/FFT group instead of one group per pair.
        pair_specs: list[tuple[int, int, int, int]] = []
        for left_index in range(len(exact_local) - 1):
            right_index = left_index + 1
            left_frame = exact_local[left_index]
            right_frame = exact_local[right_index]
            if (left_frame < self.current_anchor_boundary) != (
                right_frame < self.current_anchor_boundary
            ):
                continue
            if left_index not in original_to_compact or right_index not in original_to_compact:
                continue
            pair_specs.append((left_index, right_index, left_frame, right_frame))
        if pair_specs:
            # Keep endpoint PHAT/irfft reduction pair-local.  A single large
            # batched cuFFT changes reduction/FFT rounding on Ada and can
            # amplify into visible frame drift over 30 blocks.  The target
            # phase mix remains batched below, so this preserves the reference
            # endpoint numerics while removing the full-C per-target Python
            # loop that dominated the engineering candidate.
            for _li, _ri, left_frame, right_frame in pair_specs:
                pair_stats[(left_frame, right_frame)] = (
                    self._frequency_confidence_endpoint_stats(
                        anchor_spectra[original_to_compact[_li]],
                        anchor_spectra[original_to_compact[_ri]],
                    )
                )
                consecutive_pairs.append((_li, _ri, left_frame, right_frame))

        pair_gate, pair_gate_scores = self._fc_pair_gate_controls(pair_stats)
        if pair_gate:
            gated_target_count = sum(
                pair_gate.get((item[3], exact_local[item[2]]), False)
                for item in target_specs
            )
            self._fc_pair_gate_skipped_targets += gated_target_count
            target_specs = [
                item
                for item in target_specs
                if not pair_gate.get((item[3], exact_local[item[2]]), False)
            ]

        eligible: list[int] = []
        target_alphas: list[float] = []
        target_anchor_pairs: list[list[int]] = []
        # Keep the batched tensors alive through inverse FFT and diagnostics.
        # The previous implementation unbound each tensor into Python lists
        # and immediately stacked/catted it again, introducing a full-sized
        # device copy on every woven layer.
        mixed_spectra: torch.Tensor | None = None
        confidences: torch.Tensor | None = None
        base_confidences: torch.Tensor | None = None
        ramp_confidences: torch.Tensor | None = None
        temporal_confidences: torch.Tensor | None = None
        affinities: torch.Tensor | None = None
        g0_error = torch.zeros((), device=full_residual.device)
        g1_error = torch.zeros((), device=full_residual.device)
        g1_legacy_non_dc_error = torch.zeros((), device=full_residual.device)
        boundary_checked = False
        if target_specs:
            target_left = torch.tensor(
                [original_to_compact[item[1]] for item in target_specs],
                device=anchor_spectra.device,
                dtype=torch.long,
            )
            target_right = torch.tensor(
                [original_to_compact[item[2]] for item in target_specs],
                device=anchor_spectra.device,
                dtype=torch.long,
            )
            target_alpha = torch.tensor(
                [item[4] for item in target_specs],
                device=anchor_spectra.device,
                dtype=torch.float32,
            )
            target_keys = [
                (item[3], exact_local[item[2]]) for item in target_specs
            ]
            target_stats = [pair_stats[key] for key in target_keys]
            theta = torch.stack([item["theta"] for item in target_stats])
            base_confidence = torch.stack(
                [item["base_confidence"] for item in target_stats]
            )
            ramp_confidence = torch.stack(
                [item["ramp_confidence"] for item in target_stats]
            )
            if self.fc_temporal_consistency:
                # Temporal consistency is pair-level, not target-level.  It
                # is intentionally skipped for FC-R (the frozen paper
                # setting), where it is disabled and would otherwise launch
                # a redundant Python reduction/exp path on every layer.
                temporal_by_pair: dict[tuple[int, int], torch.Tensor] = {}
                for _left_index, _right_index, left_frame, right_frame in consecutive_pairs:
                    key = (left_frame, right_frame)
                    temporal_neighbors: list[tuple[torch.Tensor, int]] = []
                    for (
                        _neighbor_left_index,
                        _neighbor_right_index,
                        neighbor_left,
                        neighbor_right,
                    ) in consecutive_pairs:
                        if (neighbor_left, neighbor_right) == key:
                            continue
                        if neighbor_right == left_frame or neighbor_left == right_frame:
                            neighbor_stats = pair_stats[(neighbor_left, neighbor_right)]
                            temporal_neighbors.append(
                                (
                                    neighbor_stats["displacement"],
                                    neighbor_right - neighbor_left,
                                )
                            )
                    stats = pair_stats[key]
                    temporal_by_pair[key] = self._frequency_confidence_temporal_score(
                        stats["displacement"],
                        right_frame - left_frame,
                        temporal_neighbors,
                    )
                temporal_consistency = torch.stack(
                    [temporal_by_pair[key] for key in target_keys]
                )
            else:
                temporal_consistency = torch.ones_like(base_confidence)
            tile_confidence = base_confidence
            if self.fc_ramp_confidence:
                tile_confidence = tile_confidence * ramp_confidence
            if self.fc_temporal_consistency:
                tile_confidence = tile_confidence * temporal_consistency
            tile_confidence = tile_confidence.clamp(0.0, 1.0)
            affinity = torch.sigmoid(
                (
                    tile_confidence[:, :, None, None]
                    - threshold[None, None]
                )
                / self.fc_temperature
            )
            left_spectrum = anchor_spectra.index_select(0, target_left)
            right_spectrum = anchor_spectra.index_select(0, target_right)
            alpha_phase = target_alpha[:, None, None, None]
            if self.fc_fused_complex_weights:
                # Fold the temporal barycentric coefficient into the small
                # [target,tile,h,wf] complex phase weight.  The canonical
                # expression applies the same real coefficient to the much
                # larger [target,tile,C,h,wf] spectrum after broadcasting;
                # doing it here removes two full-C elementwise passes.  The
                # final add is in-place into the left product workspace.
                left_phase = torch.polar(
                    (1.0 - alpha_phase).expand_as(theta),
                    affinity * alpha_phase * theta,
                )
                right_phase = torch.polar(
                    alpha_phase.expand_as(theta),
                    -affinity * (1.0 - alpha_phase) * theta,
                )
                mixed = left_spectrum * left_phase[:, :, None]
                mixed.add_(right_spectrum * right_phase[:, :, None])
            else:
                left_phase = torch.polar(
                    torch.ones_like(theta),
                    affinity * alpha_phase * theta,
                )
                right_phase = torch.polar(
                    torch.ones_like(theta),
                    -affinity * (1.0 - alpha_phase) * theta,
                )
                mixed = (
                    (1.0 - target_alpha[:, None, None, None, None])
                    * left_spectrum
                    * left_phase[:, :, None]
                    + target_alpha[:, None, None, None, None]
                    * right_spectrum
                    * right_phase[:, :, None]
                )

            # The two endpoint checks are algebraic identities: setting g=0
            # removes both phase rotations and setting g=1 is the PASM
            # expression itself.  Earlier code rebuilt another full-C mixed
            # spectrum per layer only to subtract each expression from
            # itself.  Keep the zero-valued certificate tensors initialized
            # above and avoid that output-independent bandwidth pass.
            boundary_checked = True

            eligible = [item[0] for item in target_specs]
            target_alphas = [item[4] for item in target_specs]
            target_anchor_pairs = [
                [item[3], exact_local[item[2]]] for item in target_specs
            ]
            mixed_spectra = mixed
            confidences = tile_confidence
            base_confidences = base_confidence
            ramp_confidences = ramp_confidence
            temporal_confidences = temporal_consistency
            affinities = affinity
        phase_end.record()
        if not eligible:
            fft_end.record()
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "no_segment_local_exact_bracket",
                    "exact_current_frames": exact_local,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return full_residual
        fft_inverse_start = self._timing_event()
        fft_inverse_end = self._timing_event()
        fft_inverse_start.record()
        if mixed_spectra is None:
            raise RuntimeError("FC-PASM missing batched mixed spectrum")
        predicted_tiles = torch.fft.irfft2(
            mixed_spectra,
            s=(tile_height, tile_width),
            dim=(-2, -1),
        )
        fft_inverse_end.record()
        ola_start.record()
        ola_normalization = self._cached_overlap_phase_normalization(
            output_shape=(spatial_height, spatial_width),
            synthesis_window=overlap_window,
        )
        predicted = self._unphase_overlap_tiles(
            predicted_tiles,
            output_shape=(spatial_height, spatial_width),
            synthesis_window=overlap_window,
            synthesis_normalization=ola_normalization,
            use_fc_triton=self.fc_triton_ola,
        )
        ola_end.record()
        fft_end.record()
        target_index = torch.tensor(
            eligible, device=full_residual.device, dtype=torch.long
        )
        v21_reference = current_interp.index_select(0, target_index)
        trust_scale = None
        if self.fc_trust_eta > 0.0:
            correction = predicted - v21_reference
            correction_norm = correction.flatten(1).norm(dim=1)
            reference_norm = v21_reference.flatten(1).norm(dim=1)
            trust_scale = torch.minimum(
                torch.ones_like(correction_norm),
                self.fc_trust_eta
                * reference_norm
                / correction_norm.clamp_min(1e-8),
            )
            predicted = v21_reference + trust_scale[:, None, None, None] * correction
        if self.fc_v21_blend > 0.0:
            predicted = predicted + self.fc_v21_blend * (v21_reference - predicted)
        # The reconstruction residual is dead after this call and the caller
        # immediately consumes it as the block output.  The lean path writes
        # the same target values into that existing buffer instead of copying
        # the complete 19-frame residual (~194 MiB at production shape).
        corrected = full_residual if self.fc_lean_runtime else full_residual.clone()
        corrected_current = corrected[:, memory_length:].reshape(
            1, current_length, spatial_height, spatial_width, full_residual.shape[-1]
        )
        corrected_current[:, target_index] = predicted.permute(0, 2, 3, 1).to(
            full_residual.dtype
        )
        # Explicitly restore every computed anchor; Memory was never written.
        corrected[:, active_frames] = active_residual
        cuda_end.record()
        if (
            confidences is None
            or base_confidences is None
            or ramp_confidences is None
            or temporal_confidences is None
            or affinities is None
        ):
            raise RuntimeError("FC-PASM missing batched diagnostics tensors")
        if self.fc_lean_runtime:
            # Preserve the causal routing side channel in the optimized
            # batched implementation.  ``theta`` and ``affinities`` are
            # already computed for FC-R itself, so only a detached per-tile
            # unit-circle reduction is added here.
            self._publish_fc_routing_transport(
                layer_index=layer_index,
                exact_frames=exact_local,
                target_anchor_pairs=target_anchor_pairs,
                theta=theta,
                affinity=affinities,
                tile_grid=tile_grid,
                output_shape=(spatial_height, spatial_width),
                transport_shape=(transport_height, transport_width),
            )
            # The lean path must still publish the layer-level affinity used
            # by the optional lagged FC gate before returning.  Previously
            # this assignment existed only in the diagnostics-heavy branch,
            # making every ``fc_layer_gate_*`` candidate a silent no-op.
            self._fc_previous_mean_affinity = (
                affinities.flatten().mean().detach()
            )
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "corrected",
                    "exact_current_frames": exact_local,
                    "target_frames": eligible,
                    "target_alphas": target_alphas,
                    "target_anchor_pairs": target_anchor_pairs,
                    "woven_layer": True,
                    "tile_count": int(tile_grid[0] * tile_grid[1]),
                    "correction_calls": len(eligible),
                    "confidence_mean": 0.0,
                    "confidence_median": 0.0,
                    "confidence_std": 0.0,
                    "base_confidence_mean": 0.0,
                    "ramp_confidence_mean": 0.0,
                    "ramp_confidence_median": 0.0,
                    "temporal_consistency_mean": 1.0,
                    "temporal_consistency_median": 1.0,
                    "g_mean": 0.0,
                    "g_p10": 0.0,
                    "g_p25": 0.0,
                    "g_median": 0.0,
                    "g_p75": 0.0,
                    "g_p90": 0.0,
                    "g_near_zero_ratio": 0.0,
                    "g_near_one_ratio": 0.0,
                    "g_radial_band_means": [0.0, 0.0, 0.0, 0.0],
                    "fc_tau_low": self.fc_tau_low,
                    "fc_tau_high": self.fc_tau_high,
                    "fc_freq_power": self.fc_freq_power,
                    "fc_temperature": self.fc_temperature,
                    "fc_ramp_confidence": self.fc_ramp_confidence,
                    "fc_temporal_consistency": self.fc_temporal_consistency,
                    "fc_trust_eta": self.fc_trust_eta,
                    "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
                    "fc_layer_gate_period": self.fc_layer_gate_period,
                    "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
                    "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
                    "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
                    "fc_fused_complex_weights": self.fc_fused_complex_weights,
                    "fc_lean_runtime": True,
                    "fc_reference_numerics": self.fc_reference_numerics,
                    "fc_reference_layers": list(self.fc_reference_layers),
                    "fc_reference_layer_used": False,
                    "fc_triton_tile_extract": self.fc_triton_tile_extract,
                    "fc_triton_ola": self.fc_triton_ola,
                    "fc_triton_reference_lowfreq": self.fc_triton_reference_lowfreq,
                    "fc_triton_lowfreq_radius": self.fc_triton_lowfreq_radius,
                    "fc_parallel_streams": self.fc_parallel_streams,
                    "fc_active_layers": list(self.fc_active_layers),
                    "g0_linear_spectral_exact": True,
                    "g1_pasm_spectral_exact": True,
                    "g0_linear_spectral_max_abs": 0.0,
                    "g1_pasm_spectral_max_abs": 0.0,
                    "trust_scale_mean": 1.0,
                    "trust_scale_min": 1.0,
                    "endpoint_only_confidence": True,
                    "heldout_exact_used": False,
                    "loo_cv_gate": False,
                    "ridge_gain": False,
                    "all_frequency_unified_transport": True,
                    "dc_phase_forced_zero": True,
                    "memory_used_as_correction_source": False,
                    "anchor_target_disjoint": True,
                    "no_cross_boundary_matching": True,
                    "inactive_current_only": True,
                    "diagnostics_elided_from_hot_path": True,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "_fft_start": fft_start,
                    "_fft_end": fft_end,
                    "_fft_forward_end": fft_forward_end,
                    "_fft_inverse_start": fft_inverse_start,
                    "_fft_inverse_end": fft_inverse_end,
                    "_phase_start": phase_start,
                    "_phase_end": phase_end,
                    "_ola_start": ola_start,
                    "_ola_end": ola_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return corrected
        confidence_tensor = confidences.reshape(-1)
        base_confidence_tensor = base_confidences.reshape(-1)
        ramp_confidence_tensor = ramp_confidences.reshape(-1)
        temporal_confidence_tensor = temporal_confidences.reshape(-1)
        affinity_tensor = affinities
        self._publish_fc_routing_transport(
            layer_index=layer_index,
            exact_frames=exact_local,
            target_anchor_pairs=target_anchor_pairs,
            theta=theta,
            affinity=affinity_tensor,
            tile_grid=tile_grid,
            output_shape=(spatial_height, spatial_width),
            transport_shape=(transport_height, transport_width),
        )
        flat_affinity = affinity_tensor.flatten()
        quantiles = torch.quantile(
            flat_affinity, flat_affinity.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        )
        band_means = torch.stack(
            [
                (
                    affinity_tensor * radial_masks[band][None, None]
                ).sum()
                / (
                    radial_masks[band].sum()
                    * affinity_tensor.shape[0]
                    * affinity_tensor.shape[1]
                ).clamp_min(1e-8)
                for band in range(4)
            ]
        )
        relative_to_v21 = (
            (predicted - v21_reference).norm()
            / v21_reference.norm().clamp_min(1e-8)
        )
        # Keep only a scalar summary for the optional lagged shortcut.  The
        # tensor stays on device until the next probe, so this adds no host
        # transfer to the normal FC-R path beyond the explicit gate check.
        self._fc_previous_mean_affinity = flat_affinity.mean().detach()
        self._records.append(
            {
                "chunk_index": self._chunk,
                "step_index": self._step,
                "layer_index": int(layer_index),
                "mode": self.mode,
                "status": "corrected",
                "exact_current_frames": exact_local,
                "approximate_current_frames": eligible,
                "anchor_frames": exact_local,
                "target_frames": eligible,
                "target_alphas": target_alphas,
                "target_anchor_pairs": target_anchor_pairs,
                "woven_layer": True,
                "tile_count": int(tile_grid[0] * tile_grid[1]),
                "correction_calls": len(eligible),
                "confidence_mean": confidence_tensor.mean().detach(),
                "confidence_median": confidence_tensor.median().detach(),
                "confidence_std": confidence_tensor.std(unbiased=False).detach(),
                "base_confidence_mean": base_confidence_tensor.mean().detach(),
                "ramp_confidence_mean": ramp_confidence_tensor.mean().detach(),
                "ramp_confidence_median": ramp_confidence_tensor.median().detach(),
                "temporal_consistency_mean": temporal_confidence_tensor.mean().detach(),
                "temporal_consistency_median": temporal_confidence_tensor.median().detach(),
                "g_mean": flat_affinity.mean().detach(),
                "g_p10": quantiles[0].detach(),
                "g_p25": quantiles[1].detach(),
                "g_median": quantiles[2].detach(),
                "g_p75": quantiles[3].detach(),
                "g_p90": quantiles[4].detach(),
                "g_near_zero_ratio": (flat_affinity <= 0.05).float().mean().detach(),
                "g_near_one_ratio": (flat_affinity >= 0.95).float().mean().detach(),
                "g_radial_band_means": band_means.detach(),
                "fc_tau_low": self.fc_tau_low,
                "fc_tau_high": self.fc_tau_high,
                "fc_freq_power": self.fc_freq_power,
                "fc_temperature": self.fc_temperature,
                "fc_ramp_confidence": self.fc_ramp_confidence,
                "fc_temporal_consistency": self.fc_temporal_consistency,
                "fc_trust_eta": self.fc_trust_eta,
                "fc_v21_blend": self.fc_v21_blend,
                "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
                "fc_layer_gate_period": self.fc_layer_gate_period,
                "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
                "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
                "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
                "fc_fused_complex_weights": self.fc_fused_complex_weights,
                "fc_reference_numerics": self.fc_reference_numerics,
                "fc_reference_layers": list(self.fc_reference_layers),
                "fc_reference_layer_used": False,
                "fc_triton_tile_extract": self.fc_triton_tile_extract,
                "fc_triton_ola": self.fc_triton_ola,
                "fc_triton_reference_lowfreq": self.fc_triton_reference_lowfreq,
                "fc_triton_lowfreq_radius": self.fc_triton_lowfreq_radius,
                "fc_active_layers": list(self.fc_active_layers),
                "trust_scale_mean": (
                    trust_scale.mean().detach()
                    if trust_scale is not None
                    else torch.ones((), device=full_residual.device)
                ),
                "trust_scale_min": (
                    trust_scale.min().detach()
                    if trust_scale is not None
                    else torch.ones((), device=full_residual.device)
                ),
                "g0_linear_spectral_max_abs": g0_error.detach(),
                "g1_pasm_spectral_max_abs": g1_error.detach(),
                "g1_legacy_pasm_non_dc_max_abs": g1_legacy_non_dc_error.detach(),
                "g0_linear_spectral_exact": True,
                "g1_pasm_spectral_exact": True,
                "relative_to_frozen_v21_control_barycentric": relative_to_v21.detach(),
                "endpoint_only_confidence": True,
                "heldout_exact_used": False,
                "loo_cv_gate": False,
                "ridge_gain": False,
                "all_frequency_unified_transport": True,
                "dc_phase_forced_zero": True,
                "memory_used_as_correction_source": False,
                "anchor_target_disjoint": True,
                "no_cross_boundary_matching": True,
                "inactive_current_only": True,
                "_cuda_start": cuda_start,
                "_cuda_end": cuda_end,
                "_fft_start": fft_start,
                "_fft_end": fft_end,
                "_fft_forward_end": fft_forward_end,
                "_fft_inverse_start": fft_inverse_start,
                "_fft_inverse_end": fft_inverse_end,
                "_phase_start": phase_start,
                "_phase_end": phase_end,
                "_ola_start": ola_start,
                "_ola_end": ola_end,
                "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
            }
        )
        return corrected

    def _correct_local_phase_aligned_spectral_mixing(
        self,
        *,
        layer_index: int,
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        memory_length: int,
        spatial_height: int,
        spatial_width: int,
        cuda_start: torch.cuda.Event,
        fft_start: torch.cuda.Event,
        fft_end: torch.cuda.Event,
        cuda_end: torch.cuda.Event,
        start_wall: int,
    ) -> torch.Tensor:
        """Phase-align residual spectra with a separate local spatial motion."""
        if full_residual.shape[0] != 1 or active_residual.shape[0] != 1:
            raise RuntimeError("local PASM requires batch size one")
        linear_phase_ramp = self.mode == "overlap_local_linear_phase_ramp_cv_gate"
        spectral_ridge_gain = (
            self.mode == "overlap_local_phase_aligned_spectral_mixing_ridge_gain"
        )
        precision_regime_gate = (
            self.mode == "overlap_local_phase_aligned_spectral_mixing_precision_regime"
        )
        overlap_window = self.mode in {
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_precision_regime",
            "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
            "overlap_local_linear_phase_ramp_cv_gate",
            "frequency_confidence_phase_aligned_spectral_mixing",
        }
        local_grid = (2, 4)
        if (
            not overlap_window
            and (spatial_height % local_grid[0] or spatial_width % local_grid[1])
        ):
            raise RuntimeError("local PASM requires a divisible 2x4 spatial grid")
        current_length = int(full_residual.shape[1]) - int(memory_length)
        active_positions = torch.nonzero(
            active_frames >= memory_length, as_tuple=False
        ).flatten()
        active_global = active_frames.index_select(0, active_positions)
        order = torch.argsort(active_global)
        active_positions = active_positions.index_select(0, order)
        exact_local = [
            int(value) - memory_length
            for value in active_global.index_select(0, order).detach().cpu().tolist()
        ]
        if not exact_local or len(exact_local) != len(set(exact_local)):
            raise RuntimeError("local PASM requires unique Exact Current anchors")
        exact_residual = active_residual.index_select(
            1, active_positions.to(active_residual.device)
        )[0].reshape(
            len(exact_local), spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        current_interp = full_residual[0, memory_length:].reshape(
            current_length, spatial_height, spatial_width, full_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        if overlap_window:
            exact_tiles, local_grid = self._overlap_phase_tiles(exact_residual)
            current_tiles, current_grid = self._overlap_phase_tiles(current_interp)
            if current_grid != local_grid:
                raise RuntimeError("overlap PASM anchor/target layouts differ")

            def merge_tiles(value: torch.Tensor) -> torch.Tensor:
                return self._unphase_overlap_tiles(
                    value,
                    output_shape=(spatial_height, spatial_width),
                    use_fc_triton=self.fc_triton_ola,
                )

            def retile(value: torch.Tensor) -> torch.Tensor:
                tiles, grid = self._overlap_phase_tiles(value)
                if grid != local_grid:
                    raise RuntimeError("overlap PASM retile layout differs")
                return tiles
        else:
            exact_tiles = self._phase_tiles(exact_residual, grid=local_grid)
            current_tiles = self._phase_tiles(current_interp, grid=local_grid)

            def merge_tiles(value: torch.Tensor) -> torch.Tensor:
                return self._unphase_tiles(value, grid=local_grid)

            def retile(value: torch.Tensor) -> torch.Tensor:
                return self._phase_tiles(value, grid=local_grid)
        tile_height, tile_width = exact_tiles.shape[-2:]
        masks = self._cached_radial_masks(
            tile_height,
            tile_width,
            self.spectral_num_bands,
            device=full_residual.device,
        )
        low_mask = masks[:2].sum(dim=0)
        fft_start.record()
        anchor_spectra = torch.fft.rfft2(exact_tiles, dim=(-2, -1))

        def bounds(target: int, *, exclude: int | None = None):
            same_segment = target < self.current_anchor_boundary
            candidates = [
                (index, frame)
                for index, frame in enumerate(exact_local)
                if frame != exclude
                and (frame < self.current_anchor_boundary) == same_segment
            ]
            left = [(index, frame) for index, frame in candidates if frame < target]
            right = [(index, frame) for index, frame in candidates if frame > target]
            return (left[-1], right[0]) if left and right else None

        inactive = [frame for frame in range(current_length) if frame not in exact_local]
        eligible: list[int] = []
        corrections: list[torch.Tensor] = []
        correction_tiles: list[torch.Tensor] = []
        coherences: list[torch.Tensor] = []
        displacements: list[torch.Tensor] = []
        peak_confidences: list[torch.Tensor] = []
        target_spectral_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        target_coherence_by_tile: list[torch.Tensor] = []
        alphas: list[float] = []
        for target in inactive:
            pair = bounds(target)
            if pair is None:
                continue
            (left_index, left_frame), (right_index, right_frame) = pair
            alpha = float(target - left_frame) / float(right_frame - left_frame)
            if linear_phase_ramp:
                transported, coherence, displacement, peak_confidence = (
                    self._local_linear_phase_ramp_pair_spectrum(
                        anchor_spectra[left_index],
                        anchor_spectra[right_index],
                        alpha,
                        spatial_shape=(tile_height, tile_width),
                    )
                )
                displacements.append(displacement)
                peak_confidences.append(peak_confidence)
            else:
                transported, coherence = self._local_phase_aligned_pair_spectrum(
                    anchor_spectra[left_index], anchor_spectra[right_index], alpha
                )
            target_coherence_by_tile.append(
                (coherence * low_mask[None]).sum(dim=(-2, -1))
                / low_mask.sum().clamp_min(1e-8)
            )
            base_spectrum = torch.fft.rfft2(
                current_tiles[target], dim=(-2, -1)
            )
            target_spectral_pairs.append((base_spectrum, transported))
            if spectral_ridge_gain:
                eligible.append(target)
                coherences.append(
                    (coherence * low_mask[None]).sum()
                    / low_mask.sum()
                    / coherence.shape[0]
                )
                alphas.append(alpha)
                continue
            mixed_spectrum = base_spectrum + low_mask[None, None] * (
                transported - base_spectrum
            )
            predicted_tiles = torch.fft.irfft2(
                mixed_spectrum,
                s=(tile_height, tile_width),
                dim=(-2, -1),
            )
            predicted = merge_tiles(predicted_tiles)
            corrections.append(predicted - current_interp[target])
            correction_tiles.append(predicted_tiles - current_tiles[target])
            eligible.append(target)
            coherences.append((coherence * low_mask[None]).sum() / low_mask.sum() / coherence.shape[0])
            alphas.append(alpha)

        cv_gate = self.mode in {
            "local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_linear_phase_ramp_cv_gate",
        }
        loo_baseline: list[torch.Tensor] = []
        loo_local: list[torch.Tensor] = []
        loo_baseline_by_tile: list[torch.Tensor] = []
        loo_local_by_tile: list[torch.Tensor] = []
        loo_spectral_directions: list[torch.Tensor] = []
        loo_spectral_errors: list[torch.Tensor] = []
        loo_gain_by_segment: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        for held_out, target in enumerate(exact_local):
            pair = bounds(target, exclude=target)
            if pair is None:
                continue
            (left_index, left_frame), (right_index, right_frame) = pair
            alpha = float(target - left_frame) / float(right_frame - left_frame)
            unaligned = (
                float(1.0 - alpha) * anchor_spectra[left_index]
                + float(alpha) * anchor_spectra[right_index]
            )
            if linear_phase_ramp:
                transported, _, _, _ = self._local_linear_phase_ramp_pair_spectrum(
                    anchor_spectra[left_index],
                    anchor_spectra[right_index],
                    alpha,
                    spatial_shape=(tile_height, tile_width),
                )
            else:
                transported, _ = self._local_phase_aligned_pair_spectrum(
                    anchor_spectra[left_index], anchor_spectra[right_index], alpha
                )
            aligned = unaligned + low_mask[None, None] * (transported - unaligned)
            loo_spectral_directions.append(transported - unaligned)
            loo_spectral_errors.append(anchor_spectra[held_out] - unaligned)
            baseline_value = merge_tiles(
                torch.fft.irfft2(
                    unaligned,
                    s=(tile_height, tile_width),
                    dim=(-2, -1),
                ),
            )
            aligned_value = merge_tiles(
                torch.fft.irfft2(
                    aligned,
                    s=(tile_height, tile_width),
                    dim=(-2, -1),
                ),
            )
            loo_baseline.append((exact_residual[held_out] - baseline_value).norm())
            loo_local.append((exact_residual[held_out] - aligned_value).norm())
            exact_heldout_tiles = exact_tiles[held_out]
            baseline_tile_error = (
                (exact_heldout_tiles - retile(baseline_value))
                .flatten(1)
                .norm(dim=1)
            )
            local_tile_error = (
                (exact_heldout_tiles - retile(aligned_value))
                .flatten(1)
                .norm(dim=1)
            )
            loo_baseline_by_tile.append(baseline_tile_error)
            loo_local_by_tile.append(local_tile_error)
            segment_index = 0 if target < self.current_anchor_boundary else 1
            loo_gain_by_segment[segment_index].append(
                (baseline_tile_error - local_tile_error)
                / baseline_tile_error.clamp_min(1e-8)
            )
        fft_end.record()

        if loo_baseline_by_tile:
            baseline_tile_mean = torch.stack(loo_baseline_by_tile).mean(dim=0)
            local_tile_mean = torch.stack(loo_local_by_tile).mean(dim=0)
            tile_cv_improvement = (
                1.0 - local_tile_mean / baseline_tile_mean.clamp_min(1e-8)
            )
            tile_cv_pass = tile_cv_improvement > 0.0
        else:
            tile_cv_improvement = torch.full(
                (local_grid[0] * local_grid[1],),
                float("nan"),
                device=full_residual.device,
            )
            tile_cv_pass = torch.zeros_like(tile_cv_improvement, dtype=torch.bool)

        regime_median_gain = torch.full(
            (2, local_grid[0] * local_grid[1]),
            float("nan"),
            device=full_residual.device,
        )
        regime_win_rate = torch.zeros_like(regime_median_gain)
        regime_calibration_count = torch.zeros(
            (2,), dtype=torch.long, device=full_residual.device
        )
        for segment_index in range(2):
            segment_gains = loo_gain_by_segment[segment_index]
            regime_calibration_count[segment_index] = len(segment_gains)
            if segment_gains:
                gain_stack = torch.stack(segment_gains)
                regime_median_gain[segment_index] = gain_stack.median(dim=0).values
                regime_win_rate[segment_index] = (gain_stack > 0.0).float().mean(dim=0)

        target_regime_transport = torch.zeros(
            (len(eligible), local_grid[0] * local_grid[1]),
            dtype=torch.bool,
            device=full_residual.device,
        )
        if precision_regime_gate:
            for target_index, (target, coherence_by_tile) in enumerate(
                zip(eligible, target_coherence_by_tile, strict=True)
            ):
                segment_index = (
                    0 if target < self.current_anchor_boundary else 1
                )
                enough_calibration = (
                    regime_calibration_count[segment_index]
                    >= self.regime_min_calibration_anchors
                )
                target_regime_transport[target_index] = (
                    enough_calibration
                    & (
                        regime_median_gain[segment_index]
                        > self.regime_gain_margin
                    )
                    & (
                        regime_win_rate[segment_index]
                        >= self.regime_win_rate
                    )
                    & (coherence_by_tile > self.regime_coherence)
                )

        spectral_gain = torch.zeros(
            (local_grid[0] * local_grid[1], 2),
            dtype=torch.float32,
            device=full_residual.device,
        )
        spectral_gain_source = "zero_no_segmented_loo_calibration"
        spectral_gain_global_fallback = torch.zeros_like(
            spectral_gain, dtype=torch.bool
        )
        if spectral_ridge_gain and loo_spectral_directions:
            directions = torch.stack(loo_spectral_directions)
            errors = torch.stack(loo_spectral_errors)
            frequency_weight = torch.full(
                (tile_width // 2 + 1,),
                2.0,
                dtype=torch.float32,
                device=full_residual.device,
            )
            frequency_weight[0] = 1.0
            if tile_width % 2 == 0:
                frequency_weight[-1] = 1.0
            for band in range(2):
                band_mask = masks[band][None, None, None]
                direction_band = directions * band_mask
                error_band = errors * band_mask
                weighted_inner = (
                    (direction_band.conj() * error_band).real
                    * frequency_weight[None, None, None, None]
                )
                weighted_energy = (
                    direction_band.abs().square()
                    * frequency_weight[None, None, None, None]
                )
                numerator = weighted_inner.sum(dim=(0, 2, 3, 4))
                denominator = weighted_energy.sum(dim=(0, 2, 3, 4))
                local_gain = (
                    numerator / (denominator + self.ridge)
                ).clamp_(0.0, 1.0)
                sufficient = denominator > self.ridge
                global_denominator = denominator.sum()
                global_gain = (
                    numerator.sum() / (global_denominator + self.ridge)
                ).clamp_(0.0, 1.0)
                global_available = global_denominator > self.ridge
                fallback_gain = torch.where(
                    global_available,
                    global_gain,
                    torch.zeros_like(global_gain),
                )
                spectral_gain[:, band] = torch.where(
                    sufficient,
                    local_gain,
                    fallback_gain,
                )
                spectral_gain_global_fallback[:, band] = (
                    ~sufficient & global_available
                )
            spectral_gain_source = "segmented_loo_window_band_ridge_with_global_fallback"

            corrections.clear()
            correction_tiles.clear()
            for base_spectrum, transported in target_spectral_pairs:
                delta = transported - base_spectrum
                gained_delta = torch.zeros_like(delta)
                for band in range(2):
                    gained_delta.add_(
                        delta
                        * masks[band][None, None]
                        * spectral_gain[:, band, None, None, None]
                    )
                predicted_tiles = torch.fft.irfft2(
                    base_spectrum + gained_delta,
                    s=(tile_height, tile_width),
                    dim=(-2, -1),
                )
                target = eligible[len(corrections)]
                predicted = merge_tiles(predicted_tiles)
                corrections.append(predicted - current_interp[target])
                correction_tiles.append(predicted_tiles - current_tiles[target])
        elif spectral_ridge_gain:
            corrections = [torch.zeros_like(current_interp[target]) for target in eligible]
            correction_tiles = [
                torch.zeros_like(current_tiles[target]) for target in eligible
            ]

        corrected = full_residual.clone()
        trust_scales: list[torch.Tensor] = []
        if corrections:
            correction_stack = torch.stack(corrections)
            if precision_regime_gate:
                gated_tiles = torch.stack(correction_tiles)
                gated_tiles.mul_(
                    target_regime_transport[:, :, None, None, None]
                )
                correction_stack = merge_tiles(gated_tiles)
            elif cv_gate:
                gated_tiles = (
                    torch.stack(correction_tiles)
                    if overlap_window
                    else retile(correction_stack)
                )
                gated_tiles.mul_(
                    tile_cv_pass[None, :, None, None, None]
                )
                correction_stack = merge_tiles(gated_tiles)
            target_indices = torch.tensor(
                eligible, device=full_residual.device, dtype=torch.long
            )
            reference = current_interp.index_select(0, target_indices)
            correction_norm = correction_stack.flatten(1).norm(dim=1)
            reference_norm = reference.flatten(1).norm(dim=1)
            trust = torch.minimum(
                torch.ones_like(correction_norm),
                self.eta * reference_norm / correction_norm.clamp_min(1e-8),
            )
            correction_stack.mul_(trust[:, None, None, None])
            target_view = corrected[:, memory_length:].reshape(
                1,
                current_length,
                spatial_height,
                spatial_width,
                corrected.shape[-1],
            )
            target_view[0].index_add_(
                0,
                target_indices,
                correction_stack.permute(0, 2, 3, 1),
            )
            trust_scales = list(trust.unbind())
        corrected[:, active_frames] = active_residual
        cuda_end.record()
        baseline_mean = (
            torch.stack(loo_baseline).mean()
            if loo_baseline
            else torch.full((), float("nan"), device=full_residual.device)
        )
        local_mean = (
            torch.stack(loo_local).mean()
            if loo_local
            else torch.full((), float("nan"), device=full_residual.device)
        )
        self._records.append(
            {
                "chunk_index": self._chunk,
                "step_index": self._step,
                "layer_index": int(layer_index),
                "mode": self.mode,
                "status": "corrected" if corrections else "temporal_fallback",
                "fallback_reason": None if corrections else "no_segment_local_bracket",
                "exact_current_frames": exact_local,
                "approximate_current_frames": inactive,
                "target_frames": eligible,
                "target_alphas": alphas,
                "shared_phase_across_channels": True,
                "local_shared_phase": True,
                "local_phase_grid": list(local_grid),
                "local_phase_tile_shape": [tile_height, tile_width],
                "overlap_window": overlap_window,
                "overlap_window_shape": [11, 10] if overlap_window else None,
                "overlap_stride": [8, 8] if overlap_window else None,
                "overlap_padding": [4, 5] if overlap_window else None,
                "partition_of_unity_overlap_add": overlap_window,
                "linear_phase_ramp": linear_phase_ramp,
                "phase_parameters_per_window": 2 if linear_phase_ramp else None,
                "mean_abs_displacement_xy": (
                    torch.stack(displacements).abs().mean(dim=(0, 1))
                    if displacements
                    else torch.zeros(2, device=full_residual.device)
                ),
                "max_abs_displacement_xy": (
                    torch.stack(displacements).abs().amax(dim=(0, 1))
                    if displacements
                    else torch.zeros(2, device=full_residual.device)
                ),
                "mean_phase_correlation_peak_confidence": (
                    torch.stack(peak_confidences).mean()
                    if peak_confidences
                    else torch.zeros((), device=full_residual.device)
                ),
                "heldout_cv_tile_gate": cv_gate,
                "heldout_cv_tile_pass": tile_cv_pass,
                "heldout_cv_tile_improvement": tile_cv_improvement,
                "self_calibrated_spectral_gain": spectral_ridge_gain,
                "spectral_gain_granularity": (
                    "window_low_frequency_band" if spectral_ridge_gain else None
                ),
                "spectral_gain_source": (
                    spectral_gain_source if spectral_ridge_gain else None
                ),
                "spectral_gain": spectral_gain if spectral_ridge_gain else None,
                "spectral_gain_shape": [
                    local_grid[0] * local_grid[1],
                    2,
                ] if spectral_ridge_gain else None,
                "spectral_gain_bounds": [0.0, 1.0]
                if spectral_ridge_gain
                else None,
                "spectral_gain_global_fallback_fraction": (
                    spectral_gain_global_fallback.float().mean()
                    if spectral_ridge_gain
                    else None
                ),
                "binary_tile_gate_applied": cv_gate,
                "precision_regime_selector": precision_regime_gate,
                "regime_gain_margin": (
                    self.regime_gain_margin if precision_regime_gate else None
                ),
                "regime_win_rate_threshold": (
                    self.regime_win_rate if precision_regime_gate else None
                ),
                "regime_coherence_threshold": (
                    self.regime_coherence if precision_regime_gate else None
                ),
                "regime_min_calibration_anchors": (
                    self.regime_min_calibration_anchors
                    if precision_regime_gate
                    else None
                ),
                "regime_median_gain_by_segment": (
                    regime_median_gain if precision_regime_gate else None
                ),
                "regime_win_rate_by_segment": (
                    regime_win_rate if precision_regime_gate else None
                ),
                "regime_calibration_count_by_segment": (
                    regime_calibration_count if precision_regime_gate else None
                ),
                "target_low_band_coherence": (
                    torch.stack(target_coherence_by_tile)
                    if precision_regime_gate and target_coherence_by_tile
                    else None
                ),
                "target_transport_regime": (
                    target_regime_transport if precision_regime_gate else None
                ),
                "transport_regime_fraction": (
                    target_regime_transport.float().mean()
                    if precision_regime_gate and target_regime_transport.numel()
                    else torch.zeros((), device=full_residual.device)
                    if precision_regime_gate
                    else None
                ),
                "value_regime_fallback_on_insufficient_evidence": (
                    True if precision_regime_gate else None
                ),
                "unit_circle_transport": True,
                "phase_align_before_mix": True,
                "wrapped_angle_regression": False,
                "memory_used_as_correction_source": False,
                "no_cross_boundary_matching": True,
                "inactive_current_only": True,
                "approximate_high_frequency_preserved": True,
                "low_frequency_bands_corrected": [0, 1],
                "mean_shared_cross_spectrum_coherence": (
                    torch.stack(coherences).mean()
                    if coherences
                    else torch.zeros((), device=full_residual.device)
                ),
                "trust_scale_min": (
                    torch.stack(trust_scales).min()
                    if trust_scales
                    else torch.ones((), device=full_residual.device)
                ),
                "loo_calibration_frames": len(loo_baseline),
                "loo_unaligned_residual_l2": baseline_mean.detach(),
                "loo_phase_aligned_residual_l2": local_mean.detach(),
                "loo_phase_aligned_improvement": (
                    1.0 - local_mean / baseline_mean.clamp_min(1e-8)
                ).detach(),
                "_cuda_start": cuda_start,
                "_cuda_end": cuda_end,
                "_fft_start": fft_start,
                "_fft_end": fft_end,
                "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
            }
        )
        return corrected

    def correct(
        self,
        *,
        layer_index: int,
        full_input: torch.Tensor,
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        memory_length: int,
        spatial_height: int,
        spatial_width: int,
        coarse_probe_residual: torch.Tensor | None = None,
        active_frame_values: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if (
            self.mode == "frequency_confidence_phase_aligned_spectral_mixing"
            and self.fc_active_layers
            and int(layer_index) not in self.fc_active_layers
        ):
            # This layer is an explicit V21 identity branch.  Bypass before
            # allocating timing events or materializing active frame IDs so
            # profile-guided execution pruning has effectively zero runtime
            # cost on inactive layers.
            self._fc_layer_schedule_bypasses += 1
            return full_residual
        if (
            self.mode == "frequency_confidence_phase_aligned_spectral_mixing"
            and not self._fc_runtime_active
        ):
            # Zero-overhead causal branch: the V21 residual interpolation is
            # already materialized in ``full_residual``.  Record the decision
            # without allocating CUDA timing events or touching FFT state.
            self._fc_causal_v21_bypass_calls += 1
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": "causal_pre_camera_v21",
                    "fc_runtime_active": False,
                    "wall_ms": 0.0,
                }
            )
            return full_residual
        start_wall = time.perf_counter_ns()
        cuda_start = self._timing_event()
        fft_start = self._timing_event()
        fft_end = self._timing_event()
        cuda_end = self._timing_event()
        cuda_start.record()
        temporal = int(full_residual.shape[1])
        current_length = temporal - memory_length
        geometry = self._geometry
        coarse_probe_mode = self.mode == "coarse_probe_lowfreq_self_calibrated"
        phase_aligned_mode = self.mode in {
            "phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_precision_regime",
            "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
            "overlap_local_linear_phase_ramp_cv_gate",
            "frequency_confidence_phase_aligned_spectral_mixing",
        }
        fallback_reason = None
        if self._chunk == 0:
            fallback_reason = "first_chunk"
        elif memory_length <= 0:
            fallback_reason = "no_memory"
        elif geometry is None and not coarse_probe_mode and not phase_aligned_mode:
            fallback_reason = "missing_absolute_geometry"
        if fallback_reason is not None:
            cuda_end.record()
            self._records.append(
                {
                    "chunk_index": self._chunk,
                    "step_index": self._step,
                    "layer_index": int(layer_index),
                    "mode": self.mode,
                    "status": "temporal_fallback",
                    "fallback_reason": fallback_reason,
                    "_cuda_start": cuda_start,
                    "_cuda_end": cuda_end,
                    "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
                }
            )
            return full_residual
        if phase_aligned_mode:
            if self.mode == "frequency_confidence_phase_aligned_spectral_mixing":
                use_reference_layer = bool(
                    self.fc_reference_numerics
                    or int(layer_index) in self.fc_reference_layers
                )
                if use_reference_layer:
                    return self._correct_frequency_confidence_pasm_reference(
                        layer_index=layer_index,
                        force_reference_numerics=use_reference_layer,
                        full_residual=full_residual,
                        active_residual=active_residual,
                        active_frames=active_frames,
                        active_frame_values=active_frame_values,
                        memory_length=memory_length,
                        spatial_height=spatial_height,
                        spatial_width=spatial_width,
                        cuda_start=cuda_start,
                        fft_start=fft_start,
                        fft_end=fft_end,
                        cuda_end=cuda_end,
                        start_wall=start_wall,
                    )
                return self._correct_frequency_confidence_pasm(
                    layer_index=layer_index,
                    full_residual=full_residual,
                    active_residual=active_residual,
                    active_frames=active_frames,
                    active_frame_values=active_frame_values,
                    memory_length=memory_length,
                    spatial_height=spatial_height,
                    spatial_width=spatial_width,
                    cuda_start=cuda_start,
                    fft_start=fft_start,
                    fft_end=fft_end,
                    cuda_end=cuda_end,
                    start_wall=start_wall,
                )
            if self.mode in {
                "local_phase_aligned_spectral_mixing",
                "local_phase_aligned_spectral_mixing_cv_gate",
                "overlap_local_phase_aligned_spectral_mixing_cv_gate",
                "overlap_local_phase_aligned_spectral_mixing_precision_regime",
                "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
                "overlap_local_linear_phase_ramp_cv_gate",
            }:
                return self._correct_local_phase_aligned_spectral_mixing(
                    layer_index=layer_index,
                    full_residual=full_residual,
                    active_residual=active_residual,
                    active_frames=active_frames,
                    memory_length=memory_length,
                    spatial_height=spatial_height,
                    spatial_width=spatial_width,
                    cuda_start=cuda_start,
                    fft_start=fft_start,
                    fft_end=fft_end,
                    cuda_end=cuda_end,
                    start_wall=start_wall,
                )
            return self._correct_phase_aligned_spectral_mixing(
                layer_index=layer_index,
                full_residual=full_residual,
                active_residual=active_residual,
                active_frames=active_frames,
                memory_length=memory_length,
                spatial_height=spatial_height,
                spatial_width=spatial_width,
                cuda_start=cuda_start,
                fft_start=fft_start,
                fft_end=fft_end,
                cuda_end=cuda_end,
                start_wall=start_wall,
            )
        if geometry is not None:
            geometry.validate(memory_length=memory_length, current_length=current_length)
        _, tile_h, tile_w = self.block_shape
        pooled_input = self._pool_tiles(
            full_input,
            temporal=temporal,
            height=spatial_height,
            width=spatial_width,
            tile_h=tile_h,
            tile_w=tile_w,
        )
        pooled_residual = self._pool_tiles(
            full_residual,
            temporal=temporal,
            height=spatial_height,
            width=spatial_width,
            tile_h=tile_h,
            tile_w=tile_w,
        )
        active_current_positions = torch.nonzero(
            active_frames >= memory_length, as_tuple=False
        ).flatten()
        active_current_global = active_frames.index_select(
            0, active_current_positions
        )
        order = torch.argsort(active_current_global)
        active_current_positions = active_current_positions.index_select(0, order)
        active_current_global = active_current_global.index_select(0, order)
        exact_global = [
            int(frame) for frame in active_current_global.detach().cpu().tolist()
        ]
        exact_local = [frame - memory_length for frame in exact_global]
        exact_indices = active_current_global.to(
            device=full_residual.device, dtype=torch.long
        )
        if coarse_probe_mode:
            if coarse_probe_residual is None or coarse_probe_residual.ndim != 5:
                raise RuntimeError(
                    "coarse-probe correction requires [B,T,C,Hb,Wb] residuals"
                )
            if (
                int(coarse_probe_residual.shape[0]) != 1
                or int(coarse_probe_residual.shape[1]) != temporal
                or int(coarse_probe_residual.shape[2]) != int(full_residual.shape[-1])
                or tuple(coarse_probe_residual.shape[-2:])
                != tuple(pooled_residual.shape[-2:])
            ):
                raise RuntimeError("coarse-probe residual/grid mismatch")
            aligned = coarse_probe_residual[0, memory_length:].float()
            confidence = torch.ones(
                current_length,
                pooled_residual.shape[-2],
                pooled_residual.shape[-1],
                device=full_residual.device,
                dtype=torch.float32,
            )
            active_current_residual = active_residual.index_select(
                1, active_current_positions.to(active_residual.device)
            )
            exact_pooled = self._pool_tiles(
                active_current_residual,
                temporal=len(exact_local),
                height=spatial_height,
                width=spatial_width,
                tile_h=tile_h,
                tile_w=tile_w,
            )[0]
            match_stats = {
                "valid_projection_ratio": torch.ones(
                    (), device=full_residual.device
                ),
                "matching_entropy": torch.zeros((), device=full_residual.device),
                "coarse_probe_grid_height": torch.tensor(
                    pooled_residual.shape[-2], device=full_residual.device
                ),
                "coarse_probe_grid_width": torch.tensor(
                    pooled_residual.shape[-1], device=full_residual.device
                ),
            }
        elif self.mode in {
            "current_anchored_self_calibrated",
            "current_anchored_phase_transport",
            "current_anchored_polar_mixing",
        }:
            if full_input.shape[0] != 1 or active_residual.shape[0] != 1:
                raise RuntimeError("Current-anchor correction requires batch size one")
            if not exact_local:
                raise RuntimeError("Current-anchor correction has no Exact Current source")
            current_hidden_fine = full_input[0, memory_length:].reshape(
                current_length, spatial_height, spatial_width, full_input.shape[-1]
            ).permute(0, 3, 1, 2)
            exact_hidden_fine = full_input[0].index_select(
                0, exact_indices
            ).reshape(
                len(exact_local),
                spatial_height,
                spatial_width,
                full_input.shape[-1],
            ).permute(0, 3, 1, 2)
            active_current_residual = active_residual.index_select(
                1, active_current_positions.to(active_residual.device)
            )
            exact_pooled = self._pool_tiles(
                active_current_residual,
                temporal=len(exact_local),
                height=spatial_height,
                width=spatial_width,
                tile_h=tile_h,
                tile_w=tile_w,
            )[0]
            aligned, confidence, match_stats = self._align_current_anchors(
                current_hidden=current_hidden_fine,
                exact_hidden=exact_hidden_fine,
                exact_residual=exact_pooled,
                exact_local_frames=exact_local,
                geometry=geometry,
            )
        else:
            aligned, confidence, match_stats = self._align_memory(
                current_hidden=pooled_input[:, memory_length:],
                memory_hidden=pooled_input[:, :memory_length],
                memory_residual=pooled_residual[:, :memory_length],
                geometry=geometry,
            )
            exact_pooled = pooled_residual[0].index_select(0, exact_indices)
        masks = self._cached_radial_masks(
            pooled_residual.shape[-2],
            pooled_residual.shape[-1],
            self.spectral_num_bands,
            device=full_residual.device,
        )
        gamma = torch.ones(self.spectral_num_bands, device=full_residual.device)
        rho = torch.ones(
            self.spectral_num_bands, device=full_residual.device
        )
        calibration_count = 0
        gamma_source = "not_used"
        phase_gain = torch.zeros_like(gamma)
        amplitude_gain = torch.zeros_like(gamma)
        diagnostic: dict[str, torch.Tensor] = {}
        current_interp = pooled_residual[0, memory_length:]
        difference = aligned - current_interp
        fft_start.record()
        if self.mode == "world_aligned":
            block_correction = confidence[:, None] * difference
        elif self.mode in {
            "current_anchored_phase_transport",
            "current_anchored_polar_mixing",
        }:
            (
                phase_gain,
                amplitude_gain,
                rho,
                calibration_count,
                gamma_source,
                diagnostic,
            ) = self._calibrate_current_anchor_polar(
                layer_index=int(layer_index),
                exact_residual=exact_pooled,
                exact_local_frames=exact_local,
                aligned=aligned,
                confidence=confidence,
                masks=masks,
                mix_amplitude=self.mode == "current_anchored_polar_mixing",
            )
            interp_spectrum = torch.fft.rfft2(
                current_interp.float(), dim=(-2, -1)
            )
            aligned_spectrum = torch.fft.rfft2(
                aligned.float(), dim=(-2, -1)
            )
            block_correction = torch.zeros_like(current_interp, dtype=torch.float32)
            for band in range(2):
                predicted_spectrum = self._polar_band_prediction(
                    interp_spectrum,
                    aligned_spectrum,
                    masks[band][None, None],
                    phase_gain[band],
                    amplitude_gain[band],
                    mix_amplitude=self.mode == "current_anchored_polar_mixing",
                )
                predicted = torch.fft.irfft2(
                    predicted_spectrum,
                    s=current_interp.shape[-2:],
                    dim=(-2, -1),
                )
                block_correction.add_(
                    rho[band] * (predicted - current_interp.float())
                )
            block_correction.mul_(confidence[:, None])
        else:
            projected = self._project_bands(difference, masks)
            if self.mode == "fixed_lowpass":
                gamma = torch.zeros_like(gamma)
                gamma[0] = 1.0
                gamma_source = "fixed_lowpass_band0"
            elif self.mode in {
                "current_anchored_self_calibrated",
                "coarse_probe_lowfreq_self_calibrated",
            }:
                gamma, rho, calibration_count, gamma_source, diagnostic = (
                    self._calibrate_current_anchor_gamma(
                        layer_index=int(layer_index),
                        exact_residual=exact_pooled,
                        exact_local_frames=exact_local,
                        aligned=aligned,
                        confidence=confidence,
                        masks=masks,
                    )
                )
                if coarse_probe_mode:
                    gamma = gamma.clone()
                    rho = rho.clone()
                    gamma[2:] = 0.0
                    rho[2:] = 0.0
            else:
                gamma, calibration_count, gamma_source, diagnostic = (
                    self._calibrate_gamma(
                        layer_index=int(layer_index),
                        exact_residual=exact_pooled,
                        exact_local_frames=exact_local,
                        aligned=aligned,
                        confidence=confidence,
                        masks=masks,
                    )
                )
            block_correction = confidence[:, None] * sum(
                rho[band] * gamma[band] * projected[band]
                for band in range(self.spectral_num_bands)
            )
        fft_end.record()
        inactive_local = torch.ones(
            current_length, device=full_residual.device, dtype=torch.bool
        )
        if exact_local:
            inactive_local[torch.tensor(exact_local, device=full_residual.device)] = False
        valid_match = confidence.flatten(1).amax(dim=1) > 0.0
        apply_mask = inactive_local & valid_match
        if self.mode in {
            "current_anchored_self_calibrated",
            "current_anchored_phase_transport",
            "current_anchored_polar_mixing",
            "coarse_probe_lowfreq_self_calibrated",
        }:
            apply_mask &= rho.amax() > 0.0
        correction_norm = block_correction.flatten(1).norm(dim=1)
        residual_norm = current_interp.flatten(1).norm(dim=1)
        trust_scale = torch.minimum(
            torch.ones_like(correction_norm),
            self.eta * residual_norm / correction_norm.clamp_min(1e-8),
        )
        block_correction = block_correction * trust_scale[:, None, None, None]
        block_correction = block_correction * apply_mask[:, None, None, None]
        upsampled = F.interpolate(
            block_correction,
            size=(spatial_height, spatial_width),
            mode="bilinear",
            align_corners=False,
        )
        corrected = full_residual.clone()
        corrected_current = corrected[:, memory_length:].reshape(
            1, current_length, spatial_height, spatial_width, corrected.shape[-1]
        )
        corrected_current.add_(upsampled.permute(0, 2, 3, 1).unsqueeze(0))
        # Exact Current and every Memory residual are strict identities.
        corrected[:, active_frames] = active_residual
        cuda_end.record()
        post_norm = upsampled.flatten(1).norm()
        reference_norm = full_residual[:, memory_length:].float().norm()
        self._records.append(
            {
                "chunk_index": self._chunk,
                "step_index": self._step,
                "layer_index": int(layer_index),
                "mode": self.mode,
                "status": (
                    "corrected" if bool(apply_mask.any()) else "temporal_fallback"
                ),
                "fallback_reason": None
                if bool(apply_mask.any())
                else (
                    "zero_heldout_cv_reliability"
                    if self.mode in {
                        "current_anchored_self_calibrated",
                        "current_anchored_phase_transport",
                        "current_anchored_polar_mixing",
                        "coarse_probe_lowfreq_self_calibrated",
                    }
                    and float(rho.amax()) <= 0.0
                    else "no_valid_ray_match"
                ),
                "exact_current_frames": exact_local,
                "approximate_current_frames": torch.nonzero(apply_mask).flatten().cpu().tolist(),
                "anchor_frames": exact_local,
                "target_frames": torch.nonzero(apply_mask).flatten().cpu().tolist(),
                "calibration_frames": calibration_count,
                "gamma_source": gamma_source,
                "gamma": gamma.detach(),
                "phase_gain": phase_gain.detach(),
                "amplitude_gain": amplitude_gain.detach(),
                "rho": rho.mean().detach(),
                "rho_b": rho.detach(),
                "current_anchor_only": self.mode in {
                    "current_anchored_self_calibrated",
                    "current_anchored_phase_transport",
                    "current_anchored_polar_mixing",
                },
                "polar_spectral_mixing": self.mode in {
                    "current_anchored_phase_transport",
                    "current_anchored_polar_mixing",
                },
                "phase_transport": self.mode in {
                    "current_anchored_phase_transport",
                    "current_anchored_polar_mixing",
                },
                "amplitude_mixing": self.mode
                == "current_anchored_polar_mixing",
                "coarse_probe_current_only": coarse_probe_mode,
                "memory_used_as_correction_source": self.mode not in {
                    "current_anchored_self_calibrated",
                    "coarse_probe_lowfreq_self_calibrated",
                },
                "anchor_target_disjoint": True,
                "no_cross_boundary_matching": self.mode in {
                    "current_anchored_self_calibrated",
                    "current_anchored_phase_transport",
                    "current_anchored_polar_mixing",
                    "coarse_probe_lowfreq_self_calibrated",
                },
                "inactive_current_only": True,
                "correction_residual_norm_ratio": (
                    post_norm / reference_norm.clamp_min(1e-8)
                ).detach(),
                "trust_scale_min": trust_scale[apply_mask].min().detach()
                if bool(apply_mask.any())
                else torch.ones((), device=full_residual.device),
                **match_stats,
                **diagnostic,
                "_cuda_start": cuda_start,
                "_cuda_end": cuda_end,
                "_fft_start": fft_start,
                "_fft_end": fft_end,
                "wall_ms": (time.perf_counter_ns() - start_wall) / 1_000_000.0,
            }
        )
        return corrected

    def summary(self) -> dict[str, Any]:
        current_anchor = self.mode in {
            "current_anchored_self_calibrated",
            "current_anchored_phase_transport",
            "current_anchored_polar_mixing",
        }
        polar = self.mode in {
            "current_anchored_phase_transport",
            "current_anchored_polar_mixing",
        }
        phase_aligned = self.mode in {
            "phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_precision_regime",
            "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
            "overlap_local_linear_phase_ramp_cv_gate",
            "frequency_confidence_phase_aligned_spectral_mixing",
        }
        local_phase_aligned = self.mode in {
            "local_phase_aligned_spectral_mixing",
            "local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_precision_regime",
            "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
            "overlap_local_linear_phase_ramp_cv_gate",
            "frequency_confidence_phase_aligned_spectral_mixing",
        }
        local_phase_cv_gate = (
            self.mode == "local_phase_aligned_spectral_mixing_cv_gate"
            or self.mode
            == "overlap_local_phase_aligned_spectral_mixing_cv_gate"
            or self.mode == "overlap_local_linear_phase_ramp_cv_gate"
        )
        overlap_local_phase = self.mode in {
            "overlap_local_phase_aligned_spectral_mixing_cv_gate",
            "overlap_local_phase_aligned_spectral_mixing_precision_regime",
            "overlap_local_phase_aligned_spectral_mixing_ridge_gain",
            "overlap_local_linear_phase_ramp_cv_gate",
            "frequency_confidence_phase_aligned_spectral_mixing",
        }
        linear_phase_ramp = self.mode == "overlap_local_linear_phase_ramp_cv_gate"
        spectral_ridge_gain = (
            self.mode == "overlap_local_phase_aligned_spectral_mixing_ridge_gain"
        )
        precision_regime_gate = (
            self.mode == "overlap_local_phase_aligned_spectral_mixing_precision_regime"
        )
        coarse_probe = self.mode == "coarse_probe_lowfreq_self_calibrated"
        frequency_confidence = (
            self.mode == "frequency_confidence_phase_aligned_spectral_mixing"
        )
        contract = {
            "current_anchor_only": current_anchor,
            "coarse_probe_current_only": coarse_probe,
            "memory_used_as_correction_source": not (
                current_anchor or coarse_probe or phase_aligned
            ),
            "world_geometry_required": not (coarse_probe or phase_aligned),
            "current_abs_c2ws_required": not (coarse_probe or phase_aligned),
            "base_intrinsics_required": not (coarse_probe or phase_aligned),
            "no_cross_boundary_matching": (
                current_anchor or coarse_probe or phase_aligned
            ),
            "anchor_target_disjoint": True,
            "inactive_current_only": True,
            "align_depth_samples": self.align_depth_samples,
            "align_top_l": self.align_top_l,
            "spectral_num_bands": self.spectral_num_bands,
            "gamma_max": self.gamma_max,
            "ridge": self.ridge,
            "eta": self.eta,
            "gamma_ema": self.gamma_ema,
            "current_anchor_boundary": self.current_anchor_boundary,
            "current_anchor_match_tile": list(self.current_anchor_match_tile),
            "current_anchor_descriptor_groups": self.current_anchor_descriptor_groups,
            "current_anchor_consensus_mix": self.current_anchor_consensus_mix,
            "heldout_cv_reliability": current_anchor or coarse_probe,
            "ema_key": "step_layer" if current_anchor or coarse_probe else "layer",
            "low_frequency_bands_corrected": [0, 1]
            if coarse_probe or polar or phase_aligned
            else None,
            "polar_spectral_mixing": polar,
            "phase_transport": polar,
            "amplitude_mixing": self.mode == "current_anchored_polar_mixing",
            "approximate_high_frequency_preserved": polar or phase_aligned,
            "shared_phase_across_channels": phase_aligned,
            "local_shared_phase": local_phase_aligned,
            "local_phase_grid": (
                [3, 6] if overlap_local_phase else [2, 4]
            ) if local_phase_aligned else None,
            "local_phase_tile_shape": [11, 10] if local_phase_aligned else None,
            "heldout_cv_tile_gate": local_phase_cv_gate,
            "self_calibrated_spectral_gain": spectral_ridge_gain,
            "spectral_gain_granularity": (
                "window_low_frequency_band" if spectral_ridge_gain else None
            ),
            "spectral_gain_bounds": [0.0, 1.0] if spectral_ridge_gain else None,
            "binary_tile_gate_applied": local_phase_cv_gate,
            "precision_regime_selector": precision_regime_gate,
            "regime_gain_margin": (
                self.regime_gain_margin if precision_regime_gate else None
            ),
            "regime_win_rate_threshold": (
                self.regime_win_rate if precision_regime_gate else None
            ),
            "regime_coherence_threshold": (
                self.regime_coherence if precision_regime_gate else None
            ),
            "regime_min_calibration_anchors": (
                self.regime_min_calibration_anchors
                if precision_regime_gate
                else None
            ),
            "value_regime_fallback_on_insufficient_evidence": (
                True if precision_regime_gate else None
            ),
            "overlap_window": overlap_local_phase,
            "overlap_window_shape": [11, 10] if overlap_local_phase else None,
            "overlap_stride": [8, 8] if overlap_local_phase else None,
            "overlap_padding": [4, 5] if overlap_local_phase else None,
            "partition_of_unity_overlap_add": overlap_local_phase,
            "linear_phase_ramp": linear_phase_ramp,
            "phase_parameters_per_window": 2 if linear_phase_ramp else None,
            "unit_circle_transport": phase_aligned,
            "phase_align_before_mix": phase_aligned,
            "wrapped_angle_regression": False if phase_aligned else None,
            "frequency_confidence_pasm": frequency_confidence,
            "endpoint_only_confidence": frequency_confidence,
            "heldout_exact_used": False if frequency_confidence else None,
            "loo_cv_gate": False if frequency_confidence else None,
            "ridge_gain": False if frequency_confidence else None,
            "all_frequency_unified_transport": frequency_confidence,
            "fc_tau_low": self.fc_tau_low if frequency_confidence else None,
            "fc_tau_high": self.fc_tau_high if frequency_confidence else None,
            "fc_freq_power": self.fc_freq_power if frequency_confidence else None,
            "fc_temperature": self.fc_temperature if frequency_confidence else None,
            "fc_ramp_confidence": (
                self.fc_ramp_confidence if frequency_confidence else None
            ),
            "fc_temporal_consistency": (
                self.fc_temporal_consistency if frequency_confidence else None
            ),
            "fc_trust_eta": self.fc_trust_eta if frequency_confidence else None,
            "fc_v21_blend": self.fc_v21_blend if frequency_confidence else None,
            "fc_layer_gate_threshold": (
                self.fc_layer_gate_threshold if frequency_confidence else None
            ),
            "fc_layer_gate_period": (
                self.fc_layer_gate_period if frequency_confidence else None
            ),
            "fc_pair_gate_threshold": (
                self.fc_pair_gate_threshold if frequency_confidence else None
            ),
            "fc_transport_tile_topk": (
                self.fc_transport_tile_topk if frequency_confidence else None
            ),
            "fc_fused_complex_weights": (
                self.fc_fused_complex_weights if frequency_confidence else None
            ),
            "fc_lean_runtime": (
                self.fc_lean_runtime if frequency_confidence else None
            ),
            "fc_reference_numerics": (
                self.fc_reference_numerics if frequency_confidence else None
            ),
            "fc_reference_layers": (
                list(self.fc_reference_layers) if frequency_confidence else None
            ),
            "fc_zero_transport_fastpath": (
                self.fc_zero_transport_fastpath if frequency_confidence else None
            ),
            "fc_elide_scalar_readback": (
                self.fc_elide_scalar_readback if frequency_confidence else None
            ),
            "fc_parallel_streams": (
                self.fc_parallel_streams if frequency_confidence else None
            ),
            "fc_prealloc_targets": (
                self.fc_prealloc_targets if frequency_confidence else None
            ),
            "fc_batched_endpoint_stats": (
                self.fc_batched_endpoint_stats if frequency_confidence else None
            ),
            "fc_active_layers": (
                list(self.fc_active_layers) if frequency_confidence else None
            ),
            "fc_causal_v21_bypass_calls": (
                self._fc_causal_v21_bypass_calls if frequency_confidence else None
            ),
            "fc_coarse_transport": (
                self.fc_coarse_transport if frequency_confidence else None
            ),
            "fc_triton_phase_mix": (
                self.fc_triton_phase_mix if frequency_confidence else None
            ),
            "fc_triton_accurate_phase_mix": (
                self.fc_triton_accurate_phase_mix
                if frequency_confidence
                else None
            ),
            "fc_triton_batched_mix": (
                self.fc_triton_batched_mix if frequency_confidence else None
            ),
            "fc_triton_tile_extract": (
                self.fc_triton_tile_extract if frequency_confidence else None
            ),
            "fc_triton_ola": (
                self.fc_triton_ola if frequency_confidence else None
            ),
            "fc_triton_phat_peak": (
                self.fc_triton_phat_peak if frequency_confidence else None
            ),
            "fc_triton_ramp_confidence": (
                self.fc_triton_ramp_confidence if frequency_confidence else None
            ),
            "fc_triton_reference_lowfreq": (
                self.fc_triton_reference_lowfreq if frequency_confidence else None
            ),
            "fc_triton_lowfreq_radius": (
                self.fc_triton_lowfreq_radius if frequency_confidence else None
            ),
            "fc_legacy_lowfreq_index_bug": (
                self.fc_legacy_lowfreq_index_bug if frequency_confidence else None
            ),
            "fc_profile_timing": (
                self.fc_profile_timing if frequency_confidence else None
            ),
            "fc_batched_reference_mix": (
                self.fc_batched_reference_mix if frequency_confidence else None
            ),
        }
        if not self._records:
            return {
                "enabled": True,
                "mode": self.mode,
                "records": [],
                # A camera-guarded sample may execute no woven layer at all.
                # The worker independently proves calls == woven_layers, so
                # zero calls is a complete, explicitly not-applicable run
                # rather than a missing-instrumentation failure.
                "status": "not_applicable_no_approximate_residual",
                "complete": True,
                "calls": 0,
                "corrected_calls": 0,
                "fallback_calls": 0,
                "fc_layer_gate_skipped_calls": 0,
                "fc_layer_schedule_bypassed_calls": (
                    self._fc_layer_schedule_bypasses
                ),
                "total_cuda_ms": 0.0,
                "total_fft_ms": 0.0,
                "total_phase_transport_ms": 0.0,
                "total_ola_ms": 0.0,
                "mean_valid_projection_ratio": 0.0,
                "mean_matching_entropy": 0.0,
                "memory_always_exact": True,
                "exact_current_always_restored": True,
                "residual_only": True,
                "spatial_fft_only": True,
                "relative_pluecker_used_for_alignment": False,
                "additional_dit_attention_ffn_forwards": 0,
                **contract,
                # A camera-guarded sample can legitimately have no
                # approximate residual site.  Keep the zero-call summary
                # schema complete so profile-guided FC schedules are
                # certified as a valid no-op rather than rejected for missing
                # optional configuration fields.
                "fc_pair_gate_skipped_targets": 0,
                "fc_pair_gate_skipped_pairs": 0,
                "fc_zero_transport_fastpath": self.fc_zero_transport_fastpath,
                "fc_elide_scalar_readback": self.fc_elide_scalar_readback,
                "fc_parallel_streams": self.fc_parallel_streams,
                "fc_prealloc_targets": self.fc_prealloc_targets,
                "fc_batched_endpoint_stats": self.fc_batched_endpoint_stats,
                "fc_active_layers": list(self.fc_active_layers),
                "fc_reference_layers": list(self.fc_reference_layers),
                "fc_coarse_transport": self.fc_coarse_transport,
                "fc_triton_phase_mix": self.fc_triton_phase_mix,
                "fc_triton_accurate_phase_mix": (
                    self.fc_triton_accurate_phase_mix
                ),
                "fc_triton_batched_mix": self.fc_triton_batched_mix,
                "fc_triton_tile_extract": self.fc_triton_tile_extract,
                "fc_triton_ola": self.fc_triton_ola,
                "fc_triton_phat_peak": self.fc_triton_phat_peak,
                "fc_triton_ramp_confidence": self.fc_triton_ramp_confidence,
            }
        torch.cuda.synchronize()
        rows: list[dict[str, Any]] = []
        total_cuda_ms = 0.0
        total_fft_ms = 0.0
        total_phase_transport_ms = 0.0
        total_ola_ms = 0.0
        for raw in self._records:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, torch.Tensor):
                    value_cpu = value.detach().float().cpu()
                    row[key] = (
                        float(value_cpu.item())
                        if value_cpu.numel() == 1
                        else [float(v) for v in value_cpu.flatten().tolist()]
                    )
                else:
                    row[key] = value
            cuda_start = raw.get("_cuda_start")
            cuda_end = raw.get("_cuda_end")
            if isinstance(cuda_start, torch.cuda.Event) and isinstance(cuda_end, torch.cuda.Event):
                row["cuda_ms"] = float(cuda_start.elapsed_time(cuda_end))
                total_cuda_ms += row["cuda_ms"]
            fft_start = raw.get("_fft_start")
            fft_end = raw.get("_fft_end")
            fft_forward_end = raw.get("_fft_forward_end")
            fft_inverse_start = raw.get("_fft_inverse_start")
            fft_inverse_end = raw.get("_fft_inverse_end")
            if (
                isinstance(fft_start, torch.cuda.Event)
                and isinstance(fft_forward_end, torch.cuda.Event)
                and isinstance(fft_inverse_start, torch.cuda.Event)
                and isinstance(fft_inverse_end, torch.cuda.Event)
            ):
                row["fft_forward_ms"] = float(
                    fft_start.elapsed_time(fft_forward_end)
                )
                row["fft_inverse_ms"] = float(
                    fft_inverse_start.elapsed_time(fft_inverse_end)
                )
                row["fft_ms"] = row["fft_forward_ms"] + row["fft_inverse_ms"]
                total_fft_ms += row["fft_ms"]
            elif isinstance(fft_start, torch.cuda.Event) and isinstance(fft_end, torch.cuda.Event):
                row["fft_ms"] = float(fft_start.elapsed_time(fft_end))
                total_fft_ms += row["fft_ms"]
            else:
                row["fft_ms"] = 0.0
            phase_start = raw.get("_phase_start")
            phase_end = raw.get("_phase_end")
            if isinstance(phase_start, torch.cuda.Event) and isinstance(phase_end, torch.cuda.Event):
                row["phase_transport_ms"] = float(phase_start.elapsed_time(phase_end))
                total_phase_transport_ms += row["phase_transport_ms"]
            else:
                row["phase_transport_ms"] = 0.0
            ola_start = raw.get("_ola_start")
            ola_end = raw.get("_ola_end")
            if isinstance(ola_start, torch.cuda.Event) and isinstance(ola_end, torch.cuda.Event):
                row["ola_ms"] = float(ola_start.elapsed_time(ola_end))
                total_ola_ms += row["ola_ms"]
            else:
                row["ola_ms"] = 0.0
            rows.append(row)
        corrected = [row for row in rows if row["status"] == "corrected"]
        return {
            "enabled": True,
            "mode": self.mode,
            "complete": True,
            "records": rows,
            "calls": len(rows),
            "corrected_calls": len(corrected),
            "fallback_calls": len(rows) - len(corrected),
            "fc_pair_gate_runtime_active_calls": self._fc_pair_gate_active_calls,
            "fc_pair_gate_runtime_inactive_calls": self._fc_pair_gate_inactive_calls,
            "total_cuda_ms": total_cuda_ms,
            "total_fft_ms": total_fft_ms,
            "total_phase_transport_ms": total_phase_transport_ms,
            "total_ola_ms": total_ola_ms,
            "mean_valid_projection_ratio": (
                sum(float(row.get("valid_projection_ratio", 0.0)) for row in corrected)
                / len(corrected)
                if corrected
                else 0.0
            ),
            "mean_matching_entropy": (
                sum(float(row.get("matching_entropy", 0.0)) for row in corrected)
                / len(corrected)
                if corrected
                else 0.0
            ),
            "memory_always_exact": True,
            "exact_current_always_restored": True,
            "residual_only": True,
            "spatial_fft_only": True,
            "relative_pluecker_used_for_alignment": False,
            "additional_dit_attention_ffn_forwards": len(rows) if coarse_probe else 0,
            "mean_heldout_cv_reliability_rho": (
                sum(float(row.get("rho", 0.0)) for row in corrected)
                / len(corrected)
                if corrected
                else 0.0
            ),
            "fc_layer_gate_threshold": self.fc_layer_gate_threshold,
            "fc_layer_gate_period": self.fc_layer_gate_period,
            "fc_layer_gate_skipped_calls": self._fc_layer_gate_skips,
            "fc_pair_gate_threshold": self.fc_pair_gate_threshold,
            "fc_transport_tile_topk": self.fc_transport_tile_topk,
            "fc_pair_gate_skipped_targets": self._fc_pair_gate_skipped_targets,
            "fc_pair_gate_skipped_pairs": self._fc_pair_gate_skipped_pairs,
            "fc_layer_schedule_bypassed_calls": (
                self._fc_layer_schedule_bypasses
            ),
            **contract,
        }
