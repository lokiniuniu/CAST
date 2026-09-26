"""Curvature-Phased Frame Weaving for Matrix-Game-3.0.

The operator is deliberately temporal rather than spatial.  R4 Memory frames
remain exact world-state anchors.  For the current rollout, CWCA's
action-induced camera-worldline curvature determines which temporal cells are
refined in every DiT layer.  The remaining low-curvature frames are woven over
three complementary layer phases.  A skipped frame receives the block
residual interpolated between the nearest exact frames in the same current
worldline section; complete spatial frames are never broken into patches.

This is an independent proposed method.  It is not presented as FIS-DiT or
JiT, and it keeps the released Light Interaction q1 prediction cache intact.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from statistics import median
import time
from types import MethodType
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .matrix_jit_official_semantics import MatrixJiTOfficialSemanticsAcceleration
from .matrix_jit_spatial_acceleration import MatrixJiTSpatialAcceleration
from .matrix_mod5_gather_adaln_kernel import (
    CompiledActiveFrameLayout,
    NativeLNGatherAdaLN1Workspace,
    allocate_native_ln_gather_adaln1_workspace,
    compile_active_frame_layout,
    native_ln_gather_adaln1,
)
from .matrix_control_barycentric_pair_kernel import (
    direct_control_barycentric_pair_write,
)
from .matrix_world_spectral_residual import (
    MatrixWorldAlignmentGeometry,
    WorldAlignedSpectralResidualCorrector,
)
from .fc_pasm_swap_routing import (
    FCPASMLocalCoordinateSwapRouter,
    FCPASMSwapRoutingConfig,
)


@dataclass
class _FrameLayout:
    total_temporal: int
    spatial_height: int
    spatial_width: int
    memory_temporal: int
    full_sequence_length: int
    active_frames: torch.Tensor
    active_frame_values: tuple[int, ...]
    compact_indices: torch.Tensor
    compact_coordinates: torch.Tensor
    packed_indices: torch.Tensor | None = None
    compact_to_packed: torch.Tensor | None = None
    packed_valid: torch.Tensor | None = None
    packed_block_size: int = 0
    compact_block_indices: torch.Tensor | None = None
    compact_block_valid: torch.Tensor | None = None
    compact_token_to_packed: torch.Tensor | None = None
    compact_block_shape: tuple[int, int, int] | None = None


@dataclass
class _AsymmetricFrameLayout:
    """Full K/V world state with a compact set of complete-frame queries."""

    total_temporal: int
    spatial_height: int
    spatial_width: int
    memory_temporal: int
    full_coordinates: torch.Tensor
    query_indices: torch.Tensor
    query_coordinates: torch.Tensor
    key_packed_indices: torch.Tensor
    key_valid: torch.Tensor
    query_block_ids: torch.Tensor
    query_packed_indices: torch.Tensor
    query_valid: torch.Tensor
    block_size: int


@dataclass
class _BarycentricContractionWorkspace:
    sources: torch.Tensor
    output: torch.Tensor
    batched_sources: torch.Tensor | None
    batched_output: torch.Tensor | None
    batched_weights: torch.Tensor | None
    source_position_indices: dict[tuple[int, int], torch.Tensor]
    source_frame_indices: dict[tuple[int, int], torch.Tensor]
    temporal_priors: dict[tuple[int, int, int], torch.Tensor]
    batched_source_position_indices: dict[tuple[int, ...], torch.Tensor]
    batched_target_frame_indices: dict[tuple[int, ...], torch.Tensor]
    direct_source_position_pairs: dict[tuple[int, ...], torch.Tensor]
    direct_target_frame_indices: dict[tuple[int, ...], torch.Tensor]
    segment_position_indices: dict[tuple[int, ...], torch.Tensor]


class MatrixCurvaturePhaseFrameWeave:
    """Update complete world frames on complementary DiT-depth phases."""

    name = "matrix_curvature_phase_frame_weave_v1"

    def __init__(
        self,
        geometry_provider: Any,
        *,
        phase_period: int = 3,
        high_curvature_fraction: float = 0.25,
        high_curvature_anchor_count: int | None = None,
        force_current_endpoints_exact: bool = True,
        active_phases: int | None = None,
        scheduler_phase_schedule: dict[int, tuple[int, int]] | None = None,
        scheduler_phase_offsets: dict[int, int] | None = None,
        sparse_steps: tuple[int, ...] | None = None,
        sparse_layers: tuple[int, ...] | None = None,
        sparse_layers_by_step: dict[int, tuple[int, ...]] | None = None,
        reconstruction: str = "residual",
        compact_cwca_topology: bool = False,
        sparse_density: float = 0.20,
        camera_guard_weave_period: int | None = None,
        camera_guard_weave_active_phases: int = 1,
        camera_guard_q2_weave_period: int | None = None,
        camera_guard_exact_layers: tuple[int, ...] | None = None,
        weave_domain: str = "current",
        camera_action_threshold: float = 0.02,
        witness_probe_layers: tuple[int, ...] | None = None,
        witness_probe_epsilon: float | None = None,
        spectral_polar_probe_layers: tuple[int, ...] | None = None,
        spectral_polar_probe_chunk: int = 1,
        spectral_polar_probe_step: int = 0,
        window_forward_targets: dict[int, tuple[int, ...]] | None = None,
        window_router_metric: str = "phase_input_normalized_defect",
        ray_transport_radius: int = 1,
        ray_transport_feature_groups: int = 16,
        current_segment_boundary: int | None = None,
        curvature_anchor_scope: str = "all_current",
        curvature_anchor_cell_size: int = 1,
        temporal_interpolation: str = "linear",
        secant_scope: str = "token",
        output_dc_scope: str = "all_current",
        asymmetric_topology_selector: Callable[..., tuple[torch.Tensor, torch.Tensor, Any]]
        | None = None,
        target_q_attention_correction: bool = False,
        feature_curvature_fallback: bool = False,
        li_denoise_cache_enabled: bool = True,
        compact_ingress_kernel: str | None = None,
        control_residual_lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
        dynamic_frame_selection: str | None = None,
        dynamic_exact_cell_budget: int = 2,
        dynamic_exact_cell_minimum: int = 2,
        runtime_optimized: bool = False,
        runtime_vectorized_reconstruction: bool = False,
        runtime_vectorized_linear_lift: bool = False,
        runtime_reuse_residual_output: bool = False,
        runtime_single_anchor_write: bool = False,
        runtime_preallocated_barycentric: bool = False,
        runtime_cache_barycentric_control: bool = False,
        runtime_cache_barycentric_weights: bool = False,
        runtime_batched_barycentric_contraction: bool = False,
        runtime_direct_barycentric_pair_kernel: bool = False,
        runtime_reuse_dynamic_active_frame_list: bool = False,
        runtime_shared_int8_qkv: bool = False,
        runtime_shared_int8_qkv_all_paths: bool = False,
        runtime_cached_compact_rope_phase: bool = False,
        runtime_cached_native_rope_phase: bool = False,
        runtime_cross_attention_kv_cache: bool = False,
        compact_active_response_only: bool = False,
        runtime_gate_unused_control_response: bool = False,
        runtime_fine_only_control_response_reduction: bool = False,
        runtime_profile: bool = False,
        enable_world_spectral_residual: bool = False,
        enable_fc_pasm: bool = False,
        world_spectral_variant: str = "self_calibrated",
        align_depth_samples: int = 10,
        align_top_l: int = 4,
        spectral_num_bands: int = 4,
        gamma_max: float = 1.5,
        ridge: float = 1e-4,
        eta: float = 0.25,
        gamma_ema: float = 0.9,
        current_anchor_boundary: int = 4,
        match_tile: tuple[int, int] = (2, 4),
        descriptor_groups: int = 96,
        consensus_mix: float = 0.5,
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
        fc_pair_gate_post_camera_only: bool = False,
        fc_v21_until_camera_seen: bool = False,
        fc_v21_after_no_camera_chunks: int = 0,
        fc_profile_timing: bool = True,
        fc_batched_reference_mix: bool = False,
        fc_fused_complex_weights: bool = False,
        fc_transport_tile_topk: int = 0,
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
        routing_mode: str = "independent_topk",
        routing_candidate_multiplier: int = 2,
        routing_num_swap_rounds: int = 1,
        routing_swap_eps: float = 1e-6,
        routing_lambda_anchor: float = 1.0,
        routing_lambda_reconstruction: float = 1.0,
        routing_sketch_groups: int = 16,
        routing_require_fc_transport: bool = False,
        routing_min_transport_affinity: float = 0.0,
        routing_min_pair_transport_affinity: float = 0.0,
        routing_pair_alias_max_distance: int = -1,
        routing_max_swaps_per_call: int = 0,
        routing_active_layers: tuple[int, ...] = (),
        routing_historical_only: bool = False,
        routing_same_bank_only: bool = False,
        routing_skip_inactive_layers: bool = False,
        sol_attention_runtime: Any | None = None,
        variant_name: str | None = None,
    ) -> None:
        if phase_period < 2:
            raise ValueError("frame weave requires at least two layer phases")
        if not 0.0 <= high_curvature_fraction <= 1.0:
            raise ValueError("high-curvature fraction must be in [0, 1]")
        self.geometry_provider = geometry_provider
        self.feature_curvature_fallback = bool(feature_curvature_fallback)
        self.li_denoise_cache_enabled = bool(li_denoise_cache_enabled)
        self.phase_period = int(phase_period)
        self.active_phases = (
            self.phase_period - 1 if active_phases is None else int(active_phases)
        )
        if not 1 <= self.active_phases < self.phase_period:
            raise ValueError("active phases must lie in [1, phase_period)")
        self.scheduler_phase_schedule = {
            int(step): (int(period), int(active))
            for step, (period, active) in (scheduler_phase_schedule or {}).items()
        }
        self.scheduler_phase_offsets = {
            int(step): int(offset)
            for step, offset in (scheduler_phase_offsets or {}).items()
        }
        if any(step not in {0, 1, 2} for step in self.scheduler_phase_offsets):
            raise ValueError("frame weave phase offsets must target q0, q1, or q2")
        self.sparse_steps = (
            None if sparse_steps is None else frozenset(int(v) for v in sparse_steps)
        )
        self.sparse_layers = (
            None if sparse_layers is None else frozenset(int(v) for v in sparse_layers)
        )
        self.sparse_layers_by_step = {
            int(step): frozenset(int(v) for v in layers)
            for step, layers in (sparse_layers_by_step or {}).items()
        }
        if self.sparse_steps is not None and not self.sparse_steps <= {0, 1, 2}:
            raise ValueError("frame weave sparse steps must be q0, q1, and/or q2")
        if self.sparse_layers is not None and (
            not self.sparse_layers or min(self.sparse_layers) < 0
        ):
            raise ValueError("frame weave sparse layers must be non-negative")
        if self.sparse_layers is not None and self.sparse_layers_by_step:
            raise ValueError("use either shared or scheduler-specific sparse layers")
        if any(
            step not in {0, 1, 2} or not layers or min(layers) < 0
            for step, layers in self.sparse_layers_by_step.items()
        ):
            raise ValueError("invalid scheduler-specific sparse layers")
        self.camera_guard_exact_layers = frozenset(
            int(layer) for layer in (camera_guard_exact_layers or ())
        )
        self.camera_guard_weave_active_phases = int(
            camera_guard_weave_active_phases
        )
        if self.camera_guard_weave_active_phases < 1:
            raise ValueError("camera-guard active phases must be positive")
        if (
            camera_guard_weave_period is not None
            and self.camera_guard_weave_active_phases
            >= int(camera_guard_weave_period)
        ):
            raise ValueError(
                "camera-guard active phases must be smaller than its period"
            )
        if self.camera_guard_exact_layers and min(
            self.camera_guard_exact_layers
        ) < 0:
            raise ValueError("camera-guard exact layers must be non-negative")
        if reconstruction not in {
            "residual",
            "output",
            "action_bridge",
            "depth_transport",
            "ray_aligned_residual",
            "secant_residual",
            "q1_secant_residual",
            "multi_secant_residual",
            "multi_secant_polar_residual",
            "output_dc_residual_detail",
            "input_tangent_residual",
            "input_parallel_residual",
            "input_velocity_residual",
            "attention_delta",
            "convex_chord_residual",
            "target_q_ffn_lift",
            "feature_attention_residual",
            "feature_barycentric_residual",
            "control_space_residual",
            "control_barycentric_residual",
            "scheduler_defect_residual",
            "scheduler_feature_defect_residual",
            "scheduler_affine_residual",
            "scheduler_chord_defect_residual",
        }:
            raise ValueError(
                "frame weave reconstruction must be residual, output, "
                "action_bridge, depth_transport, ray_aligned_residual, "
                "secant_residual, q1_secant_residual, multi_secant_residual, "
                "multi_secant_polar_residual, "
                "output_dc_residual_detail, or "
                "input_tangent_residual, input_parallel_residual, "
                "input_velocity_residual, or "
                "attention_delta, convex_chord_residual, or target_q_ffn_lift"
                ", feature_attention_residual, feature_barycentric_residual, "
                "control_space_residual, control_barycentric_residual, "
                "scheduler_defect_residual, scheduler_feature_defect_residual, "
                "scheduler_affine_residual, or scheduler_chord_defect_residual"
            )
        self.reconstruction = str(reconstruction)
        if (
            len(control_residual_lambdas) != 3
            or any(not math.isfinite(float(value)) or float(value) < 0.0
                   for value in control_residual_lambdas)
            or not any(float(value) > 0.0 for value in control_residual_lambdas)
        ):
            raise ValueError(
                "control residual lambdas must be three finite non-negative values "
                "with at least one positive value"
            )
        self.control_residual_lambdas = tuple(
            float(value) for value in control_residual_lambdas
        )
        if dynamic_frame_selection not in {
            None,
            "response_topk_fixed",
            "response_effective_quota",
            "response_causal_q25_thin1",
            "response_same_layer_q25_thin1",
            "response_same_layer_q25_thin1_gated",
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
            "response_call_q50_nested_mod10_lagged_batch",
            "response_credit_fair_mod5",
            "response_credit_std_scaled_mod5",
        }:
            raise ValueError("unknown dynamic frame-selection policy")
        if dynamic_exact_cell_budget < 1 or dynamic_exact_cell_minimum < 1:
            raise ValueError("dynamic exact-cell counts must be positive")
        if (
            dynamic_frame_selection is not None
            and reconstruction != "control_barycentric_residual"
        ):
            raise ValueError(
                "dynamic response selection requires control-barycentric residuals"
            )
        self.dynamic_frame_selection = dynamic_frame_selection
        self.runtime_optimized = bool(runtime_optimized)
        self.runtime_vectorized_reconstruction = bool(
            runtime_vectorized_reconstruction
        )
        if self.runtime_vectorized_reconstruction and not self.runtime_optimized:
            raise ValueError(
                "vectorized reconstruction requires runtime optimization"
            )
        self.runtime_vectorized_linear_lift = bool(
            runtime_vectorized_linear_lift
        )
        if self.runtime_vectorized_linear_lift and not self.runtime_optimized:
            raise ValueError(
                "vectorized linear lift requires runtime optimization"
            )
        self.runtime_reuse_residual_output = bool(
            runtime_reuse_residual_output
        )
        if self.runtime_reuse_residual_output and not self.runtime_optimized:
            raise ValueError(
                "residual-output reuse requires runtime optimization"
            )
        self.runtime_single_anchor_write = bool(runtime_single_anchor_write)
        if self.runtime_single_anchor_write and not self.runtime_optimized:
            raise ValueError("single anchor write requires runtime optimization")
        self.runtime_preallocated_barycentric = bool(
            runtime_preallocated_barycentric
        )
        if self.runtime_preallocated_barycentric and not self.runtime_optimized:
            raise ValueError(
                "preallocated barycentric contraction requires runtime optimization"
            )
        self.runtime_cache_barycentric_control = bool(
            runtime_cache_barycentric_control
        )
        if self.runtime_cache_barycentric_control and not self.runtime_optimized:
            raise ValueError(
                "barycentric control-state caching requires runtime optimization"
            )
        self.runtime_cache_barycentric_weights = bool(
            runtime_cache_barycentric_weights
        )
        if self.runtime_cache_barycentric_weights and not self.runtime_optimized:
            raise ValueError(
                "barycentric weight caching requires runtime optimization"
            )
        self.runtime_batched_barycentric_contraction = bool(
            runtime_batched_barycentric_contraction
        )
        if (
            self.runtime_batched_barycentric_contraction
            and not self.runtime_preallocated_barycentric
        ):
            raise ValueError(
                "batched barycentric contraction requires preallocated contraction"
            )
        self.runtime_direct_barycentric_pair_kernel = bool(
            runtime_direct_barycentric_pair_kernel
        )
        if (
            self.runtime_direct_barycentric_pair_kernel
            and not self.runtime_preallocated_barycentric
        ):
            raise ValueError(
                "direct barycentric pair kernel requires preallocated contraction"
            )
        if (
            self.runtime_direct_barycentric_pair_kernel
            and self.runtime_batched_barycentric_contraction
        ):
            raise ValueError(
                "direct and batched barycentric contractions are mutually exclusive"
            )
        self.runtime_reuse_dynamic_active_frame_list = bool(
            runtime_reuse_dynamic_active_frame_list
        )
        self.runtime_shared_int8_qkv = bool(runtime_shared_int8_qkv)
        self.runtime_shared_int8_qkv_all_paths = bool(
            runtime_shared_int8_qkv_all_paths
        )
        self.runtime_cached_compact_rope_phase = bool(
            runtime_cached_compact_rope_phase
        )
        self.runtime_cached_native_rope_phase = bool(runtime_cached_native_rope_phase)
        self._native_rope_module = None
        self._native_rope_function = None
        self._native_rope_cache = None
        if self.runtime_shared_int8_qkv_all_paths and not self.runtime_shared_int8_qkv:
            raise ValueError("all-path shared QKV requires compact shared QKV")
        self._native_qkv_coordinators: list[Any] = []
        self.runtime_cross_attention_kv_cache = bool(runtime_cross_attention_kv_cache)
        self._cross_attention_kv_runtime = None
        self._native_cross_attention_forwards: list[tuple[Any, Any]] = []
        if (
            self.runtime_reuse_dynamic_active_frame_list
            and not self.runtime_optimized
        ):
            raise ValueError(
                "dynamic active-frame list reuse requires runtime optimization"
            )
        if (
            self.runtime_reuse_dynamic_active_frame_list
            and self.dynamic_frame_selection is None
        ):
            raise ValueError(
                "dynamic active-frame list reuse requires dynamic frame selection"
            )
        self.compact_active_response_only = bool(compact_active_response_only)
        if self.compact_active_response_only:
            active_response_recorder = getattr(
                self.geometry_provider,
                "record_compact_active_layer_response",
                None,
            )
            if not callable(active_response_recorder):
                raise ValueError(
                    "active-only compact response requires a compatible provider"
                )
            if float(getattr(self.geometry_provider, "response_weight", 1.0)) != 0.0:
                raise ValueError(
                    "active-only compact response requires curvature-only attention"
                )
        self.runtime_gate_unused_control_response = bool(
            runtime_gate_unused_control_response
        )
        self.runtime_fine_only_control_response_reduction = bool(
            runtime_fine_only_control_response_reduction
        )
        if (
            self.runtime_fine_only_control_response_reduction
            and not self.runtime_gate_unused_control_response
        ):
            raise ValueError(
                "fine-only control-response reduction requires response gating"
            )
        if self.runtime_gate_unused_control_response:
            if self.dynamic_frame_selection is None:
                raise ValueError(
                    "control-response gating requires dynamic frame selection"
                )
            enable_gating = getattr(
                self.geometry_provider,
                "enable_curvature_only_response_gating",
                None,
            )
            if not callable(enable_gating):
                raise ValueError(
                    "control-response gating requires a compatible provider"
                )
        self.runtime_profile = bool(runtime_profile)
        self.enable_world_spectral_residual = bool(
            enable_world_spectral_residual
        )
        self.enable_fc_pasm = bool(enable_fc_pasm)
        if self.enable_world_spectral_residual and self.enable_fc_pasm:
            raise ValueError("FC-PASM and legacy world-spectral modes are exclusive")
        self.world_spectral_variant = (
            "frequency_confidence_phase_aligned_spectral_mixing"
            if self.enable_fc_pasm
            else str(world_spectral_variant)
        )
        if routing_mode not in {"independent_topk", "fc_pasm_swap"}:
            raise ValueError("unknown historical-KV routing mode")
        if routing_mode == "fc_pasm_swap" and not self.enable_fc_pasm:
            raise ValueError("FC-PASM swap routing requires FC-PASM reconstruction")
        if routing_mode == "fc_pasm_swap" and not compact_cwca_topology:
            raise ValueError("FC-PASM swap routing requires compact CWCA topology")
        self.routing_mode = str(routing_mode)
        self.fc_pair_gate_post_camera_only = bool(
            fc_pair_gate_post_camera_only
        )
        self.fc_v21_until_camera_seen = bool(fc_v21_until_camera_seen)
        self.fc_v21_after_no_camera_chunks = int(fc_v21_after_no_camera_chunks)
        if self.fc_v21_after_no_camera_chunks < 0:
            raise ValueError("FC no-camera cutoff must be non-negative")
        # Runtime-only optimization: when ROCSA-A has an explicit active-layer
        # allowlist, disabled layers are guaranteed to return native Top-K.
        # The candidate skips the transport lookup/host materialization for
        # those layers while preserving the same fixed-budget selection and
        # diagnostic fallback record.
        self.routing_skip_inactive_layers = bool(routing_skip_inactive_layers)
        self.routing_pair_alias_max_distance = int(routing_pair_alias_max_distance)
        if self.routing_pair_alias_max_distance < -1:
            raise ValueError("routing pair alias max distance must be >= -1")
        self.fc_pasm_swap_router = (
            FCPASMLocalCoordinateSwapRouter(
                FCPASMSwapRoutingConfig(
                    candidate_multiplier=int(routing_candidate_multiplier),
                    num_swap_rounds=int(routing_num_swap_rounds),
                    swap_eps=float(routing_swap_eps),
                    lambda_anchor=float(routing_lambda_anchor),
                    lambda_reconstruction=float(routing_lambda_reconstruction),
                    sketch_groups=int(routing_sketch_groups),
                    require_fc_transport=bool(routing_require_fc_transport),
                    min_transport_affinity=float(routing_min_transport_affinity),
                    min_pair_transport_affinity=float(routing_min_pair_transport_affinity),
                    max_swaps_per_call=int(routing_max_swaps_per_call),
                    active_layers=tuple(int(layer) for layer in routing_active_layers),
                    historical_only=bool(routing_historical_only),
                    same_bank_only=bool(routing_same_bank_only),
                )
            )
            if self.routing_mode == "fc_pasm_swap"
            else None
        )
        self.current_anchor_boundary = int(current_anchor_boundary)
        self.current_anchor_match_tile = tuple(match_tile)
        self.current_anchor_descriptor_groups = int(descriptor_groups)
        self.current_anchor_consensus_mix = float(consensus_mix)
        self.world_spectral_corrector = (
            WorldAlignedSpectralResidualCorrector(
                mode=self.world_spectral_variant,
                align_depth_samples=int(align_depth_samples),
                align_top_l=int(align_top_l),
                spectral_num_bands=int(spectral_num_bands),
                gamma_max=float(gamma_max),
                ridge=float(ridge),
                eta=float(eta),
                gamma_ema=float(gamma_ema),
                current_anchor_boundary=self.current_anchor_boundary,
                current_anchor_match_tile=self.current_anchor_match_tile,
                current_anchor_descriptor_groups=(
                    self.current_anchor_descriptor_groups
                ),
                current_anchor_consensus_mix=self.current_anchor_consensus_mix,
                regime_gain_margin=float(regime_gain_margin),
                regime_win_rate=float(regime_win_rate),
                regime_coherence=float(regime_coherence),
                regime_min_calibration_anchors=int(
                    regime_min_calibration_anchors
                ),
                fc_tau_low=float(fc_tau_low),
                fc_tau_high=float(fc_tau_high),
                fc_freq_power=float(fc_freq_power),
                fc_temperature=float(fc_temperature),
                fc_ramp_confidence=bool(fc_ramp_confidence),
                fc_temporal_consistency=bool(fc_temporal_consistency),
                fc_trust_eta=float(fc_trust_eta),
                fc_layer_gate_threshold=float(fc_layer_gate_threshold),
                fc_layer_gate_period=int(fc_layer_gate_period),
                fc_pair_gate_threshold=float(fc_pair_gate_threshold),
                fc_transport_tile_topk=int(fc_transport_tile_topk),
                fc_fused_complex_weights=bool(fc_fused_complex_weights),
                fc_lean_runtime=bool(fc_lean_runtime),
                fc_reference_numerics=bool(fc_reference_numerics),
                fc_reference_layers=tuple(int(layer) for layer in fc_reference_layers),
                fc_zero_transport_fastpath=bool(fc_zero_transport_fastpath),
                fc_elide_scalar_readback=bool(fc_elide_scalar_readback),
                fc_v21_bypass=bool(fc_v21_bypass),
                fc_v21_blend=float(fc_v21_blend),
                fc_parallel_streams=int(fc_parallel_streams),
                fc_prealloc_targets=bool(fc_prealloc_targets),
                fc_prune_unused_pairs=bool(fc_prune_unused_pairs),
                fc_batched_endpoint_stats=bool(fc_batched_endpoint_stats),
                fc_active_layers=tuple(int(layer) for layer in fc_active_layers),
                fc_coarse_transport=bool(fc_coarse_transport),
                fc_triton_phase_mix=bool(fc_triton_phase_mix),
                fc_triton_accurate_phase_mix=bool(
                    fc_triton_accurate_phase_mix
                ),
                fc_triton_batched_mix=bool(fc_triton_batched_mix),
                fc_triton_tile_extract=bool(fc_triton_tile_extract),
                fc_triton_ola=bool(fc_triton_ola),
                fc_triton_phat_peak=bool(fc_triton_phat_peak),
                fc_triton_ramp_confidence=bool(fc_triton_ramp_confidence),
                fc_triton_reference_lowfreq=bool(fc_triton_reference_lowfreq),
                fc_triton_lowfreq_radius=float(fc_triton_lowfreq_radius),
                fc_legacy_lowfreq_index_bug=bool(fc_legacy_lowfreq_index_bug),
                fc_profile_timing=bool(fc_profile_timing),
                fc_batched_reference_mix=bool(fc_batched_reference_mix),
            )
            if self.enable_world_spectral_residual or self.enable_fc_pasm
            else None
        )
        if (
            self.world_spectral_corrector is not None
            and self.routing_skip_inactive_layers
            and self.fc_pasm_swap_router is not None
        ):
            self.world_spectral_corrector._fc_routing_publish_layers = frozenset(
                int(layer) - 1
                for layer in self.fc_pasm_swap_router.config.active_layers
                if int(layer) > 0
            )
        if self.runtime_optimized:
            enable_runtime_optimization = getattr(
                self.geometry_provider, "enable_runtime_optimization", None
            )
            if not callable(enable_runtime_optimization):
                raise RuntimeError(
                    "runtime-optimized frame weaving requires a compatible "
                    "control-response provider"
                )
            enable_runtime_optimization()
        if self.dynamic_frame_selection in {
            "response_causal_q25_thin1",
            "response_credit_fair_mod5",
            "response_credit_std_scaled_mod5",
        }:
            enable_fine_response = getattr(
                self.geometry_provider, "enable_fine_frame_response", None
            )
            if not callable(enable_fine_response):
                raise RuntimeError(
                    "fine response thinning requires a compatible Closed-Loop provider"
                )
            enable_fine_response()
        if self.runtime_gate_unused_control_response:
            enable_gating()
        if self.runtime_fine_only_control_response_reduction:
            enable_fine_only_reduction = getattr(
                self.geometry_provider,
                "enable_fine_only_response_reduction",
                None,
            )
            if not callable(enable_fine_only_reduction):
                raise RuntimeError(
                    "fine-only response reduction requires a compatible provider"
                )
            enable_fine_only_reduction()
        if self.dynamic_frame_selection == "response_same_layer_q25_thin1":
            enable_temporal_history = getattr(
                self.geometry_provider, "enable_temporal_history_response", None
            )
            if not callable(enable_temporal_history):
                raise RuntimeError(
                    "temporal response thinning requires a compatible Closed-Loop provider"
                )
            enable_temporal_history()
        if self.dynamic_frame_selection == "response_same_layer_q25_thin1_gated":
            enable_gated_history = getattr(
                self.geometry_provider,
                "enable_gated_temporal_history_response",
                None,
            )
            if not callable(enable_gated_history):
                raise RuntimeError(
                    "gated temporal thinning requires a compatible Closed-Loop provider"
                )
            enable_gated_history()
        if self.dynamic_frame_selection in {
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
        }:
            enable_lagged_history = getattr(
                self.geometry_provider,
                "enable_lagged_batched_temporal_history_response",
                None,
            )
            if not callable(enable_lagged_history):
                raise RuntimeError(
                    "lagged temporal thinning requires a compatible "
                    "Closed-Loop provider"
                )
            enable_lagged_history()
        if (
            self.dynamic_frame_selection
            == "response_call_q50_nested_mod10_lagged_batch"
        ):
            enable_call_router = getattr(
                self.geometry_provider,
                "enable_lagged_call_period_router",
                None,
            )
            if not callable(enable_call_router):
                raise RuntimeError(
                    "nested call-period routing requires a compatible "
                    "Closed-Loop provider"
                )
            enable_call_router()
        self.dynamic_exact_cell_budget = int(dynamic_exact_cell_budget)
        self.dynamic_exact_cell_minimum = int(dynamic_exact_cell_minimum)
        if ray_transport_radius < 0:
            raise ValueError("ray-transport radius must be non-negative")
        if ray_transport_feature_groups < 1:
            raise ValueError("ray-transport feature groups must be positive")
        self.ray_transport_radius = int(ray_transport_radius)
        self.ray_transport_feature_groups = int(ray_transport_feature_groups)
        self.current_segment_boundary = (
            None
            if current_segment_boundary is None
            else int(current_segment_boundary)
        )
        if (
            self.current_segment_boundary is not None
            and self.current_segment_boundary < 1
        ):
            raise ValueError("current segment boundary must be positive")
        if curvature_anchor_scope not in {"all_current", "new_generation"}:
            raise ValueError("unknown curvature anchor scope")
        if (
            curvature_anchor_scope == "new_generation"
            and self.current_segment_boundary is None
        ):
            raise ValueError("new-generation curvature scope requires segmentation")
        self.curvature_anchor_scope = str(curvature_anchor_scope)
        if curvature_anchor_cell_size < 1:
            raise ValueError("curvature anchor cell size must be positive")
        self.curvature_anchor_cell_size = int(curvature_anchor_cell_size)
        if temporal_interpolation not in {
            "linear",
            "quadratic",
            "polar_linear",
            "spherical_linear",
            "spherical_chord",
            "spherical_neville_chord",
            "nearest",
            "dc_linear_detail_nearest",
            "natural_cubic",
        }:
            raise ValueError("unsupported temporal interpolation")
        self.temporal_interpolation = str(temporal_interpolation)
        if secant_scope not in {"token", "frame"}:
            raise ValueError("unknown secant scope")
        self.secant_scope = str(secant_scope)
        if output_dc_scope not in {"all_current", "overlap", "new_generation"}:
            raise ValueError("unknown output-DC scope")
        if output_dc_scope != "all_current" and self.current_segment_boundary is None:
            raise ValueError("segmented output-DC scope requires a Current boundary")
        self.output_dc_scope = str(output_dc_scope)
        self.asymmetric_topology_selector = asymmetric_topology_selector
        self.target_q_attention_correction = bool(target_q_attention_correction)
        if self.asymmetric_topology_selector is not None and not compact_cwca_topology:
            raise ValueError("asymmetric full-K/V attention requires compact CWCA")
        if self.target_q_attention_correction and self.asymmetric_topology_selector is None:
            raise ValueError("target-Q correction requires asymmetric full-K/V attention")
        self.compact_cwca_topology = bool(compact_cwca_topology)
        self.sol_attention_runtime = sol_attention_runtime
        if not 0.0 < sparse_density <= 1.0:
            raise ValueError("compact CWCA sparse density must lie in (0, 1]")
        self.sparse_density = float(sparse_density)
        if camera_guard_weave_period not in {None, 2, 3}:
            raise ValueError("camera guard weave period must be 2, 3, or None")
        self.camera_guard_weave_period = camera_guard_weave_period
        if camera_guard_q2_weave_period not in {None, 2}:
            raise ValueError("camera guard q2 weave period must be 2 or None")
        self.camera_guard_q2_weave_period = camera_guard_q2_weave_period
        if compact_ingress_kernel not in {None, "c1a_native_ln_gather_adaln1"}:
            raise ValueError("unknown compact frame-ingress kernel")
        if compact_ingress_kernel is not None and (
            not self.compact_cwca_topology
            or self.asymmetric_topology_selector is not None
            or self.reconstruction != "residual"
        ):
            raise ValueError(
                "c1a ingress requires compact symmetric CWCA residual weaving"
            )
        self.compact_ingress_kernel = compact_ingress_kernel
        if weave_domain not in {
            "current",
            "r4_memory",
            "stationary_dual",
            "camera_guarded_current",
            "camera_history_dominant_current",
        }:
            raise ValueError(
                "frame weave domain must be current, r4_memory, "
                "stationary_dual, camera_guarded_current, or "
                "camera_history_dominant_current"
            )
        self.weave_domain = str(weave_domain)
        if camera_action_threshold < 0.0:
            raise ValueError("camera action threshold must be non-negative")
        self.camera_action_threshold = float(camera_action_threshold)
        self.witness_probe_layers = frozenset(
            int(layer) for layer in (witness_probe_layers or ())
        )
        if self.witness_probe_layers and min(self.witness_probe_layers) < 0:
            raise ValueError("witness-probe layers must be non-negative")
        self.spectral_polar_probe_layers = frozenset(
            int(layer) for layer in (spectral_polar_probe_layers or ())
        )
        if self.spectral_polar_probe_layers and (
            min(self.spectral_polar_probe_layers) < 0
            or max(self.spectral_polar_probe_layers) >= 30
        ):
            raise ValueError("spectral-polar probe layers must be in [0, 29]")
        self.spectral_polar_probe_chunk = int(spectral_polar_probe_chunk)
        self.spectral_polar_probe_step = int(spectral_polar_probe_step)
        if self.spectral_polar_probe_chunk < 0:
            raise ValueError("spectral-polar probe chunk must be non-negative")
        if self.spectral_polar_probe_step not in (0, 1, 2):
            raise ValueError("spectral-polar probe step must be q0, q1, or q2")
        self.window_forward_targets = {
            int(observation): tuple(int(layer) for layer in targets)
            for observation, targets in (window_forward_targets or {}).items()
        }
        if any(
            observation < 0
            or not targets
            or any(layer <= observation for layer in targets)
            for observation, targets in self.window_forward_targets.items()
        ):
            raise ValueError(
                "window-forward targets must be non-empty layers after observation"
            )
        flattened_targets = [
            layer
            for targets in self.window_forward_targets.values()
            for layer in targets
        ]
        if len(flattened_targets) != len(set(flattened_targets)):
            raise ValueError("window-forward target layers may not overlap")
        if window_router_metric not in {
            "phase_input_normalized_defect",
            "multi_input_normalized_defect",
        }:
            raise ValueError("unsupported window-router metric")
        self.window_router_metric = str(window_router_metric)
        if self.witness_probe_layers or self.window_forward_targets:
            if witness_probe_epsilon is None or witness_probe_epsilon <= 0.0:
                raise ValueError(
                    "witness-probe epsilon must be positive when probing or routing is enabled"
                )
            self.witness_probe_epsilon = float(witness_probe_epsilon)
        else:
            self.witness_probe_epsilon = None
        for step, (period, active) in self.scheduler_phase_schedule.items():
            if step not in (0, 1, 2) or period < 2 or not 1 <= active < period:
                raise ValueError("invalid scheduler-conditioned frame phase schedule")
        self.high_curvature_fraction = float(high_curvature_fraction)
        self.high_curvature_anchor_count = (
            None
            if high_curvature_anchor_count is None
            else int(high_curvature_anchor_count)
        )
        if self.high_curvature_anchor_count is not None and self.high_curvature_anchor_count < 0:
            raise ValueError("high-curvature anchor count must be non-negative")
        self.force_current_endpoints_exact = bool(force_current_endpoints_exact)
        if variant_name is not None:
            self.name = str(variant_name)
        self.model: Any = None
        self._native_forwards: list[tuple[Any, Any]] = []
        self.reset_runtime_state()

    def install(self, model: Any) -> None:
        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) != 30:
            raise TypeError("frame weaving requires Matrix's 30 DiT blocks")
        if self.model is not None:
            raise RuntimeError("frame weaving was installed twice")
        self.model = model
        self._num_blocks = len(blocks)
        if self.runtime_cached_native_rope_phase:
            import wan.modules.model as matrix_model

            from .matrix_native_rope_phase_cache import MatrixNativeRoPEPhaseCache

            self._native_rope_module = matrix_model
            self._native_rope_function = matrix_model.rope_apply_with_indices
            self._native_rope_cache = MatrixNativeRoPEPhaseCache(
                self._native_rope_function
            )
            matrix_model.rope_apply_with_indices = self._native_rope_cache
        if self.runtime_cross_attention_kv_cache:
            from wan.modules.attention import attention

            from .matrix_cross_attention_kv_cache import MatrixCrossAttentionKVCache

            self._cross_attention_kv_runtime = MatrixCrossAttentionKVCache()
        if self.runtime_shared_int8_qkv_all_paths:
            from .matrix_shared_int8_qkv import SharedInt8QKVForwardCoordinator
        for block_index, block in enumerate(blocks):
            block.self_attn.block_idx = block_index
            native = block.forward
            self._native_forwards.append((block, native))

            def wrapped(
                block_self: Any,
                x: torch.Tensor,
                *args: Any,
                _index: int = block_index,
                _native: Any = native,
                **kwargs: Any,
            ) -> torch.Tensor:
                return self._profile_forward_block(
                    _index, block_self, _native, x, *args, **kwargs
                )

            block.forward = MethodType(wrapped, block)
            if self.runtime_shared_int8_qkv_all_paths:
                coordinator = SharedInt8QKVForwardCoordinator(block.self_attn)
                coordinator.install()
                self._native_qkv_coordinators.append(coordinator)
            if self.runtime_cross_attention_kv_cache:
                cross_module = block.cross_attn
                native_cross = cross_module.forward
                self._native_cross_attention_forwards.append(
                    (cross_module, native_cross)
                )

                def cached_cross_forward(
                    cross_self: Any,
                    x: torch.Tensor,
                    context: torch.Tensor,
                    context_lens: torch.Tensor,
                    fa_version: Any = None,
                    _index: int = block_index,
                    _attention: Any = attention,
                ) -> torch.Tensor:
                    if self._cross_attention_kv_runtime is None:
                        raise RuntimeError("Matrix cross-attention cache was not initialized")
                    return self._cross_attention_kv_runtime.forward(
                        _index,
                        cross_self,
                        _attention,
                        x,
                        context,
                        context_lens,
                        fa_version,
                    )

                cross_module.forward = MethodType(cached_cross_forward, cross_module)

    def uninstall(self) -> None:
        if self._native_rope_module is not None:
            self._native_rope_module.rope_apply_with_indices = (
                self._native_rope_function
            )
        if self._native_rope_cache is not None:
            self._native_rope_cache.clear()
        self._native_rope_module = None
        self._native_rope_function = None
        self._native_rope_cache = None
        for coordinator in self._native_qkv_coordinators:
            coordinator.uninstall()
        self._native_qkv_coordinators.clear()
        for module, native in self._native_cross_attention_forwards:
            module.forward = native
        self._native_cross_attention_forwards.clear()
        if self._cross_attention_kv_runtime is not None:
            self._cross_attention_kv_runtime.clear()
        self._cross_attention_kv_runtime = None
        for block, native in self._native_forwards:
            block.forward = native
        self._native_forwards.clear()
        self._c1a_compiled_layouts.clear()
        self._c1a_workspaces.clear()
        self._barycentric_workspaces.clear()
        self.model = None

    def reset_runtime_state(self) -> None:
        self._active = False
        self._chunk = -1
        self._step = -1
        self._memory_atoms = 0
        self._records: list[dict[str, Any]] = []
        self._call_layers: list[dict[str, Any]] = []
        self._layout_cache: dict[tuple[Any, ...], _FrameLayout] = {}
        self._curvature_profile: dict[str, Any] | None = None
        self._last_compact_attention: dict[str, Any] | None = None
        self._last_stationary_worldline: bool | None = None
        self._current_action_signature: tuple[float, ...] | None = None
        self._camera_guard_exact = False
        self._camera_seen = False
        self._camera_seen_before_call = False
        self._fc_runtime_active_for_call = True
        self._camera_active_chunks = 0
        self._keyboard_active_chunks = 0
        self._action_history_last_chunk = -1
        self._previous_full_residual: torch.Tensor | None = None
        self._scheduler_q0_residuals: dict[int, torch.Tensor] = {}
        self._window_routing_history: dict[int, list[float]] = {
            observation: [] for observation in self.window_forward_targets
        }
        self._window_router_records: list[dict[str, Any]] = []
        self._window_forward_exact_layers: set[int] = set()
        self._last_ray_transport: dict[str, Any] | None = None
        self._last_control_space_reconstruction: dict[str, Any] | None = None
        self._last_dynamic_frame_selection: dict[str, Any] | None = None
        self._response_selection_credit: torch.Tensor | None = None
        self._temporal_history_capture_active = False
        self._fine_frame_response_capture_active = False
        self._call_current_temporal: int | None = None
        self._compact_rope_phase_key: tuple[Any, ...] | None = None
        self._compact_rope_phase: torch.Tensor | None = None
        self._c1a_compiled_layouts: dict[
            tuple[Any, ...], CompiledActiveFrameLayout
        ] = {}
        self._c1a_workspaces: dict[
            tuple[Any, ...], NativeLNGatherAdaLN1Workspace
        ] = {}
        self._c1a_calls = 0
        self._c1a_layout_compilations = 0
        self._c1a_workspace_allocations = 0
        self._c1a_active_frame_histogram: dict[int, int] = {}
        self._c1a_hidden_dtype_histogram: dict[str, int] = {}
        self._barycentric_workspaces: dict[
            tuple[Any, ...], _BarycentricContractionWorkspace
        ] = {}
        self._barycentric_workspace_allocations = 0
        self._barycentric_preallocated_contractions = 0
        self._barycentric_batched_contraction_calls = 0
        self._barycentric_direct_pair_kernel_calls = 0
        self._barycentric_control_state_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, bool]
        ] = {}
        self._barycentric_weight_cache: dict[
            tuple[int, int, int], torch.Tensor
        ] = {}
        self._barycentric_control_cache_hits = 0
        self._barycentric_control_cache_misses = 0
        self._barycentric_weight_cache_hits = 0
        self._barycentric_weight_cache_misses = 0
        self._barycentric_active_frame_list_reuse_calls = 0
        self._runtime_profile_records: list[dict[str, Any]] = []
        self._runtime_profile_summary_cache: dict[str, Any] | None = None
        self._world_alignment_geometry: MatrixWorldAlignmentGeometry | None = None
        if self.world_spectral_corrector is not None:
            self.world_spectral_corrector.reset_runtime_state()
        if self.fc_pasm_swap_router is not None:
            self.fc_pasm_swap_router.reset()

    def _profile_call(
        self,
        category: str,
        profile_layer_index: int | None,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not self.runtime_profile:
            return function(*args, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError("frame-weave runtime profiling requires CUDA")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter_ns()
        failed = False
        try:
            return function(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
            end.record()
            self._runtime_profile_records.append(
                {
                    "category": str(category),
                    "chunk_index": int(self._chunk),
                    "step_index": int(self._step),
                    "layer_index": (
                        None
                        if profile_layer_index is None
                        else int(profile_layer_index)
                    ),
                    "wall_ms": float(wall_ms),
                    "failed": bool(failed),
                    "_cuda_start": start,
                    "_cuda_end": end,
                }
            )

    def _profile_forward_block(
        self,
        block_index: int,
        block: Any,
        native_forward: Any,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self.runtime_profile:
            return self._forward_block(
                block_index, block, native_forward, x, *args, **kwargs
            )
        layer_count = len(self._call_layers)
        output = self._profile_call(
            "frame_block_total",
            block_index,
            self._forward_block,
            block_index,
            block,
            native_forward,
            x,
            *args,
            **kwargs,
        )
        if len(self._call_layers) != layer_count + 1:
            raise RuntimeError("runtime profiler lost a frame-layer record")
        if not self._runtime_profile_records:
            raise RuntimeError("runtime profiler lost its block timing")
        layer = self._call_layers[-1]
        total_record = self._runtime_profile_records[-1]
        if total_record.get("category") != "frame_block_total":
            raise RuntimeError("runtime profiler block timing is not terminal")
        total_record["execution_path"] = (
            "full" if layer.get("full_layer") is True else "woven"
        )
        total_record["active_frames"] = int(layer.get("active_frames", 0))
        total_record["total_frames"] = int(layer.get("total_frames", 0))
        return output

    def _runtime_profile_summary(self) -> dict[str, Any] | None:
        if not self.runtime_profile:
            return None
        if self._runtime_profile_summary_cache is not None:
            return self._runtime_profile_summary_cache
        torch.cuda.synchronize()
        rows: list[dict[str, Any]] = []
        summaries: dict[str, dict[str, float | int]] = {}
        for raw in self._runtime_profile_records:
            start = raw.get("_cuda_start")
            end = raw.get("_cuda_end")
            if not isinstance(start, torch.cuda.Event) or not isinstance(
                end, torch.cuda.Event
            ):
                raise RuntimeError("runtime profiler lost CUDA events")
            cuda_ms = float(start.elapsed_time(end))
            wall_ms = float(raw["wall_ms"])
            row = {
                key: value
                for key, value in raw.items()
                if not key.startswith("_cuda_")
            }
            row["cuda_ms"] = cuda_ms
            row["host_or_sync_ms"] = max(0.0, wall_ms - cuda_ms)
            rows.append(row)
            summary = summaries.setdefault(
                str(row["category"]),
                {
                    "calls": 0,
                    "wall_ms": 0.0,
                    "cuda_ms": 0.0,
                    "host_or_sync_ms": 0.0,
                },
            )
            summary["calls"] = int(summary["calls"]) + 1
            summary["wall_ms"] = float(summary["wall_ms"]) + wall_ms
            summary["cuda_ms"] = float(summary["cuda_ms"]) + cuda_ms
            summary["host_or_sync_ms"] = (
                float(summary["host_or_sync_ms"])
                + max(0.0, wall_ms - cuda_ms)
            )
        self._runtime_profile_summary_cache = {
            "enabled": True,
            "complete": bool(rows) and not any(row["failed"] for row in rows),
            "output_mutation": False,
            "timing_semantics": {
                "wall_ms": (
                    "host wall time inside the marked region; CUDA scalar reads "
                    "therefore expose synchronization stalls"
                ),
                "cuda_ms": "same-stream CUDA event time for kernels in the region",
                "host_or_sync_ms": (
                    "max(wall_ms-cuda_ms,0); diagnostic sync/host estimate, not "
                    "an additive whole-run decomposition"
                ),
                "nested_regions": True,
                "profiling_overhead": (
                    "CUDA events are diagnostic-only and this run is not a "
                    "promotion timing"
                ),
            },
            "summaries": summaries,
            "records": rows,
        }
        return self._runtime_profile_summary_cache

    def begin_model_call(
        self,
        *,
        chunk_index: int,
        step_index: int,
        current_action: torch.Tensor | None = None,
        memory_atoms: int = 0,
        world_alignment_geometry: MatrixWorldAlignmentGeometry | None = None,
        model_device: torch.device | None = None,
        **_: Any,
    ) -> str:
        if self._active:
            raise RuntimeError("frame-weave model calls may not overlap")
        if int(step_index) not in (0, 1, 2):
            raise ValueError(f"invalid Matrix scheduler step {step_index}")
        self._active = True
        self._chunk = int(chunk_index)
        self._step = int(step_index)
        self._memory_atoms = int(memory_atoms)
        self._world_alignment_geometry = world_alignment_geometry
        if self.world_spectral_corrector is not None:
            if model_device is None:
                raise RuntimeError(
                    "world-spectral correction requires the model-call device"
                )
            self.world_spectral_corrector.begin_model_call(
                geometry=world_alignment_geometry,
                chunk_index=self._chunk,
                step_index=self._step,
                device=torch.device(model_device),
            )
        self._call_layers = []
        self._call_current_temporal = None
        self._barycentric_control_state_cache = {}
        self._barycentric_weight_cache = {}
        if not self.runtime_optimized:
            self._layout_cache = {}
        self._curvature_profile = None
        self._last_stationary_worldline = None
        if self.reconstruction in {
            "scheduler_defect_residual",
            "scheduler_feature_defect_residual",
            "scheduler_affine_residual",
            "scheduler_chord_defect_residual",
        } and self._step == 0:
            # A q0 call starts a new same-chunk scheduler fibre.  Never allow
            # residuals from an earlier chunk to survive into its q2 call.
            self._scheduler_q0_residuals.clear()
        if isinstance(current_action, torch.Tensor):
            action = current_action.detach().float().reshape(-1)
            self._current_action_signature = tuple(
                float(value) for value in action.cpu().tolist()
            )
            camera_active = bool(
                action.numel() >= 2
                and float(action[:2].abs().max().item())
                > self.camera_action_threshold
            )
            keyboard_active = bool(
                action.numel() > 2
                and float(action[2:].abs().max().item())
                > self.camera_action_threshold
            )
            if (
                self.weave_domain == "camera_history_dominant_current"
                and self._action_history_last_chunk != self._chunk
            ):
                self._camera_active_chunks += int(camera_active)
                self._keyboard_active_chunks += int(keyboard_active)
                self._action_history_last_chunk = self._chunk
            self._camera_guard_exact = bool(
                camera_active
                and (
                    self.weave_domain != "camera_history_dominant_current"
                    or self._camera_active_chunks > self._keyboard_active_chunks
                )
            )
        else:
            self._current_action_signature = None
            self._camera_guard_exact = False
            camera_active = False
        self._camera_seen_before_call = bool(self._camera_seen)
        self._camera_seen = bool(self._camera_seen or camera_active)
        if self.world_spectral_corrector is not None:
            self.world_spectral_corrector.set_fc_pair_gate_runtime_active(
                not self.fc_pair_gate_post_camera_only or self._camera_seen
            )
            self._fc_runtime_active_for_call = bool(
                (
                    not self.fc_v21_until_camera_seen
                    or self._camera_seen
                )
                and (
                    self.fc_v21_after_no_camera_chunks <= 0
                    or self._camera_seen
                    or self._chunk < self.fc_v21_after_no_camera_chunks
                )
            )
            self.world_spectral_corrector.set_fc_runtime_active(
                self._fc_runtime_active_for_call
            )
        self._temporal_history_capture_active = bool(
            self.dynamic_frame_selection
            in {
                "response_same_layer_q25_thin1_gated",
                "response_same_layer_q25_thin1_lagged_batch",
                "response_same_layer_q25_structural_collapse_lagged_batch",
                "response_call_q50_nested_mod10_lagged_batch",
            }
            and self._step == 0
            and self._memory_atoms > 0
            and not self._camera_guard_exact
            and (self.sparse_steps is None or 0 in self.sparse_steps)
        )
        self._fine_frame_response_capture_active = bool(
            self.runtime_gate_unused_control_response
            and self.dynamic_frame_selection is not None
            and self._step == 0
            and self._memory_atoms > 0
            and (
                not self._camera_guard_exact
                or self.camera_guard_weave_period is not None
            )
            and (self.sparse_steps is None or 0 in self.sparse_steps)
        )
        if self.runtime_gate_unused_control_response:
            set_fine_capture = getattr(
                self.geometry_provider,
                "set_fine_frame_response_capture",
                None,
            )
            if not callable(set_fine_capture):
                raise RuntimeError(
                    "control-response provider lost fine capture gating"
                )
            set_fine_capture(self._fine_frame_response_capture_active)
        if self.dynamic_frame_selection in {
            "response_same_layer_q25_thin1_gated",
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
            "response_call_q50_nested_mod10_lagged_batch",
        }:
            set_capture = getattr(
                self.geometry_provider, "set_temporal_history_capture", None
            )
            if not callable(set_capture):
                raise RuntimeError("gated temporal-history provider lost its runtime API")
            set_capture(self._temporal_history_capture_active)
        self._previous_full_residual = None
        self._window_forward_exact_layers: set[int] = set()
        self._last_ray_transport = None
        self._last_control_space_reconstruction = None
        self._last_dynamic_frame_selection = None
        # Selection credit is local to one scheduler stack (q0 or q2).
        # Carrying it across steps/chunks would compare unrelated Current
        # trajectories and would let q0 history bias q2 selection.
        self._response_selection_credit = None
        return (
            "released_li_q1_prediction_cache"
            if self._step == 1
            and (self.sparse_steps is None or 1 not in self.sparse_steps)
            else "curvature_phase_full_frame_weave"
        )

    def abort_model_call(self) -> None:
        if self.reconstruction in {
            "scheduler_defect_residual",
            "scheduler_feature_defect_residual",
            "scheduler_affine_residual",
            "scheduler_chord_defect_residual",
        } and self._step == 0:
            self._scheduler_q0_residuals.clear()
        self._active = False
        if not self.runtime_optimized:
            self._layout_cache = {}
        if self.dynamic_frame_selection in {
            "response_same_layer_q25_thin1_gated",
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
            "response_call_q50_nested_mod10_lagged_batch",
        }:
            self.geometry_provider.set_temporal_history_capture(False)
            self._temporal_history_capture_active = False
        if self.runtime_gate_unused_control_response:
            self.geometry_provider.set_fine_frame_response_capture(False)
            self._fine_frame_response_capture_active = False

    def finish_model_call(self, prediction: Any) -> Any:
        if not self._active:
            raise RuntimeError("finished an inactive frame-weave call")
        ratios = [float(row["active_ratio"]) for row in self._call_layers]
        full_layers = sum(bool(row["full_layer"]) for row in self._call_layers)
        woven_layers = sum(not bool(row["full_layer"]) for row in self._call_layers)
        camera_period = (
            self.camera_guard_weave_period
            if self._step == 0
            else self.camera_guard_q2_weave_period
            if self._step == 2
            else None
        )
        camera_woven = bool(
            self._camera_guard_exact
            and camera_period is not None
            and woven_layers > 0
        )
        nested_selected_period = None
        if (
            self.dynamic_frame_selection
            == "response_call_q50_nested_mod10_lagged_batch"
            and woven_layers > 0
        ):
            nested_periods = {
                int(layer["dynamic_frame_selection"]["selected_period"])
                for layer in self._call_layers
                if isinstance(layer.get("dynamic_frame_selection"), dict)
            }
            if len(nested_periods) != 1:
                raise RuntimeError(
                    "nested response routing changed period within one q0 stack"
                )
            nested_selected_period = next(iter(nested_periods))
        effective_weave_period = None
        if woven_layers > 0:
            effective_weave_period = (
                camera_period
                if camera_woven
                else self.scheduler_phase_schedule.get(
                    self._step, (self.phase_period, self.active_phases)
                )[0]
            )
        record = {
            "chunk_index": self._chunk,
            "step_index": self._step,
            "mode": (
                "released_li_q1_prediction_cache"
                if not self._call_layers and self._step == 1
                else "curvature_phase_full_frame_weave"
            ),
            "layers_executed": len(self._call_layers),
            "full_layers": full_layers,
            "woven_layers": woven_layers,
            "mean_active_frame_ratio": (
                float(sum(ratios) / len(ratios)) if ratios else 0.0
            ),
            "action_signature": self._current_action_signature,
            "camera_guard_exact": self._camera_guard_exact,
            "camera_seen_before_call": self._camera_seen_before_call,
            "camera_seen_after_call": self._camera_seen,
            "fc_runtime_active": self._fc_runtime_active_for_call,
            "original_camera_guard_triggered": self._camera_guard_exact,
            "camera_guard_weave_period": self.camera_guard_weave_period,
            "camera_guard_weave_active_phases": (
                self.camera_guard_weave_active_phases
            ),
            "camera_guard_q2_weave_period": self.camera_guard_q2_weave_period,
            "camera_woven": camera_woven,
            "effective_weave_period": effective_weave_period,
            "dynamic_nested_selected_period": nested_selected_period,
            "memory_atoms": self._memory_atoms,
            "camera_active_chunks": self._camera_active_chunks,
            "keyboard_active_chunks": self._keyboard_active_chunks,
            "temporal_history_capture_active": (
                self._temporal_history_capture_active
            ),
            "fine_frame_response_capture_active": (
                self._fine_frame_response_capture_active
            ),
            "layers": list(self._call_layers),
        }
        if (
            self._step != 1
            or (self.sparse_steps is not None and 1 in self.sparse_steps)
        ) and len(self._call_layers) != getattr(self, "_num_blocks", 30):
            raise RuntimeError(
                "frame weaving did not account for all Matrix layers: "
                f"chunk={self._chunk} step={self._step} "
                f"observed={len(self._call_layers)} "
                f"expected={getattr(self, '_num_blocks', 30)}"
            )
        if (
            self.reconstruction
            in {
                "scheduler_defect_residual",
                "scheduler_feature_defect_residual",
                "scheduler_affine_residual",
                "scheduler_chord_defect_residual",
            }
            and self._step == 2
            and self._scheduler_q0_residuals
        ):
            raise RuntimeError("q2 did not consume every cached q0 layer residual")
        if (
            self.dynamic_frame_selection
            == "response_call_q50_nested_mod10_lagged_batch"
            and self._temporal_history_capture_active
        ):
            if self._call_current_temporal is None:
                raise RuntimeError("nested q0 response capture lacks frame geometry")
            finalize = getattr(
                self.geometry_provider,
                "finalize_lagged_call_period_route",
                None,
            )
            if not callable(finalize):
                raise RuntimeError("nested call-period provider lost its finalizer")
            record["temporal_history_batch_finalize"] = finalize(
                current_temporal=self._call_current_temporal,
                layer_count=getattr(self, "_num_blocks", 30),
                source_action_signature=self._current_action_signature,
                causal_quantile=0.50,
                minimum_history=2,
                base_period=5,
                low_response_period=10,
            )
        elif (
            self.dynamic_frame_selection
            in {
                "response_same_layer_q25_thin1_lagged_batch",
                "response_same_layer_q25_structural_collapse_lagged_batch",
            }
            and self._temporal_history_capture_active
        ):
            if self._call_current_temporal is None:
                raise RuntimeError("lagged q0 response capture lacks frame geometry")
            finalize = getattr(
                self.geometry_provider,
                "finalize_lagged_temporal_history_call",
                None,
            )
            if not callable(finalize):
                raise RuntimeError("lagged temporal-history provider lost its finalizer")
            record["temporal_history_batch_finalize"] = finalize(
                current_temporal=self._call_current_temporal,
                layer_count=getattr(self, "_num_blocks", 30),
                causal_quantile=0.25,
                minimum_history=2,
            )
        elif self.dynamic_frame_selection in {
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
            "response_call_q50_nested_mod10_lagged_batch",
        }:
            record["temporal_history_batch_finalize"] = None
        self._records.append(record)
        self._active = False
        if not self.runtime_optimized:
            self._layout_cache = {}
        if self.dynamic_frame_selection in {
            "response_same_layer_q25_thin1_gated",
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
            "response_call_q50_nested_mod10_lagged_batch",
        }:
            self.geometry_provider.set_temporal_history_capture(False)
            self._temporal_history_capture_active = False
        if self.runtime_gate_unused_control_response:
            self.geometry_provider.set_fine_frame_response_capture(False)
            self._fine_frame_response_capture_active = False
        return prediction

    def _current_curvature(self, current_temporal: int) -> torch.Tensor | None:
        if current_temporal <= 0:
            return None
        if self._curvature_profile is None:
            try:
                profile = self.geometry_provider.current_block_curvature_profile()
            except RuntimeError:
                return None
            curvature = profile.get("curvature")
            if not isinstance(curvature, torch.Tensor) or not curvature.numel():
                return None
            self._curvature_profile = {
                **profile,
                "curvature": curvature.detach().float(),
            }
        profile = self._curvature_profile
        if profile is None:
            return None
        frame_curvature = profile.get("frame_curvature")
        if isinstance(frame_curvature, torch.Tensor):
            if int(frame_curvature.numel()) != current_temporal:
                raise RuntimeError("feature curvature does not match Current frames")
            return frame_curvature
        tt, th, tw = (int(value) for value in profile["block_shape"])
        token_h, token_w = (int(value) for value in profile["token_hw"])
        spatial_blocks = math.ceil(token_h / th) * math.ceil(token_w / tw)
        values = profile["curvature"]
        if values.numel() % spatial_blocks:
            raise RuntimeError("CWCA curvature cannot be grouped by temporal cell")
        temporal_cells = values.reshape(-1, spatial_blocks).mean(dim=1)
        frame_values = torch.empty(
            current_temporal, device=values.device, dtype=torch.float32
        )
        for frame in range(current_temporal):
            cell = min(frame // tt, len(temporal_cells) - 1)
            frame_values[frame] = temporal_cells[cell]
        return frame_values

    def _dynamic_response_active_frames(
        self,
        *,
        block_index: int,
        total_temporal: int,
        memory_temporal: int,
        current_temporal: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """Select exact Current cells from the previous layer's control response.

        C0/C13 remain structural worldline anchors.  Every other exact frame is
        selected from the native Closed-Loop Current response cells, with at
        most one representative per cell so two identical tile scores cannot
        collapse the support into adjacent frames.  The fixed variant uses a
        constant number of response cells.  The quota variant rounds the
        response participation ratio, so concentrated response spends fewer
        exact frames and diffuse response spends more, without a fitted
        threshold.
        """

        provider = getattr(
            self.geometry_provider, "previous_layer_frame_selection_signal", None
        )
        if provider is None:
            raise RuntimeError(
                "dynamic frame selection requires a Closed-Loop response provider"
            )
        payload = self._profile_call(
            "dynamic_response_signal",
            block_index,
            provider,
            layer_index=int(block_index),
            current_temporal=int(current_temporal),
        )
        scores = payload.get("cell_scores")
        if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or not scores.numel():
            raise RuntimeError("dynamic frame selection received no response cells")
        scores = scores.detach().float().to(device=device)
        if not bool(torch.isfinite(scores).all()) or bool(torch.any(scores <= 0)):
            raise RuntimeError("dynamic frame-selection scores are invalid")
        offset = int(payload["current_temporal_offset"])
        group = int(payload["source_temporal_group_size"])
        if offset < 0 or group < 1:
            raise RuntimeError("dynamic frame-selection geometry is invalid")

        representatives: list[int] = []
        representative_cells: list[int] = []
        for cell in range(int(scores.numel())):
            low = max(1, offset + cell * group)
            high = min(current_temporal - 1, offset + (cell + 1) * group)
            if low >= high:
                continue
            # Lower chronological median: deterministic, interior, and never
            # duplicates the two structural Current endpoints.
            representatives.append((low + high - 1) // 2)
            representative_cells.append(cell)
        if not representatives:
            raise RuntimeError("Closed-Loop cells contain no interior Current frame")
        candidate_scores = scores[
            torch.tensor(representative_cells, device=device, dtype=torch.long)
        ]

        score_sum = candidate_scores.sum()
        effective_support = score_sum.square() / candidate_scores.square().sum().clamp_min(
            torch.finfo(candidate_scores.dtype).eps
        )
        if self.dynamic_frame_selection == "response_topk_fixed":
            quota = self.dynamic_exact_cell_budget
        elif self.dynamic_frame_selection == "response_effective_quota":
            quota = max(
                self.dynamic_exact_cell_minimum,
                int(torch.round(effective_support).item()),
            )
        else:
            raise RuntimeError("dynamic response selector was called while disabled")
        quota = min(quota, len(representatives))
        if quota < 1:
            raise RuntimeError("dynamic response selector produced an empty quota")

        order = torch.argsort(candidate_scores, descending=True, stable=True)
        selected_positions = order[:quota].tolist()
        selected_cells = [representative_cells[int(index)] for index in selected_positions]
        selected_current = [representatives[int(index)] for index in selected_positions]
        exact_current = torch.tensor(
            sorted({0, current_temporal - 1, *selected_current}),
            device=device,
            dtype=torch.long,
        )
        memory = torch.arange(memory_temporal, device=device, dtype=torch.long)
        active = torch.cat([memory, exact_current + memory_temporal]).long()
        self._last_dynamic_frame_selection = {
            "policy": self.dynamic_frame_selection,
            "source": str(payload["source"]),
            "layer": int(block_index),
            "previous_layer": int(payload["previous_layer"]),
            "native_current_temporal_offset": offset,
            "source_temporal_group_size": group,
            "candidate_cells": representative_cells,
            "candidate_representative_frames": representatives,
            "cell_scores": [float(value) for value in scores.tolist()],
            "effective_support": float(effective_support.item()),
            "quota": int(quota),
            "selected_cells": selected_cells,
            "selected_current_frames": selected_current,
            "structural_current_anchors": [0, current_temporal - 1],
            "exact_current_frames": [int(value) for value in exact_current.tolist()],
            "causal_previous_layer_only": True,
            "one_representative_per_native_response_cell": True,
            "benchmark_metric_read": False,
        }
        return active, len(selected_current)

    def _response_credit_active_frames(
        self,
        *,
        block_index: int,
        memory_temporal: int,
        current_temporal: int,
        canonical_current_active: torch.Tensor,
        protected_current: torch.Tensor,
        period: int,
        active_phases: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Replace mod5 phase IDs with causal response-credit scheduling.

        The canonical phase mask is used only as a per-layer cardinality
        budget.  Structural/curvature anchors remain exact and are excluded
        from the credit process.  For ``B`` dynamic selections among ``N``
        eligible frames, every frame first accrues ``B/N`` credit and each
        selected frame then pays one unit.  Thus an unselected frame gains
        ``B/N``, a selected frame loses ``1-B/N``, and total credit is exactly
        conserved.  Previous-layer control response is added to this credit
        before deterministic Top-B selection.
        """

        if int(period) < 2 or not 1 <= int(active_phases) < int(period):
            raise RuntimeError(
                "response-credit selection requires a nontrivial phase budget"
            )
        if canonical_current_active.dtype != torch.bool or tuple(
            canonical_current_active.shape
        ) != (current_temporal,):
            raise RuntimeError("response-credit selector received invalid mod5 mask")
        if protected_current.dtype != torch.bool or tuple(
            protected_current.shape
        ) != (current_temporal,):
            raise RuntimeError("response-credit selector received invalid anchors")
        if self.runtime_optimized:
            torch._assert_async(
                torch.all(canonical_current_active[protected_current]),
                "canonical mod5 mask lost a protected anchor",
            )
        elif not bool(torch.all(canonical_current_active[protected_current])):
            raise RuntimeError("canonical mod5 mask lost a protected anchor")

        provider = getattr(
            self.geometry_provider, "previous_layer_fair_frame_signal", None
        )
        if not callable(provider):
            raise RuntimeError(
                "response-credit selection requires a fine Closed-Loop provider"
            )
        payload = self._profile_call(
            "dynamic_response_signal",
            block_index,
            provider,
            layer_index=int(block_index),
            current_temporal=int(current_temporal),
        )
        raw_response = payload.get("frame_response")
        normalized_response = payload.get("normalized_frame_response")
        if not isinstance(raw_response, torch.Tensor) or not isinstance(
            normalized_response, torch.Tensor
        ):
            raise RuntimeError("response-credit selector received no frame response")
        raw_response = raw_response.detach().float().to(device=device)
        normalized_response = normalized_response.detach().float().to(device=device)
        if tuple(raw_response.shape) != (current_temporal,) or tuple(
            normalized_response.shape
        ) != (current_temporal,):
            raise RuntimeError("response-credit frame-response shape is invalid")
        if self.runtime_optimized:
            torch._assert_async(
                torch.all(
                    torch.isfinite(raw_response)
                    & torch.isfinite(normalized_response)
                    & (raw_response >= 0)
                    & (normalized_response >= 0)
                ),
                "response-credit frame response is invalid",
            )
        elif (
            not bool(torch.isfinite(raw_response).all())
            or not bool(torch.isfinite(normalized_response).all())
            or bool(torch.any(raw_response < 0))
            or bool(torch.any(normalized_response < 0))
        ):
            raise RuntimeError("response-credit frame response is invalid")

        eligible = ~protected_current
        eligible_frames = torch.nonzero(eligible).flatten()
        protected_count = int(protected_current.sum().item())
        target_exact = int(canonical_current_active.sum().item())
        dynamic_budget = target_exact - protected_count
        eligible_count = int(eligible_frames.numel())
        if not 0 <= dynamic_budget <= eligible_count:
            raise RuntimeError("response-credit dynamic budget is infeasible")
        if dynamic_budget == 0:
            raise RuntimeError("response-credit mod5 budget selected no dynamic frame")

        if self._response_selection_credit is None:
            self._response_selection_credit = torch.zeros(
                current_temporal, device=device, dtype=torch.float32
            )
        credit_before = self._response_selection_credit
        if tuple(credit_before.shape) != (current_temporal,):
            raise RuntimeError("response-credit state changed Current geometry")
        if credit_before.device != device:
            raise RuntimeError("response-credit state changed device within a call")
        credit_before = credit_before.detach().clone()
        credit_before[protected_current] = 0.0

        priority = normalized_response + credit_before
        candidate_priority = priority.index_select(0, eligible_frames)
        order = torch.argsort(candidate_priority, descending=True, stable=True)
        selected_dynamic = eligible_frames.index_select(0, order[:dynamic_budget])
        selected_mask = torch.zeros_like(protected_current)
        selected_mask[selected_dynamic] = True
        exact_mask = protected_current | selected_mask
        exact_current = torch.nonzero(exact_mask).flatten()
        if int(exact_current.numel()) != target_exact:
            raise RuntimeError("response-credit selection changed the mod5 cardinality")

        target_frequency = float(dynamic_budget) / float(eligible_count)
        if self.dynamic_frame_selection == "response_credit_std_scaled_mod5":
            credit_step_scale = float(
                normalized_response[eligible].std(unbiased=False).item()
            )
            credit_scale_source = (
                "eligible_normalized_previous_layer_response_population_std"
            )
        elif self.dynamic_frame_selection == "response_credit_fair_mod5":
            credit_step_scale = 1.0
            credit_scale_source = "unit_mass_conserving"
        else:
            raise RuntimeError("response-credit selector was called while disabled")
        if not math.isfinite(credit_step_scale) or credit_step_scale < 0.0:
            raise RuntimeError("response-credit update scale is invalid")
        credit_after = credit_before.clone()
        credit_after[eligible] += credit_step_scale * target_frequency
        credit_after[selected_dynamic] -= credit_step_scale
        credit_after[protected_current] = 0.0
        credit_delta = credit_after - credit_before
        if self.runtime_optimized:
            before_sum, after_sum = (
                float(value)
                for value in torch.stack(
                    [
                        credit_before[eligible].sum(),
                        credit_after[eligible].sum(),
                    ]
                ).tolist()
            )
        else:
            before_sum = float(credit_before[eligible].sum().item())
            after_sum = float(credit_after[eligible].sum().item())
        tolerance = 16.0 * torch.finfo(torch.float32).eps * max(1, eligible_count)
        if abs(after_sum - before_sum) > tolerance:
            raise RuntimeError("response-credit update failed sum conservation")
        expected_selected_delta = -credit_step_scale * (1.0 - target_frequency)
        expected_unselected_delta = credit_step_scale * target_frequency
        selected_delta_valid = torch.all(
            torch.abs(
                credit_delta[selected_dynamic] - expected_selected_delta
            )
            <= tolerance
        )
        if self.runtime_optimized:
            torch._assert_async(
                selected_delta_valid,
                "selected response-credit decrement is invalid",
            )
        elif not bool(selected_delta_valid):
            raise RuntimeError("selected response-credit decrement is invalid")
        unselected = eligible & ~selected_mask
        unselected_delta_valid = torch.all(
            torch.abs(credit_delta[unselected] - expected_unselected_delta)
            <= tolerance
        )
        if self.runtime_optimized:
            torch._assert_async(
                unselected_delta_valid,
                "unselected response-credit increment is invalid",
            )
        elif not bool(unselected_delta_valid):
            raise RuntimeError("unselected response-credit increment is invalid")
        self._response_selection_credit = credit_after.detach()

        canonical_exact = torch.nonzero(canonical_current_active).flatten()
        protected_exact = torch.nonzero(protected_current).flatten()
        camera_guard_budget = bool(
            self._camera_guard_exact
            and self._step == 0
            and self.camera_guard_weave_period is not None
            and int(period) == int(self.camera_guard_weave_period)
            and int(active_phases) == self.camera_guard_weave_active_phases
        )
        canonical_mod5_budget = bool(
            int(period) == 5 and int(active_phases) == 1
        )
        self._last_dynamic_frame_selection = {
            "policy": self.dynamic_frame_selection,
            "source": str(payload["source"]),
            "response_normalization": str(payload["normalization"]),
            "layer": int(block_index),
            "previous_layer": int(payload["previous_layer"]),
            "previous_has_action_module": bool(
                payload.get("previous_has_action_module")
            ),
            "frame_response": [float(value) for value in raw_response.tolist()],
            "normalized_frame_response": [
                float(value) for value in normalized_response.tolist()
            ],
            "credit_before": [float(value) for value in credit_before.tolist()],
            "priority_scores": [float(value) for value in priority.tolist()],
            "credit_after": [float(value) for value in credit_after.tolist()],
            "credit_delta": [float(value) for value in credit_delta.tolist()],
            "eligible_current_frames": [
                int(value) for value in eligible_frames.tolist()
            ],
            "protected_current_frames": [
                int(value) for value in protected_exact.tolist()
            ],
            "canonical_exact_current_frames": [
                int(value) for value in canonical_exact.tolist()
            ],
            "selected_dynamic_current_frames": [
                int(value) for value in selected_dynamic.tolist()
            ],
            "selected_current_frames": [
                int(value) for value in exact_current.tolist()
            ],
            "exact_current_frames": [int(value) for value in exact_current.tolist()],
            "structural_current_anchors": [0, current_temporal - 1],
            "quota": int(exact_current.numel()),
            "dynamic_quota": dynamic_budget,
            "eligible_count": eligible_count,
            "target_selection_frequency": target_frequency,
            "credit_step_scale": credit_step_scale,
            "credit_scale_source": credit_scale_source,
            "selected_credit_delta": expected_selected_delta,
            "unselected_credit_delta": expected_unselected_delta,
            "credit_sum_before": before_sum,
            "credit_sum_after": after_sum,
            "credit_sum_conserved": True,
            "cardinality_period": int(period),
            "cardinality_active_phases": int(active_phases),
            "phase_target_exact_count": target_exact,
            "phase_cardinality_matched": True,
            "phase_used_for_cardinality_only": True,
            "camera_guard_budget": camera_guard_budget,
            "mod5_target_exact_count": (
                target_exact if canonical_mod5_budget else None
            ),
            "mod5_cardinality_matched": canonical_mod5_budget,
            "mod5_phase_used_for_cardinality_only": canonical_mod5_budget,
            "anchors_excluded_from_credit": True,
            "curvature_and_structural_anchors_preserved": True,
            "causal_previous_layer_only": True,
            "one_representative_per_native_response_cell": False,
            "deterministic_topk": True,
            "benchmark_metric_read": False,
        }
        memory = torch.arange(memory_temporal, device=device, dtype=torch.long)
        return torch.cat([memory, exact_current + memory_temporal]).long()

    def _response_thin_canonical_active_frames(
        self,
        *,
        block_index: int,
        memory_temporal: int,
        current_temporal: int,
        canonical_current_active: torch.Tensor,
        protected_current: torch.Tensor,
        current_has_action_module: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Causally remove low-response canonical phase-only frames.

        Unlike the coarse response-cell candidates, this policy starts from
        the canonical mod5 phase and always keeps endpoints plus the curvature
        anchor intact.  When the prior control-response signal lies below the
        lower quartile of its causal history, conservative variants remove the
        lowest-response phase-only frame; structural collapse removes every
        phase-only frame while retaining the protected structural set.
        """

        structural_collapse = (
            self.dynamic_frame_selection
            == "response_same_layer_q25_structural_collapse_lagged_batch"
        )
        lagged_batch = self.dynamic_frame_selection in {
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
        }
        provider_name = (
            "previous_layer_lagged_temporal_history_signal"
            if lagged_batch
            else "previous_layer_temporal_history_signal"
            if self.dynamic_frame_selection
            in {
                "response_same_layer_q25_thin1",
                "response_same_layer_q25_thin1_gated",
            }
            else "previous_layer_frame_thinning_signal"
        )
        provider = getattr(self.geometry_provider, provider_name, None)
        if provider is None:
            raise RuntimeError(
                "response thinning requires a compatible Closed-Loop response provider"
            )
        payload = provider(
            layer_index=int(block_index),
            current_temporal=int(current_temporal),
            current_has_action_module=bool(current_has_action_module),
            causal_quantile=0.25,
            minimum_history=2,
        )
        scores = payload.get("frame_scores")
        if (
            not isinstance(scores, torch.Tensor)
            or scores.ndim != 1
            or int(scores.numel()) not in {0, current_temporal}
        ):
            raise RuntimeError("response thinning received an invalid frame profile")
        scores = scores.detach().float()
        if lagged_batch:
            if scores.device.type != "cpu":
                raise RuntimeError("lagged response decisions must stay on CPU")
        else:
            scores = scores.to(device=device)
        if not bool(torch.isfinite(scores).all()) or bool(torch.any(scores < 0)):
            raise RuntimeError("response-thinning frame scores are invalid")
        canonical_current_active = canonical_current_active.clone()
        if canonical_current_active.dtype != torch.bool or tuple(
            canonical_current_active.shape
        ) != (current_temporal,):
            raise RuntimeError("canonical Current mask is invalid")
        if protected_current.dtype != torch.bool or tuple(
            protected_current.shape
        ) != (current_temporal,):
            raise RuntimeError("protected Current mask is invalid")
        candidates = torch.nonzero(
            canonical_current_active & ~protected_current
        ).flatten()
        dropped = torch.empty(0, device=device, dtype=torch.long)
        if bool(payload.get("should_thin")) and int(candidates.numel()):
            if int(scores.numel()) != current_temporal:
                raise RuntimeError("thin decision lacks fine frame scores")
            if structural_collapse:
                dropped = candidates.clone()
            elif lagged_batch:
                candidate_frames = [int(value) for value in candidates.tolist()]
                lowest_frame = min(
                    candidate_frames,
                    key=lambda frame: (float(scores[frame].item()), frame),
                )
                dropped = torch.tensor(
                    [lowest_frame], device=device, dtype=torch.long
                )
            else:
                lowest = torch.argmin(scores.index_select(0, candidates))
                dropped = candidates[lowest : lowest + 1]
            canonical_current_active[dropped] = False

        canonical_exact = torch.nonzero(
            canonical_current_active
            | torch.zeros_like(canonical_current_active)
        ).flatten()
        # Reconstruct the pre-thinning set for the trace without relying on a
        # mutable alias after the one-frame deletion.
        pre_thin_exact = torch.sort(
            torch.cat([canonical_exact, dropped]).unique()
        ).values
        final_exact = torch.nonzero(canonical_current_active).flatten()
        protected_exact = torch.nonzero(protected_current).flatten()
        threshold = payload.get("causal_threshold")
        previous_energy = payload.get("previous_energy")
        self._last_dynamic_frame_selection = {
            "policy": self.dynamic_frame_selection,
            "source": str(payload["source"]),
            "layer": int(block_index),
            "previous_layer": int(payload["previous_layer"]),
            "frame_scores": [float(value) for value in scores.tolist()],
            "previous_response_energy": (
                None
                if previous_energy is None
                else float(previous_energy.item())
                if isinstance(previous_energy, torch.Tensor)
                else float(previous_energy)
            ),
            "causal_history_layers": [
                int(value) for value in payload["causal_history_layers"]
            ],
            "history_axis": str(payload["history_axis"]),
            "source_eligible_call_index": payload.get(
                "source_eligible_call_index"
            ),
            "selection_eligible_call_index": payload.get(
                "selection_eligible_call_index"
            ),
            "selection_lag_eligible_calls": payload.get(
                "selection_lag_eligible_calls"
            ),
            "payload_ready": payload.get("payload_ready"),
            "batched_host_decision": payload.get("batched_host_decision"),
            "causal_history_count": int(payload["causal_history_count"]),
            "causal_history_quantile": float(payload["causal_history_quantile"]),
            "causal_threshold": (
                None
                if threshold is None
                else float(threshold.item())
                if isinstance(threshold, torch.Tensor)
                else float(threshold)
            ),
            "minimum_history": int(payload["minimum_history"]),
            "previous_has_action_module": bool(
                payload["previous_has_action_module"]
            ),
            "current_has_action_module": bool(
                payload["current_has_action_module"]
            ),
            "action_safe": bool(payload["action_safe"]),
            "thin_decision_requested": bool(payload["should_thin"]),
            "thin_applied": bool(dropped.numel()),
            "canonical_exact_current_frames": [
                int(value) for value in pre_thin_exact.tolist()
            ],
            "protected_current_frames": [
                int(value) for value in protected_exact.tolist()
            ],
            "drop_candidates": [int(value) for value in candidates.tolist()],
            "dropped_current_frames": [int(value) for value in dropped.tolist()],
            "selected_current_frames": [
                int(value) for value in final_exact.tolist()
            ],
            "exact_current_frames": [int(value) for value in final_exact.tolist()],
            "structural_current_anchors": [0, current_temporal - 1],
            "quota": int(final_exact.numel()),
            "causal_previous_layer_only": True,
            "one_representative_per_native_response_cell": False,
            "canonical_phase_preserved_except_lowest_response_drop": (
                not structural_collapse
            ),
            "canonical_phase_preserved_except_response_gated_phase_only_drop": True,
            "phase_only_drop_rule": (
                "all_when_below_q25"
                if structural_collapse
                else "lowest_response_one_when_below_q25"
            ),
            "maximum_dropped_frames": int(candidates.numel())
            if structural_collapse
            else 1,
            "curvature_and_structural_anchors_preserved": bool(
                torch.all(canonical_current_active[protected_current])
            ),
            "benchmark_metric_read": False,
        }
        memory = torch.arange(memory_temporal, device=device, dtype=torch.long)
        current = final_exact + memory_temporal
        return torch.cat([memory, current]).long()

    def _response_nested_period_active_frames(
        self,
        *,
        block_index: int,
        memory_temporal: int,
        current_temporal: int,
        protected_current: torch.Tensor,
        base_period: int,
        active_phases: int,
        phase_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Route the whole q0 stack between nested mod5 and mod10 lattices."""

        if base_period != 5 or active_phases != 1:
            raise RuntimeError("nested response routing requires canonical mod5")
        provider = getattr(
            self.geometry_provider,
            "previous_call_lagged_period_signal",
            None,
        )
        if not callable(provider):
            raise RuntimeError("nested response routing lost its Closed-Loop provider")
        payload = provider(
            current_temporal=current_temporal,
            current_action_signature=self._current_action_signature,
            causal_quantile=0.50,
            minimum_history=2,
            base_period=5,
            low_response_period=10,
        )
        if protected_current.dtype != torch.bool or tuple(
            protected_current.shape
        ) != (current_temporal,):
            raise RuntimeError("nested response routing received invalid anchors")
        selected_period = 10 if bool(payload.get("should_route")) else 5
        local = torch.arange(current_temporal, device=device)
        canonical_phase = (
            (local + block_index - 1 + int(phase_offset)).remainder(5) < 1
        )
        selected_phase = (
            (local + block_index - 1 + int(phase_offset)).remainder(
                selected_period
            )
            < 1
        )
        if selected_period == 10 and bool(torch.any(selected_phase & ~canonical_phase)):
            raise RuntimeError("mod10 route is not a strict mod5 sub-lattice")
        canonical_current = protected_current | canonical_phase
        selected_current = protected_current | selected_phase
        dropped = torch.nonzero(canonical_current & ~selected_current).flatten()
        added = torch.nonzero(selected_current & ~canonical_current).flatten()
        if int(added.numel()):
            raise RuntimeError("nested response routing introduced a phase frame")
        response_energy = payload.get("call_response_energy")
        threshold = payload.get("causal_threshold")
        source_signature = payload.get("source_action_signature")
        current_signature = payload.get("current_action_signature")
        canonical_exact = torch.nonzero(canonical_current).flatten()
        final_exact = torch.nonzero(selected_current).flatten()
        protected_exact = torch.nonzero(protected_current).flatten()
        self._last_dynamic_frame_selection = {
            "policy": self.dynamic_frame_selection,
            "source": str(payload["source"]),
            "layer": int(block_index),
            "call_response_energy": (
                None if response_energy is None else float(response_energy)
            ),
            "causal_history_count": int(payload["causal_history_count"]),
            "causal_history_quantile": float(payload["causal_history_quantile"]),
            "causal_threshold": (
                None if threshold is None else float(threshold)
            ),
            "minimum_history": int(payload["minimum_history"]),
            "history_axis": str(payload["history_axis"]),
            "source_eligible_call_index": payload.get(
                "source_eligible_call_index"
            ),
            "selection_eligible_call_index": payload.get(
                "selection_eligible_call_index"
            ),
            "selection_lag_eligible_calls": payload.get(
                "selection_lag_eligible_calls"
            ),
            "payload_ready": bool(payload.get("payload_ready")),
            "batched_host_decision": bool(
                payload.get("batched_host_decision")
            ),
            "response_below_threshold": bool(
                payload.get("response_below_threshold")
            ),
            "source_action_signature": (
                None
                if source_signature is None
                else [float(value) for value in source_signature]
            ),
            "current_action_signature": (
                None
                if current_signature is None
                else [float(value) for value in current_signature]
            ),
            "action_signature_match": bool(
                payload.get("action_signature_match")
            ),
            "route_decision_requested": bool(
                payload.get("response_below_threshold")
            ),
            "route_applied": selected_period == 10,
            "base_period": 5,
            "low_response_period": 10,
            "selected_period": selected_period,
            "phase_offset": int(phase_offset),
            "canonical_exact_current_frames": [
                int(value) for value in canonical_exact.tolist()
            ],
            "protected_current_frames": [
                int(value) for value in protected_exact.tolist()
            ],
            "drop_candidates": [
                int(value)
                for value in torch.nonzero(
                    canonical_phase & ~protected_current
                ).flatten().tolist()
            ],
            "dropped_current_frames": [
                int(value) for value in dropped.tolist()
            ],
            "added_current_frames": [int(value) for value in added.tolist()],
            "selected_current_frames": [
                int(value) for value in final_exact.tolist()
            ],
            "exact_current_frames": [
                int(value) for value in final_exact.tolist()
            ],
            "structural_current_anchors": [0, current_temporal - 1],
            "quota": int(final_exact.numel()),
            "causal_previous_layer_only": False,
            "causal_previous_eligible_call_only": True,
            "nested_phase_subset": True,
            "hardware_layout_family": "mod5_mod10_nested",
            "curvature_and_structural_anchors_preserved": bool(
                torch.all(selected_current[protected_current])
            ),
            "one_representative_per_native_response_cell": False,
            "benchmark_metric_read": False,
        }
        memory = torch.arange(memory_temporal, device=device, dtype=torch.long)
        current = final_exact + memory_temporal
        return torch.cat([memory, current]).long()

    def _active_frames(
        self,
        *,
        block_index: int,
        total_temporal: int,
        memory_temporal: int,
        device: torch.device,
        ignore_window_router: bool = False,
        current_has_action_module: bool = False,
    ) -> tuple[torch.Tensor, int]:
        self._last_curvature_anchor_indices: list[int] = []
        self._last_dynamic_frame_selection = None
        # Chunk zero has no retrieved world state and native Matrix deliberately
        # uses dense attention there.  The first/last DiT layers are exact
        # boundary conditions for every later chunk.
        if (
            memory_temporal <= 0
            or block_index == 0
            or block_index == self._num_blocks - 1
            or (self.sparse_steps is not None and self._step not in self.sparse_steps)
            or (
                self.sparse_layers is not None
                and block_index not in self.sparse_layers
            )
            or (
                self.sparse_layers_by_step
                and block_index
                not in self.sparse_layers_by_step.get(self._step, frozenset())
            )
            or (
                self._camera_guard_exact
                and self._step == 0
                and self.camera_guard_weave_period is not None
                and block_index in self.camera_guard_exact_layers
            )
        ):
            return torch.arange(total_temporal, device=device), 0
        if (
            not ignore_window_router
            and block_index in self._window_forward_exact_layers
        ):
            return torch.arange(total_temporal, device=device), 0
        current_temporal = total_temporal - memory_temporal
        if current_temporal <= 2:
            return torch.arange(total_temporal, device=device), 0

        if (
            self.weave_domain
            in {"camera_guarded_current", "camera_history_dominant_current"}
            and self._camera_guard_exact
            and (
                (self._step == 0 and self.camera_guard_weave_period is None)
                or (
                    self._step == 2
                    and self.camera_guard_q2_weave_period is None
                )
            )
        ):
            return torch.arange(total_temporal, device=device), 0
        if (
            self._step == 2
            and self.camera_guard_q2_weave_period is not None
            and not self._camera_guard_exact
        ):
            return torch.arange(total_temporal, device=device), 0

        if self.dynamic_frame_selection in {
            "response_topk_fixed",
            "response_effective_quota",
        }:
            return self._dynamic_response_active_frames(
                block_index=block_index,
                total_temporal=total_temporal,
                memory_temporal=memory_temporal,
                current_temporal=current_temporal,
                device=device,
            )

        if self.weave_domain == "stationary_dual":
            current_curvature = self._current_curvature(current_temporal)
            if current_curvature is None:
                self._last_stationary_worldline = False
                return torch.arange(total_temporal, device=device), 0
            threshold = math.sqrt(torch.finfo(torch.float32).eps)
            stationary = float(current_curvature.mean().item()) <= threshold
            self._last_stationary_worldline = stationary
            if not stationary:
                return torch.arange(total_temporal, device=device), 0
            if self._step == 2:
                # The terminal scheduler phase keeps every generated frame
                # exact.  Only the persistent R4 sink is evolved exactly; its
                # layer residual lifts the stationary retrieved world state.
                memory = torch.zeros(1, device=device, dtype=torch.long)
                current = torch.arange(
                    memory_temporal, total_temporal, device=device, dtype=torch.long
                )
                return torch.cat([memory, current]), 0
            # q0 encodes a stationary micro-flow: retain the complete R4 state,
            # current endpoints, and one depth-staggered interior phase.
            local = torch.arange(current_temporal, device=device)
            phase = (local + block_index - 1).remainder(self.phase_period)
            keep = phase < self.active_phases
            keep[0] = True
            keep[-1] = True
            memory = torch.arange(memory_temporal, device=device)
            current = torch.nonzero(keep).flatten() + memory_temporal
            return torch.cat([memory, current]).long(), 2

        if self.weave_domain == "r4_memory":
            # R4 orders its state as a persistent sink followed by reciprocal
            # pose-simplex endpoints.  The sink is the global world anchor;
            # rotate one non-sink endpoint through network depth so every
            # retrieved state is refreshed periodically while the complete
            # current rollout remains exact at every layer.
            if memory_temporal <= 1:
                return torch.arange(total_temporal, device=device), 0
            endpoint = 1 + ((block_index - 1) % (memory_temporal - 1))
            memory = torch.tensor([0, endpoint], device=device, dtype=torch.long)
            current = torch.arange(
                memory_temporal, total_temporal, device=device, dtype=torch.long
            )
            return torch.cat([memory, current]).sort().values, 0

        current_curvature = self._current_curvature(current_temporal)
        high = torch.zeros(current_temporal, device=device, dtype=torch.bool)
        if current_curvature is not None and self.high_curvature_fraction > 0.0:
            count = (
                self.high_curvature_anchor_count
                if self.high_curvature_anchor_count is not None
                else max(
                    1,
                    int(math.ceil(current_temporal * self.high_curvature_fraction)),
                )
            )
            if count <= 0:
                selected = torch.empty(0, device=device, dtype=torch.long)
            else:
                scoped_curvature = current_curvature.to(device=device)
                offset = 0
                if self.curvature_anchor_scope == "new_generation":
                    offset = int(self.current_segment_boundary or 0)
                    scoped_curvature = scoped_curvature[offset:]
                if self.curvature_anchor_cell_size == 1:
                    selected = torch.topk(
                        scoped_curvature,
                        k=min(count, int(scoped_curvature.numel())),
                    ).indices + offset
                else:
                    cell_size = self.curvature_anchor_cell_size
                    cell_scores = []
                    representatives = []
                    for start in range(0, int(scoped_curvature.numel()), cell_size):
                        end = min(start + cell_size, int(scoped_curvature.numel()))
                        cell_scores.append(scoped_curvature[start:end].max())
                        representatives.append(offset + start + (end - start - 1) // 2)
                    stacked_scores = torch.stack(cell_scores)
                    chosen_cells = torch.topk(
                        stacked_scores,
                        k=min(count, int(stacked_scores.numel())),
                    ).indices.tolist()
                    selected = torch.tensor(
                        [representatives[int(index)] for index in chosen_cells],
                        device=device,
                        dtype=torch.long,
                    )
            self._last_curvature_anchor_indices = [
                int(index) for index in selected.tolist()
            ]
            high[selected] = True

        # Current endpoints close the local worldline section.  Every other
        # low-curvature frame is omitted in exactly one of three layer phases.
        if self.force_current_endpoints_exact:
            high[0] = True
            high[-1] = True
        if self.current_segment_boundary is not None:
            if self.current_segment_boundary >= current_temporal:
                raise RuntimeError("current segment boundary lies outside Current")
            high[self.current_segment_boundary - 1] = True
            high[self.current_segment_boundary] = True
        local = torch.arange(current_temporal, device=device)
        period, active_phases = self.scheduler_phase_schedule.get(
            self._step, (self.phase_period, self.active_phases)
        )
        if (
            self._camera_guard_exact
            and (
                (self._step == 0 and self.camera_guard_weave_period is not None)
                or (
                    self._step == 2
                    and self.camera_guard_q2_weave_period is not None
                )
            )
        ):
            period = (
                self.camera_guard_weave_period
                if self._step == 0
                else self.camera_guard_q2_weave_period
            )
            active_phases = (
                self.camera_guard_weave_active_phases
                if self._step == 0
                else 1
            )
        phase_offset = self.scheduler_phase_offsets.get(self._step, 0)
        if (
            self.dynamic_frame_selection
            == "response_call_q50_nested_mod10_lagged_batch"
        ):
            if self._step != 0:
                raise RuntimeError("nested response routing is restricted to q0")
            active = self._response_nested_period_active_frames(
                block_index=block_index,
                memory_temporal=memory_temporal,
                current_temporal=current_temporal,
                protected_current=high,
                base_period=int(period),
                active_phases=int(active_phases),
                phase_offset=int(phase_offset),
                device=device,
            )
            return active, int(high.sum().item())
        phase = (local + block_index - 1 + phase_offset).remainder(period)
        current_active = high | (phase < active_phases)
        if self.dynamic_frame_selection in {
            "response_credit_fair_mod5",
            "response_credit_std_scaled_mod5",
        }:
            def select_response_credit_frames() -> tuple[torch.Tensor, int]:
                active = self._response_credit_active_frames(
                    block_index=block_index,
                    memory_temporal=memory_temporal,
                    current_temporal=current_temporal,
                    canonical_current_active=current_active,
                    protected_current=high,
                    period=int(period),
                    active_phases=int(active_phases),
                    device=device,
                )
                return active, int(high.sum().item())

            return self._profile_call(
                "dynamic_selection_total",
                block_index,
                select_response_credit_frames,
            )
        if self.dynamic_frame_selection in {
            "response_causal_q25_thin1",
            "response_same_layer_q25_thin1",
            "response_same_layer_q25_thin1_gated",
            "response_same_layer_q25_thin1_lagged_batch",
            "response_same_layer_q25_structural_collapse_lagged_batch",
        }:
            active = self._response_thin_canonical_active_frames(
                block_index=block_index,
                memory_temporal=memory_temporal,
                current_temporal=current_temporal,
                canonical_current_active=current_active,
                protected_current=high,
                current_has_action_module=current_has_action_module,
                device=device,
            )
            return active, int(high.sum().item())
        memory = torch.arange(memory_temporal, device=device)
        current = torch.nonzero(current_active).flatten() + memory_temporal
        active = torch.cat([memory, current]).long()
        return active, int(high.sum().item())

    def _update_window_router(
        self, block_index: int, probe: dict[str, Any]
    ) -> dict[str, Any]:
        """Route a later layer window using only causally prior chunk scores."""

        targets = self.window_forward_targets.get(int(block_index))
        if targets is None:
            raise ValueError(f"layer {block_index} is not a router observation")
        history = self._window_routing_history[int(block_index)]
        raw_signal = probe.get(self.window_router_metric)
        finite_signal = (
            isinstance(raw_signal, (int, float)) and math.isfinite(float(raw_signal))
        )
        threshold = float(median(history)) if history else None
        # The first observation and any invalid probe fail closed to exact.  A
        # later score is compared only with earlier chunks from this sample;
        # screen8 labels and paired-full errors are never read by the router.
        exact = bool(
            not finite_signal
            or threshold is None
            or float(raw_signal) >= threshold
        )
        if exact:
            self._window_forward_exact_layers.update(targets)
        if finite_signal:
            history.append(float(raw_signal))
        record = {
            "observation_layer": int(block_index),
            "metric": self.window_router_metric,
            "signal": float(raw_signal) if finite_signal else None,
            "threshold": threshold,
            "history_count_before": len(history) - int(finite_signal),
            "decision": "exact" if exact else "woven",
            "target_layers": list(targets),
            "causal_running_median": True,
            "fail_closed": bool(not finite_signal or threshold is None),
        }
        self._window_router_records.append(
            {
                "chunk_index": self._chunk,
                "step_index": self._step,
                **record,
            }
        )
        return record

    @staticmethod
    def _lift_temporal(
        active_values: torch.Tensor,
        active_frames: torch.Tensor,
        total_temporal: int,
        memory_temporal: int,
        current_segment_boundary: int | None = None,
        temporal_interpolation: str = "linear",
        runtime_optimized: bool = False,
        runtime_vectorized: bool = False,
    ) -> torch.Tensor:
        """Piecewise-linear lift inside the current worldline section."""

        batch, active_count, spatial, dim = active_values.shape
        if active_count != int(active_frames.numel()):
            raise RuntimeError("active frame/value cardinality mismatch")
        full = active_values.new_empty(batch, total_temporal, spatial, dim)
        full[:, active_frames] = active_values
        def fill_segment(start: int, end: int) -> None:
            segment_mask = (active_frames >= start) & (active_frames < end)
            segment_frames = active_frames[segment_mask]
            if not int(segment_frames.numel()):
                raise RuntimeError("frame weave segment has no exact anchor")
            segment_values = active_values[:, segment_mask]
            positions = segment_frames - start
            if runtime_optimized and temporal_interpolation == "linear":
                # Materialize the tiny chronological anchor list once.  The
                # original implementation queried CUDA separately for every
                # target frame (any/nonzero/int), introducing several host
                # synchronizations per skipped frame.  This branch changes
                # only host-side bracketing; the tensor arithmetic and write
                # order below are byte-for-byte identical to the linear path.
                anchor_frames = [int(value) for value in segment_frames.tolist()]
                anchor_to_index = {
                    frame: index for index, frame in enumerate(anchor_frames)
                }
                if runtime_vectorized and active_values.dtype == torch.float32:
                    target_frames: list[int] = []
                    left_indices: list[int] = []
                    right_indices: list[int] = []
                    left_weights: list[float] = []
                    right_weights: list[float] = []
                    copy_frames: list[int] = []
                    copy_indices: list[int] = []
                    for frame in range(start, end):
                        if frame in anchor_to_index:
                            continue
                        right_i = bisect_left(anchor_frames, frame)
                        if right_i == 0:
                            copy_frames.append(frame)
                            copy_indices.append(0)
                            continue
                        if right_i == len(anchor_frames):
                            copy_frames.append(frame)
                            copy_indices.append(len(anchor_frames) - 1)
                            continue
                        left_i = right_i - 1
                        left_t = anchor_frames[left_i]
                        right_t = anchor_frames[right_i]
                        weight = float(frame - left_t) / float(
                            right_t - left_t
                        )
                        target_frames.append(frame)
                        left_indices.append(left_i)
                        right_indices.append(right_i)
                        left_weights.append(1.0 - weight)
                        right_weights.append(weight)
                    if copy_frames:
                        full[:, torch.tensor(
                            copy_frames,
                            device=active_frames.device,
                            dtype=torch.long,
                        )] = segment_values.index_select(
                            1,
                            torch.tensor(
                                copy_indices,
                                device=active_frames.device,
                                dtype=torch.long,
                            ),
                        )
                    if target_frames:
                        target_index = torch.tensor(
                            target_frames,
                            device=active_frames.device,
                            dtype=torch.long,
                        )
                        left_index = torch.tensor(
                            left_indices,
                            device=active_frames.device,
                            dtype=torch.long,
                        )
                        right_index = torch.tensor(
                            right_indices,
                            device=active_frames.device,
                            dtype=torch.long,
                        )
                        weight_shape = (1, len(target_frames), 1, 1)
                        left_weight = torch.tensor(
                            left_weights,
                            device=active_values.device,
                            dtype=active_values.dtype,
                        ).reshape(weight_shape)
                        right_weight = torch.tensor(
                            right_weights,
                            device=active_values.device,
                            dtype=active_values.dtype,
                        ).reshape(weight_shape)
                        full[:, target_index] = (
                            left_weight
                            * segment_values.index_select(1, left_index)
                            + right_weight
                            * segment_values.index_select(1, right_index)
                        )
                    return
                for frame in range(start, end):
                    if frame in anchor_to_index:
                        continue
                    right_i = bisect_left(anchor_frames, frame)
                    if right_i == 0:
                        full[:, frame] = segment_values[:, 0]
                    elif right_i == len(anchor_frames):
                        full[:, frame] = segment_values[:, -1]
                    else:
                        left_i = right_i - 1
                        left_t = anchor_frames[left_i]
                        right_t = anchor_frames[right_i]
                        weight = float(frame - left_t) / float(right_t - left_t)
                        full[:, frame] = (
                            (1.0 - weight) * segment_values[:, left_i]
                            + weight * segment_values[:, right_i]
                        )
                return
            second_derivatives: list[torch.Tensor] | None = None
            if temporal_interpolation == "natural_cubic" and len(positions) >= 3:
                knots = [float(value) for value in positions.tolist()]
                values = segment_values.float()
                intervals = [
                    knots[index + 1] - knots[index]
                    for index in range(len(knots) - 1)
                ]
                internal_count = len(knots) - 2
                c_prime: list[float] = []
                d_prime: list[torch.Tensor] = []
                for row in range(internal_count):
                    knot = row + 1
                    lower = intervals[knot - 1] if row > 0 else 0.0
                    diagonal = 2.0 * (
                        intervals[knot - 1] + intervals[knot]
                    )
                    upper = intervals[knot] if row + 1 < internal_count else 0.0
                    rhs = 6.0 * (
                        (values[:, knot + 1] - values[:, knot])
                        / intervals[knot]
                        - (values[:, knot] - values[:, knot - 1])
                        / intervals[knot - 1]
                    )
                    denominator = diagonal - (
                        lower * c_prime[row - 1] if row else 0.0
                    )
                    c_prime.append(upper / denominator)
                    d_prime.append(
                        (rhs - (lower * d_prime[row - 1] if row else 0.0))
                        / denominator
                    )
                internal = [torch.empty(0)] * internal_count
                for row in range(internal_count - 1, -1, -1):
                    internal[row] = d_prime[row] - (
                        c_prime[row] * internal[row + 1]
                        if row + 1 < internal_count
                        else 0.0
                    )
                zero = torch.zeros_like(values[:, 0])
                second_derivatives = [zero, *internal, zero]
            for frame in range(start, end):
                if bool(torch.any(segment_frames == frame)):
                    continue
                position = frame - start
                left = torch.nonzero(positions < position).flatten()
                right = torch.nonzero(positions > position).flatten()
                if not len(left):
                    full[:, frame] = segment_values[:, int(right[0])]
                elif not len(right):
                    full[:, frame] = segment_values[:, int(left[-1])]
                else:
                    left_i, right_i = int(left[-1]), int(right[0])
                    left_t, right_t = int(positions[left_i]), int(positions[right_i])
                    weight = float(position - left_t) / float(right_t - left_t)
                    linear = (
                        (1.0 - weight) * segment_values[:, left_i]
                        + weight * segment_values[:, right_i]
                    )
                    if temporal_interpolation in {
                        "nearest",
                        "dc_linear_detail_nearest",
                    }:
                        nearest_i = (
                            left_i
                            if position - left_t <= right_t - position
                            else right_i
                        )
                        if temporal_interpolation == "nearest":
                            full[:, frame] = segment_values[:, nearest_i]
                        else:
                            left_value = segment_values[:, left_i]
                            right_value = segment_values[:, right_i]
                            dc = (
                                (1.0 - weight)
                                * left_value.mean(dim=1, keepdim=True)
                                + weight
                                * right_value.mean(dim=1, keepdim=True)
                            )
                            nearest_value = segment_values[:, nearest_i]
                            detail = nearest_value - nearest_value.mean(
                                dim=1, keepdim=True
                            )
                            full[:, frame] = dc + detail
                    elif temporal_interpolation == "linear":
                        full[:, frame] = linear
                    elif temporal_interpolation == "natural_cubic":
                        if second_derivatives is None:
                            full[:, frame] = linear
                        else:
                            interval = float(right_t - left_t)
                            left_weight = float(right_t - position) / interval
                            right_weight = float(position - left_t) / interval
                            cubic = (
                                left_weight * segment_values[:, left_i].float()
                                + right_weight * segment_values[:, right_i].float()
                                + (
                                    (left_weight**3 - left_weight)
                                    * second_derivatives[left_i]
                                    + (right_weight**3 - right_weight)
                                    * second_derivatives[right_i]
                                )
                                * (interval**2 / 6.0)
                            )
                            full[:, frame] = cubic.to(dtype=segment_values.dtype)
                    elif temporal_interpolation == "polar_linear":
                        left_value = segment_values[:, left_i]
                        right_value = segment_values[:, right_i]
                        left_radius = torch.linalg.vector_norm(
                            left_value.float(), dim=-1, keepdim=True
                        )
                        right_radius = torch.linalg.vector_norm(
                            right_value.float(), dim=-1, keepdim=True
                        )
                        target_radius = (
                            (1.0 - weight) * left_radius
                            + weight * right_radius
                        )
                        chord_radius = torch.linalg.vector_norm(
                            linear.float(), dim=-1, keepdim=True
                        )
                        epsilon = torch.finfo(torch.float32).eps
                        scale = torch.where(
                            chord_radius > epsilon,
                            target_radius / chord_radius.clamp_min(epsilon),
                            torch.ones_like(chord_radius),
                        )
                        full[:, frame] = (linear.float() * scale).to(
                            dtype=linear.dtype
                        )
                    elif temporal_interpolation in {
                        "spherical_linear",
                        "spherical_chord",
                    }:
                        # Interpolate each token residual on its channel sphere.
                        # Unlike polar_linear (normalized chord / nlerp), this
                        # follows the constant-angular-speed geodesic.  Radius
                        # remains the unique affine interpolation of endpoint
                        # radii, so the only changed object is residual direction.
                        left_value = segment_values[:, left_i].float()
                        right_value = segment_values[:, right_i].float()
                        epsilon = 1e-6
                        left_radius = torch.linalg.vector_norm(
                            left_value, dim=-1, keepdim=True
                        )
                        right_radius = torch.linalg.vector_norm(
                            right_value, dim=-1, keepdim=True
                        )
                        left_unit = left_value / left_radius.clamp_min(epsilon)
                        right_unit = right_value / right_radius.clamp_min(epsilon)
                        cosine = (left_unit * right_unit).sum(
                            dim=-1, keepdim=True
                        ).clamp(-1.0 + epsilon, 1.0 - epsilon)
                        angle = torch.acos(cosine)
                        sine = torch.sin(angle)
                        geodesic = (
                            torch.sin((1.0 - weight) * angle)
                            / sine.clamp_min(epsilon)
                            * left_unit
                            + torch.sin(weight * angle)
                            / sine.clamp_min(epsilon)
                            * right_unit
                        )
                        # Near-coincident directions are better evaluated as a
                        # normalized chord, avoiding acos/sin cancellation.
                        chord = (1.0 - weight) * left_unit + weight * right_unit
                        chord = chord / torch.linalg.vector_norm(
                            chord, dim=-1, keepdim=True
                        ).clamp_min(epsilon)
                        direction = torch.where(angle < 1e-3, chord, geodesic)
                        target_radius = (
                            torch.linalg.vector_norm(
                                linear.float(), dim=-1, keepdim=True
                            )
                            if temporal_interpolation == "spherical_chord"
                            else (1.0 - weight) * left_radius
                            + weight * right_radius
                        )
                        full[:, frame] = (direction * target_radius).to(
                            dtype=segment_values.dtype
                        )
                    elif temporal_interpolation == "spherical_neville_chord":
                        # Three-anchor Riemannian Neville interpolation of the
                        # channel direction, paired with the same local chord
                        # radius as the linear baseline.  This is the spherical
                        # quadratic analogue of Lagrange/Neville interpolation;
                        # all three anchors are selected inside the segment.
                        epsilon = 1e-6

                        def unit(index: int) -> torch.Tensor:
                            value = segment_values[:, index].float()
                            return value / torch.linalg.vector_norm(
                                value, dim=-1, keepdim=True
                            ).clamp_min(epsilon)

                        def sphere_lerp(
                            start_value: torch.Tensor,
                            end_value: torch.Tensor,
                            fraction: float,
                        ) -> torch.Tensor:
                            cosine = (start_value * end_value).sum(
                                dim=-1, keepdim=True
                            ).clamp(-1.0 + epsilon, 1.0 - epsilon)
                            angle = torch.acos(cosine)
                            sine = torch.sin(angle)
                            geodesic = (
                                torch.sin((1.0 - fraction) * angle)
                                / sine.clamp_min(epsilon)
                                * start_value
                                + torch.sin(fraction * angle)
                                / sine.clamp_min(epsilon)
                                * end_value
                            )
                            chord = (
                                (1.0 - fraction) * start_value
                                + fraction * end_value
                            )
                            chord = chord / torch.linalg.vector_norm(
                                chord, dim=-1, keepdim=True
                            ).clamp_min(epsilon)
                            return torch.where(angle < 1e-3, chord, geodesic)

                        if len(positions) < 3:
                            direction = sphere_lerp(
                                unit(left_i), unit(right_i), weight
                            )
                        else:
                            nearest = torch.argsort(
                                torch.abs(positions - position), stable=True
                            )[:3]
                            nearest = nearest[torch.argsort(positions[nearest])]
                            knots = [float(positions[index]) for index in nearest]
                            first, middle, last = (
                                int(index) for index in nearest
                            )
                            q01 = sphere_lerp(
                                unit(first),
                                unit(middle),
                                (float(position) - knots[0])
                                / (knots[1] - knots[0]),
                            )
                            q12 = sphere_lerp(
                                unit(middle),
                                unit(last),
                                (float(position) - knots[1])
                                / (knots[2] - knots[1]),
                            )
                            direction = sphere_lerp(
                                q01,
                                q12,
                                (float(position) - knots[0])
                                / (knots[2] - knots[0]),
                            )
                        chord_radius = torch.linalg.vector_norm(
                            linear.float(), dim=-1, keepdim=True
                        )
                        full[:, frame] = (direction * chord_radius).to(
                            dtype=segment_values.dtype
                        )
                    elif temporal_interpolation == "quadratic" and len(positions) < 3:
                        full[:, frame] = linear
                    elif temporal_interpolation == "quadratic":
                        nearest = torch.argsort(
                            torch.abs(positions - position), stable=True
                        )[:3]
                        knots = positions[nearest].tolist()
                        weights = []
                        for knot_index, knot in enumerate(knots):
                            numerator = 1.0
                            denominator = 1.0
                            for other_index, other in enumerate(knots):
                                if other_index == knot_index:
                                    continue
                                numerator *= float(position - other)
                                denominator *= float(knot - other)
                            weights.append(numerator / denominator)
                        full[:, frame] = sum(
                            weight_value * segment_values[:, int(anchor_index)]
                            for weight_value, anchor_index in zip(weights, nearest.tolist())
                        )
                    else:
                        raise ValueError("unsupported temporal interpolation")

        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid segmented Current boundary")
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)
        return full

    @staticmethod
    def _scheduler_defect_residual_lift(
        q0_residual: torch.Tensor,
        active_q2_residual: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Lift only the same-frame q0→q2 scheduler defect.

        q0 supplies a target-frame-specific nonlinear response at every
        temporal position.  Exact q2 anchors observe the scheduler defect;
        only that defect is interpolated inside each Current segment.  Exact
        anchors are restored bit-for-bit and Memory remains exact.
        """

        if q0_residual.ndim != 4 or active_q2_residual.ndim != 4:
            raise RuntimeError("scheduler-defect residuals must be frame grids")
        if q0_residual.shape[0] != active_q2_residual.shape[0]:
            raise RuntimeError("q0/q2 scheduler-defect batch mismatch")
        if q0_residual.shape[2:] != active_q2_residual.shape[2:]:
            raise RuntimeError("q0/q2 scheduler-defect spatial/channel mismatch")
        if int(active_frames.numel()) != active_q2_residual.shape[1]:
            raise RuntimeError("q2 scheduler-defect anchor/value mismatch")
        total_temporal = int(q0_residual.shape[1])
        if bool((active_frames < 0).any()) or bool(
            (active_frames >= total_temporal).any()
        ):
            raise RuntimeError("q2 scheduler-defect anchor lies outside q0 grid")

        q0_active = q0_residual[:, active_frames]
        active_defect = active_q2_residual - q0_active
        full_defect = MatrixCurvaturePhaseFrameWeave._lift_temporal(
            active_defect,
            active_frames,
            total_temporal,
            memory_temporal,
            current_segment_boundary,
            "linear",
        )
        full_q2 = q0_residual + full_defect
        full_q2[:, active_frames] = active_q2_residual
        return full_q2

    @staticmethod
    def _scheduler_feature_defect_residual_lift(
        q0_residual: torch.Tensor,
        active_q2_residual: torch.Tensor,
        q2_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Use target-q2 feature affinity only for the q0→q2 defect."""

        if q0_residual.shape != q2_input.shape:
            raise RuntimeError("q0 residual/q2 input grid mismatch")
        if int(active_frames.numel()) != active_q2_residual.shape[1]:
            raise RuntimeError("q2 scheduler-feature anchor/value mismatch")
        if q0_residual.shape[0] != active_q2_residual.shape[0] or (
            q0_residual.shape[2:] != active_q2_residual.shape[2:]
        ):
            raise RuntimeError("q0/q2 scheduler-feature residual mismatch")
        active_defect = active_q2_residual - q0_residual[:, active_frames]
        full_defect = MatrixCurvaturePhaseFrameWeave._feature_attention_residual_lift(
            active_defect,
            q2_input,
            active_frames,
            memory_temporal,
            current_segment_boundary,
        )
        full_q2 = q0_residual + full_defect
        full_q2[:, active_frames] = active_q2_residual
        return full_q2

    @staticmethod
    def _scheduler_affine_residual_lift(
        q0_residual: torch.Tensor,
        active_q2_residual: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Fit and interpolate a per-channel q0→q2 affine transport."""

        if q0_residual.ndim != 4 or active_q2_residual.ndim != 4:
            raise RuntimeError("scheduler-affine residuals must be frame grids")
        if int(active_frames.numel()) != active_q2_residual.shape[1]:
            raise RuntimeError("scheduler-affine anchor/value mismatch")
        q0_active = q0_residual[:, active_frames].float()
        q2_active = active_q2_residual.float()
        if q0_active.shape != q2_active.shape:
            raise RuntimeError("q0/q2 scheduler-affine residual mismatch")

        q0_mean = q0_active.mean(dim=2, keepdim=True)
        q2_mean = q2_active.mean(dim=2, keepdim=True)
        q0_centered = q0_active - q0_mean
        q2_centered = q2_active - q2_mean
        eps = torch.finfo(q0_active.dtype).eps
        scale = (q0_centered * q2_centered).mean(
            dim=2, keepdim=True
        ) / q0_centered.square().mean(dim=2, keepdim=True).clamp_min(eps)
        bias = q2_mean - scale * q0_mean

        total_temporal = int(q0_residual.shape[1])
        full_scale = MatrixCurvaturePhaseFrameWeave._lift_temporal(
            scale,
            active_frames,
            total_temporal,
            memory_temporal,
            current_segment_boundary,
            "linear",
        )
        full_bias = MatrixCurvaturePhaseFrameWeave._lift_temporal(
            bias,
            active_frames,
            total_temporal,
            memory_temporal,
            current_segment_boundary,
            "linear",
        )
        full_q2 = (
            full_scale * q0_residual.float() + full_bias
        ).to(q0_residual.dtype)
        full_q2[:, active_frames] = active_q2_residual
        return full_q2

    @staticmethod
    def _scheduler_chord_defect_residual_lift(
        q0_residual: torch.Tensor,
        active_q2_residual: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Interpolate the q0→q2 defect in q0-trajectory coordinates.

        The q0 residual remains the exact per-frame carrier.  For an inactive
        q2 frame, its scalar coordinate is the least-squares projection of the
        target q0 residual onto the chord between the adjacent exact q0
        anchors.  Only the scheduler defect is blended at that coordinate.
        """

        if q0_residual.ndim != 4 or active_q2_residual.ndim != 4:
            raise RuntimeError("scheduler-chord residuals must be frame grids")
        if int(active_frames.numel()) != active_q2_residual.shape[1]:
            raise RuntimeError("scheduler-chord anchor/value mismatch")
        q0_active = q0_residual[:, active_frames]
        if q0_active.shape != active_q2_residual.shape:
            raise RuntimeError("q0/q2 scheduler-chord residual mismatch")
        active_defect = active_q2_residual - q0_active
        full_defect = MatrixCurvaturePhaseFrameWeave._lift_temporal(
            active_defect,
            active_frames,
            int(q0_residual.shape[1]),
            memory_temporal,
            current_segment_boundary,
            "linear",
        )

        def fill_segment(start: int, end: int) -> None:
            mask = (active_frames >= start) & (active_frames < end)
            segment_frames = active_frames[mask]
            segment_defect = active_defect[:, mask]
            if not int(segment_frames.numel()):
                raise RuntimeError("scheduler-chord segment has no exact anchor")
            for frame in range(start, end):
                if bool(torch.any(segment_frames == frame)):
                    continue
                left = torch.nonzero(segment_frames < frame).flatten()
                right = torch.nonzero(segment_frames > frame).flatten()
                if not len(left) or not len(right):
                    continue
                left_i, right_i = int(left[-1]), int(right[0])
                left_frame = int(segment_frames[left_i])
                right_frame = int(segment_frames[right_i])
                q0_left = q0_residual[:, left_frame].float()
                q0_right = q0_residual[:, right_frame].float()
                q0_target = q0_residual[:, frame].float()
                chord = q0_right - q0_left
                numerator = ((q0_target - q0_left) * chord).sum(
                    dim=(1, 2), keepdim=True
                )
                denominator = chord.square().sum(dim=(1, 2), keepdim=True)
                eps = torch.finfo(chord.dtype).eps
                temporal_weight = float(frame - left_frame) / float(
                    right_frame - left_frame
                )
                projected_weight = numerator / denominator.clamp_min(eps)
                fallback = torch.full_like(projected_weight, temporal_weight)
                weight = torch.where(
                    denominator > eps, projected_weight, fallback
                ).clamp_(0.0, 1.0)
                full_defect[:, frame] = (
                    (1.0 - weight) * segment_defect[:, left_i].float()
                    + weight * segment_defect[:, right_i].float()
                ).to(full_defect.dtype)

        total_temporal = int(q0_residual.shape[1])
        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid scheduler-chord Current boundary")
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)

        full_q2 = q0_residual + full_defect
        full_q2[:, active_frames] = active_q2_residual
        return full_q2

    @staticmethod
    def _secant_correct_residual(
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        x_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
        secant_scope: str = "token",
    ) -> torch.Tensor:
        """Apply the minimum rank-one endpoint secant inside each segment."""

        corrected = full_residual.clone()
        total_temporal = int(x_input.shape[1])

        def correct_segment(start: int, end: int) -> None:
            mask = (active_frames >= start) & (active_frames < end)
            segment_frames = active_frames[mask]
            segment_residual = active_residual[:, mask]
            positions = segment_frames - start
            for frame in range(start, end):
                if bool(torch.any(segment_frames == frame)):
                    continue
                position = frame - start
                left = torch.nonzero(positions < position).flatten()
                right = torch.nonzero(positions > position).flatten()
                if not len(left) or not len(right):
                    continue
                left_i, right_i = int(left[-1]), int(right[0])
                left_frame = int(segment_frames[left_i])
                right_frame = int(segment_frames[right_i])
                weight = float(frame - left_frame) / float(
                    right_frame - left_frame
                )
                left_input = x_input[:, left_frame].float()
                right_input = x_input[:, right_frame].float()
                input_chord = right_input - left_input
                input_linear = (1.0 - weight) * left_input + weight * right_input
                input_defect = x_input[:, frame].float() - input_linear
                reduce_dims = (-1,) if secant_scope == "token" else (-2, -1)
                denominator = input_chord.square().sum(
                    dim=reduce_dims, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).eps)
                coefficient = (input_defect * input_chord).sum(
                    dim=reduce_dims, keepdim=True
                ) / denominator
                residual_chord = (
                    segment_residual[:, right_i].float()
                    - segment_residual[:, left_i].float()
                )
                corrected[:, frame] = (
                    full_residual[:, frame].float()
                    + coefficient * residual_chord
                ).to(dtype=full_residual.dtype)

        if memory_temporal:
            correct_segment(0, memory_temporal)
        if current_segment_boundary is None:
            correct_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            correct_segment(memory_temporal, split)
            correct_segment(split, total_temporal)
        return corrected

    @staticmethod
    def _feature_attention_residual_lift(
        active_residual: torch.Tensor,
        x_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Recover inactive residuals by same-site attention over exact frames.

        Every target frame keeps its own current block input as the query.  The
        exact frames inside the same chronological segment provide keys and
        exact same-layer residual values.  Attention is only over the frame
        axis at a fixed spatial token, so no spatial correspondence, temporal
        extrapolation, cross-boundary mixing, or approximate-as-source state is
        introduced.  Cosine logits use the standard sqrt(channel) concentration
        scale; there is no learned or benchmark-fitted temperature.
        """

        batch, total_temporal, spatial, channels = x_input.shape
        if int(active_frames.numel()) != int(active_residual.shape[1]):
            raise RuntimeError("active frame/residual cardinality mismatch")
        full = active_residual.new_empty(
            batch, total_temporal, spatial, active_residual.shape[-1]
        )
        full[:, active_frames] = active_residual
        epsilon = torch.finfo(torch.float32).eps
        logit_scale = math.sqrt(float(channels))

        def fill_segment(start: int, end: int) -> None:
            segment_mask = (active_frames >= start) & (active_frames < end)
            segment_frames = active_frames[segment_mask]
            if not int(segment_frames.numel()):
                raise RuntimeError("feature-attention segment has no exact anchor")
            anchor_inputs = x_input[:, segment_frames].float()
            anchor_values = active_residual[:, segment_mask].float()
            anchor_unit = anchor_inputs / torch.linalg.vector_norm(
                anchor_inputs, dim=-1, keepdim=True
            ).clamp_min(epsilon)
            for frame in range(start, end):
                if bool(torch.any(segment_frames == frame)):
                    continue
                query = x_input[:, frame].float()
                query_unit = query / torch.linalg.vector_norm(
                    query, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                logits = (
                    anchor_unit * query_unit.unsqueeze(1)
                ).sum(dim=-1) * logit_scale
                weights = torch.softmax(logits, dim=1)
                full[:, frame] = (
                    weights.unsqueeze(-1) * anchor_values
                ).sum(dim=1).to(dtype=active_residual.dtype)

        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid feature-attention Current boundary")
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)
        return full

    @staticmethod
    def _feature_barycentric_residual_lift(
        active_residual: torch.Tensor,
        x_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Content-conditioned two-anchor barycentric residual recovery.

        The chronological barycentric weights form a strict local prior.  At
        each spatial site, the target block input supplies a content likelihood
        for the left and right exact inputs.  Their normalized product is the
        unique two-anchor posterior used to combine exact residuals.  This keeps
        local temporal support while allowing the target frame's own feature to
        move the interpolation coordinate; no fitted blend or threshold exists.
        """

        batch, total_temporal, spatial, channels = x_input.shape
        if int(active_frames.numel()) != int(active_residual.shape[1]):
            raise RuntimeError("active frame/residual cardinality mismatch")
        full = active_residual.new_empty(
            batch, total_temporal, spatial, active_residual.shape[-1]
        )
        full[:, active_frames] = active_residual
        epsilon = torch.finfo(torch.float32).eps
        logit_scale = math.sqrt(float(channels))

        def fill_segment(start: int, end: int) -> None:
            segment_mask = (active_frames >= start) & (active_frames < end)
            segment_frames = active_frames[segment_mask]
            if not int(segment_frames.numel()):
                raise RuntimeError("feature-barycentric segment has no exact anchor")
            segment_inputs = x_input[:, segment_frames].float()
            segment_values = active_residual[:, segment_mask].float()
            positions = segment_frames - start
            for frame in range(start, end):
                if bool(torch.any(segment_frames == frame)):
                    continue
                position = frame - start
                left = torch.nonzero(positions < position).flatten()
                right = torch.nonzero(positions > position).flatten()
                if not len(left):
                    full[:, frame] = segment_values[:, int(right[0])]
                    continue
                if not len(right):
                    full[:, frame] = segment_values[:, int(left[-1])]
                    continue
                left_i, right_i = int(left[-1]), int(right[0])
                left_t, right_t = int(positions[left_i]), int(positions[right_i])
                alpha = float(position - left_t) / float(right_t - left_t)
                anchor_inputs = torch.stack(
                    [segment_inputs[:, left_i], segment_inputs[:, right_i]], dim=1
                )
                anchor_values = torch.stack(
                    [segment_values[:, left_i], segment_values[:, right_i]], dim=1
                )
                anchor_unit = anchor_inputs / torch.linalg.vector_norm(
                    anchor_inputs, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                query = x_input[:, frame].float()
                query_unit = query / torch.linalg.vector_norm(
                    query, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                content_logits = (
                    anchor_unit * query_unit.unsqueeze(1)
                ).sum(dim=-1) * logit_scale
                temporal_log_prior = torch.tensor(
                    [math.log1p(-alpha), math.log(alpha)],
                    device=x_input.device,
                    dtype=torch.float32,
                ).view(1, 2, 1)
                weights = torch.softmax(
                    content_logits + temporal_log_prior, dim=1
                )
                full[:, frame] = (
                    weights.unsqueeze(-1) * anchor_values
                ).sum(dim=1).to(dtype=active_residual.dtype)

        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid feature-barycentric Current boundary")
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)
        return full

    @staticmethod
    def _condition_to_latents(
        value: torch.Tensor | None,
        latent_count: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Reduce a frame-rate control sequence to chronological latent cells."""

        if not isinstance(value, torch.Tensor) or latent_count <= 0:
            return None
        condition = value.detach().float().to(device=device)
        if condition.ndim != 3:
            raise RuntimeError("control condition must have shape [B,T,C]")
        if int(condition.shape[1]) == latent_count:
            return condition
        # Matrix controls arrive at decoded-frame rate (40/57 frames), whereas
        # FrameWeave operates on 10/14 Current latent cells.  Area pooling is a
        # fixed chronological reduction and introduces no learned parameters.
        return torch.nn.functional.adaptive_avg_pool1d(
            condition.transpose(1, 2), latent_count
        ).transpose(1, 2)

    def _barycentric_contraction_workspace(
        self, active_residual: torch.Tensor, total_temporal: int
    ) -> _BarycentricContractionWorkspace | None:
        if not self.runtime_preallocated_barycentric:
            return None
        if active_residual.ndim != 4 or active_residual.dtype != torch.float32:
            # The production Matrix hidden path is FP32.  Other dtypes retain
            # the exact reference contraction instead of changing promotion
            # or accumulation semantics.
            return None
        batch, active_count, spatial, channels = active_residual.shape
        maximum_targets = max(0, int(total_temporal) - int(active_count))
        key = (
            str(active_residual.device),
            str(active_residual.dtype),
            int(batch),
            int(spatial),
            int(channels),
            int(maximum_targets),
        )
        workspace = self._barycentric_workspaces.get(key)
        if workspace is None:
            workspace = _BarycentricContractionWorkspace(
                sources=torch.empty(
                    batch,
                    2,
                    spatial,
                    channels,
                    device=active_residual.device,
                    dtype=active_residual.dtype,
                ),
                output=torch.empty(
                    batch,
                    1,
                    spatial * channels,
                    device=active_residual.device,
                    dtype=torch.float32,
                ),
                batched_sources=(
                    torch.empty(
                        batch,
                        2 * maximum_targets,
                        spatial,
                        channels,
                        device=active_residual.device,
                        dtype=active_residual.dtype,
                    )
                    if self.runtime_batched_barycentric_contraction
                    and maximum_targets
                    else None
                ),
                batched_output=(
                    torch.empty(
                        batch * maximum_targets,
                        1,
                        spatial * channels,
                        device=active_residual.device,
                        dtype=torch.float32,
                    )
                    if self.runtime_batched_barycentric_contraction
                    and maximum_targets
                    else None
                ),
                batched_weights=(
                    torch.empty(
                        batch,
                        maximum_targets,
                        2,
                        device=active_residual.device,
                        dtype=torch.float32,
                    )
                    if (
                        self.runtime_batched_barycentric_contraction
                        or self.runtime_direct_barycentric_pair_kernel
                    )
                    and maximum_targets
                    else None
                ),
                source_position_indices={},
                source_frame_indices={},
                temporal_priors={},
                batched_source_position_indices={},
                batched_target_frame_indices={},
                direct_source_position_pairs={},
                direct_target_frame_indices={},
                segment_position_indices={},
            )
            self._barycentric_workspaces[key] = workspace
            self._barycentric_workspace_allocations += 1
        return workspace

    @classmethod
    def _control_space_residual_lift(
        cls,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        total_temporal: int,
        memory_temporal: int,
        current_segment_boundary: int | None,
        full_plucker: torch.Tensor | None,
        mouse_cond: torch.Tensor | None,
        keyboard_cond: torch.Tensor | None,
        mouse_cond_memory: torch.Tensor | None,
        keyboard_cond_memory: torch.Tensor | None,
        lambdas: tuple[float, float, float],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Trajectory-conditioned convex reconstruction of exact residuals.

        The support is every exact frame in the same chronological segment.
        Log weights combine temporal distance, the already-computed camera
        (Plucker) embedding, and action distance.  The operation is training
        free and never uses an approximate frame as a source.
        """

        batch, active_count, spatial, channels = active_residual.shape
        if active_count != int(active_frames.numel()):
            raise RuntimeError("active frame/residual cardinality mismatch")
        if full_plucker is None:
            raise RuntimeError("control-space reconstruction requires camera state")
        if full_plucker.ndim != 3 or int(full_plucker.shape[1]) < total_temporal * spatial:
            raise RuntimeError("camera state does not cover the full temporal grid")
        camera = full_plucker[:, : total_temporal * spatial].detach().float().reshape(
            batch, total_temporal, spatial, -1
        ).mean(dim=2)
        camera = torch.nn.functional.normalize(camera, dim=-1, eps=1e-8)

        current_count = total_temporal - memory_temporal
        current_parts = []
        for value in (mouse_cond, keyboard_cond):
            pooled = cls._condition_to_latents(
                value, current_count, device=active_residual.device
            )
            if pooled is not None:
                current_parts.append(pooled)
        memory_parts = []
        for value in (mouse_cond_memory, keyboard_cond_memory):
            pooled = cls._condition_to_latents(
                value, memory_temporal, device=active_residual.device
            )
            if pooled is not None:
                memory_parts.append(pooled)
        action_available = bool(current_parts)
        action_dim = sum(int(part.shape[-1]) for part in current_parts)
        if action_available:
            current_action = torch.cat(current_parts, dim=-1)
            if memory_temporal:
                if memory_parts:
                    memory_action = torch.cat(memory_parts, dim=-1)
                    if int(memory_action.shape[-1]) != action_dim:
                        raise RuntimeError("Memory/Current action dimensions disagree")
                else:
                    memory_action = current_action.new_zeros(
                        batch, memory_temporal, action_dim
                    )
                action = torch.cat([memory_action, current_action], dim=1)
            else:
                action = current_action
            action = torch.nn.functional.normalize(action, dim=-1, eps=1e-8)
        else:
            action = camera.new_zeros(batch, total_temporal, 1)

        lambda_t, lambda_p, lambda_a = lambdas
        full = active_residual.new_empty(
            batch, total_temporal, spatial, channels
        )
        full[:, active_frames] = active_residual
        target_count = 0
        support_min = total_temporal
        support_max = 0

        def fill_segment(start: int, end: int) -> None:
            nonlocal target_count, support_min, support_max
            mask = (active_frames >= start) & (active_frames < end)
            anchors = active_frames[mask]
            if not int(anchors.numel()):
                raise RuntimeError("control-space segment has no exact anchor")
            values = active_residual[:, mask]
            support_min = min(support_min, int(anchors.numel()))
            support_max = max(support_max, int(anchors.numel()))
            for frame in range(start, end):
                if bool(torch.any(anchors == frame)):
                    continue
                temporal_distance = (anchors - frame).abs().float().view(1, -1)
                camera_distance = 1.0 - (
                    camera[:, frame].unsqueeze(1) * camera[:, anchors]
                ).sum(dim=-1).clamp(-1.0, 1.0)
                action_distance = 1.0 - (
                    action[:, frame].unsqueeze(1) * action[:, anchors]
                ).sum(dim=-1).clamp(-1.0, 1.0)
                logits = -(
                    lambda_t * temporal_distance
                    + lambda_p * camera_distance
                    + lambda_a * action_distance
                )
                weights = torch.softmax(logits, dim=1)
                full[:, frame] = torch.einsum(
                    "ba,bash->bsh", weights, values.float()
                ).to(dtype=active_residual.dtype)
                target_count += 1

        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid control-space Current boundary")
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)
        full[:, active_frames] = active_residual
        return full, {
            "mode": "trajectory_conditioned_exact_residual_convex_sum",
            "lambdas": [lambda_t, lambda_p, lambda_a],
            "camera_source": "mean_encoded_plucker_per_latent",
            "action_source": "chronological_area_pooled_mouse_keyboard",
            "camera_available": True,
            "action_available": action_available,
            "targets": target_count,
            "support_min": support_min,
            "support_max": support_max,
            "same_segment_only": True,
            "exact_anchors_restored": True,
        }

    @classmethod
    def _control_barycentric_residual_lift(
        cls,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        total_temporal: int,
        memory_temporal: int,
        current_segment_boundary: int | None,
        full_plucker: torch.Tensor | None,
        mouse_cond: torch.Tensor | None,
        keyboard_cond: torch.Tensor | None,
        mouse_cond_memory: torch.Tensor | None,
        keyboard_cond_memory: torch.Tensor | None,
        lambdas: tuple[float, float, float],
        runtime_optimized: bool = False,
        runtime_vectorized: bool = False,
        runtime_single_write: bool = False,
        contraction_workspace: _BarycentricContractionWorkspace | None = None,
        control_state_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, bool]
        ]
        | None = None,
        weight_cache: dict[tuple[int, int, int], torch.Tensor] | None = None,
        runtime_batched_contraction: bool = False,
        runtime_direct_pair_kernel: bool = False,
        runtime_active_frame_values: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Control-conditioned two-anchor residual interpolation.

        The chronological linear weights remain a strict local prior. Camera
        and action similarity may only redistribute mass between the nearest
        exact anchors bracketing the target; no remote or approximate frame is
        ever used as a source.
        """

        batch, active_count, spatial, channels = active_residual.shape
        if active_count != int(active_frames.numel()):
            raise RuntimeError("active frame/residual cardinality mismatch")
        if runtime_active_frame_values is not None:
            if len(runtime_active_frame_values) != active_count:
                raise RuntimeError(
                    "reused active-frame list/residual cardinality mismatch"
                )
            if tuple(sorted(set(runtime_active_frame_values))) != tuple(
                runtime_active_frame_values
            ):
                raise RuntimeError(
                    "reused active-frame list must be strictly chronological"
                )
            if (
                runtime_active_frame_values[0] < 0
                or runtime_active_frame_values[-1] >= total_temporal
            ):
                raise RuntimeError("reused active-frame list is out of range")
        if full_plucker is None:
            raise RuntimeError("control-barycentric reconstruction requires camera state")
        if full_plucker.ndim != 3 or int(full_plucker.shape[1]) < total_temporal * spatial:
            raise RuntimeError("camera state does not cover the full temporal grid")
        def tensor_signature(value: torch.Tensor | None) -> tuple[Any, ...] | None:
            if not isinstance(value, torch.Tensor):
                return None
            # Inference tensors deliberately do not allocate a version
            # counter.  Their storage is immutable for this reconstruction
            # call, so a tagged sentinel is the corresponding stable cache
            # identity; ordinary tensors retain the stricter mutation guard.
            version: int | str = (
                "inference"
                if torch.is_inference(value)
                else int(value._version)
            )
            return (
                int(value.data_ptr()),
                version,
                tuple(int(v) for v in value.shape),
                str(value.dtype),
                str(value.device),
            )

        control_cache_key = (
            int(total_temporal),
            int(memory_temporal),
            int(spatial),
            tensor_signature(full_plucker),
            tensor_signature(mouse_cond),
            tensor_signature(keyboard_cond),
            tensor_signature(mouse_cond_memory),
            tensor_signature(keyboard_cond_memory),
        )
        cached_control = (
            control_state_cache.get(control_cache_key)
            if control_state_cache is not None
            else None
        )
        control_cache_hit = cached_control is not None
        if cached_control is not None:
            camera, action, action_available = cached_control
        else:
            camera = (
                full_plucker[:, : total_temporal * spatial]
                .detach()
                .float()
                .reshape(batch, total_temporal, spatial, -1)
                .mean(dim=2)
            )
            camera = torch.nn.functional.normalize(camera, dim=-1, eps=1e-8)

            current_count = total_temporal - memory_temporal
            current_parts = []
            for value in (mouse_cond, keyboard_cond):
                pooled = cls._condition_to_latents(
                    value, current_count, device=active_residual.device
                )
                if pooled is not None:
                    current_parts.append(pooled)
            memory_parts = []
            for value in (mouse_cond_memory, keyboard_cond_memory):
                pooled = cls._condition_to_latents(
                    value, memory_temporal, device=active_residual.device
                )
                if pooled is not None:
                    memory_parts.append(pooled)
            action_available = bool(current_parts)
            action_dim = sum(int(part.shape[-1]) for part in current_parts)
            if action_available:
                current_action = torch.cat(current_parts, dim=-1)
                if memory_temporal:
                    if memory_parts:
                        memory_action = torch.cat(memory_parts, dim=-1)
                        if int(memory_action.shape[-1]) != action_dim:
                            raise RuntimeError(
                                "Memory/Current action dimensions disagree"
                            )
                    else:
                        memory_action = current_action.new_zeros(
                            batch, memory_temporal, action_dim
                        )
                    action = torch.cat([memory_action, current_action], dim=1)
                else:
                    action = current_action
                action = torch.nn.functional.normalize(action, dim=-1, eps=1e-8)
            else:
                action = camera.new_zeros(batch, total_temporal, 1)
            if control_state_cache is not None:
                control_state_cache[control_cache_key] = (
                    camera,
                    action,
                    action_available,
                )

        _, lambda_p, lambda_a = lambdas
        full = active_residual.new_empty(
            batch, total_temporal, spatial, channels
        )
        if not runtime_single_write:
            full[:, active_frames] = active_residual
        target_count = 0
        support_sizes: set[int] = set()
        batched_contraction_calls = 0
        direct_pair_kernel_calls = 0
        weight_cache_hits = 0
        weight_cache_misses = 0

        def fill_segment(start: int, end: int) -> None:
            nonlocal target_count, batched_contraction_calls
            nonlocal direct_pair_kernel_calls
            nonlocal weight_cache_hits, weight_cache_misses
            if runtime_active_frame_values is not None:
                # Preserve the frozen boolean-index gather and its physical
                # layout exactly.  Only replace the subsequent CUDA->host
                # ``anchors.tolist()`` with the compiled CPU tuple.
                mask = (active_frames >= start) & (active_frames < end)
                anchors = active_frames[mask]
                values = active_residual[:, mask]
                anchor_frames = [
                    frame
                    for frame in runtime_active_frame_values
                    if start <= frame < end
                ]
                if not anchor_frames:
                    raise RuntimeError(
                        "control-barycentric segment has no exact anchor"
                    )
            else:
                mask = (active_frames >= start) & (active_frames < end)
                anchors = active_frames[mask]
                values = active_residual[:, mask]
                if not int(anchors.numel()):
                    raise RuntimeError(
                        "control-barycentric segment has no exact anchor"
                    )
                anchor_frames = (
                    [int(value) for value in anchors.tolist()]
                    if runtime_optimized
                    else None
                )
            positions = anchors - start
            anchor_set = set(anchor_frames) if anchor_frames is not None else None
            vectorized_rows: dict[int, int] = {}
            vectorized_camera_distance: torch.Tensor | None = None
            vectorized_action_distance: torch.Tensor | None = None
            vectorized_temporal_prior: torch.Tensor | None = None
            batched_target_frames: list[int] = []
            batched_source_positions: list[int] = []
            batched_weight_rows: list[torch.Tensor] = []
            if runtime_batched_contraction and (
                contraction_workspace is None
                or contraction_workspace.batched_sources is None
                or contraction_workspace.batched_output is None
                or contraction_workspace.batched_weights is None
            ):
                raise RuntimeError(
                    "batched barycentric contraction lacks its preallocated workspace"
                )
            if runtime_direct_pair_kernel and (
                contraction_workspace is None
                or contraction_workspace.batched_weights is None
            ):
                raise RuntimeError(
                    "direct barycentric pair kernel lacks its preallocated workspace"
                )
            if runtime_vectorized:
                if anchor_frames is None:
                    raise RuntimeError(
                        "vectorized barycentric reconstruction requires "
                        "runtime-optimized anchor materialization"
                    )
                target_frames: list[int] = []
                source_frame_pairs: list[list[int]] = []
                temporal_priors: list[list[float]] = []
                for target_frame in range(start, end):
                    if target_frame in anchor_set:
                        continue
                    target_right = bisect_left(anchor_frames, target_frame)
                    if target_right == 0 or target_right == len(anchor_frames):
                        raise RuntimeError(
                            "control-barycentric target is not bracketed by exact anchors"
                        )
                    target_left = target_right - 1
                    target_left_frame = anchor_frames[target_left]
                    target_right_frame = anchor_frames[target_right]
                    if (
                        weight_cache is not None
                        and (
                            target_left_frame,
                            target_frame,
                            target_right_frame,
                        )
                        in weight_cache
                    ):
                        continue
                    target_alpha = float(
                        target_frame - target_left_frame
                    ) / float(target_right_frame - target_left_frame)
                    vectorized_rows[target_frame] = len(target_frames)
                    target_frames.append(target_frame)
                    source_frame_pairs.append(
                        [target_left_frame, target_right_frame]
                    )
                    temporal_priors.append(
                        [math.log1p(-target_alpha), math.log(target_alpha)]
                    )
                if target_frames:
                    target_index = torch.tensor(
                        target_frames,
                        device=active_frames.device,
                        dtype=torch.long,
                    )
                    source_index = torch.tensor(
                        source_frame_pairs,
                        device=active_frames.device,
                        dtype=torch.long,
                    )
                    vectorized_camera_distance = 1.0 - (
                        camera[:, target_index].unsqueeze(2)
                        * camera[:, source_index]
                    ).sum(dim=-1).clamp(-1.0, 1.0)
                    vectorized_action_distance = 1.0 - (
                        action[:, target_index].unsqueeze(2)
                        * action[:, source_index]
                    ).sum(dim=-1).clamp(-1.0, 1.0)
                    vectorized_temporal_prior = torch.tensor(
                        temporal_priors,
                        device=active_frames.device,
                        dtype=torch.float32,
                    )
            for frame in range(start, end):
                if anchor_frames is not None:
                    if frame in anchor_set:
                        continue
                    right_i = bisect_left(anchor_frames, frame)
                    if right_i == 0 or right_i == len(anchor_frames):
                        raise RuntimeError(
                            "control-barycentric target is not bracketed by exact anchors"
                        )
                    left_i = right_i - 1
                    left_frame = anchor_frames[left_i]
                    right_frame = anchor_frames[right_i]
                else:
                    if bool(torch.any(anchors == frame)):
                        continue
                    position = frame - start
                    left = torch.nonzero(positions < position).flatten()
                    right = torch.nonzero(positions > position).flatten()
                    if not len(left) or not len(right):
                        raise RuntimeError(
                            "control-barycentric target is not bracketed by exact anchors"
                        )
                    left_i, right_i = int(left[-1]), int(right[0])
                    left_frame = int(anchors[left_i])
                    right_frame = int(anchors[right_i])
                alpha = float(frame - left_frame) / float(right_frame - left_frame)
                weight_key = (left_frame, frame, right_frame)
                weights = (
                    weight_cache.get(weight_key)
                    if weight_cache is not None
                    else None
                )
                if weights is not None:
                    weight_cache_hits += 1
                elif runtime_vectorized:
                    row = vectorized_rows[frame]
                    if (
                        vectorized_camera_distance is None
                        or vectorized_action_distance is None
                        or vectorized_temporal_prior is None
                    ):
                        raise RuntimeError(
                            "vectorized barycentric distances are incomplete"
                        )
                    camera_distance = vectorized_camera_distance[:, row]
                    action_distance = vectorized_action_distance[:, row]
                    temporal_log_prior = vectorized_temporal_prior[row].view(1, 2)
                elif weights is None:
                    source_frame_key = (left_frame, right_frame)
                    source_frames = (
                        contraction_workspace.source_frame_indices.get(
                            source_frame_key
                        )
                        if contraction_workspace is not None
                        else None
                    )
                    if source_frames is None:
                        source_frames = torch.tensor(
                            [left_frame, right_frame],
                            device=active_frames.device,
                            dtype=torch.long,
                        )
                        if contraction_workspace is not None:
                            contraction_workspace.source_frame_indices[
                                source_frame_key
                            ] = source_frames
                    camera_distance = 1.0 - (
                        camera[:, frame].unsqueeze(1) * camera[:, source_frames]
                    ).sum(dim=-1).clamp(-1.0, 1.0)
                    action_distance = 1.0 - (
                        action[:, frame].unsqueeze(1) * action[:, source_frames]
                    ).sum(dim=-1).clamp(-1.0, 1.0)
                    temporal_key = (left_frame, frame, right_frame)
                    temporal_log_prior = (
                        contraction_workspace.temporal_priors.get(temporal_key)
                        if contraction_workspace is not None
                        else None
                    )
                    if temporal_log_prior is None:
                        temporal_log_prior = torch.tensor(
                            [math.log1p(-alpha), math.log(alpha)],
                            device=active_frames.device,
                            dtype=torch.float32,
                        ).view(1, 2)
                        if contraction_workspace is not None:
                            contraction_workspace.temporal_priors[
                                temporal_key
                            ] = temporal_log_prior
                if weights is None:
                    weights = torch.softmax(
                        temporal_log_prior
                        - lambda_p * camera_distance
                        - lambda_a * action_distance,
                        dim=1,
                    )
                    if weight_cache is not None:
                        weight_cache[weight_key] = weights
                        weight_cache_misses += 1
                if runtime_batched_contraction or runtime_direct_pair_kernel:
                    batched_target_frames.append(frame)
                    batched_source_positions.extend([left_i, right_i])
                    batched_weight_rows.append(weights)
                elif contraction_workspace is None:
                    source_values = torch.stack(
                        [values[:, left_i], values[:, right_i]], dim=1
                    )
                    reconstructed = torch.einsum(
                        "ba,bash->bsh", weights, source_values.float()
                    ).to(dtype=active_residual.dtype)
                else:
                    position_key = (left_i, right_i)
                    source_positions = (
                        contraction_workspace.source_position_indices.get(
                            position_key
                        )
                    )
                    if source_positions is None:
                        source_positions = torch.tensor(
                            [left_i, right_i],
                            device=active_frames.device,
                            dtype=torch.long,
                        )
                        contraction_workspace.source_position_indices[
                            position_key
                        ] = source_positions
                    torch.index_select(
                        values,
                        1,
                        source_positions,
                        out=contraction_workspace.sources,
                    )
                    torch.bmm(
                        weights.unsqueeze(1),
                        contraction_workspace.sources.reshape(batch, 2, -1),
                        out=contraction_workspace.output,
                    )
                    reconstructed = contraction_workspace.output.reshape(
                        batch, spatial, channels
                    )
                if not (
                    runtime_batched_contraction or runtime_direct_pair_kernel
                ):
                    full[:, frame] = reconstructed
                target_count += 1
                support_sizes.add(2)
            if batched_target_frames:
                if contraction_workspace is None:
                    raise RuntimeError("batched contraction workspace disappeared")
                count = len(batched_target_frames)
                source_key = tuple(batched_source_positions)
                target_key = tuple(batched_target_frames)
                batched_weights = contraction_workspace.batched_weights[
                    :, :count
                ]
                torch.stack(batched_weight_rows, dim=1, out=batched_weights)
                if runtime_direct_pair_kernel:
                    source_pairs = (
                        contraction_workspace.direct_source_position_pairs.get(
                            source_key
                        )
                    )
                    if source_pairs is None:
                        source_pairs = torch.tensor(
                            batched_source_positions,
                            device=active_frames.device,
                            dtype=torch.int32,
                        ).reshape(count, 2)
                        contraction_workspace.direct_source_position_pairs[
                            source_key
                        ] = source_pairs
                    direct_targets = (
                        contraction_workspace.direct_target_frame_indices.get(
                            target_key
                        )
                    )
                    if direct_targets is None:
                        direct_targets = torch.tensor(
                            batched_target_frames,
                            device=active_frames.device,
                            dtype=torch.int32,
                        )
                        contraction_workspace.direct_target_frame_indices[
                            target_key
                        ] = direct_targets
                    direct_control_barycentric_pair_write(
                        values,
                        batched_weights,
                        source_pairs,
                        direct_targets,
                        full,
                        block=1024,
                        arithmetic_mode="left_then_right_fma",
                    )
                    direct_pair_kernel_calls += 1
                    return
                source_positions = (
                    contraction_workspace.batched_source_position_indices.get(
                        source_key
                    )
                )
                if source_positions is None:
                    source_positions = torch.tensor(
                        batched_source_positions,
                        device=active_frames.device,
                        dtype=torch.long,
                    )
                    contraction_workspace.batched_source_position_indices[
                        source_key
                    ] = source_positions
                target_frames = (
                    contraction_workspace.batched_target_frame_indices.get(
                        target_key
                    )
                )
                if target_frames is None:
                    target_frames = torch.tensor(
                        batched_target_frames,
                        device=active_frames.device,
                        dtype=torch.long,
                    )
                    contraction_workspace.batched_target_frame_indices[
                        target_key
                    ] = target_frames
                batched_sources = contraction_workspace.batched_sources[
                    :, : 2 * count
                ]
                batched_output = contraction_workspace.batched_output[
                    : batch * count
                ]
                torch.index_select(
                    values, 1, source_positions, out=batched_sources
                )
                torch.bmm(
                    batched_weights.reshape(batch * count, 1, 2),
                    batched_sources.reshape(batch * count, 2, -1),
                    out=batched_output,
                )
                full[:, target_frames] = batched_output.reshape(
                    batch, count, spatial, channels
                )
                batched_contraction_calls += 1

        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid control-barycentric Current boundary")
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)
        full[:, active_frames] = active_residual
        return full, {
            "mode": "control_conditioned_local_two_anchor_residual",
            "control_lambdas": [lambda_p, lambda_a],
            "temporal_prior": "chronological_linear_barycentric",
            "camera_source": "mean_encoded_plucker_per_latent",
            "action_source": "chronological_area_pooled_mouse_keyboard",
            "camera_available": True,
            "action_available": action_available,
            "control_state_cache_hit": control_cache_hit,
            "weight_cache_hits": weight_cache_hits,
            "weight_cache_misses": weight_cache_misses,
            "active_frame_source": (
                "dynamic_selector_cpu_trace"
                if runtime_active_frame_values is not None
                else "cuda_anchor_readback"
                if runtime_optimized
                else "cuda_predicates"
            ),
            "batched_contraction_calls": batched_contraction_calls,
            "direct_pair_kernel_calls": direct_pair_kernel_calls,
            "targets": target_count,
            "support_sizes": sorted(support_sizes),
            "nearest_bracketing_exact_anchors_only": True,
            "same_segment_only": True,
            "exact_anchors_restored": True,
        }

    @staticmethod
    def _multi_secant_correct_residual(
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        x_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Map target input curvature through a rank-2 anchor secant operator.

        The first basis is the local bracketing input chord.  The second is
        the third-nearest anchor's deviation from that same affine chord, so
        it represents input curvature rather than another time coordinate.
        Residual chords use the identical coefficients.  The 2x2 Gram solve
        is per spatial token and falls back to the rank-1 Moore--Penrose solve
        only when the second direction is numerically dependent.
        """

        corrected = full_residual.clone()
        total_temporal = int(x_input.shape[1])
        epsilon = torch.finfo(torch.float32).eps

        def correct_segment(start: int, end: int) -> None:
            mask = (active_frames >= start) & (active_frames < end)
            segment_frames = active_frames[mask]
            segment_residual = active_residual[:, mask]
            if int(segment_frames.numel()) < 3:
                return
            positions = segment_frames - start
            for frame in range(start, end):
                if bool(torch.any(segment_frames == frame)):
                    continue
                position = frame - start
                left = torch.nonzero(positions < position).flatten()
                right = torch.nonzero(positions > position).flatten()
                if not len(left) or not len(right):
                    continue
                left_i, right_i = int(left[-1]), int(right[0])
                left_frame = int(segment_frames[left_i])
                right_frame = int(segment_frames[right_i])
                alpha = float(frame - left_frame) / float(
                    right_frame - left_frame
                )
                candidates = [
                    index
                    for index in range(int(segment_frames.numel()))
                    if index not in {left_i, right_i}
                ]
                third_i = min(
                    candidates,
                    key=lambda index: (
                        abs(int(segment_frames[index]) - frame),
                        int(segment_frames[index]),
                    ),
                )
                third_frame = int(segment_frames[third_i])
                third_alpha = float(third_frame - left_frame) / float(
                    right_frame - left_frame
                )

                left_input = x_input[:, left_frame].float()
                right_input = x_input[:, right_frame].float()
                third_input = x_input[:, third_frame].float()
                target_input = x_input[:, frame].float()
                input_chord = right_input - left_input
                input_curvature = third_input - (
                    (1.0 - third_alpha) * left_input
                    + third_alpha * right_input
                )
                input_defect = target_input - (
                    (1.0 - alpha) * left_input + alpha * right_input
                )

                left_residual = segment_residual[:, left_i].float()
                right_residual = segment_residual[:, right_i].float()
                third_residual = segment_residual[:, third_i].float()
                residual_chord = right_residual - left_residual
                residual_curvature = third_residual - (
                    (1.0 - third_alpha) * left_residual
                    + third_alpha * right_residual
                )

                g11 = input_chord.square().sum(dim=-1, keepdim=True)
                g12 = (input_chord * input_curvature).sum(
                    dim=-1, keepdim=True
                )
                g22 = input_curvature.square().sum(dim=-1, keepdim=True)
                b1 = (input_defect * input_chord).sum(dim=-1, keepdim=True)
                b2 = (input_defect * input_curvature).sum(
                    dim=-1, keepdim=True
                )
                determinant = g11 * g22 - g12.square()
                independent = determinant > epsilon * (g11 * g22).clamp_min(
                    epsilon
                )
                safe_determinant = determinant.clamp_min(epsilon)
                coefficient_1_rank2 = (
                    b1 * g22 - b2 * g12
                ) / safe_determinant
                coefficient_2_rank2 = (
                    b2 * g11 - b1 * g12
                ) / safe_determinant
                coefficient_1 = torch.where(
                    independent,
                    coefficient_1_rank2,
                    b1 / g11.clamp_min(epsilon),
                )
                coefficient_2 = torch.where(
                    independent,
                    coefficient_2_rank2,
                    torch.zeros_like(coefficient_2_rank2),
                )
                corrected[:, frame] = (
                    full_residual[:, frame].float()
                    + coefficient_1 * residual_chord
                    + coefficient_2 * residual_curvature
                ).to(dtype=full_residual.dtype)

        if memory_temporal:
            correct_segment(0, memory_temporal)
        if current_segment_boundary is None:
            correct_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            correct_segment(memory_temporal, split)
            correct_segment(split, total_temporal)
        corrected[:, active_frames] = active_residual
        return corrected

    @staticmethod
    def _match_token_radius(
        proposal: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Retract a proposal direction to the reference channel radius."""

        if proposal.shape != reference.shape:
            raise ValueError("polar residual tensors must have identical shapes")
        epsilon = torch.finfo(torch.float32).eps
        target_radius = torch.linalg.vector_norm(
            reference.float(), dim=-1, keepdim=True
        )
        proposal_radius = torch.linalg.vector_norm(
            proposal.float(), dim=-1, keepdim=True
        )
        return (
            proposal.float()
            * target_radius
            / proposal_radius.clamp_min(epsilon)
        ).to(dtype=proposal.dtype)

    @classmethod
    def _input_tangent_residual_lift(
        cls,
        active_residual: torch.Tensor,
        full_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Lift residuals in each target frame's local input-tangent chart.

        A block residual is decomposed at every exact anchor into its scalar
        component parallel to that frame's block input and an orthogonal
        detail component.  Both sections are lifted independently inside each
        Current segment.  The parallel section is then rebuilt from the
        *target* frame input, while transported detail is projected into the
        target frame's orthogonal complement.  Exact anchors are restored
        byte-for-byte at the end.
        """

        active_input = full_input[:, active_frames]
        residual_f = active_residual.float()
        input_f = active_input.float()
        denominator = input_f.square().sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        )
        coefficient = (residual_f * input_f).sum(
            dim=-1, keepdim=True
        ) / denominator
        orthogonal = residual_f - coefficient * input_f
        total_temporal = int(full_input.shape[1])
        coefficient_full = cls._lift_temporal(
            coefficient.to(active_residual.dtype),
            active_frames,
            total_temporal,
            memory_temporal,
            current_segment_boundary,
            "linear",
        ).float()
        orthogonal_full = cls._lift_temporal(
            orthogonal.to(active_residual.dtype),
            active_frames,
            total_temporal,
            memory_temporal,
            current_segment_boundary,
            "linear",
        ).float()
        target_input = full_input.float()
        target_denominator = target_input.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).eps)
        transported_parallel = (
            orthogonal_full * target_input
        ).sum(dim=-1, keepdim=True) / target_denominator
        orthogonal_full = (
            orthogonal_full - transported_parallel * target_input
        )
        lifted = coefficient_full * target_input + orthogonal_full
        lifted = lifted.to(active_residual.dtype)
        lifted[:, active_frames] = active_residual
        return lifted

    @classmethod
    def _input_parallel_correct_residual(
        cls,
        full_residual: torch.Tensor,
        active_residual: torch.Tensor,
        full_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Correct only V22's one-dimensional input-parallel component."""

        active_input = full_input[:, active_frames].float()
        active_residual_f = active_residual.float()
        active_denominator = active_input.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).eps)
        active_coefficient = (
            active_residual_f * active_input
        ).sum(dim=-1, keepdim=True) / active_denominator
        coefficient_full = cls._lift_temporal(
            active_coefficient.to(active_residual.dtype),
            active_frames,
            int(full_input.shape[1]),
            memory_temporal,
            current_segment_boundary,
            "linear",
        ).float()
        target_input = full_input.float()
        target_denominator = target_input.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(torch.float32).eps)
        residual_f = full_residual.float()
        existing_coefficient = (residual_f * target_input).sum(
            dim=-1, keepdim=True
        ) / target_denominator
        corrected = residual_f + (
            coefficient_full - existing_coefficient
        ) * target_input
        corrected = corrected.to(full_residual.dtype)
        corrected[:, active_frames] = active_residual
        return corrected

    @classmethod
    def _input_velocity_residual_lift(
        cls,
        active_residual: torch.Tensor,
        full_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Correct linear residual lift with the target input's chord defect.

        For every inactive frame, the two chronological exact anchors define
        an input chord and a residual chord.  A scalar least-squares secant
        maps the target input's deviation from its chronological input chord
        into residual space.  This preserves target-frame first-order motion
        information without fitting a learned gain or reading an evaluation
        signal.  Memory and the two Current sections are processed separately.
        """

        total_temporal = int(full_input.shape[1])
        lifted = cls._lift_temporal(
            active_residual,
            active_frames,
            total_temporal,
            memory_temporal,
            current_segment_boundary,
            "linear",
        )
        corrected = lifted.clone()
        epsilon = torch.finfo(torch.float32).eps

        def correct_segment(start: int, end: int) -> None:
            mask = (active_frames >= start) & (active_frames < end)
            frames = active_frames[mask]
            residuals = active_residual[:, mask]
            if not int(frames.numel()):
                raise RuntimeError("velocity-residual segment has no exact anchor")
            positions = frames - start
            for frame in range(start, end):
                if bool(torch.any(frames == frame)):
                    continue
                position = frame - start
                left = torch.nonzero(positions < position).flatten()
                right = torch.nonzero(positions > position).flatten()
                if not len(left) or not len(right):
                    continue
                left_i, right_i = int(left[-1]), int(right[0])
                left_frame = int(frames[left_i])
                right_frame = int(frames[right_i])
                alpha = float(frame - left_frame) / float(
                    right_frame - left_frame
                )

                left_input = full_input[:, left_frame].float()
                right_input = full_input[:, right_frame].float()
                target_input = full_input[:, frame].float()
                input_chord = right_input - left_input
                input_linear = (
                    (1.0 - alpha) * left_input + alpha * right_input
                )
                input_defect = target_input - input_linear

                left_residual = residuals[:, left_i].float()
                right_residual = residuals[:, right_i].float()
                residual_chord = right_residual - left_residual
                gain = (residual_chord * input_chord).sum(
                    dim=-1, keepdim=True
                ) / input_chord.square().sum(
                    dim=-1, keepdim=True
                ).clamp_min(epsilon)
                corrected[:, frame] = (
                    lifted[:, frame].float() + gain * input_defect
                ).to(dtype=lifted.dtype)

        if memory_temporal:
            correct_segment(0, memory_temporal)
        if current_segment_boundary is None:
            correct_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            if not memory_temporal < split < total_temporal:
                raise RuntimeError("invalid velocity-residual Current boundary")
            correct_segment(memory_temporal, split)
            correct_segment(split, total_temporal)
        corrected[:, active_frames] = active_residual
        return corrected

    @staticmethod
    def _convex_input_chord_residual_lift(
        active_residual: torch.Tensor,
        full_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
    ) -> torch.Tensor:
        """Project each target input onto its local anchor chord, without extrapolation."""

        batch, _, spatial, channels = active_residual.shape
        total_temporal = int(full_input.shape[1])
        full = active_residual.new_empty(
            batch, total_temporal, spatial, channels
        )
        full[:, active_frames] = active_residual

        def fill_segment(start: int, end: int) -> None:
            mask = (active_frames >= start) & (active_frames < end)
            frames = active_frames[mask]
            residual = active_residual[:, mask]
            positions = frames - start
            if not int(frames.numel()):
                raise RuntimeError("convex chord segment has no anchor")
            for frame in range(start, end):
                if bool(torch.any(frames == frame)):
                    continue
                position = frame - start
                left = torch.nonzero(positions < position).flatten()
                right = torch.nonzero(positions > position).flatten()
                if not len(left):
                    full[:, frame] = residual[:, int(right[0])]
                    continue
                if not len(right):
                    full[:, frame] = residual[:, int(left[-1])]
                    continue
                left_i, right_i = int(left[-1]), int(right[0])
                left_frame = int(frames[left_i])
                right_frame = int(frames[right_i])
                left_input = full_input[:, left_frame].float()
                chord = full_input[:, right_frame].float() - left_input
                offset = full_input[:, frame].float() - left_input
                denominator = chord.square().sum(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(torch.float32).eps)
                weight = ((offset * chord).sum(
                    dim=-1, keepdim=True
                ) / denominator).clamp(0.0, 1.0)
                full[:, frame] = (
                    (1.0 - weight) * residual[:, left_i].float()
                    + weight * residual[:, right_i].float()
                ).to(active_residual.dtype)

        if memory_temporal:
            fill_segment(0, memory_temporal)
        if current_segment_boundary is None:
            fill_segment(memory_temporal, total_temporal)
        else:
            split = memory_temporal + int(current_segment_boundary)
            fill_segment(memory_temporal, split)
            fill_segment(split, total_temporal)
        return full

    @staticmethod
    def _replace_with_interpolated_output_dc(
        full_output: torch.Tensor,
        active_output: torch.Tensor,
        active_frames: torch.Tensor,
        memory_temporal: int,
        current_segment_boundary: int | None,
        output_dc_scope: str = "all_current",
    ) -> torch.Tensor:
        """Use output interpolation only on the spatial-constant subspace."""

        active_dc = active_output.mean(dim=2, keepdim=True)
        lifted_dc = MatrixCurvaturePhaseFrameWeave._lift_temporal(
            active_dc,
            active_frames,
            int(full_output.shape[1]),
            memory_temporal,
            current_segment_boundary,
            "linear",
        )
        adjusted = full_output + lifted_dc - full_output.mean(dim=2, keepdim=True)
        if output_dc_scope == "all_current":
            return adjusted
        if current_segment_boundary is None:
            raise ValueError("segmented output-DC scope requires a Current boundary")
        split = memory_temporal + int(current_segment_boundary)
        selected = full_output.clone()
        if output_dc_scope == "overlap":
            selected[:, memory_temporal:split] = adjusted[:, memory_temporal:split]
        elif output_dc_scope == "new_generation":
            selected[:, split:] = adjusted[:, split:]
        else:
            raise ValueError("unknown output-DC scope")
        return selected

    def _ray_aligned_residual_lift(
        self,
        active_residual: torch.Tensor,
        active_frames: torch.Tensor,
        total_temporal: int,
        memory_temporal: int,
        *,
        spatial_height: int,
        spatial_width: int,
        full_plucker: torch.Tensor | None,
        alignment_features: torch.Tensor,
        alignment_feature_source: str = "action_module",
    ) -> torch.Tensor:
        """Transport zero-mean detail to the target ray before interpolation."""

        batch, active_count, spatial, channels = active_residual.shape
        if active_count != int(active_frames.numel()):
            raise RuntimeError("ray transport active-frame cardinality mismatch")
        if spatial != spatial_height * spatial_width:
            raise RuntimeError("ray transport spatial grid mismatch")
        if alignment_features.shape[:3] != (batch, total_temporal, spatial):
            raise RuntimeError("ray transport requires dense alignment features")
        if alignment_feature_source not in {"action_module", "block_input"}:
            raise ValueError("unknown ray-transport alignment feature source")
        if full_plucker is not None and (
            full_plucker.ndim != 3
            or full_plucker.shape[1] < total_temporal * spatial
        ):
            raise RuntimeError("ray transport received malformed Plucker embeddings")

        feature_channels = int(alignment_features.shape[-1])
        groups = min(self.ray_transport_feature_groups, feature_channels)
        compressed = F.adaptive_avg_pool1d(
            alignment_features.detach().reshape(
                batch * total_temporal * spatial, 1, feature_channels
            ),
            groups,
        ).float().reshape(batch, total_temporal, spatial, groups)
        compressed = F.normalize(compressed, dim=-1, eps=1e-6)
        rays = None
        if full_plucker is not None:
            ray_channels = int(full_plucker.shape[-1])
            ray_groups = min(self.ray_transport_feature_groups, ray_channels)
            rays = F.adaptive_avg_pool1d(
                full_plucker[:, : total_temporal * spatial].detach().reshape(
                    batch * total_temporal * spatial, 1, ray_channels
                ),
                ray_groups,
            ).float().reshape(batch, total_temporal, spatial, ray_groups)
            rays = F.normalize(rays, dim=-1, eps=1e-6)

        yy, xx = torch.meshgrid(
            torch.arange(spatial_height, device=active_frames.device),
            torch.arange(spatial_width, device=active_frames.device),
            indexing="ij",
        )
        candidates = []
        radius = self.ray_transport_radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                cy = (yy + dy).clamp(0, spatial_height - 1)
                cx = (xx + dx).clamp(0, spatial_width - 1)
                candidates.append((cy * spatial_width + cx).reshape(-1))
        candidate_indices = torch.stack(candidates, dim=-1)
        identity = torch.arange(spatial, device=active_frames.device)

        full = active_residual.new_empty(batch, total_temporal, spatial, channels)
        full[:, active_frames] = active_residual
        active_list = [int(frame) for frame in active_frames.tolist()]
        active_position = {frame: position for position, frame in enumerate(active_list)}
        transported_frames = 0
        shifted_tokens = 0
        total_tokens = 0
        limited_batches = 0

        def transport(source_frame: int, target_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal shifted_tokens, total_tokens
            source_position = active_position[source_frame]
            source = active_residual[:, source_position].float()
            source_mean = source.mean(dim=1, keepdim=True)
            source_detail = source - source_mean
            source_features = compressed[:, source_frame][:, candidate_indices]
            target_features = compressed[:, target_frame].unsqueeze(2)
            correspondence_score = (target_features * source_features).sum(dim=-1)
            if rays is not None:
                source_rays = rays[:, source_frame][:, candidate_indices]
                target_rays = rays[:, target_frame].unsqueeze(2)
                correspondence_score = correspondence_score + (
                    target_rays * source_rays
                ).sum(dim=-1)
            selected_offset = correspondence_score.argmax(dim=-1)
            selected = candidate_indices.unsqueeze(0).expand(batch, -1, -1).gather(
                2, selected_offset.unsqueeze(-1)
            ).squeeze(-1)
            shifted_tokens += int((selected != identity.unsqueeze(0)).sum().item())
            total_tokens += int(selected.numel())
            gathered = source_detail.gather(
                1, selected.unsqueeze(-1).expand(-1, -1, channels)
            )
            source_norm = torch.linalg.vector_norm(source_detail, dim=(1, 2))
            return gathered, source_norm

        for target_frame in range(memory_temporal, total_temporal):
            if target_frame in active_position:
                continue
            left = [frame for frame in active_list if memory_temporal <= frame < target_frame]
            right = [frame for frame in active_list if target_frame < frame < total_temporal]
            if not left or not right:
                raise RuntimeError("ray transport target is not bracketed by exact frames")
            left_frame, right_frame = left[-1], right[0]
            alpha = float(target_frame - left_frame) / float(right_frame - left_frame)
            left_detail, left_norm = transport(left_frame, target_frame)
            right_detail, right_norm = transport(right_frame, target_frame)
            detail = (1.0 - alpha) * left_detail + alpha * right_detail
            detail = detail - detail.mean(dim=1, keepdim=True)
            detail_norm = torch.linalg.vector_norm(detail, dim=(1, 2))
            norm_bound = (1.0 - alpha) * left_norm + alpha * right_norm
            scale = torch.minimum(
                torch.ones_like(detail_norm),
                norm_bound / detail_norm.clamp_min(1e-6),
            )
            limited_batches += int((scale < 1.0).sum().item())
            detail = detail * scale[:, None, None]
            left_mean = active_residual[:, active_position[left_frame]].float().mean(
                dim=1, keepdim=True
            )
            right_mean = active_residual[:, active_position[right_frame]].float().mean(
                dim=1, keepdim=True
            )
            mean = (1.0 - alpha) * left_mean + alpha * right_mean
            full[:, target_frame] = (mean + detail).to(active_residual.dtype)
            transported_frames += 1

        # The current-domain V21 schedule keeps all R4 frames exact.  Retain a
        # defensive fallback for any future domain variant without weakening
        # the certificate for the configured method.
        for target_frame in range(memory_temporal):
            if target_frame not in active_position:
                raise RuntimeError("ray transport requires exact memory anchors")
        self._last_ray_transport = {
            "transported_frames": transported_frames,
            "candidate_count": int(candidate_indices.shape[-1]),
            "shifted_token_fraction": (
                float(shifted_tokens) / float(total_tokens) if total_tokens else 0.0
            ),
            "norm_limited_batches": limited_batches,
            "constant_preserving": True,
            "spatial_dc_interpolated": True,
            "zero_mean_detail_transported": True,
            "uses_plucker_rays": rays is not None,
            "alignment_feature_source": alignment_feature_source,
            "uses_dense_action_features": alignment_feature_source == "action_module",
        }
        return full

    @staticmethod
    def _normalized_l2_error(
        exact: torch.Tensor,
        approximate: torch.Tensor,
        *,
        epsilon: float,
    ) -> tuple[float | None, bool]:
        """Return a JSON-safe normalized error and its finite-status bit."""

        if exact.shape != approximate.shape:
            raise ValueError("normalized-error tensors must have identical shapes")
        exact_float = exact.detach().float()
        approximate_float = approximate.detach().float()
        numerator = torch.linalg.vector_norm(exact_float - approximate_float)
        denominator = torch.linalg.vector_norm(exact_float) + float(epsilon)
        finite = bool(
            torch.isfinite(numerator).item()
            and torch.isfinite(denominator).item()
        )
        if not finite:
            return None, False
        return float((numerator / denominator).item()), True

    def _witness_defect(
        self,
        *,
        active_values: torch.Tensor,
        active_inputs: torch.Tensor,
        active_frames: torch.Tensor,
        total_temporal: int,
        memory_temporal: int,
    ) -> dict[str, Any]:
        """Measure leave-one-out interpolation error at an exact current frame."""

        current_mask = active_frames >= int(memory_temporal)
        current_positions = torch.nonzero(current_mask).flatten()
        if int(current_positions.numel()) < 3:
            return {"status": "no_bracket"}
        candidate_positions = current_positions[1:-1]
        if not int(candidate_positions.numel()):
            return {"status": "no_bracket"}
        curvature = self._current_curvature(total_temporal - memory_temporal)
        if curvature is None:
            return {"status": "no_curvature"}
        candidate_frames = active_frames[candidate_positions]
        local_frames = candidate_frames - int(memory_temporal)
        candidate_curvature = curvature.to(active_frames.device)[local_frames]
        if not bool(torch.isfinite(candidate_curvature).all()):
            return {"status": "nonfinite_curvature"}
        witness_offset = int(torch.argmax(candidate_curvature).item())
        epsilon = self.witness_probe_epsilon
        if epsilon is None:
            raise RuntimeError("witness defect requested without probe epsilon")

        def measure(position: int, curvature_value: torch.Tensor) -> dict[str, Any]:
            left_position = position - 1
            right_position = position + 1
            frame = int(active_frames[position].item())
            left_frame = int(active_frames[left_position].item())
            right_frame = int(active_frames[right_position].item())
            if not left_frame < frame < right_frame:
                raise RuntimeError("witness frames are not strictly bracketed")
            weight = float(frame - left_frame) / float(right_frame - left_frame)
            exact = active_values[:, position].detach().float()
            left_value = active_values[:, left_position].detach().float()
            right_value = active_values[:, right_position].detach().float()
            reconstructed = (1.0 - weight) * left_value + weight * right_value
            absolute_error = torch.linalg.vector_norm(exact - reconstructed)
            witness_norm = torch.linalg.vector_norm(exact)
            input_norm = torch.linalg.vector_norm(
                active_inputs[:, position].detach().float()
            )
            left_norm = torch.linalg.vector_norm(left_value)
            right_norm = torch.linalg.vector_norm(right_value)
            neighbor_energy_norm = torch.sqrt(
                0.5 * (left_norm.square() + right_norm.square())
            )
            chord_norm = torch.linalg.vector_norm(right_value - left_value)
            return {
                "frame": frame,
                "left_frame": left_frame,
                "right_frame": right_frame,
                "weight": weight,
                "curvature": curvature_value,
                "absolute_error": absolute_error,
                "witness_norm": witness_norm,
                "input_norm": input_norm,
                "left_norm": left_norm,
                "right_norm": right_norm,
                "neighbor_energy_norm": neighbor_energy_norm,
                "chord_norm": chord_norm,
            }

        measurements = [
            measure(int(position.item()), candidate_curvature[offset])
            for offset, position in enumerate(candidate_positions)
        ]
        selected = measurements[witness_offset]

        def aggregate(
            selected_measurements: list[dict[str, Any]], prefix: str
        ) -> dict[str, Any]:
            if not selected_measurements:
                return {
                    f"{prefix}_witness_count": 0,
                    f"{prefix}_witness_frames": [],
                }

            def combined_norm(name: str) -> torch.Tensor:
                return torch.sqrt(
                    torch.stack(
                        [row[name].square() for row in selected_measurements]
                    ).sum()
                )

            absolute = combined_norm("absolute_error")
            witness_scale = combined_norm("witness_norm")
            input_scale = combined_norm("input_norm")
            neighbor_scale = combined_norm("neighbor_energy_norm")
            chord_scale = combined_norm("chord_norm")
            tensors = {
                f"{prefix}_absolute_l2_error": absolute,
                f"{prefix}_normalized_defect": absolute
                / (witness_scale + epsilon),
                f"{prefix}_input_normalized_defect": absolute
                / (input_scale + epsilon),
                f"{prefix}_neighbor_energy_normalized_defect": absolute
                / (neighbor_scale + epsilon),
                f"{prefix}_chord_normalized_defect": absolute
                / (chord_scale + epsilon),
            }
            return {
                f"{prefix}_witness_count": len(selected_measurements),
                f"{prefix}_witness_frames": [
                    int(row["frame"]) for row in selected_measurements
                ],
                **{
                    name: (
                        float(value.item())
                        if bool(torch.isfinite(value).item())
                        else None
                    )
                    for name, value in tensors.items()
                },
            }

        tensors = {
            "absolute_l2_error": selected["absolute_error"],
            "witness_norm": selected["witness_norm"],
            "input_norm": selected["input_norm"],
            "left_neighbor_norm": selected["left_norm"],
            "right_neighbor_norm": selected["right_norm"],
            "neighbor_energy_norm": selected["neighbor_energy_norm"],
            "neighbor_chord_norm": selected["chord_norm"],
            "normalized_defect": selected["absolute_error"]
            / (selected["witness_norm"] + epsilon),
            "input_normalized_defect": selected["absolute_error"]
            / (selected["input_norm"] + epsilon),
            "neighbor_energy_normalized_defect": selected["absolute_error"]
            / (selected["neighbor_energy_norm"] + epsilon),
            "chord_normalized_defect": selected["absolute_error"]
            / (selected["chord_norm"] + epsilon),
        }
        values = {
            name: (
                float(value.item()) if bool(torch.isfinite(value).item()) else None
            )
            for name, value in tensors.items()
        }
        phase_measurements = [
            row for offset, row in enumerate(measurements) if offset != witness_offset
        ]
        aggregate_values = {
            **aggregate(measurements, "multi"),
            **aggregate(phase_measurements, "phase"),
        }
        finite = all(value is not None for value in values.values()) and all(
            value is not None
            for name, value in aggregate_values.items()
            if name.endswith("_defect") or name.endswith("_error")
        )
        return {
            "status": "ok" if finite else "nonfinite_defect",
            "witness_frame": int(selected["frame"]),
            "left_frame": int(selected["left_frame"]),
            "right_frame": int(selected["right_frame"]),
            "interpolation_weight": float(selected["weight"]),
            "witness_curvature": float(selected["curvature"].item()),
            **values,
            **aggregate_values,
            "finite": finite,
        }

    def _layout(
        self,
        *,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        memory_length: int,
        active_frames: torch.Tensor,
        orbit_sparse_layout: Any,
    ) -> _FrameLayout:
        temporal, height, width = (int(value) for value in grid_sizes[0].tolist())
        active_frame_values = tuple(
            int(value) for value in active_frames.tolist()
        )
        if (
            self.runtime_optimized
            and orbit_sparse_layout is not None
            and self.compact_cwca_topology
            and self.sol_attention_runtime is None
        ):
            orbit_layout_key: Any = (
                "compact_cwca_block_shape",
                tuple(int(value) for value in orbit_sparse_layout.block_shape),
                str(x.device),
            )
        else:
            orbit_layout_key = id(orbit_sparse_layout)
        key = (
            temporal,
            height,
            width,
            int(memory_length),
            active_frame_values,
            orbit_layout_key,
        )
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached
        spatial = height * width
        space = torch.arange(spatial, device=x.device)
        compact = (active_frames[:, None] * spatial + space[None, :]).reshape(-1)
        coords = torch.stack(
            [
                active_frames.repeat_interleave(spatial),
                (space // width).repeat(len(active_frames)),
                (space % width).repeat(len(active_frames)),
            ],
            dim=-1,
        ).long()
        result = _FrameLayout(
            total_temporal=temporal,
            spatial_height=height,
            spatial_width=width,
            memory_temporal=int(memory_length),
            full_sequence_length=int(x.shape[1]),
            active_frames=active_frames,
            active_frame_values=active_frame_values,
            compact_indices=compact.long(),
            compact_coordinates=coords,
        )
        if (
            orbit_sparse_layout is not None
            and self.compact_cwca_topology
            and self.sol_attention_runtime is None
        ):
            tt, th, tw = (int(value) for value in orbit_sparse_layout.block_shape)
            compact_t = int(active_frames.numel())
            nh, nw = math.ceil(height / th), math.ceil(width / tw)
            local_token = torch.arange(compact_t * spatial, device=x.device)
            local_t = local_token // spatial
            local_space = local_token % spatial
            local_h, local_w = local_space // width, local_space % width
            compact_memory = (
                bisect_left(active_frame_values, int(memory_length))
                if self.runtime_optimized
                else int(
                    torch.count_nonzero(
                        active_frames < int(memory_length)
                    ).item()
                )
            )
            if compact_memory:
                memory_cells = math.ceil(compact_memory / tt)
                temporal_cell = torch.where(
                    local_t < compact_memory,
                    local_t // tt,
                    memory_cells + (local_t - compact_memory) // tt,
                )
                compact_blocks = (
                    memory_cells + math.ceil((compact_t - compact_memory) / tt)
                ) * nh * nw
            else:
                temporal_cell = local_t // tt
                compact_blocks = math.ceil(compact_t / tt) * nh * nw
            compact_block_id = (
                temporal_cell * nh * nw
                + (local_h // th) * nw
                + local_w // tw
            )
            block_size = tt * th * tw
            block_indices = torch.full(
                (compact_blocks, block_size),
                -1,
                device=x.device,
                dtype=torch.long,
            )
            if self.runtime_optimized:
                # Stable grouping reproduces the reference's per-block
                # ``nonzero`` order while replacing 90--120 tiny CUDA launches
                # with a fixed vectorized pipeline.
                order = torch.argsort(compact_block_id, stable=True)
                sorted_blocks = compact_block_id.index_select(0, order)
                counts = torch.bincount(
                    sorted_blocks, minlength=compact_blocks
                )
                torch._assert_async(
                    torch.all(counts <= block_size),
                    "compact frame block exceeds its packed capacity",
                )
                starts = torch.cumsum(counts, dim=0) - counts
                ranks = torch.arange(
                    int(order.numel()), device=x.device, dtype=torch.long
                ) - torch.repeat_interleave(
                    starts, counts, output_size=int(order.numel())
                )
                flat_positions = sorted_blocks * block_size + ranks
                block_indices.view(-1)[flat_positions] = order
            else:
                for index in range(compact_blocks):
                    members = torch.nonzero(compact_block_id == index).flatten()
                    block_indices[index, : members.numel()] = members
            block_valid = block_indices >= 0
            token_to_packed = torch.empty(
                compact_t * spatial, device=x.device, dtype=torch.long
            )
            packed_position = torch.arange(
                compact_blocks * block_size, device=x.device
            ).reshape(compact_blocks, block_size)
            token_to_packed[block_indices[block_valid]] = packed_position[block_valid]
            result.compact_block_indices = block_indices
            result.compact_block_valid = block_valid
            result.compact_token_to_packed = token_to_packed
            result.compact_block_shape = (tt, th, tw)
            result.packed_block_size = block_size
            self._layout_cache[key] = result
            return result
        if orbit_sparse_layout is not None:
            tt, th, tw = (int(value) for value in orbit_sparse_layout.block_shape)
            nh, nw = math.ceil(height / th), math.ceil(width / tw)
            memory_t = int(memory_length)
            current_t = temporal - memory_t
            t_coord, h_coord, w_coord = coords.unbind(dim=1)
            if bool(getattr(orbit_sparse_layout, "protected_current", False)):
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
                raise RuntimeError("frame-weave/CWCA block count mismatch")
            per_block = [
                torch.nonzero(block_id == index).flatten()
                for index in range(total_blocks)
            ]
            maximum = max(int(values.numel()) for values in per_block)
            packed_size = 1 << max(0, maximum - 1).bit_length()
            packed = torch.full(
                (total_blocks, packed_size), -1, device=x.device, dtype=torch.long
            )
            for index, values in enumerate(per_block):
                packed[index, : values.numel()] = values
            valid = packed >= 0
            compact_to_packed = torch.empty(
                compact.numel(), device=x.device, dtype=torch.long
            )
            positions = torch.arange(
                total_blocks * packed_size, device=x.device
            ).reshape(total_blocks, packed_size)
            compact_to_packed[packed[valid]] = positions[valid]
            result.packed_indices = packed
            result.compact_to_packed = compact_to_packed
            result.packed_valid = valid
            result.packed_block_size = packed_size
        self._layout_cache[key] = result
        return result

    def _self_attention(
        self,
        module: Any,
        x: torch.Tensor,
        layout: _FrameLayout,
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
        heads, head_dim = module.num_heads, module.head_dim
        if self.runtime_shared_int8_qkv:
            from .matrix_shared_int8_qkv import shared_int8_qkv

            q_raw, k_raw, v_raw = shared_int8_qkv(
                module.q, module.k, module.v, x.contiguous()
            )
        else:
            q_raw, k_raw, v_raw = module.q(x), module.k(x), module.v(x)
        q = module.norm_q(q_raw).view(batch, sequence, heads, head_dim)
        k = module.norm_k(k_raw).view(batch, sequence, heads, head_dim)
        v = v_raw.view(batch, sequence, heads, head_dim)
        if self.runtime_cached_compact_rope_phase:
            phase = self._cached_compact_rope_phase(
                layout.compact_coordinates,
                freqs,
                memory_length,
                memory_latent_idx,
                predict_latent_idx,
            )
            q = self._apply_compact_rope_phase(q, phase)
            k = self._apply_compact_rope_phase(k, phase)
        else:
            apply_rope = MatrixJiTOfficialSemanticsAcceleration._apply_sparse_rope
            q = apply_rope(
                q,
                layout.compact_coordinates,
                freqs,
                memory_length,
                memory_latent_idx,
                predict_latent_idx,
            )
            k = apply_rope(
                k,
                layout.compact_coordinates,
                freqs,
                memory_length,
                memory_latent_idx,
                predict_latent_idx,
            )
        if self.sol_attention_runtime is not None:
            return self._profile_call(
                "compact_sol_attention",
                getattr(module, "block_idx", None),
                self._compact_sol_attention,
                module,
                q,
                k,
                v,
                layout,
            )
        if orbit_sparse_layout is not None and self.compact_cwca_topology:
            return self._profile_call(
                "compact_sparse_attention",
                getattr(module, "block_idx", None),
                self._compact_cwca_attention,
                module,
                q,
                k,
                v,
                layout,
            )
        compact_lens = torch.full_like(seq_lens, sequence)
        if orbit_sparse_layout is None or memory_length == 0:
            from wan.modules.attention import attention

            output = attention(
                q=q,
                k=k,
                v=v,
                k_lens=compact_lens,
                window_size=module.window_size,
                version=fa_version,
            )
        else:
            if (
                layout.packed_indices is None
                or layout.packed_valid is None
                or layout.compact_to_packed is None
            ):
                raise RuntimeError("frame-weave sparse attention lacks packed layout")
            packed = layout.packed_indices.clamp_min(0).reshape(-1)
            valid = layout.packed_valid
            q_packed = q[:, packed].transpose(1, 2).contiguous()
            k_packed = k[:, packed].transpose(1, 2).contiguous()
            v_packed = v[:, packed].transpose(1, 2).contiguous()
            mask = valid.reshape(-1)
            q_packed[:, :, ~mask] = 0
            k_packed[:, :, ~mask] = 0
            v_packed[:, :, ~mask] = 0
            output_packed = MatrixJiTSpatialAcceleration._longcat_active(
                q_packed,
                k_packed,
                v_packed,
                orbit_sparse_layout.indices.expand(batch, heads, -1, -1),
                orbit_sparse_layout.counts.expand(batch, heads, -1),
                valid,
                layout.packed_block_size,
            ).transpose(1, 2)
            output = output_packed[:, layout.compact_to_packed]
        return module.o(output.flatten(2).to(x.dtype))

    def _cached_compact_rope_phase(
        self,
        coords: torch.Tensor,
        freqs: torch.Tensor,
        memory_length: int,
        memory_latent_idx: Any,
        predict_latent_idx: Any,
    ) -> torch.Tensor:
        memory_values = tuple(
            int(value) for value in (
                memory_latent_idx
                if memory_latent_idx is not None
                else range(int(memory_length))
            )
        )
        if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
            predict_values = tuple(
                range(int(predict_latent_idx[0]), int(predict_latent_idx[1]))
            )
        elif predict_latent_idx is not None:
            predict_values = tuple(int(value) for value in predict_latent_idx)
        else:
            predict_values = ()
        key = (
            int(coords.data_ptr()),
            tuple(coords.shape),
            int(freqs.data_ptr()),
            tuple(freqs.shape),
            int(memory_length),
            memory_values,
            predict_values,
        )
        if self._compact_rope_phase_key == key and self._compact_rope_phase is not None:
            return self._compact_rope_phase

        half = 128 // 2
        split = [half - 2 * (half // 3), half // 3, half // 3]
        axes = freqs.split(split, dim=2 if freqs.dim() == 3 else 1)
        raw_t = coords[:, 0].long()
        temporal_frames = int(raw_t.max().item()) + 1
        temporal = torch.empty_like(raw_t)
        if memory_length:
            if len(memory_values) < memory_length:
                raise RuntimeError("cached compact RoPE memory indices are incomplete")
            memory_tensor = torch.tensor(memory_values, device=coords.device)
            memory_mask = raw_t < memory_length
            temporal[memory_mask] = memory_tensor[raw_t[memory_mask]]
        else:
            memory_mask = torch.zeros_like(raw_t, dtype=torch.bool)
        current_frames = temporal_frames - int(memory_length)
        if not predict_values:
            predict_values = tuple(range(current_frames))
        if len(predict_values) < current_frames:
            raise RuntimeError("cached compact RoPE prediction indices are incomplete")
        if current_frames:
            predict_tensor = torch.tensor(predict_values, device=coords.device)
            temporal[~memory_mask] = predict_tensor[
                raw_t[~memory_mask] - memory_length
            ]
        indices = (temporal.long(), coords[:, 1].long(), coords[:, 2].long())
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
                [
                    axis[index].unsqueeze(1)
                    for axis, index in zip(axes, indices)
                ],
                dim=-1,
            )
        self._compact_rope_phase_key = key
        self._compact_rope_phase = phase
        return phase

    @staticmethod
    def _apply_compact_rope_phase(
        value: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        complex_value = torch.view_as_complex(
            value.float().reshape(
                value.shape[0], value.shape[1], value.shape[2], -1, 2
            )
        )
        rotated = complex_value * phase.unsqueeze(0).to(complex_value.dtype)
        return torch.view_as_real(rotated).flatten(3).float()

    def _compact_sol_attention(
        self,
        module: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: _FrameLayout,
    ) -> torch.Tensor:
        """Run official Sol-Attn on chronological compact Exact frames.

        The Closed-Loop provider remains only as the causal response source for
        the unchanged exact-frame selector.  It no longer allocates attention
        budgets or ranks Q/K blocks in this path.
        """

        provider = self.geometry_provider
        layer_index = int(getattr(module, "block_idx", -1))
        if (
            getattr(provider, "mode", None) == "closed_loop"
            and hasattr(provider, "begin_compact_layer")
        ):
            if layer_index < 0:
                raise RuntimeError("compact Sol-Attn lacks a layer index")
            provider.begin_compact_layer(layer_index)
        if q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError("compact Sol-Attn requires square self-attention")
        if q.ndim != 4 or q.shape[-1] != 128:
            raise RuntimeError("compact Sol-Attn requires BTHD with D=128")

        q_bthd = q.to(torch.bfloat16).contiguous()
        k_bthd = k.to(torch.bfloat16).contiguous()
        v_bthd = v.to(torch.bfloat16).contiguous()
        output = self.sol_attention_runtime.square_bthd(
            q_bthd,
            k_bthd,
            v_bthd,
            dense=lambda: (_ for _ in ()).throw(
                RuntimeError("strict compact Sol-Attn cannot fall back to dense")
            ),
            strict=True,
        )
        token_count = int(q.shape[1])
        self._last_compact_attention = {
            "backend": "sol_attn",
            "kernel_backend": self.sol_attention_runtime.metadata()[
                "kernel_backend"
            ],
            "tokens": token_count,
            "blocks_64": math.ceil(token_count / 64),
            "active_frames": int(layout.active_frames.numel()),
            "memory_frames": int(
                torch.count_nonzero(
                    layout.active_frames < layout.memory_temporal
                ).item()
            ),
            "tau": float(self.sol_attention_runtime.tau),
            "thresh_type": str(self.sol_attention_runtime.thresh_type),
            "adaptive_route": True,
            "source_q_dtype": str(q.dtype),
            "source_k_dtype": str(k.dtype),
            "source_v_dtype": str(v.dtype),
            "official_kernel_dtype": str(q_bthd.dtype),
            "cwca_budget_used": False,
            "cwca_qk_topk_used": False,
        }
        return module.o(output.flatten(2).to(q.dtype))

    def _compact_cwca_attention(
        self,
        module: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: _FrameLayout,
    ) -> torch.Tensor:
        """Run CWCA on a genuinely compact complete-frame block graph.

        Geometry still allocates a fixed global edge budget through the
        current worldline curvature.  Native pooled Q/K alone ranks keys
        inside that budget.  Unlike the first adapter, inactive temporal
        blocks do not survive as padded kernel rows or zero-valued keys.
        """

        closed_loop_provider = (
            self.geometry_provider
            if getattr(self.geometry_provider, "mode", None) == "closed_loop"
            and hasattr(self.geometry_provider, "compact_closed_loop_degrees")
            else None
        )
        layer_index = int(getattr(module, "block_idx", -1))
        if closed_loop_provider is not None:
            if layer_index < 0:
                raise RuntimeError("compact Closed-Loop CWCA lacks a layer index")
            closed_loop_provider.begin_compact_layer(layer_index)

        packed = layout.compact_block_indices
        valid = layout.compact_block_valid
        token_to_packed = layout.compact_token_to_packed
        if packed is None or valid is None or token_to_packed is None:
            raise RuntimeError("compact CWCA layout was not constructed")
        block_size = int(layout.packed_block_size)
        flat = packed.clamp_min(0).reshape(-1)
        mask = valid.reshape(-1)
        q_packed = q[:, flat].transpose(1, 2).contiguous()
        k_packed = k[:, flat].transpose(1, 2).contiguous()
        v_packed = v[:, flat].transpose(1, 2).contiguous()
        q_packed[:, :, ~mask] = 0
        k_packed[:, :, ~mask] = 0
        v_packed[:, :, ~mask] = 0
        batch, heads = q_packed.shape[:2]
        blocks = int(packed.shape[0])
        q_blocks = q_packed.reshape(
            batch, heads, blocks, block_size, q_packed.shape[-1]
        ).float()
        k_blocks = k_packed.reshape(
            batch, heads, blocks, block_size, k_packed.shape[-1]
        ).float()
        v_blocks = v_packed.reshape(
            batch, heads, blocks, block_size, v_packed.shape[-1]
        ).float()
        weights = valid.to(q_blocks.dtype)[None, None, :, :, None]
        denominator = weights.sum(dim=3).clamp_min(1.0)
        q_content = F.normalize((q_blocks * weights).sum(dim=3) / denominator, dim=-1)
        k_content = F.normalize((k_blocks * weights).sum(dim=3) / denominator, dim=-1)
        v_content = (v_blocks * weights).sum(dim=3) / denominator
        score = torch.matmul(q_content, k_content.transpose(-2, -1))

        if layout.compact_block_shape is None:
            raise RuntimeError("compact CWCA block shape is missing")
        tt, th, tw = layout.compact_block_shape
        nh = math.ceil(layout.spatial_height / th)
        nw = math.ceil(layout.spatial_width / tw)
        spatial = nh * nw
        compact_frames = len(layout.active_frame_values)
        memory_frames = bisect_left(
            layout.active_frame_values, int(layout.memory_temporal)
        )
        expected_cells = math.ceil(memory_frames / tt) + math.ceil(
            (compact_frames - memory_frames) / tt
        )
        if blocks != expected_cells * spatial:
            raise RuntimeError("compact CWCA block-grid cardinality mismatch")
        ids = torch.arange(blocks, device=q.device)
        bt, rem = ids // spatial, ids % spatial
        bh, bw = rem // nw, rem % nw
        local = (
            (bt[:, None] - bt[None, :]).abs()
            + (bh[:, None] - bh[None, :]).abs()
            + (bw[:, None] - bw[None, :]).abs()
        ) <= 1

        mean_degree = max(
            int(local.sum(dim=-1).max().item()),
            int(round(self.sparse_density * blocks)),
        )
        mean_degree = min(mean_degree, blocks)
        degrees = torch.full(
            (blocks,), mean_degree, device=q.device, dtype=torch.long
        )
        memory_cells = math.ceil(memory_frames / tt)
        memory_blocks = min(blocks, memory_cells * spatial)
        current_blocks = blocks - memory_blocks
        active_current_global = layout.active_frames[
            layout.active_frames >= layout.memory_temporal
        ]
        active_current_values = tuple(
            int(value)
            for value in layout.active_frame_values
            if int(value) >= int(layout.memory_temporal)
        )
        compact_budget_metadata = None
        if current_blocks > 0 and mean_degree > int(local.sum(dim=-1).max().item()):
            local_degree = int(local.sum(dim=-1).max().item())
            base = max(local_degree, math.ceil(4 * mean_degree / 5))
            current_frames = compact_frames - memory_frames
            active_current_local = (
                active_current_global - layout.memory_temporal
            )
            if closed_loop_provider is not None:
                current_degrees, compact_budget_metadata = (
                    closed_loop_provider.compact_closed_loop_degrees(
                        layer_index=layer_index,
                        active_current=active_current_global,
                        current_frame_origin=layout.memory_temporal,
                        temporal_group_size=tt,
                        spatial_blocks=spatial,
                        mean_degree=mean_degree,
                        base_degree=base,
                        total_blocks=blocks,
                        memory_blocks=memory_blocks,
                    )
                )
                if int(current_degrees.numel()) != current_blocks:
                    raise RuntimeError(
                        "compact Closed-Loop degree layout differs from packed rows"
                    )
                degrees[memory_blocks:] = current_degrees.reshape(-1)
            else:
                residual = current_blocks * (mean_degree - base)
                curvature = self._current_curvature(
                    layout.total_temporal - layout.memory_temporal
                )
                if curvature is not None and current_frames > 0:
                    frame_weight = curvature.to(q.device)[active_current_local]
                    time_cells = current_blocks // spatial
                    cell_weight = []
                    for cell in range(time_cells):
                        low = cell * tt
                        high = min(current_frames, (cell + 1) * tt)
                        cell_weight.append(
                            frame_weight[low:high].mean()
                            if low < high
                            else frame_weight.mean()
                        )
                    weight = torch.stack(cell_weight)[:, None].expand(-1, spatial)
                    weight = 1.0 + torch.log1p(
                        weight / weight.mean().clamp_min(1e-8)
                    )
                    raw = residual * weight.reshape(-1) / weight.sum().clamp_min(1e-8)
                    bonus = torch.floor(raw).long()
                    left = residual - int(bonus.sum().item())
                    if left > 0:
                        order = torch.argsort(
                            raw - bonus, descending=True, stable=True
                        )
                        bonus[order[:left]] += 1
                    degrees[memory_blocks:] = base + bonus
        # A very peaked curvature profile can assign more edges to one query
        # than there are compact key blocks.  Saturate those queries and
        # deterministically redistribute the overflow so the global budget is
        # preserved whenever capacity remains.
        degrees.clamp_(max=blocks)
        target_edges = blocks * mean_degree
        missing = target_edges - int(degrees.sum().item())
        while missing > 0:
            capacity = blocks - degrees
            candidates = torch.nonzero(capacity > 0).flatten()
            if not int(candidates.numel()):
                break
            take = min(missing, int(candidates.numel()))
            degrees[candidates[:take]] += 1
            missing -= take
        if closed_loop_provider is not None:
            if compact_budget_metadata is None:
                raise RuntimeError(
                    "compact Closed-Loop CWCA did not allocate a feedback budget"
                )
            closed_loop_provider.record_compact_budget(
                compact_budget_metadata,
                degrees,
                local_support_preserved=True,
                native_qk_topk=True,
            )
        score = score + local[None, None].to(score.dtype) * 1e6
        maximum = int(degrees.max().item())
        selected = torch.topk(score, k=maximum, dim=-1, largest=True).indices
        counts = degrees.to(torch.int32)[None, None].expand(batch, heads, -1).contiguous()
        routing_record = None
        if (
            self.fc_pasm_swap_router is not None
            and self.world_spectral_corrector is not None
            and bool(
                getattr(
                    self.world_spectral_corrector,
                    "fc_v21_bypass",
                    False,
                )
            )
        ):
            # The V21-bypass control explicitly disables FC-PASM transport.
            # In that regime the FC-aware router has no admissible causal
            # transport sketch and must return native pooled-Q/K Top-K.  Do
            # not materialize probabilities/omission proxies or run the
            # quadratic swap objective just to rediscover that no-op.  The
            # selected tensor and fixed row counts are left untouched.
            routing_record = {
                "layer_index": int(layer_index),
                "status": "independent_topk_fallback",
                "fallback_reason": "fc_v21_bypass",
                "swap_count": 0,
                "fc_cache_used": False,
                "g0_fallback": True,
                "fixed_budget": True,
                "fc_v21_bypass": True,
            }
            self.fc_pasm_swap_router._records.append(routing_record)
        elif self.fc_pasm_swap_router is not None:
            if self.world_spectral_corrector is None:
                raise RuntimeError("FC-PASM routing has no spectral corrector")
            active_routing_layers = tuple(
                int(layer)
                for layer in self.fc_pasm_swap_router.config.active_layers
            )
            if (
                self.routing_skip_inactive_layers
                and active_routing_layers
                and layer_index not in active_routing_layers
            ):
                # The router's normal inactive-layer path performs a causal
                # transport lookup only to return this same independent Top-K
                # result.  No tensor used by attention is changed here.
                transport_step = int(
                    getattr(self.world_spectral_corrector, "_step", -1)
                )
                transport_cache = getattr(
                    self.world_spectral_corrector,
                    "_fc_routing_transport",
                    {},
                )
                available_transport_layers = sorted(
                    int(key[1])
                    for key in transport_cache
                    if isinstance(key, tuple)
                    and len(key) == 2
                    and int(key[0]) == transport_step
                )
                routing_record = {
                    "layer_index": int(layer_index),
                    "status": "independent_topk_fallback",
                    "fallback_reason": "layer_not_enabled",
                    "swap_count": 0,
                    "transport_query_step": transport_step,
                    "transport_query_source_layer": int(layer_index) - 1,
                    "transport_available_layers": available_transport_layers,
                    "transport_pair_aliases": [],
                    "fc_cache_used": False,
                    "fixed_budget": True,
                    "routing_skip_inactive_layers": True,
                }
                self.fc_pasm_swap_router._records.append(routing_record)
                # The route is guaranteed to be the native Top-K result on a
                # disabled layer.  Finish this attention call immediately so
                # the candidate does not fall through to transport lookup,
                # endpoint aliasing, or omission-proxy construction.
                output_packed = MatrixJiTSpatialAcceleration._longcat_active(
                    q_packed,
                    k_packed,
                    v_packed,
                    selected.to(torch.int32),
                    counts,
                    valid,
                    block_size,
                ).transpose(1, 2)
                output = output_packed[:, token_to_packed]
                self._last_compact_attention = {
                    "compact_blocks": blocks,
                    "requested_density": self.sparse_density,
                    "mean_degree": float(degrees.float().mean().item()),
                    "minimum_degree": int(degrees.min().item()),
                    "maximum_degree": maximum,
                    "edge_budget": int(degrees.sum().item()),
                    "effective_density": float(degrees.sum().item())
                    / float(blocks * blocks),
                    "empty_blocks": 0,
                    "budget_source": (
                        compact_budget_metadata["source"]
                        if compact_budget_metadata is not None
                        else "compact_curvature"
                    ),
                    "closed_loop_previous_layer": (
                        compact_budget_metadata["previous_layer"]
                        if compact_budget_metadata is not None
                        else None
                    ),
                    "closed_loop_feedback_consumed": (
                        compact_budget_metadata is not None
                        and compact_budget_metadata["previous_layer"] is not None
                    ),
                    "native_qk_topk": True,
                    "local_support_preserved": True,
                    "routing_mode": self.routing_mode,
                    "routing": routing_record,
                }
                return module.o(output.flatten(2).to(q.dtype))
            else:
                # Keep a read-only causal-cache diagnostic next to the routing
                # record.  The router is required to consume the previous layer's
                # detached FC transport sketch; when that sketch is unavailable we
                # must be able to distinguish a real causal miss from a pair-level
                # mismatch without changing the Top-K path.
                transport_cache = getattr(
                    self.world_spectral_corrector, "_fc_routing_transport", {}
                )
                transport_step = int(
                    getattr(self.world_spectral_corrector, "_step", -1)
                )
                available_transport_layers = sorted(
                    int(key[1])
                    for key in transport_cache
                    if isinstance(key, tuple)
                    and len(key) == 2
                    and int(key[0]) == transport_step
                )
                transport = self.world_spectral_corrector.fc_pasm_routing_transport(
                    layer_index=layer_index
                )
            transport_pair_aliases: list[dict[str, tuple[int, int]]] = []
            if isinstance(transport, dict):
                # Dynamic frame selection is intentionally layer-local, so the
                # exact endpoint pair at layer ``l`` need not be present in
                # the detached FC sketch published by ``l-1``.  Preserve the
                # causal contract by aliasing only to the nearest pair from
                # the previous layer within the same Current segment.  This
                # never computes or reads current-layer U/g and never crosses
                # the C3|C4 boundary.
                previous_pairs = transport.get("pairs", {})
                if isinstance(previous_pairs, dict) and previous_pairs:
                    segment_boundary = int(self.current_segment_boundary or 4)
                    current_pairs = []
                    exact_values = active_current_values
                    exact_local_values = tuple(
                        int(value) - int(layout.memory_temporal)
                        for value in exact_values
                    )
                    for left, right in zip(exact_local_values, exact_local_values[1:]):
                        if right - left > 1 and (left < segment_boundary) == (right < segment_boundary):
                            current_pairs.append((left, right))
                    missing_pairs = [
                        pair for pair in current_pairs if pair not in previous_pairs
                    ]
                    if missing_pairs:
                        candidates = [
                            pair for pair in previous_pairs
                            if isinstance(pair, tuple)
                            and len(pair) == 2
                            and int(pair[1]) > int(pair[0])
                            and (int(pair[0]) < segment_boundary)
                            == (int(pair[1]) < segment_boundary)
                        ]
                        if candidates:
                            transport = dict(transport)
                            transport["pairs"] = dict(previous_pairs)
                            for pair in missing_pairs:
                                source = min(
                                    candidates,
                                    key=lambda candidate: (
                                        abs(int(candidate[0]) - int(pair[0]))
                                        + abs(int(candidate[1]) - int(pair[1])),
                                        abs((int(candidate[0]) + int(candidate[1])) - (int(pair[0]) + int(pair[1]))),
                                        int(candidate[0]),
                                        int(candidate[1]),
                                    ),
                                )
                                if (
                                    self.routing_pair_alias_max_distance >= 0
                                    and max(
                                        abs(int(source[0]) - int(pair[0])),
                                        abs(int(source[1]) - int(pair[1])),
                                    )
                                    > self.routing_pair_alias_max_distance
                                ):
                                    continue
                                transport["pairs"][pair] = previous_pairs[source]
                                transport_pair_aliases.append(
                                    {"requested": pair, "source": source}
                                )
            if transport is not None:
                transport = dict(transport)
                transport["attention_spatial_h"] = nh
                transport["attention_spatial_w"] = nw
            exact_current_frames = tuple(
                int(value) - int(layout.memory_temporal)
                for value in active_current_values
            )
            selected, routing_record = self.fc_pasm_swap_router.refine(
                score=score - local[None, None].to(score.dtype) * 1e6,
                pooled_value=v_content,
                selected=selected,
                counts=counts,
                local=local,
                memory_blocks=memory_blocks,
                spatial_shape=(nh, nw),
                exact_current_frames=exact_current_frames,
                temporal_group_size=tt,
                current_boundary=self.current_segment_boundary or 4,
                transport=transport,
                layer_index=layer_index,
            )
            routing_record = {
                key: (
                    float(value.detach().float().cpu().item())
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                    else [
                        float(item)
                        for item in value.detach().float().cpu().flatten().tolist()
                    ]
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in routing_record.items()
            }
            routing_record["transport_query_step"] = transport_step
            routing_record["transport_query_source_layer"] = int(layer_index) - 1
            routing_record["transport_available_layers"] = (
                available_transport_layers
            )
            # ``refine`` appends its own immutable snapshot before returning;
            # mirror the diagnostic-only cache fields into that stored row so
            # the final historical-routing trace exposes the actual causal
            # lookup state rather than only the compact-attention last record.
            router_records = getattr(self.fc_pasm_swap_router, "_records", None)
            if isinstance(router_records, list) and router_records:
                router_records[-1].update(
                    {
                        "transport_query_step": transport_step,
                        "transport_query_source_layer": int(layer_index) - 1,
                        "transport_available_layers": available_transport_layers,
                        "transport_pair_aliases": transport_pair_aliases,
                    }
                )
        output_packed = MatrixJiTSpatialAcceleration._longcat_active(
            q_packed,
            k_packed,
            v_packed,
            selected.to(torch.int32),
            counts,
            valid,
            block_size,
        ).transpose(1, 2)
        output = output_packed[:, token_to_packed]
        self._last_compact_attention = {
            "compact_blocks": blocks,
            "requested_density": self.sparse_density,
            "mean_degree": float(degrees.float().mean().item()),
            "minimum_degree": int(degrees.min().item()),
            "maximum_degree": maximum,
            "edge_budget": int(degrees.sum().item()),
            "effective_density": float(degrees.sum().item()) / float(blocks * blocks),
            "empty_blocks": 0,
            "budget_source": (
                compact_budget_metadata["source"]
                if compact_budget_metadata is not None
                else "compact_curvature"
            ),
            "closed_loop_previous_layer": (
                compact_budget_metadata["previous_layer"]
                if compact_budget_metadata is not None
                else None
            ),
            "closed_loop_feedback_consumed": (
                compact_budget_metadata is not None
                and compact_budget_metadata["previous_layer"] is not None
            ),
            "native_qk_topk": True,
            "local_support_preserved": True,
            "routing_mode": self.routing_mode,
            "routing": routing_record,
        }
        return module.o(output.flatten(2).to(q.dtype))

    def _c1a_compact_ingress(
        self,
        *,
        block_index: int,
        block: Any,
        x_valid: torch.Tensor,
        embedding: torch.Tensor,
        active_frames: torch.Tensor,
        temporal: int,
        height: int,
        width: int,
        memory_length: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        """Execute validated c1a only at canonical q0 woven ingress sites."""

        if self.compact_ingress_kernel != "c1a_native_ln_gather_adaln1":
            raise RuntimeError("c1a ingress was called while disabled")
        active_count = int(active_frames.numel())
        production_shape = (
            x_valid.shape[0] == 1
            and temporal == 19
            and memory_length == 5
            and height == 22
            and width == 40
            and x_valid.shape[-1] == 3072
            and active_count in {9, 10, 11}
        )
        if not production_shape:
            raise RuntimeError(
                "c1a ingress is validated only for Matrix B1/T19/M5/"
                "H22/W40/C3072 with 9, 10, or 11 active frames; got "
                f"B{x_valid.shape[0]}/T{temporal}/M{memory_length}/"
                f"H{height}/W{width}/C{x_valid.shape[-1]}/A{active_count}"
            )
        if self._step != 0 or block_index not in range(1, 29):
            raise RuntimeError("c1a ingress may execute only on q0 woven layers 1-28")
        frame_ids = tuple(int(value) for value in active_frames.tolist())
        if frame_ids[:memory_length] != tuple(range(memory_length)):
            raise RuntimeError("c1a ingress requires all five Memory frames exact")

        device_key = (x_valid.device.type, x_valid.device.index)
        spatial_tokens = height * width
        compiled_key = (
            device_key,
            temporal,
            spatial_tokens,
            frame_ids,
        )
        compiled = self._c1a_compiled_layouts.get(compiled_key)
        layout_compiled = compiled is None
        if compiled is None:
            compiled = compile_active_frame_layout(
                frame_ids,
                total_frames=temporal,
                spatial_tokens=spatial_tokens,
                device=x_valid.device,
            )
            self._c1a_compiled_layouts[compiled_key] = compiled
            self._c1a_layout_compilations += 1

        # Buffers depend on the layout geometry, device and dtype, not on the
        # frame-ID values.  Keep one grow-to-largest allocation per geometry
        # and slice it for smaller active sets; caching 9/10/11 separately
        # would retain several GiB without adding concurrency or reuse value.
        workspace_key = (
            device_key,
            x_valid.dtype,
            temporal,
            spatial_tokens,
            x_valid.shape[-1],
        )
        workspace_base = self._c1a_workspaces.get(workspace_key)
        workspace_allocated = (
            workspace_base is None
            or int(workspace_base.active_x.shape[1]) < compiled.active_tokens
        )
        if workspace_allocated:
            workspace_base = allocate_native_ln_gather_adaln1_workspace(
                compiled,
                channels=x_valid.shape[-1],
                dtype=x_valid.dtype,
                device=x_valid.device,
            )
            self._c1a_workspaces[workspace_key] = workspace_base
            self._c1a_workspace_allocations += 1
        if workspace_base is None:
            raise RuntimeError("c1a workspace allocation did not produce buffers")
        active_tokens = compiled.active_tokens
        workspace = NativeLNGatherAdaLN1Workspace(
            active_x=workspace_base.active_x[:, :active_tokens],
            shift_scale_gates=workspace_base.shift_scale_gates[:, :active_tokens],
            residual_gates=workspace_base.residual_gates[:, :active_tokens],
        )

        x_active, normalized, residual_gates = native_ln_gather_adaln1(
            x_valid,
            embedding,
            block.modulation,
            compiled,
            workspace=workspace,
            eps=float(getattr(block.norm1, "eps", 1e-6)),
        )
        # Preserve the six-entry downstream interface without recreating the
        # original 6-way active-e materialization.  Each entry remains an FP32
        # [B, compact_tokens, 1, C] view in canonical gate order.
        modulation = (
            *workspace.shift_scale_gates.chunk(2, dim=2),
            *residual_gates.chunk(4, dim=2),
        )
        self._c1a_calls += 1
        self._c1a_active_frame_histogram[active_count] = (
            self._c1a_active_frame_histogram.get(active_count, 0) + 1
        )
        hidden_dtype = str(x_valid.dtype)
        self._c1a_hidden_dtype_histogram[hidden_dtype] = (
            self._c1a_hidden_dtype_histogram.get(hidden_dtype, 0) + 1
        )
        record = {
            "candidate": "c1a_native_ln_gather_adaln1",
            "call_index": self._c1a_calls - 1,
            "active_frames": active_count,
            "active_frame_ids": list(frame_ids),
            "active_tokens": int(compiled.active_tokens),
            "layout_compiled": layout_compiled,
            "workspace_allocated": workspace_allocated,
            "workspace_capacity_active_tokens": int(
                workspace_base.active_x.shape[1]
            ),
            "workspace_shape_key": {
                "total_frames": temporal,
                "spatial_tokens": spatial_tokens,
                "active_frames": active_count,
                "channels": int(x_valid.shape[-1]),
                "dtype": str(x_valid.dtype),
                "device_type": x_valid.device.type,
                "device_index": x_valid.device.index,
            },
            "hidden_dtype": hidden_dtype,
            "native_fp32_layer_norm": True,
            "native_input_dtype_roundtrip": True,
            "native_bf16_rounding": x_valid.dtype == torch.bfloat16,
            "native_fp32_hidden": x_valid.dtype == torch.float32,
            "native_post_ln_modulation": True,
            "schedule_changed": False,
        }
        return x_active, normalized, modulation, record

    @staticmethod
    def _coarse_probe_block_residual(
        native_forward: Any,
        x_input: torch.Tensor,
        *,
        temporal: int,
        height: int,
        width: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Run one low-resolution target-frame probe through the same block.

        This is an explicit additional partial block forward.  It observes all
        Current frames on the native CWCA spatial tile grid (6x5 for 22x40)
        without changing the full-resolution Exact-frame schedule.
        """

        tile_h, tile_w = 4, 8

        def pool(value: torch.Tensor, trailing: tuple[int, ...]) -> torch.Tensor:
            batch = int(value.shape[0])
            grid = value.reshape(batch, temporal, height, width, *trailing)
            channels = math.prod(trailing)
            grid = grid.reshape(batch * temporal, height, width, channels).permute(
                0, 3, 1, 2
            )
            pooled = F.avg_pool2d(
                grid.float(),
                kernel_size=(tile_h, tile_w),
                stride=(tile_h, tile_w),
                ceil_mode=True,
                count_include_pad=False,
            )
            return pooled.permute(0, 2, 3, 1).reshape(
                batch, temporal, pooled.shape[-2], pooled.shape[-1], *trailing
            )

        batch, _, _, channels = x_input.shape
        pooled_x = pool(x_input, (channels,))
        coarse_h, coarse_w = int(pooled_x.shape[2]), int(pooled_x.shape[3])
        coarse_x = pooled_x.reshape(batch, temporal * coarse_h * coarse_w, channels)
        full_tokens = temporal * height * width
        embedding = kwargs["e"][:, :full_tokens]
        pooled_e = pool(embedding, tuple(int(v) for v in embedding.shape[2:]))
        coarse_kwargs = dict(kwargs)
        coarse_kwargs["e"] = pooled_e.reshape(
            batch, temporal * coarse_h * coarse_w, *embedding.shape[2:]
        )
        plucker = kwargs.get("plucker_emb")
        if isinstance(plucker, torch.Tensor):
            plucker = plucker[:, :full_tokens]
            pooled_plucker = pool(plucker, (int(plucker.shape[-1]),))
            coarse_kwargs["plucker_emb"] = pooled_plucker.reshape(
                batch, temporal * coarse_h * coarse_w, plucker.shape[-1]
            ).to(plucker.dtype)
        coarse_grid = kwargs["grid_sizes"].new_tensor(
            [[temporal, coarse_h, coarse_w]]
        )
        coarse_kwargs["grid_sizes"] = coarse_grid
        coarse_kwargs["seq_lens"] = kwargs["seq_lens"].new_full(
            kwargs["seq_lens"].shape, temporal * coarse_h * coarse_w
        )
        # Dense coarse attention avoids reusing full-grid CWCA metadata.  The
        # absolute Memory/Current content, camera injection, ActionModule,
        # cross-attention and FFN still run through the original block.
        coarse_kwargs["memory_length"] = 0
        coarse_kwargs["memory_latent_idx"] = None
        coarse_kwargs["predict_latent_idx"] = None
        coarse_kwargs["orbit_sparse_layout"] = None
        output = native_forward(coarse_x, *args, **coarse_kwargs)
        residual = (output - coarse_x).reshape(
            batch, temporal, coarse_h, coarse_w, channels
        )
        return residual.permute(0, 1, 4, 2, 3).detach()

    @staticmethod
    def _spectral_polar_residual_oracle(
        approximate_residual: torch.Tensor,
        exact_output: torch.Tensor,
        x_input: torch.Tensor,
        active_frames: torch.Tensor,
        memory_length: int,
        height: int,
        width: int,
        *,
        channel_group: int = 256,
    ) -> dict[str, Any]:
        """Read-only low-frequency amplitude/phase Oracle on inactive Current.

        Both arms start from the identical block input.  Exact frames and
        Memory are excluded from the metric, and no Oracle tensor is returned
        to the generation path.
        """
        temporal = int(approximate_residual.shape[1])
        inactive = torch.ones(temporal, dtype=torch.bool, device=active_frames.device)
        inactive[:memory_length] = False
        inactive[active_frames.to(torch.long)] = False
        target_frames = torch.nonzero(inactive, as_tuple=False).flatten()
        if int(target_frames.numel()) == 0:
            raise RuntimeError("spectral-polar probe has no inactive Current frames")
        exact_residual = exact_output.reshape_as(x_input) - x_input
        approx = approximate_residual[:, target_frames].reshape(
            -1, height, width, approximate_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()
        exact = exact_residual[:, target_frames].reshape(
            -1, height, width, exact_residual.shape[-1]
        ).permute(0, 3, 1, 2).float()

        fy = torch.fft.fftfreq(height, device=approx.device)[:, None]
        fx = torch.fft.rfftfreq(width, device=approx.device)[None, :]
        radius = torch.sqrt(fx.square() + fy.square())
        radius = radius / radius.max().clamp_min(1e-12)
        centers = torch.linspace(0.0, 1.0, 4, device=approx.device)
        bands = torch.exp(
            -0.5 * ((radius[None] - centers[:, None, None]) / 0.20).square()
        )
        bands = bands / bands.sum(dim=0, keepdim=True).clamp_min(1e-12)
        low_mask = bands[:2].sum(dim=0)[None, None]

        sums = {
            "baseline": 0.0,
            "low_error": 0.0,
            "amplitude_only": 0.0,
            "phase_only": 0.0,
            "amplitude_and_phase": 0.0,
            "exact_energy": 0.0,
        }
        channels = int(approx.shape[1])
        for start in range(0, channels, channel_group):
            stop = min(start + channel_group, channels)
            a = approx[:, start:stop]
            e = exact[:, start:stop]
            fa = torch.fft.rfft2(a, dim=(-2, -1))
            fe = torch.fft.rfft2(e, dim=(-2, -1))
            unit_a = fa / fa.abs().clamp_min(1e-20)
            unit_e = fe / fe.abs().clamp_min(1e-20)
            baseline = a - e
            sums["baseline"] += float(baseline.square().sum().item())
            sums["exact_energy"] += float(e.square().sum().item())
            low_error = torch.fft.irfft2(
                (fa - fe) * low_mask,
                s=(height, width),
                dim=(-2, -1),
            )
            sums["low_error"] += float(low_error.square().sum().item())
            oracle_spectra = {
                "amplitude_only": (
                    (1.0 - low_mask) * fa + low_mask * (fe.abs() * unit_a)
                ),
                "phase_only": (
                    (1.0 - low_mask) * fa + low_mask * (fa.abs() * unit_e)
                ),
                "amplitude_and_phase": (
                    (1.0 - low_mask) * fa + low_mask * fe
                ),
            }
            for name, spectrum in oracle_spectra.items():
                reconstructed = torch.fft.irfft2(
                    spectrum, s=(height, width), dim=(-2, -1)
                )
                sums[name] += float((reconstructed - e).square().sum().item())

        baseline = max(sums["baseline"], 1e-30)
        return {
            "diagnostic_only": True,
            "output_mutation": False,
            "reference": "dense_same_block_same_input",
            "domain": "inactive_current_whole_block_residual",
            "high_frequency": "approximate",
            "low_frequency_bands": [0, 1],
            "smooth_radial_bands": 4,
            "target_current_frames": [
                int(value) - memory_length for value in target_frames.tolist()
            ],
            "active_current_frames": [
                int(value) - memory_length
                for value in active_frames.tolist()
                if int(value) >= memory_length
            ],
            "baseline_relative_l2": float(
                (sums["baseline"] / max(sums["exact_energy"], 1e-30)) ** 0.5
            ),
            "low_error_energy_fraction": float(sums["low_error"] / baseline),
            "squared_error_recovery": {
                name: float(1.0 - sums[name] / baseline)
                for name in (
                    "amplitude_only", "phase_only", "amplitude_and_phase"
                )
            },
        }

    def _forward_block(
        self,
        block_index: int,
        block: Any,
        native_forward: Any,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self._active:
            return native_forward(x, *args, **kwargs)
        grid_sizes = kwargs["grid_sizes"]
        temporal, height, width = (int(value) for value in grid_sizes[0].tolist())
        memory_length = int(kwargs.get("memory_length", 0))
        current_temporal = temporal - memory_length
        if self._call_current_temporal is None:
            self._call_current_temporal = current_temporal
        elif self._call_current_temporal != current_temporal:
            raise RuntimeError("Matrix layers disagree on Current-frame geometry")
        if (
            block_index == 0
            and self.feature_curvature_fallback
            and self._curvature_profile is None
        ):
            valid_tokens = temporal * height * width
            frames = x[:, :valid_tokens].detach().float().reshape(
                x.shape[0], temporal, height * width, x.shape[-1]
            )
            current = frames[:, memory_length:]
            magnitudes = torch.linalg.vector_norm(current, dim=-1)
            curvature = torch.zeros(
                current.shape[0], current.shape[1],
                device=x.device, dtype=torch.float32,
            )
            if current.shape[1] >= 3:
                second = magnitudes[:, 2:] - 2.0 * magnitudes[:, 1:-1] + magnitudes[:, :-2]
                numerator = torch.linalg.vector_norm(second, dim=-1)
                denominator = torch.linalg.vector_norm(
                    magnitudes[:, 1:-1], dim=-1
                ).clamp_min(torch.finfo(torch.float32).eps)
                curvature[:, 1:-1] = numerator / denominator
            self._curvature_profile = {
                "frame_curvature": curvature.mean(dim=0),
                "source": "current_feature_channel_l2_second_difference",
            }
        # Historical frame-weave variants leave q1 to released LI: cache hits
        # return before this wrapper and cache misses execute q1 exactly.  A
        # variant that explicitly lists step 1 instead applies the same
        # complete-frame weave used by q0/q2.
        if self._step == 1 and (
            self.sparse_steps is None or 1 not in self.sparse_steps
        ):
            output = native_forward(x, *args, **kwargs)
            self._call_layers.append(
                {
                    "block_index": block_index,
                    "full_layer": True,
                    "active_frames": temporal,
                    "total_frames": temporal,
                    "active_ratio": 1.0,
                    "high_curvature_current_frames": 0,
                    "bootstrap_q1_exact": True,
                }
            )
            return output
        active_frames, high_count = self._active_frames(
            block_index=block_index,
            total_temporal=temporal,
            memory_temporal=memory_length,
            device=x.device,
            current_has_action_module=block.action_model is not None,
        )
        full_layer = int(active_frames.numel()) == temporal
        if full_layer:
            output = native_forward(x, *args, **kwargs)
            if self.reconstruction in {
                "scheduler_defect_residual",
                "scheduler_feature_defect_residual",
                "scheduler_affine_residual",
                "scheduler_chord_defect_residual",
            }:
                valid_tokens = temporal * height * width
                full_residual = (
                    output[:, :valid_tokens] - x[:, :valid_tokens]
                ).reshape(
                    x.shape[0], temporal, height * width, x.shape[-1]
                )
                if self._step == 0:
                    self._scheduler_q0_residuals[block_index] = (
                        full_residual.detach()
                    )
                elif self._step == 2:
                    # Full q2 boundary/guard layers do not need the q0 control
                    # variate, but must consume it to keep the cache contract
                    # auditable and bounded.
                    self._scheduler_q0_residuals.pop(block_index, None)
            if self.reconstruction == "depth_transport":
                valid_tokens = temporal * height * width
                self._previous_full_residual = (
                    output[:, :valid_tokens] - x[:, :valid_tokens]
                ).reshape(
                    x.shape[0], temporal, height * width, x.shape[-1]
                )
            layer_record = {
                "block_index": block_index,
                "full_layer": True,
                "active_frames": temporal,
                "total_frames": temporal,
                "active_ratio": 1.0,
                "high_curvature_current_frames": high_count,
                "curvature_anchor_current_indices": list(
                    self._last_curvature_anchor_indices
                ),
                "dynamic_frame_selection": self._last_dynamic_frame_selection,
                "window_forward_exact": (
                    block_index in self._window_forward_exact_layers
                ),
            }
            # Layer 15 is both the end of the first routed window and the
            # observation for the second.  When the first window is exact,
            # measure its witness on the hypothetical V8 active anchors from
            # the already-computed dense output; no paired shadow execution is
            # needed and the second decision remains causal.
            if block_index in self.window_forward_targets and memory_length > 0:
                router_frames, _ = self._active_frames(
                    block_index=block_index,
                    total_temporal=temporal,
                    memory_temporal=memory_length,
                    device=x.device,
                    ignore_window_router=True,
                    current_has_action_module=block.action_model is not None,
                )
                if int(router_frames.numel()) < temporal:
                    valid_tokens = temporal * height * width
                    input_grid = x[:, :valid_tokens].reshape(
                        x.shape[0], temporal, height * width, x.shape[-1]
                    )
                    output_grid = output[:, :valid_tokens].reshape_as(input_grid)
                    probe = self._witness_defect(
                        active_values=(output_grid - input_grid)[:, router_frames],
                        active_inputs=input_grid[:, router_frames],
                        active_frames=router_frames,
                        total_temporal=temporal,
                        memory_temporal=memory_length,
                    )
                    layer_record["window_router"] = self._update_window_router(
                        block_index, probe
                    )
            self._call_layers.append(layer_record)
            return output

        probe_reference = None
        if block_index in self.witness_probe_layers:
            # The probe is development-only and deliberately pays for an
            # additional dense temporal execution from the identical block
            # input.  It never feeds the dense result into generation.
            probe_reference = native_forward(x, *args, **kwargs)

        polar_reference = None
        polar_oracle = None
        polar_probe_active = bool(
            block_index in self.spectral_polar_probe_layers
            and self._chunk == self.spectral_polar_probe_chunk
            and self._step == self.spectral_polar_probe_step
        )
        if polar_probe_active:
            # Suppress Closed-Loop response observation during the shadow
            # execution.  It is a read-only Exact reference, not part of the
            # causal generation trajectory or the next-layer CWCA budget.
            probe_observer = (
                self.geometry_provider
                if getattr(self.geometry_provider, "mode", None) == "closed_loop"
                and hasattr(self.geometry_provider, "begin_compact_layer")
                else None
            )
            if probe_observer is not None:
                probe_observer.begin_compact_layer(block_index)
            try:
                polar_reference = native_forward(x, *args, **kwargs)
            finally:
                if probe_observer is not None:
                    probe_observer.end_compact_layer(block_index)

        valid_tokens = temporal * height * width
        if x.shape[1] < valid_tokens:
            raise RuntimeError("frame-weave input is shorter than the Matrix grid")
        x_valid = x[:, :valid_tokens]
        tail = x[:, valid_tokens:]
        orbit_layout = kwargs.get("orbit_sparse_layout")
        x_input = x_valid.reshape(
            x_valid.shape[0], temporal, height * width, x_valid.shape[-1]
        )
        coarse_probe_residual = None
        if (
            self.world_spectral_corrector is not None
            and self.world_spectral_variant
            == "coarse_probe_lowfreq_self_calibrated"
        ):
            probe_observer = (
                self.geometry_provider
                if getattr(self.geometry_provider, "mode", None) == "closed_loop"
                and hasattr(self.geometry_provider, "begin_compact_layer")
                else None
            )
            if probe_observer is not None:
                probe_observer.begin_compact_layer(block_index)
            try:
                coarse_probe_residual = self._profile_call(
                    "coarse_probe_block_forward",
                    block_index,
                    self._coarse_probe_block_residual,
                    native_forward,
                    x_input,
                    temporal=temporal,
                    height=height,
                    width=width,
                    args=args,
                    kwargs=kwargs,
                )
            finally:
                if probe_observer is not None:
                    probe_observer.end_compact_layer(block_index)
        full_plucker = kwargs.get("plucker_emb")
        self._last_compact_attention = None
        self._last_ray_transport = None
        full_query_attention_delta = None
        compact_ingress_record = None
        if self.asymmetric_topology_selector is None:
            layout = self._profile_call(
                "compact_layout_build",
                block_index,
                self._layout,
                x=x_valid,
                grid_sizes=grid_sizes,
                memory_length=memory_length,
                active_frames=active_frames,
                orbit_sparse_layout=orbit_layout,
            )
            compact_indices = layout.compact_indices
            if self.compact_ingress_kernel is not None:
                (
                    x_active_input,
                    normalized,
                    modulation,
                    compact_ingress_record,
                ) = self._c1a_compact_ingress(
                    block_index=block_index,
                    block=block,
                    x_valid=x_valid,
                    embedding=kwargs["e"][:, :valid_tokens],
                    active_frames=active_frames,
                    temporal=temporal,
                    height=height,
                    width=width,
                    memory_length=memory_length,
                )
            else:
                x_active_input = x_valid[:, compact_indices]
                e = kwargs["e"][:, compact_indices]
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    modulation = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
                    normalized = (
                        block.norm1(x_active_input).float()
                        * (1 + modulation[1].squeeze(2))
                        + modulation[0].squeeze(2)
                    ).to(x_active_input.dtype)
            with torch.amp.autocast("cuda", dtype=torch.float32):
                attention_out = self._profile_call(
                    "compact_self_attention",
                    block_index,
                    self._self_attention,
                    block.self_attn,
                    normalized,
                    layout,
                    seq_lens=kwargs["seq_lens"],
                    freqs=kwargs["freqs"],
                    memory_length=memory_length,
                    memory_latent_idx=kwargs.get("memory_latent_idx"),
                    predict_latent_idx=kwargs.get("predict_latent_idx"),
                    fa_version=kwargs.get("fa_version"),
                    orbit_sparse_layout=orbit_layout,
                )
                active = x_active_input + attention_out * modulation[2].squeeze(2)
        else:
            # Active Q rows retain the unchanged full 19-frame CWCA K/V state.
            # The existing asymmetric kernel is reused without changing the
            # selector or its fixed sparse edge budget.
            self.topology_selector = self.asymmetric_topology_selector
            query_frames = (
                torch.arange(temporal, device=x.device)
                if self.target_q_attention_correction
                else active_frames
            )
            layout = MatrixCausalFrameLayerRelay._asymmetric_layout(
                self,
                x=x_valid,
                grid_sizes=grid_sizes,
                memory_length=memory_length,
                orbit_sparse_layout=orbit_layout,
                query_frames=query_frames,
                allow_memory_queries=True,
            )
            if self.target_q_attention_correction:
                spatial_index = torch.arange(height * width, device=x.device)
                compact_indices = (
                    active_frames[:, None] * (height * width)
                    + spatial_index[None, :]
                ).reshape(-1)
            else:
                compact_indices = layout.query_indices
            x_active_input = x_valid[:, compact_indices]
            e = kwargs["e"][:, compact_indices]
            with torch.amp.autocast("cuda", dtype=torch.float32):
                modulation = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
                full_e = (block.modulation.unsqueeze(0) + kwargs["e"][:, :valid_tokens]).chunk(
                    6, dim=2
                )
                normalized_full = (
                    block.norm1(x_valid).float()
                    * (1 + full_e[1].squeeze(2))
                    + full_e[0].squeeze(2)
                ).to(x_valid.dtype)
                query_attention_out = MatrixCausalFrameLayerRelay._current_query_attention(
                    self,
                    block.self_attn,
                    normalized_full,
                    layout,
                    freqs=kwargs["freqs"],
                    memory_length=memory_length,
                    memory_latent_idx=kwargs.get("memory_latent_idx"),
                    predict_latent_idx=kwargs.get("predict_latent_idx"),
                    orbit_sparse_layout=orbit_layout,
                )
                if self.target_q_attention_correction:
                    attention_out = query_attention_out[:, compact_indices]
                    full_query_attention_delta = (
                        query_attention_out * full_e[2].squeeze(2)
                    ).reshape(
                        x_valid.shape[0],
                        temporal,
                        height * width,
                        x_valid.shape[-1],
                    )
                else:
                    attention_out = query_attention_out
                active = x_active_input + attention_out * modulation[2].squeeze(2)
        plucker = full_plucker
        if plucker is not None:
            plucker = plucker[:, compact_indices]
        if self.reconstruction == "target_q_ffn_lift":
            if full_query_attention_delta is None:
                raise RuntimeError("target-Q FFN lift lacks full attention output")
            dense = x_valid + full_query_attention_delta.reshape_as(x_valid)
            if full_plucker is not None:
                full_camera_input = full_plucker[:, :valid_tokens]
                full_camera = block.cam_injector_layer2(
                    F.silu(block.cam_injector_layer1(full_camera_input))
                ) + full_camera_input
                dense = (
                    (1.0 + block.cam_scale_layer(full_camera)) * dense
                    + block.cam_shift_layer(full_camera)
                )
            dense = block.norm3(dense)
            dense = dense + block.cross_attn(
                dense,
                kwargs["context"],
                kwargs.get("context_lens"),
                fa_version=kwargs.get("fa_version"),
            )
            if block.action_model is not None:
                dense = block.action_model(
                    dense.to(block.ffn[0].weight.dtype),
                    temporal,
                    height,
                    width,
                    kwargs.get("mouse_cond"),
                    kwargs.get("keyboard_cond"),
                    kwargs.get("mouse_cond_memory"),
                    kwargs.get("keyboard_cond_memory"),
                )
            active_pre_ffn = dense[:, compact_indices]
            active_ffn = block.ffn(
                (
                    block.norm2(active_pre_ffn).float()
                    * (1 + modulation[4].squeeze(2))
                    + modulation[3].squeeze(2)
                ).to(block.ffn[0].weight.dtype)
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                active_ffn_delta = (
                    active_ffn * modulation[5].squeeze(2)
                ).reshape(
                    x_valid.shape[0],
                    len(active_frames),
                    height * width,
                    x_valid.shape[-1],
                )
            full_ffn_delta = self._lift_temporal(
                active_ffn_delta,
                active_frames,
                temporal,
                memory_length,
                self.current_segment_boundary,
                "linear",
            )
            output_valid = dense + full_ffn_delta.reshape_as(dense)
            output = (
                torch.cat([output_valid, tail], dim=1)
                if tail.numel()
                else output_valid
            )
            self._call_layers.append(
                {
                    "block_index": block_index,
                    "full_layer": False,
                    "active_frames": int(active_frames.numel()),
                    "total_frames": temporal,
                    "active_ratio": float(active_frames.numel()) / float(temporal),
                    "high_curvature_current_frames": high_count,
                    "curvature_anchor_current_indices": list(
                        self._last_curvature_anchor_indices
                    ),
                    "dynamic_frame_selection": self._last_dynamic_frame_selection,
                    "phase": (
                        block_index - 1
                        + self.scheduler_phase_offsets.get(self._step, 0)
                    )
                    % self.scheduler_phase_schedule.get(
                        self._step, (self.phase_period, self.active_phases)
                    )[0],
                    "action_bridge_dense": block.action_model is not None,
                    "stationary_worldline": self._last_stationary_worldline,
                    "complete_spatial_frames": True,
                    "compact_attention": self._last_compact_attention,
                    "ray_aligned_transport": None,
                    "target_q_condition_path_dense": True,
                    "ffn_delta_segmented_lift": True,
                }
            )
            return output
        if self.reconstruction == "attention_delta":
            # Approximate only the expensive temporal self-attention update.
            # Every subsequent world-model operation remains native on every
            # Memory/Current frame, so inactive frames retain their own
            # camera, text, action, and nonlinear FFN response.
            active_attention_delta = (
                attention_out * modulation[2].squeeze(2)
            ).reshape(
                x_valid.shape[0],
                len(active_frames),
                height * width,
                x_valid.shape[-1],
            )
            full_attention_delta = self._lift_temporal(
                active_attention_delta,
                active_frames,
                temporal,
                memory_length,
                self.current_segment_boundary,
                self.temporal_interpolation,
            )
            dense = x_valid + full_attention_delta.reshape_as(x_valid)
            full_e = kwargs["e"][:, :valid_tokens]
            with torch.amp.autocast("cuda", dtype=torch.float32):
                full_modulation = (
                    block.modulation.unsqueeze(0) + full_e
                ).chunk(6, dim=2)
            if full_plucker is not None:
                full_camera = block.cam_injector_layer2(
                    F.silu(block.cam_injector_layer1(full_plucker[:, :valid_tokens]))
                ) + full_plucker[:, :valid_tokens]
                dense = (
                    (1.0 + block.cam_scale_layer(full_camera)) * dense
                    + block.cam_shift_layer(full_camera)
                )
            dense = block.norm3(dense)
            dense = dense + block.cross_attn(
                dense,
                kwargs["context"],
                kwargs.get("context_lens"),
                fa_version=kwargs.get("fa_version"),
            )
            if block.action_model is not None:
                dense = block.action_model(
                    dense.to(block.ffn[0].weight.dtype),
                    temporal,
                    height,
                    width,
                    kwargs.get("mouse_cond"),
                    kwargs.get("keyboard_cond"),
                    kwargs.get("mouse_cond_memory"),
                    kwargs.get("keyboard_cond_memory"),
                )
            dense_ffn = block.ffn(
                (
                    block.norm2(dense).float()
                    * (1 + full_modulation[4].squeeze(2))
                    + full_modulation[3].squeeze(2)
                ).to(block.ffn[0].weight.dtype)
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                output_valid = (
                    dense + dense_ffn * full_modulation[5].squeeze(2)
                )
            output = (
                torch.cat([output_valid, tail], dim=1)
                if tail.numel()
                else output_valid
            )
            self._call_layers.append(
                {
                    "block_index": block_index,
                    "full_layer": False,
                    "active_frames": int(active_frames.numel()),
                    "total_frames": temporal,
                    "active_ratio": float(active_frames.numel()) / float(temporal),
                    "high_curvature_current_frames": high_count,
                    "curvature_anchor_current_indices": list(
                        self._last_curvature_anchor_indices
                    ),
                    "dynamic_frame_selection": self._last_dynamic_frame_selection,
                    "phase": (
                        block_index - 1
                        + self.scheduler_phase_offsets.get(self._step, 0)
                    )
                    % self.scheduler_phase_schedule.get(
                        self._step, (self.phase_period, self.active_phases)
                    )[0],
                    "action_bridge_dense": block.action_model is not None,
                    "stationary_worldline": self._last_stationary_worldline,
                    "complete_spatial_frames": True,
                    "compact_attention": self._last_compact_attention,
                    "ray_aligned_transport": None,
                    "attention_delta_segmented_lift": True,
                    "post_attention_world_model_dense": True,
                }
            )
            return output
        compact_response_provider = (
            self.geometry_provider
            if getattr(self.geometry_provider, "mode", None) == "closed_loop"
            and hasattr(self.geometry_provider, "record_compact_layer_response")
            and self.geometry_provider.observing_compact_layer(block_index)
            else None
        )
        compact_response_capture_active = bool(
            compact_response_provider is not None
            and (
                not self.runtime_gate_unused_control_response
                or self.geometry_provider.response_observation_active()
            )
        )
        camera_before_dense = None
        camera_after_dense = None
        response_active_frame_values = None
        if compact_response_capture_active and self.compact_active_response_only:
            selection = self._last_dynamic_frame_selection
            if not isinstance(selection, dict):
                raise RuntimeError("active-only response lacks dynamic selection")
            exact_current_frames = selection.get("exact_current_frames")
            if not isinstance(exact_current_frames, list) or not all(
                isinstance(value, int) for value in exact_current_frames
            ):
                raise RuntimeError(
                    "active-only response lacks exact Current frame IDs"
                )
            response_active_frame_values = (
                *range(memory_length),
                *(
                    memory_length + int(value)
                    for value in exact_current_frames
                ),
            )
            expected_active_frames = torch.tensor(
                response_active_frame_values,
                device=active_frames.device,
                dtype=active_frames.dtype,
            )
            torch._assert_async(
                torch.all(active_frames == expected_active_frames),
                "active-only response frame IDs disagree with selection",
            )
        if compact_response_capture_active and not self.compact_active_response_only:
            if full_plucker is None:
                raise RuntimeError(
                    "compact Closed-Loop response requires Camera Injection"
                )
            # FrameWeave's inactive hidden state already has a defined,
            # segmented reconstruction.  Observe the real Camera Injection
            # module on that full approximate state, without feeding the
            # diagnostic inactive rows back into generation.
            def dense_camera_response() -> tuple[torch.Tensor, torch.Tensor]:
                before = self._profile_call(
                    "response_camera_dense_lift",
                    block_index,
                    self._lift_temporal,
                    active.reshape(
                        active.shape[0],
                        len(active_frames),
                        height * width,
                        active.shape[-1],
                    ),
                    active_frames,
                    temporal,
                    memory_length,
                    self.current_segment_boundary,
                    self.temporal_interpolation,
                    self.runtime_optimized,
                    (
                        self.runtime_vectorized_reconstruction
                        or self.runtime_vectorized_linear_lift
                    ),
                ).reshape(active.shape[0], valid_tokens, active.shape[-1])
                full_camera_input = full_plucker[:, :valid_tokens]
                full_camera = block.cam_injector_layer2(
                    F.silu(block.cam_injector_layer1(full_camera_input))
                ) + full_camera_input
                after = (
                    (1.0 + block.cam_scale_layer(full_camera)) * before
                    + block.cam_shift_layer(full_camera)
                )
                return before, after

            camera_before_dense, camera_after_dense = self._profile_call(
                "response_camera_dense_total",
                block_index,
                dense_camera_response,
            )
        camera_before_active = active
        if plucker is not None:
            def compact_camera_update() -> torch.Tensor:
                camera = block.cam_injector_layer2(
                    F.silu(block.cam_injector_layer1(plucker))
                ) + plucker
                return (
                    (1.0 + block.cam_scale_layer(camera)) * active
                    + block.cam_shift_layer(camera)
                )

            active = self._profile_call(
                "compact_camera_update",
                block_index,
                compact_camera_update,
            )
        camera_after_active = active
        active = block.norm3(active)
        active = active + self._profile_call(
            "compact_cross_attention",
            block_index,
            block.cross_attn,
            active,
            kwargs["context"],
            kwargs.get("context_lens"),
            fa_version=kwargs.get("fa_version"),
        )
        dense_action_bridge = None
        action_before_dense = None
        action_after_dense = None
        action_before_active = None
        action_after_active = None
        if block.action_model is not None:
            active_grid = active.reshape(
                active.shape[0], len(active_frames), height * width, active.shape[-1]
            )
            action_input = self._profile_call(
                "action_dense_lift",
                block_index,
                self._lift_temporal,
                active_grid,
                active_frames,
                temporal,
                memory_length,
                self.current_segment_boundary,
                self.temporal_interpolation,
                self.runtime_optimized,
                (
                    self.runtime_vectorized_reconstruction
                    or self.runtime_vectorized_linear_lift
                ),
            ).reshape(active.shape[0], valid_tokens, active.shape[-1])
            action_output = self._profile_call(
                "dense_action_model",
                block_index,
                block.action_model,
                action_input.to(block.ffn[0].weight.dtype),
                temporal,
                height,
                width,
                kwargs.get("mouse_cond"),
                kwargs.get("keyboard_cond"),
                kwargs.get("mouse_cond_memory"),
                kwargs.get("keyboard_cond_memory"),
            )
            dense_action_bridge = action_output.reshape(
                action_output.shape[0], temporal, height * width, action_output.shape[-1]
            )
            action_before_dense = action_input
            action_after_dense = action_output
            action_before_active = action_input[:, compact_indices]
            action_after_active = action_output[:, compact_indices]
            active = action_output[:, compact_indices]
        if compact_response_provider is not None:
            if not compact_response_capture_active:
                pass
            elif self.compact_active_response_only:
                if response_active_frame_values is None:
                    raise RuntimeError(
                        "compact active response lacks chronological frame IDs"
                    )
                self._profile_call(
                    "fine_control_response_observer_active",
                    block_index,
                    compact_response_provider.record_compact_active_layer_response,
                    block_index,
                    camera_before=camera_before_active,
                    camera_after=camera_after_active,
                    action_before=action_before_active,
                    action_after=action_after_active,
                    active_frame_values=response_active_frame_values,
                    grid=(temporal, height, width),
                    memory_length=memory_length,
                )
            else:
                if camera_before_dense is None or camera_after_dense is None:
                    raise RuntimeError(
                        "compact Closed-Loop camera response is incomplete"
                    )
                self._profile_call(
                    "fine_control_response_observer",
                    block_index,
                    compact_response_provider.record_compact_layer_response,
                    block_index,
                    camera_before=camera_before_dense,
                    camera_after=camera_after_dense,
                    action_before=action_before_dense,
                    action_after=action_after_dense,
                    grid=(temporal, height, width),
                    memory_length=memory_length,
                )
            compact_response_provider.end_compact_layer(block_index)
        ffn = self._profile_call(
            "compact_ffn",
            block_index,
            block.ffn,
            (
                block.norm2(active).float()
                * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            ).to(block.ffn[0].weight.dtype),
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            active = active + ffn * modulation[5].squeeze(2)
        active_output = active.reshape(
            active.shape[0], len(active_frames), height * width, active.shape[-1]
        )
        witness_values = active_output
        if self.reconstruction == "depth_transport":
            active_residual = active_output - x_input[:, active_frames]
            witness_values = active_residual
            previous = self._previous_full_residual
            if previous is None or previous.shape != x_input.shape:
                full_residual = self._lift_temporal(
                    active_residual,
                    active_frames,
                    temporal,
                    memory_length,
                    self.current_segment_boundary,
                    self.temporal_interpolation,
                )
            else:
                # Fit the layer-to-layer residual law only on exact anchors in
                # the domain being woven.  The complementary domain is exact
                # and should not dominate the transport calibration.
                if self.weave_domain == "r4_memory" or (
                    self.weave_domain == "stationary_dual" and self._step == 2
                ):
                    fit_mask = active_frames < memory_length
                else:
                    fit_mask = active_frames >= memory_length
                previous_active = previous[:, active_frames]
                if bool(fit_mask.any()):
                    source = previous_active[:, fit_mask].float()
                    target = active_residual[:, fit_mask].float()
                    dimensions = tuple(range(target.ndim - 1))
                    numerator = (source * target).sum(dim=dimensions)
                    denominator = source.square().sum(dim=dimensions).clamp_min(1e-8)
                    scale = (numerator / denominator).to(previous.dtype)
                    full_residual = previous * scale
                else:
                    full_residual = previous.clone()
                full_residual = full_residual.clone()
                full_residual[:, active_frames] = active_residual
            full_output = x_input + full_residual
            self._previous_full_residual = full_residual
        elif self.reconstruction == "action_bridge" and dense_action_bridge is not None:
            # Matrix's native ActionModule has already re-synchronized every
            # complete world frame with the current controls.  Preserve that
            # dense control response exactly and lift only the subsequent FFN
            # update.  Discarding the inactive action outputs would erase the
            # very world-model-specific signal we paid to compute.
            active_residual = (
                active_output - dense_action_bridge[:, active_frames]
            )
            witness_values = active_residual
            full_residual = self._lift_temporal(
                active_residual,
                active_frames,
                temporal,
                memory_length,
                self.current_segment_boundary,
                self.temporal_interpolation,
            )
            full_output = dense_action_bridge + full_residual
        elif self.reconstruction == "ray_aligned_residual":
            active_residual = active_output - x_input[:, active_frames]
            witness_values = active_residual
            alignment_features = (
                dense_action_bridge
                if dense_action_bridge is not None
                else x_input
            )
            full_residual = self._ray_aligned_residual_lift(
                active_residual,
                active_frames,
                temporal,
                memory_length,
                spatial_height=height,
                spatial_width=width,
                full_plucker=full_plucker,
                alignment_features=alignment_features,
                alignment_feature_source=(
                    "action_module"
                    if dense_action_bridge is not None
                    else "block_input"
                ),
            )
            full_output = x_input + full_residual
        elif self.reconstruction in {
            "residual",
            "action_bridge",
            "secant_residual",
            "q1_secant_residual",
            "multi_secant_residual",
            "multi_secant_polar_residual",
            "output_dc_residual_detail",
            "input_tangent_residual",
            "input_parallel_residual",
            "input_velocity_residual",
            "convex_chord_residual",
            "feature_attention_residual",
            "feature_barycentric_residual",
            "control_space_residual",
            "control_barycentric_residual",
            "scheduler_defect_residual",
            "scheduler_feature_defect_residual",
            "scheduler_affine_residual",
            "scheduler_chord_defect_residual",
        }:
            active_residual = active_output - x_input[:, active_frames]
            witness_values = active_residual
            runtime_active_frame_values = None
            if self.reconstruction == "input_tangent_residual":
                full_residual = self._input_tangent_residual_lift(
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
            elif self.reconstruction == "input_velocity_residual":
                full_residual = self._input_velocity_residual_lift(
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
            elif self.reconstruction == "convex_chord_residual":
                full_residual = self._convex_input_chord_residual_lift(
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
            elif self.reconstruction == "feature_attention_residual":
                full_residual = self._feature_attention_residual_lift(
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
            elif self.reconstruction == "feature_barycentric_residual":
                full_residual = self._feature_barycentric_residual_lift(
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
            elif self.reconstruction == "control_space_residual":
                (
                    full_residual,
                    self._last_control_space_reconstruction,
                ) = self._control_space_residual_lift(
                    active_residual,
                    active_frames,
                    temporal,
                    memory_length,
                    self.current_segment_boundary,
                    full_plucker,
                    kwargs.get("mouse_cond"),
                    kwargs.get("keyboard_cond"),
                    kwargs.get("mouse_cond_memory"),
                    kwargs.get("keyboard_cond_memory"),
                    self.control_residual_lambdas,
                )
            elif self.reconstruction == "control_barycentric_residual":
                barycentric_workspace = self._barycentric_contraction_workspace(
                    active_residual, temporal
                )
                runtime_active_frame_values = None
                if self.runtime_reuse_dynamic_active_frame_list:
                    if not hasattr(layout, "active_frame_values"):
                        raise RuntimeError(
                            "active-frame reuse requires the compiled FrameLayout"
                        )
                    runtime_active_frame_values = layout.active_frame_values
                (
                    full_residual,
                    self._last_control_space_reconstruction,
                ) = self._profile_call(
                    "control_barycentric_reconstruction",
                    block_index,
                    self._control_barycentric_residual_lift,
                    active_residual,
                    active_frames,
                    temporal,
                    memory_length,
                    self.current_segment_boundary,
                    full_plucker,
                    kwargs.get("mouse_cond"),
                    kwargs.get("keyboard_cond"),
                    kwargs.get("mouse_cond_memory"),
                    kwargs.get("keyboard_cond_memory"),
                    self.control_residual_lambdas,
                    self.runtime_optimized,
                    self.runtime_vectorized_reconstruction,
                    (
                        self.runtime_reuse_residual_output
                        or self.runtime_single_anchor_write
                    ),
                    barycentric_workspace,
                    (
                        self._barycentric_control_state_cache
                        if self.runtime_cache_barycentric_control
                        else None
                    ),
                    (
                        self._barycentric_weight_cache
                        if self.runtime_cache_barycentric_weights
                        else None
                    ),
                    self.runtime_batched_barycentric_contraction,
                    self.runtime_direct_barycentric_pair_kernel,
                    runtime_active_frame_values,
                )
                if self.runtime_reuse_dynamic_active_frame_list:
                    if (
                        self._last_control_space_reconstruction[
                            "active_frame_source"
                        ]
                        != "dynamic_selector_cpu_trace"
                    ):
                        raise RuntimeError(
                            "dynamic active-frame list reuse was not applied"
                        )
                    self._barycentric_active_frame_list_reuse_calls += 1
                if self.runtime_cache_barycentric_control:
                    if self._last_control_space_reconstruction[
                        "control_state_cache_hit"
                    ]:
                        self._barycentric_control_cache_hits += 1
                    else:
                        self._barycentric_control_cache_misses += 1
                if self.runtime_cache_barycentric_weights:
                    self._barycentric_weight_cache_hits += int(
                        self._last_control_space_reconstruction["weight_cache_hits"]
                    )
                    self._barycentric_weight_cache_misses += int(
                        self._last_control_space_reconstruction["weight_cache_misses"]
                    )
                if barycentric_workspace is not None:
                    self._barycentric_preallocated_contractions += int(
                        self._last_control_space_reconstruction["targets"]
                    )
                self._barycentric_batched_contraction_calls += int(
                    self._last_control_space_reconstruction[
                        "batched_contraction_calls"
                    ]
                )
                self._barycentric_direct_pair_kernel_calls += int(
                    self._last_control_space_reconstruction[
                        "direct_pair_kernel_calls"
                    ]
                )
            elif self.reconstruction in {
                "scheduler_defect_residual",
                "scheduler_feature_defect_residual",
                "scheduler_affine_residual",
                "scheduler_chord_defect_residual",
            }:
                if self._step == 0:
                    # q0 itself uses the unchanged segmented residual lift;
                    # its full per-frame result becomes q2's control variate.
                    full_residual = self._lift_temporal(
                        active_residual,
                        active_frames,
                        temporal,
                        memory_length,
                        self.current_segment_boundary,
                        "linear",
                    )
                elif self._step == 2:
                    q0_residual = self._scheduler_q0_residuals.pop(
                        block_index, None
                    )
                    if q0_residual is None:
                        raise RuntimeError(
                            f"q2 layer {block_index} lacks its same-chunk q0 residual"
                        )
                    if q0_residual.shape != x_input.shape:
                        raise RuntimeError(
                            f"q0/q2 layer {block_index} residual-grid mismatch"
                        )
                    if self.reconstruction == "scheduler_defect_residual":
                        full_residual = self._scheduler_defect_residual_lift(
                            q0_residual,
                            active_residual,
                            active_frames,
                            memory_length,
                            self.current_segment_boundary,
                        )
                    elif self.reconstruction == "scheduler_feature_defect_residual":
                        full_residual = self._scheduler_feature_defect_residual_lift(
                            q0_residual,
                            active_residual,
                            x_input,
                            active_frames,
                            memory_length,
                            self.current_segment_boundary,
                        )
                    elif self.reconstruction == "scheduler_affine_residual":
                        full_residual = self._scheduler_affine_residual_lift(
                            q0_residual,
                            active_residual,
                            active_frames,
                            memory_length,
                            self.current_segment_boundary,
                        )
                    else:
                        full_residual = self._scheduler_chord_defect_residual_lift(
                            q0_residual,
                            active_residual,
                            active_frames,
                            memory_length,
                            self.current_segment_boundary,
                        )
                else:
                    raise RuntimeError(
                        "scheduler-defect reconstruction is defined only on q0/q2"
                    )
            else:
                full_residual = self._lift_temporal(
                    active_residual,
                    active_frames,
                    temporal,
                    memory_length,
                    self.current_segment_boundary,
                    self.temporal_interpolation,
                )
            if self.reconstruction == "input_parallel_residual":
                full_residual = self._input_parallel_correct_residual(
                    full_residual,
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
            if self.reconstruction == "secant_residual" or (
                self.reconstruction == "q1_secant_residual" and self._step == 1
            ):
                full_residual = self._secant_correct_residual(
                    full_residual,
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                    self.secant_scope,
                )
            if self.reconstruction in {
                "multi_secant_residual",
                "multi_secant_polar_residual",
            }:
                linear_residual = full_residual
                multi_secant_residual = self._multi_secant_correct_residual(
                    linear_residual,
                    active_residual,
                    x_input,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                )
                if self.reconstruction == "multi_secant_polar_residual":
                    full_residual = self._match_token_radius(
                        multi_secant_residual, linear_residual
                    )
                    full_residual[:, active_frames] = active_residual
                else:
                    full_residual = multi_secant_residual
            if self.target_q_attention_correction:
                if full_query_attention_delta is None:
                    raise RuntimeError("target-Q correction lacks full attention delta")
                anchor_attention_delta = full_query_attention_delta[:, active_frames]
                lifted_attention_delta = self._lift_temporal(
                    anchor_attention_delta,
                    active_frames,
                    temporal,
                    memory_length,
                    self.current_segment_boundary,
                    "linear",
                )
                full_residual = (
                    full_residual.float()
                    + full_query_attention_delta.float()
                    - lifted_attention_delta.float()
                ).to(full_residual.dtype)
                full_residual[:, active_frames] = active_residual
            if (
                self.reconstruction
                in {
                    "scheduler_defect_residual",
                    "scheduler_feature_defect_residual",
                    "scheduler_affine_residual",
                    "scheduler_chord_defect_residual",
                }
                and self._step == 0
            ):
                self._scheduler_q0_residuals[block_index] = full_residual.detach()
            if self.world_spectral_corrector is not None:
                # Preserve the existing Module-3 estimator as R_interp, then
                # correct only inactive Current residuals.  Module 1, CWCA,
                # the exact-frame selector/credit and the exact budget have
                # already completed and are not consulted or changed here.
                full_residual = self.world_spectral_corrector.correct(
                    layer_index=block_index,
                    full_input=x_input,
                    full_residual=full_residual,
                    active_residual=active_residual,
                    active_frames=active_frames,
                    memory_length=memory_length,
                    spatial_height=height,
                    spatial_width=width,
                    coarse_probe_residual=coarse_probe_residual,
                    active_frame_values=runtime_active_frame_values,
                )
            if polar_reference is not None:
                polar_oracle = self._spectral_polar_residual_oracle(
                    approximate_residual=full_residual,
                    exact_output=polar_reference[:, :valid_tokens],
                    x_input=x_input,
                    active_frames=active_frames,
                    memory_length=memory_length,
                    height=height,
                    width=width,
                )
            if (
                self.runtime_reuse_residual_output
                and self.reconstruction == "control_barycentric_residual"
            ):
                # The residual buffer is dead after reconstruction.  Reuse it
                # as the block output instead of allocating a second full
                # 19-frame tensor; torch.add_ preserves the same FP32 element
                # addition as the former out-of-place expression.
                full_residual.add_(x_input)
                full_output = full_residual
            else:
                full_output = x_input + full_residual
            if self.reconstruction == "output_dc_residual_detail":
                full_output = self._replace_with_interpolated_output_dc(
                    full_output,
                    active_output,
                    active_frames,
                    memory_length,
                    self.current_segment_boundary,
                    self.output_dc_scope,
                )
        else:
            full_output = self._lift_temporal(
                active_output,
                active_frames,
                temporal,
                memory_length,
                self.current_segment_boundary,
                self.temporal_interpolation,
            )
        output_valid = full_output.reshape(
            x_valid.shape[0], valid_tokens, x_valid.shape[-1]
        )
        output = torch.cat([output_valid, tail], dim=1) if tail.numel() else output_valid
        layer_record = {
            "block_index": block_index,
            "full_layer": False,
            "active_frames": int(active_frames.numel()),
            "total_frames": temporal,
            "active_ratio": float(active_frames.numel()) / float(temporal),
            "high_curvature_current_frames": high_count,
            "curvature_anchor_current_indices": list(
                self._last_curvature_anchor_indices
            ),
            "dynamic_frame_selection": self._last_dynamic_frame_selection,
            "phase": (
                block_index - 1
                + self.scheduler_phase_offsets.get(self._step, 0)
            )
            % self.scheduler_phase_schedule.get(
                self._step, (self.phase_period, self.active_phases)
            )[0],
            "action_bridge_dense": block.action_model is not None,
            "stationary_worldline": self._last_stationary_worldline,
            "complete_spatial_frames": True,
            "compact_attention": self._last_compact_attention,
            "ray_aligned_transport": self._last_ray_transport,
            "control_space_reconstruction": self._last_control_space_reconstruction,
            "scheduler_defect_control_variate": (
                self.reconstruction
                in {
                    "scheduler_defect_residual",
                    "scheduler_feature_defect_residual",
                    "scheduler_affine_residual",
                    "scheduler_chord_defect_residual",
                }
            ),
            "compact_ingress_kernel": compact_ingress_record,
            "world_spectral_residual": (
                {
                    "enabled": True,
                    "variant": self.world_spectral_variant,
                    "residual_only": True,
                    "exact_selector_unchanged": True,
                    "exact_budget_unchanged": True,
                    **(
                        {
                            "current_anchor": {
                                "boundary": self.current_anchor_boundary,
                                "match_tile": self.current_anchor_match_tile,
                                "descriptor_groups": (
                                    self.current_anchor_descriptor_groups
                                ),
                                "consensus_mix": (
                                    self.current_anchor_consensus_mix
                                ),
                            }
                        }
                        if self.world_spectral_variant in {
                            "current_anchored_self_calibrated",
                            "current_anchored_phase_transport",
                            "current_anchored_polar_mixing",
                        }
                        else {}
                    ),
                }
                if self.world_spectral_corrector is not None
                else {"enabled": False}
            ),
        }
        if polar_reference is not None:
            if polar_oracle is None:
                raise RuntimeError("spectral-polar reference was not consumed")
            layer_record["spectral_polar_oracle"] = polar_oracle
        probe = None
        if (
            probe_reference is not None
            or block_index in self.window_forward_targets
        ):
            probe = self._witness_defect(
                active_values=witness_values,
                active_inputs=x_input[:, active_frames],
                active_frames=active_frames,
                total_temporal=temporal,
                memory_temporal=memory_length,
            )
        if probe_reference is not None and probe is not None:
            reference_grid = probe_reference[:, :valid_tokens].reshape_as(x_input)
            epsilon = self.witness_probe_epsilon
            if epsilon is None:
                raise RuntimeError("paired full error requested without probe epsilon")
            full_error, full_error_finite = self._normalized_l2_error(
                reference_grid[:, memory_length:],
                full_output[:, memory_length:],
                epsilon=epsilon,
            )
            probe.update(
                {
                    "paired_full_normalized_error": full_error,
                    "paired_full_error_finite": full_error_finite,
                    "schedule": "periodic_sparse",
                    "reference_schedule": "all_temporal_frames",
                    "current_frames": temporal - memory_length,
                    "active_current_frames": int(
                        torch.count_nonzero(active_frames >= memory_length).item()
                    ),
                }
            )
            layer_record["witness_probe"] = probe
        if block_index in self.window_forward_targets:
            if probe is None:
                raise RuntimeError("window router observation did not produce a probe")
            layer_record["window_router"] = self._update_window_router(
                block_index, probe
            )
        self._call_layers.append(layer_record)
        return output

    def metadata(self) -> dict[str, Any]:
        runtime_profile = self._runtime_profile_summary()
        world_spectral_runtime = (
            self.world_spectral_corrector.summary()
            if self.world_spectral_corrector is not None
            else {
                "enabled": False,
                "mode": None,
                "records": [],
                "output_identity_path": True,
            }
        )
        exact_ratios = [
            float(row["mean_active_frame_ratio"])
            for row in self._records
            if row["layers_executed"]
        ]
        period_histogram: dict[str, int] = {}
        for row in self._records:
            period = row.get("effective_weave_period")
            if period is not None:
                key = str(int(period))
                period_histogram[key] = period_histogram.get(key, 0) + 1
        dynamic_rows = [
            layer["dynamic_frame_selection"]
            for record in self._records
            for layer in record["layers"]
            if isinstance(layer.get("dynamic_frame_selection"), dict)
        ]
        dynamic_quota_histogram: dict[str, int] = {}
        dynamic_frame_histogram: dict[str, int] = {}
        dynamic_drop_histogram: dict[str, int] = {}
        dynamic_period_histogram: dict[str, int] = {}
        lagged_finalize_rows = [
            record["temporal_history_batch_finalize"]
            for record in self._records
            if isinstance(record.get("temporal_history_batch_finalize"), dict)
        ]
        for row in dynamic_rows:
            quota_key = str(int(row["quota"]))
            dynamic_quota_histogram[quota_key] = (
                dynamic_quota_histogram.get(quota_key, 0) + 1
            )
            for frame in row["selected_current_frames"]:
                frame_key = str(int(frame))
                dynamic_frame_histogram[frame_key] = (
                    dynamic_frame_histogram.get(frame_key, 0) + 1
                )
            for frame in row.get("dropped_current_frames", []):
                frame_key = str(int(frame))
                dynamic_drop_histogram[frame_key] = (
                    dynamic_drop_histogram.get(frame_key, 0) + 1
                )
            selected_period = row.get("selected_period")
            if selected_period is not None:
                period_key = str(int(selected_period))
                dynamic_period_histogram[period_key] = (
                    dynamic_period_histogram.get(period_key, 0) + 1
                )
        lagged_payloads_per_call = (
            1
            if self.dynamic_frame_selection
            == "response_call_q50_nested_mod10_lagged_batch"
            else getattr(self, "_num_blocks", 30) - 2
        )
        if self.fc_pasm_swap_router is not None:
            historical_kv_routing = dict(
                self.fc_pasm_swap_router.summary()
            )
            # Make the non-bypass contract explicit in every paper trace.
            # ROCSA-A is a fixed-budget refinement of FC-R; it must never be
            # reported as active while the corrector is returning V21 output.
            historical_kv_routing.update(
                {
                    "fc_r_path_enabled": bool(self.enable_fc_pasm),
                    "fc_v21_bypass": bool(
                        getattr(self.world_spectral_corrector, "fc_v21_bypass", False)
                    ),
                    "rocsa_a_enabled": self.routing_mode == "fc_pasm_swap",
                    "routing_skip_inactive_layers": self.routing_skip_inactive_layers,
                }
            )
        else:
            historical_kv_routing = {
                "routing_mode": "independent_topk",
                "records": [],
                "fixed_cwca_budget": True,
                "fc_r_path_enabled": bool(self.enable_fc_pasm),
                "fc_v21_bypass": bool(
                    getattr(self.world_spectral_corrector, "fc_v21_bypass", False)
                ),
                "rocsa_a_enabled": False,
                "routing_skip_inactive_layers": self.routing_skip_inactive_layers,
            }
        return {
            "name": self.name,
            "runtime_profile": runtime_profile,
            "world_spectral_residual_runtime": world_spectral_runtime,
            "historical_kv_routing": historical_kv_routing,
            "execution_optimization": {
                "enabled": self.runtime_optimized,
                "mathematical_path_unchanged": True,
                "temporal_lift": (
                    "single_anchor_materialization_python_bisect"
                    if self.runtime_optimized
                    else "per_target_cuda_predicates"
                ),
                "control_barycentric_support": (
                    "single_anchor_materialization_python_bisect"
                    if self.runtime_optimized
                    else "per_target_cuda_predicates"
                ),
                "compact_layout_build": (
                    "stable_vectorized_block_grouping"
                    if self.runtime_optimized
                    else "per_block_cuda_nonzero"
                ),
                "compact_layout_cache_scope": (
                    "sample_runtime_stable_geometry"
                    if self.runtime_optimized
                    else "single_model_call_orbit_object_identity"
                ),
                "compact_layout_cache_entries": len(self._layout_cache),
                "response_validation": (
                    "cuda_async_assert"
                    if self.runtime_optimized
                    else "per_layer_host_boolean"
                ),
                "response_credit_conservation_readback": (
                    "single_batched_scalar_readback"
                    if self.runtime_optimized
                    else "independent_scalar_readbacks"
                ),
                "vectorized_reconstruction": (
                    self.runtime_vectorized_reconstruction
                ),
                "vectorized_linear_lift": (
                    self.runtime_vectorized_linear_lift
                ),
                "linear_lift_launches": (
                    "batched_targets"
                    if (
                        self.runtime_vectorized_reconstruction
                        or self.runtime_vectorized_linear_lift
                    )
                    else "per_target"
                ),
                "control_distance_launches": (
                    "batched_targets"
                    if self.runtime_vectorized_reconstruction
                    else "per_target"
                ),
                "control_barycentric_anchor_write": (
                    "single_final_write"
                    if (
                        self.runtime_reuse_residual_output
                        or self.runtime_single_anchor_write
                    )
                    else "initial_and_final_write"
                ),
                "residual_output_buffer": (
                    "in_place_full_residual_add"
                    if self.runtime_reuse_residual_output
                    else "out_of_place_add"
                ),
                "barycentric_contraction": (
                    "direct_pair_triton_left_then_right_fma"
                    if self.runtime_direct_barycentric_pair_kernel
                    else
                    "preallocated_batched_index_select_bmm_out"
                    if self.runtime_batched_barycentric_contraction
                    else "preallocated_index_select_bmm_out"
                    if self.runtime_preallocated_barycentric
                    else "allocating_stack_einsum"
                ),
                "barycentric_workspace_allocations": (
                    self._barycentric_workspace_allocations
                ),
                "barycentric_preallocated_contractions": (
                    self._barycentric_preallocated_contractions
                ),
                "barycentric_batched_contraction_calls": (
                    self._barycentric_batched_contraction_calls
                ),
                "barycentric_direct_pair_kernel_calls": (
                    self._barycentric_direct_pair_kernel_calls
                ),
                "barycentric_control_state": (
                    "model_call_identity_version_cache"
                    if self.runtime_cache_barycentric_control
                    else "recomputed_per_layer"
                ),
                "barycentric_control_cache_hits": (
                    self._barycentric_control_cache_hits
                ),
                "barycentric_control_cache_misses": (
                    self._barycentric_control_cache_misses
                ),
                "barycentric_weight_cache": (
                    "model_call_exact_triplet_cache"
                    if self.runtime_cache_barycentric_weights
                    else "disabled"
                ),
                "barycentric_weight_cache_hits": self._barycentric_weight_cache_hits,
                "barycentric_weight_cache_misses": self._barycentric_weight_cache_misses,
                "barycentric_active_frame_source": (
                    "dynamic_selector_cpu_trace"
                    if self.runtime_reuse_dynamic_active_frame_list
                    else "cuda_anchor_readback"
                ),
                "barycentric_active_frame_list_reuse_calls": (
                    self._barycentric_active_frame_list_reuse_calls
                ),
                "compact_control_response_source": (
                    "exact_active_frames_local_scalar_interpolation"
                    if self.compact_active_response_only
                    else "segmented_full_frameweave_hidden"
                ),
                "unused_control_response_gating": {
                    "enabled": self.runtime_gate_unused_control_response,
                    "fine_only_response_reduction": (
                        self.runtime_fine_only_control_response_reduction
                    ),
                    "capture_model_calls": sum(
                        bool(row.get("fine_frame_response_capture_active"))
                        for row in self._records
                    ),
                    "skipped_model_calls": sum(
                        not bool(row.get("fine_frame_response_capture_active"))
                        for row in self._records
                    ),
                    "only_q0_memory_noncamera_or_camera_woven": True,
                    "attention_budget_response_independent": (
                        float(
                            getattr(self.geometry_provider, "response_weight", 1.0)
                        )
                        == 0.0
                    ),
                },
                "selection_policy_unchanged": True,
                "attention_budget_unchanged": True,
                "interpolation_tensor_arithmetic_unchanged": (
                    self.world_spectral_corrector is None
                ),
            },
            "records": list(self._records),
            "compact_cwca_sparse_density": self.sparse_density,
            "camera_guard_weave_policy": {
                "period": self.camera_guard_weave_period,
                "active_phases": self.camera_guard_weave_active_phases,
                "camera_only_exact_layers": sorted(
                    self.camera_guard_exact_layers
                ),
                "periods_by_step": {
                    "0": self.camera_guard_weave_period,
                    "2": self.camera_guard_q2_weave_period,
                },
                "original_guard_calls": sum(
                    bool(row.get("original_camera_guard_triggered"))
                    for row in self._records
                ),
                "original_guard_q0_calls": sum(
                    bool(row.get("original_camera_guard_triggered"))
                    and int(row["step_index"]) == 0
                    for row in self._records
                ),
                "actual_camera_woven_calls": sum(
                    bool(row.get("camera_woven")) for row in self._records
                ),
                "period_histogram": period_histogram,
                "full_layers": sum(
                    int(row["full_layers"]) for row in self._records
                ),
                "woven_layers": sum(
                    int(row["woven_layers"]) for row in self._records
                ),
            },
            "dynamic_frame_selection_runtime": {
                "enabled": self.dynamic_frame_selection is not None,
                "policy": self.dynamic_frame_selection,
                "records": len(dynamic_rows),
                "fixed_cell_budget": self.dynamic_exact_cell_budget,
                "minimum_cell_quota": self.dynamic_exact_cell_minimum,
                "quota_histogram": dynamic_quota_histogram,
                "selected_current_frame_histogram": dynamic_frame_histogram,
                "dropped_current_frame_histogram": dynamic_drop_histogram,
                "selected_period_histogram": dynamic_period_histogram,
                "thin_decisions": sum(
                    row.get("thin_applied") is True for row in dynamic_rows
                ),
                "period_route_decisions": sum(
                    row.get("route_applied") is True for row in dynamic_rows
                ),
                "all_causal_previous_layer_only": bool(dynamic_rows)
                and all(
                    row.get("causal_previous_layer_only") is True
                    and int(row["previous_layer"]) == int(row["layer"]) - 1
                    for row in dynamic_rows
                ),
                "all_causal_previous_eligible_call_only": bool(dynamic_rows)
                and all(
                    row.get("causal_previous_eligible_call_only") is True
                    for row in dynamic_rows
                ),
                "all_one_per_native_response_cell": bool(dynamic_rows)
                and all(
                    row.get("one_representative_per_native_response_cell") is True
                    and len(row["selected_cells"])
                    == len(set(int(cell) for cell in row["selected_cells"]))
                    for row in dynamic_rows
                ),
                "all_structural_endpoints_exact": bool(dynamic_rows)
                and all(
                    row["exact_current_frames"][0]
                    == row["structural_current_anchors"][0]
                    and row["exact_current_frames"][-1]
                    == row["structural_current_anchors"][-1]
                    for row in dynamic_rows
                ),
                "all_canonical_phase_thin_only": bool(dynamic_rows)
                and all(
                    row.get(
                        "canonical_phase_preserved_except_response_gated_phase_only_drop"
                    )
                    is True
                    and len(row.get("dropped_current_frames", []))
                    <= int(row.get("maximum_dropped_frames", 1))
                    for row in dynamic_rows
                ),
                "all_curvature_and_structural_anchors_preserved": bool(dynamic_rows)
                and all(
                    row.get("curvature_and_structural_anchors_preserved") is True
                    for row in dynamic_rows
                ),
                "all_nested_mod5_mod10_period_routing": bool(dynamic_rows)
                and all(
                    row.get("nested_phase_subset") is True
                    and row.get("hardware_layout_family")
                    == "mod5_mod10_nested"
                    and int(row.get("base_period", -1)) == 5
                    and int(row.get("low_response_period", -1)) == 10
                    and int(row.get("selected_period", -1)) in {5, 10}
                    and not row.get("added_current_frames")
                    for row in dynamic_rows
                ),
                "all_response_credit_fair_mod5": bool(dynamic_rows)
                and all(
                    row.get("policy") == "response_credit_fair_mod5"
                    and row.get("credit_scale_source") == "unit_mass_conserving"
                    and float(row.get("credit_step_scale", float("nan"))) == 1.0
                    and row.get("mod5_cardinality_matched") is True
                    and row.get("mod5_phase_used_for_cardinality_only") is True
                    and row.get("anchors_excluded_from_credit") is True
                    and row.get("credit_sum_conserved") is True
                    and row.get("deterministic_topk") is True
                    and int(row.get("quota", -1))
                    == int(row.get("mod5_target_exact_count", -2))
                    and int(row.get("dynamic_quota", -1))
                    == len(row.get("selected_dynamic_current_frames", []))
                    for row in dynamic_rows
                ),
                "all_response_credit_std_scaled_mod5": bool(dynamic_rows)
                and all(
                    row.get("policy") == "response_credit_std_scaled_mod5"
                    and row.get("credit_scale_source")
                    == "eligible_normalized_previous_layer_response_population_std"
                    and isinstance(row.get("credit_step_scale"), (int, float))
                    and not isinstance(row.get("credit_step_scale"), bool)
                    and math.isfinite(float(row["credit_step_scale"]))
                    and float(row["credit_step_scale"]) >= 0.0
                    and row.get("mod5_cardinality_matched") is True
                    and row.get("mod5_phase_used_for_cardinality_only") is True
                    and row.get("anchors_excluded_from_credit") is True
                    and row.get("credit_sum_conserved") is True
                    and row.get("deterministic_topk") is True
                    and int(row.get("quota", -1))
                    == int(row.get("mod5_target_exact_count", -2))
                    and int(row.get("dynamic_quota", -1))
                    == len(row.get("selected_dynamic_current_frames", []))
                    for row in dynamic_rows
                ),
                "all_response_credit_std_scaled_phase_budget": bool(dynamic_rows)
                and all(
                    row.get("policy") == "response_credit_std_scaled_mod5"
                    and row.get("credit_scale_source")
                    == "eligible_normalized_previous_layer_response_population_std"
                    and isinstance(row.get("credit_step_scale"), (int, float))
                    and not isinstance(row.get("credit_step_scale"), bool)
                    and math.isfinite(float(row["credit_step_scale"]))
                    and float(row["credit_step_scale"]) >= 0.0
                    and row.get("phase_cardinality_matched") is True
                    and row.get("phase_used_for_cardinality_only") is True
                    and row.get("anchors_excluded_from_credit") is True
                    and row.get("credit_sum_conserved") is True
                    and row.get("deterministic_topk") is True
                    and int(row.get("quota", -1))
                    == int(row.get("phase_target_exact_count", -2))
                    and int(row.get("dynamic_quota", -1))
                    == len(row.get("selected_dynamic_current_frames", []))
                    for row in dynamic_rows
                ),
                "all_lagged_one_eligible_q0_call": bool(dynamic_rows)
                and all(
                    row.get("selection_lag_eligible_calls") == 1
                    and row.get("batched_host_decision") is True
                    and row.get("history_axis")
                    in {
                        "same_source_layer_across_prior_q0_woven_calls_lag1",
                        "aggregate_closed_loop_response_across_prior_q0_calls_lag1",
                    }
                    and (
                        row.get("source_eligible_call_index") is None
                        or int(row["selection_eligible_call_index"])
                        == int(row["source_eligible_call_index"]) + 1
                    )
                    for row in dynamic_rows
                ),
                "lagged_batch_finalize_records": lagged_finalize_rows,
                "lagged_batch_finalize_calls": len(lagged_finalize_rows),
                "lagged_batch_single_d2h": bool(lagged_finalize_rows)
                and all(
                    row.get("single_device_to_host_batch") is True
                    and int(row.get("selection_lag_eligible_calls", -1)) == 1
                    and int(row.get("payloads_written", -1))
                    == lagged_payloads_per_call
                    for row in lagged_finalize_rows
                ),
                "benchmark_metric_read": False,
            },
            "compact_ingress_kernel_runtime": {
                "enabled": self.compact_ingress_kernel is not None,
                "candidate": self.compact_ingress_kernel,
                "calls": self._c1a_calls,
                "active_frame_histogram": {
                    str(count): calls
                    for count, calls in sorted(
                        self._c1a_active_frame_histogram.items()
                    )
                },
                "hidden_dtype_histogram": dict(
                    sorted(self._c1a_hidden_dtype_histogram.items())
                ),
                "layout_compilations": self._c1a_layout_compilations,
                "compiled_layout_cache_entries": len(
                    self._c1a_compiled_layouts
                ),
                "workspace_allocations": self._c1a_workspace_allocations,
                "workspace_cache_entries": len(self._c1a_workspaces),
                "workspace_cache_key": (
                    "layout_geometry+device+dtype;grow_to_largest_active_shape"
                    if self.compact_ingress_kernel is not None
                    else None
                ),
                "q0_woven_layers_only": self.compact_ingress_kernel is not None,
                "native_fp32_layer_norm": self.compact_ingress_kernel is not None,
                "native_input_dtype_roundtrip": (
                    self.compact_ingress_kernel is not None
                ),
                "native_post_ln_modulation": self.compact_ingress_kernel is not None,
            },
            "witness_probe_rows": sum(
                "witness_probe" in layer
                for record in self._records
                for layer in record["layers"]
            ),
            "window_router_records": list(self._window_router_records),
            "window_router_exact_decisions": sum(
                row["decision"] == "exact" for row in self._window_router_records
            ),
            "window_router_woven_decisions": sum(
                row["decision"] == "woven" for row in self._window_router_records
            ),
            "ray_transport_rows": sum(
                layer.get("ray_aligned_transport") is not None
                for record in self._records
                for layer in record["layers"]
            ),
            "ray_transport_shifted_token_fraction": (
                float(
                    sum(
                        layer["ray_aligned_transport"]["shifted_token_fraction"]
                        for record in self._records
                        for layer in record["layers"]
                        if layer.get("ray_aligned_transport") is not None
                    )
                    / max(
                        1,
                        sum(
                            layer.get("ray_aligned_transport") is not None
                            for record in self._records
                            for layer in record["layers"]
                        ),
                    )
                )
            ),
            "mean_active_frame_ratio_exact_calls": (
                float(sum(exact_ratios) / len(exact_ratios))
                if exact_ratios
                else 0.0
            ),
            "released_li_q1_cached_calls": sum(
                row["mode"] == "released_li_q1_prediction_cache"
                for row in self._records
            ),
            "cache_contract": {
                "frame_by_layer": True,
                "complete_spatial_frames": True,
                "memory_world_state_anchors_exact": True,
                "cwca_curvature_controls_refresh_rate": (
                    self.high_curvature_fraction > 0.0
                    and not self.feature_curvature_fallback
                ),
                "native_qk_within_active_topology": True,
                "current_worldline_residual_interpolation": True,
                "world_aligned_self_calibrated_spectral_residual": {
                    "enabled": self.world_spectral_corrector is not None,
                    "mode": (
                        self.world_spectral_variant
                        if self.world_spectral_corrector is not None
                        else None
                    ),
                    "base_reconstruction_preserved": self.reconstruction,
                    "relative_pluecker_forbidden": True,
                    "memory_residual_exact": True,
                    "exact_current_residual_exact": True,
                    "spatial_fft_only": True,
                    "additional_dit_attention_ffn_forwards": 0,
                    "fc_pair_gate_post_camera_only": (
                        self.fc_pair_gate_post_camera_only
                    ),
                    "fc_v21_until_camera_seen": self.fc_v21_until_camera_seen,
                    "fc_v21_after_no_camera_chunks": (
                        self.fc_v21_after_no_camera_chunks
                    ),
                    **(
                        {
                            "current_anchor": {
                                "boundary": self.current_anchor_boundary,
                                "match_tile": self.current_anchor_match_tile,
                                "descriptor_groups": (
                                    self.current_anchor_descriptor_groups
                                ),
                                "consensus_mix": (
                                    self.current_anchor_consensus_mix
                                ),
                            }
                        }
                        if self.world_spectral_corrector is not None
                        and self.world_spectral_variant in {
                            "current_anchored_self_calibrated",
                            "current_anchored_phase_transport",
                            "current_anchored_polar_mixing",
                        }
                        else {}
                    ),
                },
                "first_last_layer_exact": True,
                "current_endpoints_exact": self.force_current_endpoints_exact,
                "dynamic_frame_selection": self.dynamic_frame_selection,
                "dynamic_exact_cell_budget": self.dynamic_exact_cell_budget,
                "dynamic_exact_cell_minimum": self.dynamic_exact_cell_minimum,
                "dynamic_selection_uses_previous_layer_closed_loop_response": (
                    self.dynamic_frame_selection is not None
                ),
                "dynamic_selection_reads_benchmark_metric": False,
                "periodic_selection_bypassed_when_dynamic": (
                    self.dynamic_frame_selection
                    in {"response_topk_fixed", "response_effective_quota"}
                ),
                "periodic_frame_ids_replaced_but_cardinality_preserved": (
                    self.dynamic_frame_selection
                    in {
                        "response_credit_fair_mod5",
                        "response_credit_std_scaled_mod5",
                    }
                ),
                "phase_period": self.phase_period,
                "active_phases": self.active_phases,
                "scheduler_phase_schedule": {
                    str(step): [period, active]
                    for step, (period, active) in self.scheduler_phase_schedule.items()
                },
                "scheduler_phase_offsets": {
                    str(step): offset
                    for step, offset in self.scheduler_phase_offsets.items()
                },
                "sparse_steps": (
                    None if self.sparse_steps is None else sorted(self.sparse_steps)
                ),
                "sparse_layers": (
                    None if self.sparse_layers is None else sorted(self.sparse_layers)
                ),
                "sparse_layers_by_step": {
                    str(step): sorted(layers)
                    for step, layers in self.sparse_layers_by_step.items()
                },
                "reconstruction": self.reconstruction,
                "scheduler_defect_q0_control_variate": (
                    self.reconstruction
                    in {
                        "scheduler_defect_residual",
                        "scheduler_feature_defect_residual",
                        "scheduler_affine_residual",
                        "scheduler_chord_defect_residual",
                    }
                ),
                "scheduler_defect_interpolation": (
                    "segmented_linear_q0_to_q2_defect"
                    if self.reconstruction == "scheduler_defect_residual"
                    else "segmented_feature_attention_q0_to_q2_defect"
                    if self.reconstruction == "scheduler_feature_defect_residual"
                    else "segmented_channel_affine_q0_to_q2_residual"
                    if self.reconstruction == "scheduler_affine_residual"
                    else "segmented_q0_chord_coordinate_q0_to_q2_defect"
                    if self.reconstruction == "scheduler_chord_defect_residual"
                    else None
                ),
                "scheduler_defect_cache_remaining": len(
                    self._scheduler_q0_residuals
                ),
                "compact_cwca_topology": self.compact_cwca_topology,
                "compact_ingress_kernel": self.compact_ingress_kernel,
                "compact_ingress_only_q0_woven_layers": (
                    self.compact_ingress_kernel is not None
                ),
                "compact_ingress_changes_frame_schedule": False,
                "compact_ingress_changes_attention_support": False,
                "compact_ingress_changes_action_bridge": False,
                "active_query_full_kv_attention": (
                    self.asymmetric_topology_selector is not None
                ),
                "target_q_attention_correction": self.target_q_attention_correction,
                "inactive_q_attention_exact_on_cwca_topology": (
                    self.target_q_attention_correction
                ),
                "post_attention_residual_segmented_lift": (
                    self.target_q_attention_correction
                    and self.reconstruction == "residual"
                ),
                "target_q_condition_path_dense": (
                    self.reconstruction == "target_q_ffn_lift"
                ),
                "ffn_delta_segmented_lift": (
                    self.reconstruction == "target_q_ffn_lift"
                ),
                "weave_domain": self.weave_domain,
                "camera_action_exact_guard": (
                    self.weave_domain
                    in {"camera_guarded_current", "camera_history_dominant_current"}
                    and self.camera_guard_weave_period is None
                ),
                "camera_action_guard_conservative_weave_period": (
                    self.camera_guard_weave_period
                ),
                "camera_action_guard_conservative_weave_active_phases": (
                    self.camera_guard_weave_active_phases
                ),
                "camera_action_guard_q2_conservative_weave_period": (
                    self.camera_guard_q2_weave_period
                ),
                "camera_action_threshold": (
                    self.camera_action_threshold
                    if self.weave_domain
                    in {"camera_guarded_current", "camera_history_dominant_current"}
                    else None
                ),
                "camera_history_dominance_guard": (
                    self.weave_domain == "camera_history_dominant_current"
                ),
                "witness_probe_layers": sorted(self.witness_probe_layers),
                "witness_probe_epsilon": self.witness_probe_epsilon,
                "witness_probe_paired_full_execution": bool(
                    self.witness_probe_layers
                ),
                "spectral_polar_probe": {
                    "enabled": bool(self.spectral_polar_probe_layers),
                    "layers": sorted(self.spectral_polar_probe_layers),
                    "chunk": self.spectral_polar_probe_chunk,
                    "step": self.spectral_polar_probe_step,
                    "diagnostic_only": True,
                    "output_mutation": False,
                },
                "witness_probe_changes_generation_output": False,
                "window_forward_targets": {
                    str(observation): list(targets)
                    for observation, targets in self.window_forward_targets.items()
                },
                "window_router_metric": (
                    self.window_router_metric
                    if self.window_forward_targets
                    else None
                ),
                "window_router_calibration": (
                    "causal_per_sample_running_median"
                    if self.window_forward_targets
                    else None
                ),
                "window_router_paired_full_execution": False,
                "window_router_reads_benchmark_metric": False,
                "ray_aligned_residual_transport": (
                    self.reconstruction == "ray_aligned_residual"
                ),
                "rank_one_input_residual_secant": (
                    self.reconstruction in {"secant_residual", "q1_secant_residual"}
                ),
                "rank_one_input_residual_secant_steps": (
                    [1]
                    if self.reconstruction == "q1_secant_residual"
                    else sorted(self.sparse_steps)
                    if self.reconstruction == "secant_residual"
                    and self.sparse_steps is not None
                    else None
                ),
                "rank_one_secant_scope": (
                    self.secant_scope
                    if self.reconstruction in {"secant_residual", "q1_secant_residual"}
                    else None
                ),
                "multi_secant_input_curvature_rank": (
                    2
                    if self.reconstruction
                    in {"multi_secant_residual", "multi_secant_polar_residual"}
                    else None
                ),
                "multi_secant_radius": (
                    "linear_residual_token_radius"
                    if self.reconstruction == "multi_secant_polar_residual"
                    else "unconstrained"
                    if self.reconstruction == "multi_secant_residual"
                    else None
                ),
                "feature_attention_same_site": (
                    self.reconstruction == "feature_attention_residual"
                ),
                "feature_barycentric_local_support": (
                    self.reconstruction == "feature_barycentric_residual"
                ),
                "control_space_residual_reconstruction": (
                    self.reconstruction == "control_space_residual"
                ),
                "control_space_residual_lambdas": (
                    list(self.control_residual_lambdas)
                    if self.reconstruction == "control_space_residual"
                    else None
                ),
                "control_space_uses_exact_anchors_only": (
                    self.reconstruction == "control_space_residual"
                ),
                "control_space_segmented_no_cross_boundary": (
                    self.reconstruction
                    in {"control_space_residual", "control_barycentric_residual"}
                ),
                "control_barycentric_local_two_anchor_support": (
                    self.reconstruction == "control_barycentric_residual"
                ),
                "control_barycentric_lambdas": (
                    list(self.control_residual_lambdas[1:])
                    if self.reconstruction == "control_barycentric_residual"
                    else None
                ),
                "output_dc_residual_detail_hodge_split": (
                    self.reconstruction == "output_dc_residual_detail"
                ),
                "output_dc_scope": (
                    self.output_dc_scope
                    if self.reconstruction == "output_dc_residual_detail"
                    else None
                ),
                "input_tangent_residual_lift": (
                    self.reconstruction == "input_tangent_residual"
                ),
                "input_parallel_residual_correction": (
                    self.reconstruction == "input_parallel_residual"
                ),
                "input_velocity_residual_correction": (
                    self.reconstruction == "input_velocity_residual"
                ),
                "attention_delta_segmented_lift": (
                    self.reconstruction == "attention_delta"
                ),
                "post_attention_world_model_dense": (
                    self.reconstruction == "attention_delta"
                ),
                "convex_input_chord_residual_lift": (
                    self.reconstruction == "convex_chord_residual"
                ),
                "ray_transport_radius": (
                    self.ray_transport_radius
                    if self.reconstruction == "ray_aligned_residual"
                    else None
                ),
                "ray_transport_feature_groups": (
                    self.ray_transport_feature_groups
                    if self.reconstruction == "ray_aligned_residual"
                    else None
                ),
                "ray_transport_hard_local_correspondence": (
                    self.reconstruction == "ray_aligned_residual"
                ),
                "ray_transport_spatial_dc_interpolation": (
                    self.reconstruction == "ray_aligned_residual"
                ),
                "ray_transport_zero_mean_detail": (
                    self.reconstruction == "ray_aligned_residual"
                ),
                "ray_transport_norm_limiter": (
                    self.reconstruction == "ray_aligned_residual"
                ),
                "stationarity_threshold": (
                    math.sqrt(torch.finfo(torch.float32).eps)
                    if self.weave_domain == "stationary_dual"
                    else None
                ),
                "high_curvature_fraction": self.high_curvature_fraction,
                "high_curvature_anchor_count": self.high_curvature_anchor_count,
                "feature_curvature_fallback": self.feature_curvature_fallback,
                "curvature_anchor_scope": self.curvature_anchor_scope,
                "curvature_anchor_cell_size": self.curvature_anchor_cell_size,
                "temporal_interpolation": self.temporal_interpolation,
                "current_segment_boundary": self.current_segment_boundary,
                "current_segments": (
                    [[0, self.current_segment_boundary - 1],
                     [self.current_segment_boundary, 13]]
                    if self.current_segment_boundary is not None
                    else None
                ),
                "segmented_current_interpolation": (
                    self.current_segment_boundary is not None
                ),
                "interpolation_crosses_current_segment_boundary": False,
                "li_denoise_cache_enabled": self.li_denoise_cache_enabled,
                "action_module_dense_worldline_bridge": True,
                "benchmark_metric_read": False,
                "third_party_method_claim": False,
            },
        }


class MatrixCausalFrameLayerRelay(MatrixCurvaturePhaseFrameWeave):
    """Treat retrieved frames as read-mostly state, not output queries.

    Matrix predicts only the current rollout, while retrieved frames are
    causal conditioning state.  Between periodic depth anchors, every
    retrieved frame remains available as a native CWCA-selected K/V block but
    does not spend Query, output-projection, cross-attention, or FFN compute.
    The complete current spatial frames are updated in every layer.  This is
    a frame-by-layer schedule; it never removes spatial tokens from a frame.
    """

    name = "matrix_causal_frame_layer_relay_v1"

    def __init__(
        self,
        geometry_provider: Any,
        topology_selector: Callable[..., tuple[torch.Tensor, torch.Tensor, Any]],
        *,
        memory_refresh_period: int = 3,
        current_phase_period: int | None = None,
        current_active_phases: int | None = None,
        current_high_curvature_anchors: int = 0,
        inactive_transport: str = "identity",
        variant_name: str | None = None,
    ) -> None:
        if memory_refresh_period < 2:
            raise ValueError("memory refresh period must be at least two layers")
        super().__init__(
            geometry_provider,
            phase_period=memory_refresh_period,
            active_phases=memory_refresh_period - 1,
            high_curvature_fraction=0.0,
            high_curvature_anchor_count=0,
            variant_name=self.name,
        )
        self.topology_selector = topology_selector
        self.memory_refresh_period = int(memory_refresh_period)
        self.current_phase_period = (
            None if current_phase_period is None else int(current_phase_period)
        )
        self.current_active_phases = (
            None if current_active_phases is None else int(current_active_phases)
        )
        self.current_high_curvature_anchors = int(current_high_curvature_anchors)
        if inactive_transport not in {"identity", "previous_depth_residual"}:
            raise ValueError("unknown inactive current-frame transport")
        self.inactive_transport = str(inactive_transport)
        if self.current_phase_period is not None:
            if (
                self.current_phase_period < 2
                or self.current_active_phases is None
                or not 1
                <= self.current_active_phases
                < self.current_phase_period
                or self.current_high_curvature_anchors < 0
            ):
                raise ValueError("invalid current-frame depth schedule")
        if variant_name is not None:
            self.name = str(variant_name)

    def begin_model_call(self, **kwargs: Any) -> str:
        self._last_depth_residual = None
        return super().begin_model_call(**kwargs)

    def _memory_refresh_layer(self, block_index: int) -> bool:
        return (
            block_index == 0
            or block_index == self._num_blocks - 1
            or block_index % self.memory_refresh_period == 0
        )

    def _query_frames(
        self,
        *,
        block_index: int,
        temporal: int,
        memory_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        current_count = temporal - memory_length
        current = torch.arange(current_count, device=device)
        if self.current_phase_period is None:
            return current + memory_length
        assert self.current_active_phases is not None
        phase = (current + block_index).remainder(self.current_phase_period)
        active = phase < self.current_active_phases
        # Endpoints are the boundary conditions of the generated worldline.
        active[0] = True
        active[-1] = True
        if self.current_high_curvature_anchors:
            curvature = self._current_curvature(current_count)
            if curvature is not None and curvature.numel():
                anchors = torch.topk(
                    curvature.to(device=device),
                    k=min(self.current_high_curvature_anchors, current_count),
                ).indices
                active[anchors] = True
        return torch.nonzero(active).flatten() + memory_length

    def _asymmetric_layout(
        self,
        *,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
        memory_length: int,
        orbit_sparse_layout: Any,
        query_frames: torch.Tensor,
        allow_memory_queries: bool = False,
    ) -> _AsymmetricFrameLayout:
        temporal, height, width = (int(value) for value in grid_sizes[0].tolist())
        key = (
            "causal_relay",
            temporal,
            height,
            width,
            int(memory_length),
            tuple(int(value) for value in query_frames.tolist()),
            bool(allow_memory_queries),
            id(orbit_sparse_layout),
        )
        cached = self._layout_cache.get(key)
        if cached is not None:
            if not isinstance(cached, _AsymmetricFrameLayout):
                raise RuntimeError("frame relay layout cache type mismatch")
            return cached
        if orbit_sparse_layout is None:
            raise RuntimeError("causal frame relay requires a CWCA layout")
        tt, th, tw = (int(value) for value in orbit_sparse_layout.block_shape)
        spatial = height * width
        token = torch.arange(temporal * spatial, device=x.device)
        t_coord = token // spatial
        rem = token % spatial
        h_coord = rem // width
        w_coord = rem % width
        coordinates = torch.stack([t_coord, h_coord, w_coord], dim=-1).long()

        nh, nw = math.ceil(height / th), math.ceil(width / tw)
        memory_t = int(memory_length)
        current_t = temporal - memory_t
        in_memory = t_coord < memory_t
        if bool(getattr(orbit_sparse_layout, "protected_current", False)):
            memory_blocks = math.ceil(memory_t / tt) * nh * nw if memory_t else 0
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
        if int(orbit_sparse_layout.indices.shape[-2]) != total_blocks:
            raise RuntimeError("causal frame relay/CWCA block count mismatch")
        block_size = tt * th * tw
        packed = torch.full(
            (total_blocks, block_size), -1, device=x.device, dtype=torch.long
        )
        for index in range(total_blocks):
            members = torch.nonzero(block_id == index).flatten()
            if int(members.numel()) > block_size:
                raise RuntimeError("CWCA block contains too many tokens")
            packed[index, : members.numel()] = members
        key_valid = packed >= 0
        query_frame_mask = torch.zeros(
            temporal, device=x.device, dtype=torch.bool
        )
        query_frame_mask[query_frames] = True
        is_query = query_frame_mask[t_coord]
        if not allow_memory_queries and bool(torch.any(is_query & in_memory)):
            raise RuntimeError("causal relay may not write Memory on a relay layer")
        query_block_ids = torch.unique(block_id[is_query], sorted=True)
        query_packed = torch.full(
            (len(query_block_ids), block_size),
            -1,
            device=x.device,
            dtype=torch.long,
        )
        for row, index in enumerate(query_block_ids.tolist()):
            members = torch.nonzero((block_id == index) & is_query).flatten()
            query_packed[row, : members.numel()] = members
        query_valid = query_packed >= 0
        query_indices = token[is_query]
        result = _AsymmetricFrameLayout(
            total_temporal=temporal,
            spatial_height=height,
            spatial_width=width,
            memory_temporal=memory_t,
            full_coordinates=coordinates,
            query_indices=query_indices,
            query_coordinates=coordinates[query_indices],
            key_packed_indices=packed,
            key_valid=key_valid,
            query_block_ids=query_block_ids,
            query_packed_indices=query_packed,
            query_valid=query_valid,
            block_size=block_size,
        )
        self._layout_cache[key] = result  # type: ignore[assignment]
        return result

    @staticmethod
    def _longcat_asymmetric(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
        counts: torch.Tensor,
        key_valid: torch.Tensor,
        *,
        query_block_size: int,
        key_block_size: int,
    ) -> torch.Tensor:
        import triton
        from wan.modules.longcat_kernel import _attn_fwd_bsa_align

        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()
        indices = indices.to(torch.int32).contiguous()
        counts = counts.to(torch.int32).contiguous()
        valid = key_valid.reshape(-1).to(torch.uint8).contiguous()
        batch, heads, query_length, head_dim = q.shape
        key_length = k.shape[2]
        if query_length % query_block_size or key_length % key_block_size:
            raise RuntimeError("asymmetric sparse storage is not block aligned")
        output = torch.empty_like(q)
        maximum = torch.empty(
            (batch, heads, query_length), device=q.device, dtype=torch.float32
        )
        grid = (query_length // query_block_size, batch * heads)
        _attn_fwd_bsa_align[grid](
            Q=q,
            K=k,
            V=v,
            K_valid=valid,
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
            Q_LEN=query_length,
            K_LEN=key_length,
            HEAD_DIM=head_dim,
            BLOCK_M=query_block_size,
            BLOCK_N=key_block_size,
        )
        return output

    def _current_query_attention(
        self,
        module: Any,
        normalized: torch.Tensor,
        layout: _AsymmetricFrameLayout,
        *,
        freqs: torch.Tensor,
        memory_length: int,
        memory_latent_idx: Any,
        predict_latent_idx: Any,
        orbit_sparse_layout: Any,
    ) -> torch.Tensor:
        batch, sequence = normalized.shape[:2]
        heads, head_dim = module.num_heads, module.head_dim
        # Full Q/K is intentionally formed for the unchanged CWCA selector.
        # Only current-frame Q rows enter the expensive sparse attention core.
        q = module.norm_q(module.q(normalized)).view(
            batch, sequence, heads, head_dim
        )
        k = module.norm_k(module.k(normalized)).view(
            batch, sequence, heads, head_dim
        )
        v = module.v(normalized).view(batch, sequence, heads, head_dim)
        apply_rope = MatrixJiTOfficialSemanticsAcceleration._apply_sparse_rope
        q = apply_rope(
            q,
            layout.full_coordinates,
            freqs,
            memory_length,
            memory_latent_idx,
            predict_latent_idx,
        )
        k = apply_rope(
            k,
            layout.full_coordinates,
            freqs,
            memory_length,
            memory_latent_idx,
            predict_latent_idx,
        )
        selected, counts, _ = self.topology_selector(
            q=q.transpose(1, 2).contiguous(),
            k=k.transpose(1, 2).contiguous(),
            geometry_indices=orbit_sparse_layout.indices,
            geometry_counts=orbit_sparse_layout.counts,
            latent_hw=(layout.spatial_height, layout.spatial_width),
            block_shape=tuple(int(value) for value in orbit_sparse_layout.block_shape),
        )
        key_storage = layout.key_packed_indices.clamp_min(0).reshape(-1)
        query_storage = layout.query_packed_indices.clamp_min(0).reshape(-1)
        q_packed = q[:, query_storage].transpose(1, 2).contiguous()
        k_packed = k[:, key_storage].transpose(1, 2).contiguous()
        v_packed = v[:, key_storage].transpose(1, 2).contiguous()
        query_mask = layout.query_valid.reshape(-1)
        key_mask = layout.key_valid.reshape(-1)
        q_packed[:, :, ~query_mask] = 0
        k_packed[:, :, ~key_mask] = 0
        v_packed[:, :, ~key_mask] = 0
        row_ids = layout.query_block_ids
        selected = selected[:, :, row_ids]
        counts = counts[:, :, row_ids]
        output_packed = MatrixCausalFrameLayerRelay._longcat_asymmetric(
            q_packed,
            k_packed,
            v_packed,
            selected,
            counts,
            layout.key_valid,
            query_block_size=layout.block_size,
            key_block_size=layout.block_size,
        ).transpose(1, 2)
        original = layout.query_packed_indices.reshape(-1)[query_mask]
        compact = output_packed[:, query_mask]
        full = compact.new_zeros(batch, sequence, heads, head_dim)
        full[:, original] = compact
        current = full[:, layout.query_indices]
        return module.o(current.flatten(2).to(normalized.dtype))

    def _forward_block(
        self,
        block_index: int,
        block: Any,
        native_forward: Any,
        x: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self._active:
            return native_forward(x, *args, **kwargs)
        temporal, height, width = (
            int(value) for value in kwargs["grid_sizes"][0].tolist()
        )
        memory_length = int(kwargs.get("memory_length", 0))
        # Bootstrap, released LI's exact q1 fallback, and depth anchors retain
        # the untouched native Matrix block.
        if (
            self._step == 1
            or memory_length <= 0
            or self._memory_refresh_layer(block_index)
        ):
            output = native_forward(x, *args, **kwargs)
            valid_tokens = temporal * height * width
            self._last_depth_residual = (
                output[:, :valid_tokens] - x[:, :valid_tokens]
            ).reshape(output.shape[0], temporal, height * width, output.shape[-1])
            self._call_layers.append(
                {
                    "block_index": block_index,
                    "full_layer": True,
                    "active_frames": temporal,
                    "total_frames": temporal,
                    "active_ratio": 1.0,
                    "memory_refresh": memory_length > 0,
                    "bootstrap_q1_exact": self._step == 1,
                }
            )
            return output

        valid_tokens = temporal * height * width
        x_valid, tail = x[:, :valid_tokens], x[:, valid_tokens:]
        orbit_layout = kwargs.get("orbit_sparse_layout")
        query_frames = self._query_frames(
            block_index=block_index,
            temporal=temporal,
            memory_length=memory_length,
            device=x.device,
        )
        layout = self._asymmetric_layout(
            x=x_valid,
            grid_sizes=kwargs["grid_sizes"],
            memory_length=memory_length,
            orbit_sparse_layout=orbit_layout,
            query_frames=query_frames,
        )
        current = layout.query_indices
        x_current = x_valid[:, current]
        e_current = kwargs["e"][:, current]
        with torch.amp.autocast("cuda", dtype=torch.float32):
            modulation = (block.modulation.unsqueeze(0) + e_current).chunk(6, dim=2)
            e_full = (block.modulation.unsqueeze(0) + kwargs["e"]).chunk(6, dim=2)
            normalized_full = (
                block.norm1(x_valid).float()
                * (1 + e_full[1].squeeze(2))
                + e_full[0].squeeze(2)
            ).to(x_valid.dtype)
            attention_out = self._current_query_attention(
                block.self_attn,
                normalized_full,
                layout,
                freqs=kwargs["freqs"],
                memory_length=memory_length,
                memory_latent_idx=kwargs.get("memory_latent_idx"),
                predict_latent_idx=kwargs.get("predict_latent_idx"),
                orbit_sparse_layout=orbit_layout,
            )
            active = x_current + attention_out * modulation[2].squeeze(2)
        plucker = kwargs.get("plucker_emb")
        if plucker is not None:
            plucker = plucker[:, current]
            camera = block.cam_injector_layer2(
                F.silu(block.cam_injector_layer1(plucker))
            ) + plucker
            active = (
                (1.0 + block.cam_scale_layer(camera)) * active
                + block.cam_shift_layer(camera)
            )
        active = block.norm3(active)
        active = active + block.cross_attn(
            active,
            kwargs["context"],
            kwargs.get("context_lens"),
            fa_version=kwargs.get("fa_version"),
        )
        if block.action_model is not None:
            bridge = x_valid.clone()
            bridge[:, current] = active
            bridge = block.action_model(
                bridge.to(block.ffn[0].weight.dtype),
                temporal,
                height,
                width,
                kwargs.get("mouse_cond"),
                kwargs.get("keyboard_cond"),
                kwargs.get("mouse_cond_memory"),
                kwargs.get("keyboard_cond_memory"),
            )
            active = bridge[:, current]
        ffn = block.ffn(
            (
                block.norm2(active).float()
                * (1 + modulation[4].squeeze(2))
                + modulation[3].squeeze(2)
            ).to(block.ffn[0].weight.dtype)
        )
        with torch.amp.autocast("cuda", dtype=torch.float32):
            active = active + ffn * modulation[5].squeeze(2)
        output_valid = x_valid.clone()
        output_valid[:, current] = active
        active_residual = (active - x_valid[:, current]).reshape(
            x_valid.shape[0], len(query_frames), height * width, x_valid.shape[-1]
        )
        if self.inactive_transport == "previous_depth_residual":
            previous = self._last_depth_residual
            if previous is None:
                raise RuntimeError("depth residual relay has no exact boundary state")
            current_frames = torch.arange(
                memory_length, temporal, device=x.device, dtype=torch.long
            )
            active_mask = torch.zeros(temporal, device=x.device, dtype=torch.bool)
            active_mask[query_frames] = True
            inactive_frames = current_frames[~active_mask[current_frames]]
            if inactive_frames.numel():
                output_grid = output_valid.reshape(
                    x_valid.shape[0], temporal, height * width, x_valid.shape[-1]
                )
                input_grid = x_valid.reshape_as(output_grid)
                output_grid[:, inactive_frames] = (
                    input_grid[:, inactive_frames] + previous[:, inactive_frames]
                )
                output_valid = output_grid.reshape_as(x_valid)
        if self._last_depth_residual is None:
            self._last_depth_residual = x_valid.new_zeros(
                x_valid.shape[0], temporal, height * width, x_valid.shape[-1]
            )
        self._last_depth_residual[:, query_frames] = active_residual
        output = (
            torch.cat([output_valid, tail], dim=1) if tail.numel() else output_valid
        )
        self._call_layers.append(
            {
                "block_index": block_index,
                "full_layer": False,
                "active_frames": int(query_frames.numel()),
                "total_frames": temporal,
                "active_ratio": float(query_frames.numel()) / float(temporal),
                "memory_refresh": False,
                "memory_readable_as_kv": True,
                "current_frames_exact": int(query_frames.numel())
                == temporal - memory_length,
                "inactive_current_frames_identity_relay": int(
                    temporal - memory_length - query_frames.numel()
                ),
                "complete_spatial_frames": True,
                "action_bridge_dense": block.action_model is not None,
            }
        )
        return output

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result["name"] = self.name
        contract = result["cache_contract"]
        contract.update(
            {
                "memory_world_state_anchors_exact": False,
                "cwca_curvature_controls_refresh_rate": False,
                "current_worldline_residual_interpolation": False,
                "current_frames_exact_every_layer": self.current_phase_period is None,
                "memory_readable_as_kv_every_layer": True,
                "memory_query_is_depth_phased": True,
                "memory_refresh_period": self.memory_refresh_period,
                "causal_read_write_asymmetry": True,
                "current_phase_period": self.current_phase_period,
                "current_active_phases": self.current_active_phases,
                "current_high_curvature_anchors": self.current_high_curvature_anchors,
                "inactive_current_transport": self.inactive_transport,
            }
        )
        return result
