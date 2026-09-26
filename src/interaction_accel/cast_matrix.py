"""Fixed CAST component construction for Matrix-Game 3.0.

The public factory keeps the frozen method knobs in one place. The Matrix and
Light Interaction pipelines still own model loading, memory selection, chunk
state, and video delivery; see README.md for the integration boundary.
"""

from __future__ import annotations

from .methods.proposed.control_response_cwca_attention import (
    MatrixClosedLoopCWCAAttentionCompiler,
)
from .methods.proposed.li_budget_matched_memory import (
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler,
)
from .methods.proposed.matrix_curvature_phase_frame_weave import (
    MatrixCurvaturePhaseFrameWeave,
)


VARIANT = "cast_matrix"


def make_cast_components() -> tuple[
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler,
    MatrixClosedLoopCWCAAttentionCompiler,
    MatrixCurvaturePhaseFrameWeave,
]:
    """Build the frozen Matrix method components without loading a model.

    The returned memory compiler is used only on the R4 side of the original
    FOV-gated hybrid memory route. The original Matrix route is retained when
    the Light Interaction FOV gate disables it.
    """

    memory = MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler()
    attention = MatrixClosedLoopCWCAAttentionCompiler(
        response_weight=0.0,
        feedback_form="additive",
        curvature_temporal_pooling="mean",
        curvature_softmax_temperature=1.0,
        fixed_query_budget=False,
    )
    weave = MatrixCurvaturePhaseFrameWeave(
        attention,
        phase_period=5,
        active_phases=1,
        high_curvature_fraction=1.0,
        high_curvature_anchor_count=1,
        force_current_endpoints_exact=True,
        sparse_steps=(0,),
        sparse_layers=tuple(range(1, 29)),
        reconstruction="control_barycentric_residual",
        compact_cwca_topology=True,
        sparse_density=0.2,
        weave_domain="camera_guarded_current",
        camera_action_threshold=0.02,
        li_denoise_cache_enabled=True,
        dynamic_frame_selection="response_credit_std_scaled_mod5",
        dynamic_exact_cell_budget=2,
        dynamic_exact_cell_minimum=2,
        runtime_optimized=True,
        runtime_preallocated_barycentric=True,
        runtime_cache_barycentric_control=True,
        runtime_cache_barycentric_weights=True,
        runtime_reuse_dynamic_active_frame_list=True,
        runtime_shared_int8_qkv=True,
        runtime_shared_int8_qkv_all_paths=True,
        runtime_cached_compact_rope_phase=True,
        runtime_cached_native_rope_phase=True,
        runtime_gate_unused_control_response=True,
        runtime_fine_only_control_response_reduction=True,
        enable_fc_pasm=True,
        world_spectral_variant="frequency_confidence_phase_aligned_spectral_mixing",
        fc_tau_low=0.50,
        fc_tau_high=0.75,
        fc_freq_power=2.0,
        fc_temperature=0.03,
        fc_ramp_confidence=True,
        fc_reference_numerics=True,
        fc_v21_bypass=False,
        fc_prealloc_targets=True,
        fc_prune_unused_pairs=True,
        fc_parallel_streams=0,
        fc_lean_runtime=True,
        fc_profile_timing=False,
        fc_elide_scalar_readback=True,
        fc_triton_tile_extract=True,
        fc_triton_ola=False,
        fc_triton_phase_mix=False,
        fc_fused_complex_weights=True,
        routing_mode="fc_pasm_swap",
        routing_candidate_multiplier=2,
        routing_num_swap_rounds=1,
        routing_swap_eps=0.10,
        routing_lambda_anchor=1.0,
        routing_lambda_reconstruction=1.0,
        routing_sketch_groups=16,
        routing_require_fc_transport=True,
        routing_min_transport_affinity=0.50,
        routing_max_swaps_per_call=1,
        routing_active_layers=(22, 26),
        routing_same_bank_only=True,
        routing_skip_inactive_layers=True,
        variant_name=VARIANT,
    )
    return memory, attention, weave
