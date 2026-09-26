"""Locality-preserving Q/K uncertainty-gated tangent attention."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from .command_oriented_ray_qk_product_attention import (
    MatrixCommandOrientedRayQKProductAttentionCompiler,
)
from .command_phase_ray_worldline_attention import (
    MatrixCommandPhaseRayWorldlineAttentionCompiler,
)
from .reciprocal_joint_attention import _tile_visual_tensor
from .reciprocal_ray_geodesic_attention import (
    MatrixReciprocalRayGeodesicAttentionCompiler,
)
from .plucker_worldline_attention import (
    MatrixCheiralityAwareRayCrossingAttentionCompiler,
)


@dataclass(frozen=True)
class QKUncertaintyTangentSelectionReport:
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
    mean_rank_margin_gate: float = 0.0
    max_rank_margin_gate: float = 0.0
    action_geometry_trust: float = 1.0
    budget_min: int = 0
    budget_max: int = 0
    budget_total: int = 0
    mean_ray_motion: float = 0.0
    max_ray_motion: float = 0.0
    trajectory_complexity_kind: str = "motion_magnitude"
    persistence_quota_min: int = 0
    persistence_quota_max: int = 0
    mean_first_order_motion: float = 0.0
    section_budget_min: int = 0
    section_budget_max: int = 0
    mean_query_uncertainty: float = 0.0
    max_query_uncertainty: float = 0.0
    remote_quota_min: int = 0
    remote_quota_max: int = 0
    remote_quota_violations: int = 0
    memory_budget_total: int = 0
    current_worldline_budget_min: int = 0
    current_worldline_budget_max: int = 0


class MatrixQKUncertaintyTangentAttentionCompiler(
    MatrixCommandOrientedRayQKProductAttentionCompiler
):
    """Use geometry only where the live block Q/K evidence is uncertain.

    The released Light Interaction local Manhattan-one stencil is retained
    exactly.  Every remaining edge is ranked once by

        qk_cosine + (1 - q_coherence) (1 - k_coherence) tangent_similarity.

    Coherence is the norm of the block mean divided by the mean token norm,
    so the geometry gate is derived from the live Q/K block itself and has no
    fitted mixing coefficient.  The action tangent penalizes a ray edge only
    when its temporal direction opposes the current command.  One fixed-degree
    top-k emits the final 20-percent sparse graph.
    """

    name = "matrix_qk_uncertainty_tangent_attention_compiler"
    selection_standard = "local_qk_uncertainty_tangent_single_topk"
    cache_key_suffix = "local_qk_uncertainty_tangent"
    protect_local_stencil = True

    @staticmethod
    def _coherence(blocks: torch.Tensor) -> torch.Tensor:
        block_mean = blocks.float().mean(dim=-2)
        mean_norm = torch.linalg.vector_norm(block_mean, dim=-1)
        token_norm = torch.linalg.vector_norm(blocks.float(), dim=-1).mean(dim=-1)
        return (mean_norm / token_norm.clamp_min(1e-12)).clamp(0.0, 1.0)

    @staticmethod
    def _geometry_gate(
        q_uncertainty: torch.Tensor,
        k_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        return q_uncertainty[..., :, None] * k_uncertainty[..., None, :]

    @staticmethod
    def _tangent_similarity(
        ray_score: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        return ray_score - violations[None, None].to(ray_score.dtype)

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del q_content, k_content, temporal
        return content_score + geometry_gate * tangent_score

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        del q_blocks, k_blocks
        return torch.matmul(q_content, k_content.transpose(-2, -1))

    def _select_indices(self, score: torch.Tensor) -> torch.Tensor:
        return torch.argsort(score, dim=-1, descending=True, stable=True)[
            ..., : self._degree
        ]

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        geometry_indices: torch.Tensor,
        geometry_counts: torch.Tensor,
        latent_hw: tuple[int, int],
        block_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, QKUncertaintyTangentSelectionReport]:
        del geometry_indices
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("uncertainty-tangent selector requires equal B,H,L,D Q/K")
        if self._ray_features is None or self._orientation_violations is None:
            raise RuntimeError("uncertainty-tangent selector has no compiled geometry")
        if tuple(int(value) for value in block_shape) != self._block_shape:
            raise RuntimeError("runtime block shape differs from compiled geometry")
        runtime_counts = geometry_counts[0, 0].detach().to("cpu", torch.int64)
        if runtime_counts.numel() != self._num_blocks or not torch.all(
            runtime_counts == self._degree
        ):
            raise RuntimeError("runtime sparse degree differs from compiled geometry")

        height, width = (int(value) for value in latent_hw)
        tokens_per_frame = height * width
        if tokens_per_frame <= 0 or q.shape[2] % tokens_per_frame:
            raise ValueError("Q/K tokens do not define an integral temporal grid")
        temporal = q.shape[2] // tokens_per_frame
        q_blocks = _tile_visual_tensor(
            q, temporal=temporal, height=height, width=width,
            block_shape=self._block_shape,
        )
        k_blocks = _tile_visual_tensor(
            k, temporal=temporal, height=height, width=width,
            block_shape=self._block_shape,
        )
        if q_blocks.shape[2] != self._num_blocks:
            raise RuntimeError("tiled Q/K count differs from compiled geometry")

        q_content = F.normalize(q_blocks.float().mean(dim=-2), dim=-1)
        k_content = F.normalize(k_blocks.float().mean(dim=-2), dim=-1)
        content_score = self._content_score(
            q_blocks, k_blocks, q_content, k_content
        )
        q_uncertainty = 1.0 - self._coherence(q_blocks)
        k_uncertainty = 1.0 - self._coherence(k_blocks)
        geometry_gate = self._geometry_gate(q_uncertainty, k_uncertainty)

        ray = self._ray_features.to(device=q.device)
        ray_score = torch.matmul(ray, ray.transpose(0, 1))[None, None]
        violations = self._orientation_violations.to(device=q.device)
        tangent_score = self._tangent_similarity(ray_score, violations)
        score = self._compose_score(
            content_score,
            geometry_gate,
            tangent_score,
            q_content,
            k_content,
            temporal=math.ceil(temporal / self._block_shape[0]),
        )

        tt, th, tw = self._block_shape
        del tt
        nh = (height + th - 1) // th
        nw = (width + tw - 1) // tw
        spatial = nh * nw
        block_ids = torch.arange(self._num_blocks, device=q.device)
        t = block_ids // spatial
        rem = block_ids % spatial
        h = rem // nw
        w = rem % nw
        local = (
            (t[:, None] - t[None, :]).abs()
            + (h[:, None] - h[None, :]).abs()
            + (w[:, None] - w[None, :]).abs()
        ) <= 1
        if self.protect_local_stencil and int(local.sum(dim=-1).max().item()) > self._degree:
            raise RuntimeError("LI local stencil exceeds fixed sparse degree")
        if self.protect_local_stencil:
            score = score + local[None, None].to(score.dtype) * 1e6

        selected = self._select_indices(score)
        selected = torch.sort(selected, dim=-1).values
        duplicates = int(
            torch.count_nonzero(selected[..., 1:] == selected[..., :-1]).item()
        )
        if duplicates:
            raise RuntimeError("uncertainty-tangent selector emitted duplicate edges")
        expanded_local = local[None, None].expand(*selected.shape[:2], -1, -1)
        chosen_local = torch.gather(expanded_local, -1, selected)
        expected_local = (
            int(local.sum().item()) * selected.shape[0] * selected.shape[1]
            if self.protect_local_stencil
            else 0
        )
        missing_local = expected_local - int(chosen_local.sum().item())
        if self.protect_local_stencil and missing_local:
            raise RuntimeError("uncertainty-tangent selector lost an LI local edge")
        if not self.protect_local_stencil:
            missing_local = 0

        expanded_violations = violations[None, None].expand(
            *selected.shape[:2], -1, -1
        )
        chosen_violations = torch.gather(expanded_violations, -1, selected)
        per_row_violations = torch.count_nonzero(chosen_violations, dim=-1)
        expanded_ray = ray_score.expand(*selected.shape[:2], -1, -1)
        selected_ray = torch.gather(expanded_ray, -1, selected)
        selected_content = torch.gather(content_score, -1, selected)
        batch, heads = q.shape[:2]
        counts = torch.full(
            (batch, heads, self._num_blocks), self._degree,
            device=q.device, dtype=torch.int32,
        )
        report = QKUncertaintyTangentSelectionReport(
            batch_heads=batch * heads,
            query_rows=self._num_blocks,
            edges_per_row=self._degree,
            row_duplicate_violations=duplicates,
            selected_orientation_violations=int(per_row_violations.sum().item()),
            max_orientation_violations_per_row=int(per_row_violations.max().item()),
            mean_selected_ray_cosine=float(selected_ray.mean().item()),
            mean_selected_qk_cosine=float(selected_content.mean().item()),
            qk_read_for_selection=True,
            selector_count=1,
            protected_local_edges=expected_local,
            missing_local_edges=missing_local,
            mean_geometry_gate=float(geometry_gate.mean().item()),
            max_geometry_gate=float(geometry_gate.max().item()),
        )
        return selected.to(torch.int32), counts, report


class MatrixDualSimilarityTopKAttentionCompiler(
    MatrixQKUncertaintyTangentAttentionCompiler
):
    """Select one sparse support from equally scaled content and ray cosines.

    For block-pooled queries/keys and unit camera-ray descriptors, the only
    ranking score is

        (1 + cos(q_bar_i, k_bar_j)) / 2
        + (1 + cos(r_i, r_j)) / 2.

    The inherited fixed degree is the requested 20-percent budget.  There is
    no uncertainty gate, action tangent, second selector, or post-hoc support
    merge; the released LI local stencil is retained as a numerical safety
    invariant before the same single Top-K operation.
    """

    name = "matrix_dual_similarity_topk_attention_compiler"
    selection_standard = "content_ray_dual_similarity_single_topk_20_percent"
    cache_key_suffix = "content_ray_dual_similarity_topk20"
    protect_local_stencil = False

    @staticmethod
    def _geometry_gate(
        q_uncertainty: torch.Tensor,
        k_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(
            (*q_uncertainty.shape, k_uncertainty.shape[-1]),
            device=q_uncertainty.device,
            dtype=q_uncertainty.dtype,
        )

    @staticmethod
    def _tangent_similarity(
        ray_score: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        del violations
        return ray_score

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del geometry_gate, q_content, k_content, temporal
        return 0.5 * (1.0 + content_score) + 0.5 * (1.0 + tangent_score)

    def _select_indices(self, score: torch.Tensor) -> torch.Tensor:
        _, selected = torch.topk(score, k=self._degree, dim=-1)
        return selected


class MatrixRemoteDualSimilarityTopKAttentionCompiler(
    MatrixDualSimilarityTopKAttentionCompiler
):
    """Apply ray similarity only from current queries into Memory keys.

    The score is ``C + G`` for Current-to-Memory edges and exactly ``C`` for
    every other edge, where both similarities use the affine cosine map to
    ``[0, 1]``.  Unlike the direct DualSim ablation, LI's local stencil is
    protected before the single fixed-budget Top-K.
    """

    name = "matrix_remote_dual_similarity_topk_attention_compiler"
    selection_standard = (
        "local_content_ray_remote_dual_similarity_single_topk_20_percent"
    )
    cache_key_suffix = "local_content_ray_remote_dual_similarity_topk20"
    protect_local_stencil = True

    def __init__(self) -> None:
        super().__init__()
        self._current_to_memory_mask: torch.Tensor | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        _, tile_h, tile_w = self._block_shape
        spatial_blocks = math.ceil(int(kwargs["token_h"]) / tile_h) * math.ceil(
            int(kwargs["token_w"]) / tile_w
        )
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial_blocks
        block_ids = torch.arange(self._num_blocks, device=self._ray_features.device)
        self._current_to_memory_mask = (block_ids[:, None] >= memory_blocks) & (
            block_ids[None, :] < memory_blocks
        )
        return compiled, report

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del geometry_gate, q_content, k_content, temporal
        if self._current_to_memory_mask is None:
            raise RuntimeError("RemoteDual selector has no Current-to-Memory mask")
        content = 0.5 * (1.0 + content_score)
        geometry = 0.5 * (1.0 + tangent_score)
        remote = self._current_to_memory_mask.to(
            device=content.device, dtype=content.dtype
        )
        return content + remote[None, None] * geometry


class MatrixRemoteMeanSimilarityTopKAttentionCompiler(
    MatrixRemoteDualSimilarityTopKAttentionCompiler
):
    """Use the parameter-free mean of content and ray similarity remotely.

    Current-to-Memory edges use ``(C + G) / 2`` while all other edges use
    ``C``.  Both branches therefore remain in ``[0, 1]`` before the protected
    local stencil and the single 20-percent Top-K.
    """

    name = "matrix_remote_mean_similarity_topk_attention_compiler"
    selection_standard = (
        "local_content_ray_remote_mean_similarity_single_topk_20_percent"
    )
    cache_key_suffix = "local_content_ray_remote_mean_similarity_topk20"

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del geometry_gate, q_content, k_content, temporal
        if self._current_to_memory_mask is None:
            raise RuntimeError("RemoteMean selector has no Current-to-Memory mask")
        content = 0.5 * (1.0 + content_score)
        geometry = 0.5 * (1.0 + tangent_score)
        remote = self._current_to_memory_mask.to(device=content.device)
        remote_mean = 0.5 * (content + geometry)
        return torch.where(remote[None, None], remote_mean, content)


class MatrixQKCurvatureSupportAttentionCompiler(
    MatrixQKUncertaintyTangentAttentionCompiler
):
    """Inject positive action-tangent support only at joint Q/K uncertainty.

    The squared joint uncertainty confines geometry to ambiguous Q/K edges.
    The non-negative ray kernel is restricted to the current-action tangent
    cone, so geometry can promote a plausible remote edge but cannot subtract
    content evidence from reciprocal support.  The LI local stencil, fixed
    degree, and single top-k are unchanged.
    """

    name = "matrix_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_curvature_tangent_support_single_topk"
    cache_key_suffix = "local_qk_curvature_tangent_support"

    @staticmethod
    def _geometry_gate(
        q_uncertainty: torch.Tensor,
        k_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        joint = q_uncertainty[..., :, None] * k_uncertainty[..., None, :]
        return joint.square()

    @staticmethod
    def _tangent_similarity(
        ray_score: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        ray_support = 0.5 * (ray_score + 1.0)
        return ray_support * (~violations)[None, None].to(ray_score.dtype)


class MatrixRemoteQKCurvatureSupportAttentionCompiler(
    MatrixQKCurvatureSupportAttentionCompiler
):
    """Confine action geometry to current-query/remote-memory transport.

    Current-current dynamics and archive self-reconstruction retain the exact
    LI local-plus-Q/K ranking.  The curvature-gated action-ray support is used
    only when a current query retrieves a remote memory block, matching the
    role of world geometry without perturbing the current motion carrier.
    """

    name = "matrix_remote_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_remote_curvature_support_single_topk"
    cache_key_suffix = "local_qk_remote_curvature_support"

    def __init__(self) -> None:
        super().__init__()
        self._remote_transport_mask: torch.Tensor | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        _, th, tw = self._block_shape
        token_h = int(kwargs["token_h"])
        token_w = int(kwargs["token_w"])
        spatial = math.ceil(token_h / th) * math.ceil(token_w / tw)
        memory_blocks = math.ceil(int(layout.memory_length) / self._block_shape[0]) * spatial
        ids = torch.arange(self._num_blocks, device=self._ray_features.device)
        self._remote_transport_mask = (ids[:, None] >= memory_blocks) & (
            ids[None, :] < memory_blocks
        )
        return compiled, report

    def _geometry_gate(
        self,
        q_uncertainty: torch.Tensor,
        k_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        if self._remote_transport_mask is None:
            raise RuntimeError("remote curvature selector has no transport mask")
        gate = super()._geometry_gate(q_uncertainty, k_uncertainty)
        mask = self._remote_transport_mask.to(device=gate.device, dtype=gate.dtype)
        return gate * mask[None, None]


class MatrixWorldlineQKCurvatureSupportAttentionCompiler(
    MatrixRemoteQKCurvatureSupportAttentionCompiler
):
    """Add current-time support only along the same spatial camera ray.

    Remote memory transport keeps A3's unrestricted ray support.  Inside the
    current trajectory, geometry is admitted only between temporal blocks at
    the same spatial ray cell.  Cross-ray appearance mixing therefore remains
    the released LI Q/K ranking, while temporal worldlines receive the A2
    curvature-gated action support that was responsible for motion stability.
    """

    name = "matrix_worldline_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_remote_and_worldline_curvature_single_topk"
    cache_key_suffix = "local_qk_remote_and_worldline_curvature"

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if self._remote_transport_mask is None:
            raise RuntimeError("worldline curvature selector has no remote mask")
        _, th, tw = self._block_shape
        spatial = math.ceil(int(kwargs["token_h"]) / th) * math.ceil(
            int(kwargs["token_w"]) / tw
        )
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        ids = torch.arange(self._num_blocks, device=self._remote_transport_mask.device)
        current = ids >= memory_blocks
        same_ray = (ids[:, None] % spatial) == (ids[None, :] % spatial)
        current_worldline = current[:, None] & current[None, :] & same_ray
        self._remote_transport_mask = self._remote_transport_mask | current_worldline
        return compiled, report


class MatrixCausalWorldlineProductKernelAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """The A4 selector written as one causal worldline product kernel.

    This class is intentionally selection-equivalent to A4-F123.  Its purpose
    is to expose the method as one kernel rather than as a Q/K score followed
    by a sequence of geometric corrections.  For block content ``c``, Q/K
    cancellation residue ``u``, homogeneous unit-ray feature
    ``phi(r)=[r,1]/sqrt(2)``, and the directed worldline relation ``W_a``:

        K_a(i,j) = <c_i,c_j>
                   + 1[(i,j) in W_a] <u_i^2 phi(r_i), u_j^2 phi(r_j)>.

    ``W_a`` is the single geometric object: current-to-memory transport plus
    same-ray Current worldlines, oriented by the action-induced camera
    trajectory.  The released LI local stencil is a topology invariant, and
    one fixed-degree Top-K is applied to ``K_a``.  There is no learned weight,
    second selector, support merge, or metric-dependent gate.

    The operation order below deliberately matches A4-F123 exactly so the new
    paper parameterization cannot change a boundary tie through floating-point
    reassociation.
    """

    name = "matrix_causal_worldline_product_kernel_attention_compiler"
    selection_standard = "local_causal_worldline_product_kernel_single_topk"
    cache_key_suffix = "causal_worldline_product_kernel"

    @staticmethod
    def _tangent_similarity(
        ray_score: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        # <[r_i,1]/sqrt(2), [r_j,1]/sqrt(2)> on the action-causal cone.
        homogeneous_ray_kernel = 0.5 * (ray_score + 1.0)
        return homogeneous_ray_kernel * (~violations)[None, None].to(
            ray_score.dtype
        )

    # ``MatrixRemoteQKCurvatureSupportAttentionCompiler`` normally injects the
    # worldline domain through this virtual method.  Keep that public hook so
    # both the ordinary and fused A4 runtimes execute the same single kernel.
    def _geometry_gate(
        self,
        q_uncertainty: torch.Tensor,
        k_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        if self._remote_transport_mask is None:
            raise RuntimeError("causal worldline kernel has no transport domain")
        joint = q_uncertainty[..., :, None] * k_uncertainty[..., None, :]
        gate = joint.square()
        domain = self._remote_transport_mask.to(
            device=gate.device, dtype=gate.dtype
        )
        return gate * domain[None, None]

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del q_content, k_content, temporal
        return content_score + geometry_gate * tangent_score


class MatrixWorldlineQKCurvatureNoActionAxisAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Ablate only A4's action-axis feasibility from geometric support.

    Q/K content, squared uncertainty/curvature gating, Current-to-Memory and
    same-ray Current worldline masks, the LI local stencil, fixed 20-percent
    degree and the single Top-K are unchanged.  Geometry contributes its
    non-negative ray support without the command-axis violation mask.
    """

    name = "matrix_worldline_qk_curvature_no_action_axis_attention_compiler"
    selection_standard = (
        "local_qk_remote_and_worldline_curvature_no_action_axis_single_topk"
    )
    cache_key_suffix = "local_qk_remote_worldline_curvature_no_action_axis"

    @staticmethod
    def _tangent_similarity(
        ray_score: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        del violations
        return 0.5 * (ray_score + 1.0)


class MatrixActionCausalFeasibleQKAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Let action geometry define legality and Q/K rank legal content.

    This is a clean replacement for A4's additive curvature support.  The
    action-induced camera trajectory defines a directed causal cone for
    queries in the Current trajectory.  Memory-query rows are left unchanged,
    and the released LI Manhattan-one stencil is always legal.  Within that
    binary feasible set, the selector is exactly one content-only Q/K Top-K:

        feasible(i, j) = local(i, j) or memory_query(i)
                          or action_time_consistent(i, j)
        N(i) = TopK_K { cos(q_i, k_j) : feasible(i, j) }.

    Geometry therefore never receives a mixing weight and never competes with
    content in a shared scalar score.  It answers only whether information may
    flow along an action-causal world trajectory; live Q/K answers how much
    the legal source matters.
    """

    name = "matrix_action_causal_feasible_qk_attention_compiler"
    selection_standard = "action_causal_feasible_native_qk_single_topk"
    cache_key_suffix = "action_causal_feasible_native_qk"

    def __init__(self) -> None:
        super().__init__()
        self._causal_feasible_mask: torch.Tensor | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if self._orientation_violations is None:
            raise RuntimeError("action-causal Q/K selector has no orientation mask")

        tt, th, tw = self._block_shape
        height_blocks = math.ceil(int(kwargs["token_h"]) / th)
        width_blocks = math.ceil(int(kwargs["token_w"]) / tw)
        spatial = height_blocks * width_blocks
        memory_blocks = math.ceil(int(layout.memory_length) / tt) * spatial

        ids = torch.arange(self._num_blocks, device=self._orientation_violations.device)
        time_ids = ids // spatial
        spatial_ids = ids % spatial
        h_ids = spatial_ids // width_blocks
        w_ids = spatial_ids % width_blocks
        local = (
            (time_ids[:, None] - time_ids[None, :]).abs()
            + (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
        ) <= 1
        memory_query = ids[:, None] < memory_blocks
        action_consistent = ~self._orientation_violations
        feasible = local | memory_query | action_consistent
        feasible_counts = feasible.sum(dim=-1)
        if int(feasible_counts.min().item()) < self._degree:
            raise RuntimeError(
                "action-causal geometry leaves fewer legal keys than sparse degree"
            )
        self._causal_feasible_mask = feasible.detach()
        return compiled, replace(
            report,
            selection_standard=self.selection_standard,
        )

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del geometry_gate, tangent_score, q_content, k_content, temporal
        if self._causal_feasible_mask is None:
            raise RuntimeError("action-causal Q/K selector has no feasible set")
        feasible = self._causal_feasible_mask.to(device=content_score.device)
        return content_score.masked_fill(~feasible[None, None], -torch.inf)


class MatrixWorldRayTransportQKAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Transport Memory rays into a world-space candidate topology before Q/K.

    Geometry and content have disjoint jobs.  For a Current query, positive-
    depth closest points between its camera ray and every Memory ray define a
    depth-free reprojection tube.  The closest ``2K`` Memory blocks form a
    fixed world-space transport shortlist, while all Current keys and the
    released LI local stencil remain available.  Native block-pooled Q/K then
    selects the final ``K`` edges inside that Boolean topology.

    The crossing energy is never added to, multiplied with, or used to
    calibrate the Q/K score.  Action enters through the camera trajectory that
    generated the oriented rays, so the topology represents the geometric
    consequence of control rather than a post-hoc discrete action bias.
    """

    name = "matrix_world_ray_transport_qk_attention_compiler"
    selection_standard = "world_ray_transport_topology_then_native_qk_topk"
    cache_key_suffix = "world_ray_transport_native_qk"

    def __init__(self) -> None:
        super().__init__()
        self._transport_candidate_mask: torch.Tensor | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        tt, th, tw = self._block_shape
        token_h = int(kwargs["token_h"])
        token_w = int(kwargs["token_w"])
        spatial = math.ceil(token_h / th) * math.ceil(token_w / tw)
        memory_blocks = math.ceil(int(layout.memory_length) / tt) * spatial

        poses = MatrixReciprocalRayGeodesicAttentionCompiler._temporal_block_poses(
            kwargs["active_atoms"], kwargs["current_c2ws"], temporal_tile=tt
        )
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(poses)
        rays = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            poses,
            token_h=token_h,
            token_w=token_w,
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        ).float()
        origins = rays[:, :3]
        directions = F.normalize(rays[:, 3:], dim=-1)
        crossing = (
            MatrixCheiralityAwareRayCrossingAttentionCompiler
            ._forward_ray_distance_squared(origins, directions)
        )
        direction = (
            directions[:, None, :] - directions[None, :, :]
        ).square().sum(dim=-1)
        transport_energy = crossing + direction

        ids = torch.arange(self._num_blocks, device=rays.device)
        current_query = ids >= memory_blocks
        current_key = ids >= memory_blocks
        candidate = torch.ones(
            (self._num_blocks, self._num_blocks),
            device=rays.device,
            dtype=torch.bool,
        )
        if memory_blocks:
            remote_pool = min(memory_blocks, 2 * self._degree)
            remote_order = torch.argsort(
                transport_energy[:, :memory_blocks], dim=-1, stable=True
            )
            remote = remote_order[:, :remote_pool]
            candidate[current_query] = False
            current_rows = torch.nonzero(current_query, as_tuple=False).flatten()
            current_cols = torch.nonzero(current_key, as_tuple=False).flatten()
            candidate[current_rows[:, None], current_cols[None, :]] = True
            candidate[current_rows[:, None], remote[current_query]] = True

        time_ids = ids // spatial
        spatial_ids = ids % spatial
        width_blocks = math.ceil(token_w / tw)
        h_ids = spatial_ids // width_blocks
        w_ids = spatial_ids % width_blocks
        local = (
            (time_ids[:, None] - time_ids[None, :]).abs()
            + (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
        ) <= 1
        candidate |= local
        if int(candidate.sum(dim=-1).min().item()) < self._degree:
            raise RuntimeError("world-ray transport topology is smaller than K")
        self._transport_candidate_mask = candidate.detach()
        return compiled, replace(report, selection_standard=self.selection_standard)

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del geometry_gate, tangent_score, q_content, k_content, temporal
        if self._transport_candidate_mask is None:
            raise RuntimeError("world-ray transport topology was not compiled")
        topology = self._transport_candidate_mask.to(content_score.device)
        return content_score.masked_fill(~topology[None, None], -torch.inf)


class MatrixTrajectoryAdaptiveBudgetAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Allocate a fixed global edge budget by camera-ray motion, then use Q/K.

    Geometry never ranks or removes a key.  It only assigns the number of
    edges ``k_i`` available to each query block.  Within every row, all keys
    compete under native block-pooled Q/K cosine.  Integer budgets are assigned
    by largest-remainder apportionment, preserving exactly ``N * K`` edges.
    """

    name = "matrix_trajectory_adaptive_budget_attention_compiler"
    selection_standard = "trajectory_adaptive_fixed_total_budget_native_qk"
    cache_key_suffix = "trajectory_adaptive_budget_native_qk"
    trajectory_complexity_kind = "motion_magnitude"

    def __init__(self) -> None:
        super().__init__()
        self._row_degrees: torch.Tensor | None = None
        self._ray_motion: torch.Tensor | None = None
        self._first_order_motion: torch.Tensor | None = None
        self._ray_curvature: torch.Tensor | None = None
        self._spatial_blocks: int | None = None

    @staticmethod
    def _apportion(weights: torch.Tensor, total: int) -> torch.Tensor:
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("TABA weights must be a non-empty vector")
        if total <= 0:
            raise ValueError("TABA total edge budget must be positive")
        normalized = weights.float().clamp_min(0.0)
        if not bool(normalized.sum() > 0):
            normalized = torch.ones_like(normalized)
        raw = float(total) * normalized / normalized.sum()
        result = torch.floor(raw).to(torch.long)
        remainder = int(total - int(result.sum().item()))
        if remainder:
            fractions = raw - result
            order = torch.argsort(fractions, descending=True, stable=True)
            result[order[:remainder]] += 1
        if int(result.sum().item()) != total:
            raise RuntimeError("TABA integer apportionment lost the global budget")
        return result

    @staticmethod
    def _motion_weights(ray_motion: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(ray_motion.dtype).eps
        return 1.0 + ray_motion / (ray_motion.mean() + eps)

    def _allocate_degrees(
        self, weights: torch.Tensor, total: int
    ) -> torch.Tensor:
        return self._apportion(weights, total)

    def _motion_weights_for_layout(
        self, ray_motion: torch.Tensor, spatial: int
    ) -> torch.Tensor:
        del spatial
        return self._motion_weights(ray_motion)

    @staticmethod
    def _trajectory_complexity(section: torch.Tensor) -> torch.Tensor:
        if len(section) <= 1:
            return torch.zeros(
                section.shape[1], device=section.device, dtype=section.dtype
            )
        velocity = section[1:] - section[:-1]
        return torch.linalg.vector_norm(velocity, dim=-1).mean(dim=0)

    @staticmethod
    def _motion_curvature(section: torch.Tensor) -> torch.Tensor:
        if len(section) <= 2:
            return torch.zeros(
                section.shape[1], device=section.device, dtype=section.dtype
            )
        velocity = section[1:] - section[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        return torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=0)

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        tt, th, tw = self._block_shape
        token_h = int(kwargs["token_h"])
        token_w = int(kwargs["token_w"])
        spatial = math.ceil(token_h / th) * math.ceil(token_w / tw)
        poses = [
            getattr(atom, "c2w").to(device=kwargs["current_c2ws"].device)
            for atom in kwargs["active_atoms"]
        ]
        poses.extend(pose for pose in kwargs["current_c2ws"])
        all_poses = torch.stack(poses).float()
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(
            all_poses
        )
        per_frame_rays = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            all_poses,
            token_h=token_h,
            token_w=token_w,
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        ).float().reshape(len(all_poses), spatial, 6)
        motion_rows = []
        first_order_rows = []
        curvature_rows = []
        temporal_blocks = math.ceil(len(all_poses) / tt)
        for block in range(temporal_blocks):
            section = per_frame_rays[block * tt : min((block + 1) * tt, len(all_poses))]
            motion = self._trajectory_complexity(section)
            motion_rows.append(motion)
            first_order_rows.append(
                MatrixTrajectoryAdaptiveBudgetAttentionCompiler._trajectory_complexity(
                    section
                )
            )
            curvature_rows.append(self._motion_curvature(section))
        ray_motion = torch.cat(motion_rows)
        first_order_motion = torch.cat(first_order_rows)
        ray_curvature = torch.cat(curvature_rows)
        if ray_motion.numel() != self._num_blocks:
            raise RuntimeError("TABA ray-motion rows do not match attention blocks")
        self._spatial_blocks = spatial
        weights = self._motion_weights_for_layout(ray_motion, spatial)
        total = self._num_blocks * self._degree
        row_degrees = self._allocate_degrees(weights, total)

        ids = torch.arange(self._num_blocks, device=ray_motion.device)
        time_ids = ids // spatial
        spatial_ids = ids % spatial
        width_blocks = math.ceil(token_w / tw)
        h_ids = spatial_ids // width_blocks
        w_ids = spatial_ids % width_blocks
        local = (
            (time_ids[:, None] - time_ids[None, :]).abs()
            + (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
        ) <= 1
        if int(row_degrees.min().item()) < int(local.sum(dim=-1).max().item()):
            raise RuntimeError("TABA budget is too small to protect LI locality")
        if int(row_degrees.max().item()) > self._num_blocks:
            raise RuntimeError("TABA assigned more keys than exist")
        self._row_degrees = row_degrees.detach()
        self._ray_motion = ray_motion.detach()
        self._first_order_motion = first_order_motion.detach()
        self._ray_curvature = ray_curvature.detach()
        return compiled, replace(report, selection_standard=self.selection_standard)

    def _select_by_budget(
        self,
        ranking: torch.Tensor,
        local: torch.Tensor,
        row_degrees: torch.Tensor,
    ) -> torch.Tensor:
        del local
        max_degree = int(row_degrees.max().item())
        return torch.argsort(
            ranking, dim=-1, descending=True, stable=True
        )[..., :max_degree]

    def _runtime_degrees(self, q_blocks: torch.Tensor) -> torch.Tensor:
        del q_blocks
        if self._row_degrees is None:
            raise RuntimeError("TABA row budgets were not compiled")
        return self._row_degrees

    def select(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        geometry_indices: torch.Tensor,
        geometry_counts: torch.Tensor,
        latent_hw: tuple[int, int],
        block_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, QKUncertaintyTangentSelectionReport]:
        del geometry_indices
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("TABA requires equal B,H,L,D Q/K")
        if self._row_degrees is None or self._ray_motion is None:
            raise RuntimeError("TABA budgets were not compiled")
        if tuple(int(value) for value in block_shape) != self._block_shape:
            raise RuntimeError("TABA runtime block shape differs from compilation")
        reference_counts = geometry_counts[0, 0].detach().to("cpu", torch.long)
        if reference_counts.numel() != self._num_blocks or not torch.all(
            reference_counts == self._degree
        ):
            raise RuntimeError("TABA baseline layout does not have fixed K")

        height, width = (int(value) for value in latent_hw)
        temporal = q.shape[2] // (height * width)
        q_blocks = _tile_visual_tensor(
            q, temporal=temporal, height=height, width=width,
            block_shape=self._block_shape,
        )
        k_blocks = _tile_visual_tensor(
            k, temporal=temporal, height=height, width=width,
            block_shape=self._block_shape,
        )
        q_content = F.normalize(q_blocks.float().mean(dim=-2), dim=-1)
        k_content = F.normalize(k_blocks.float().mean(dim=-2), dim=-1)
        content_score = torch.matmul(q_content, k_content.transpose(-2, -1))

        _, th, tw = self._block_shape
        nh = math.ceil(height / th)
        nw = math.ceil(width / tw)
        spatial = nh * nw
        ids = torch.arange(self._num_blocks, device=q.device)
        time_ids = ids // spatial
        spatial_ids = ids % spatial
        h_ids = spatial_ids // nw
        w_ids = spatial_ids % nw
        local = (
            (time_ids[:, None] - time_ids[None, :]).abs()
            + (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
        ) <= 1
        ranking = content_score + local[None, None].to(content_score.dtype) * 1e6
        row_degrees = self._runtime_degrees(q_blocks).to(q.device)
        max_degree = int(row_degrees.max().item())
        selected = self._select_by_budget(ranking, local, row_degrees)
        batch, heads = q.shape[:2]
        counts = row_degrees[None, None].expand(batch, heads, -1).to(torch.int32)
        valid = (
            torch.arange(max_degree, device=q.device)[None, None, None, :]
            < counts[..., None]
        )
        duplicates = int(torch.count_nonzero(
            (selected[..., 1:] == selected[..., :-1]) & valid[..., 1:]
        ).item())
        expanded_local = local[None, None].expand(batch, heads, -1, -1)
        chosen_local = torch.gather(expanded_local, -1, selected) & valid
        expected_local = int(local.sum().item()) * batch * heads
        missing_local = expected_local - int(chosen_local.sum().item())
        if duplicates or missing_local:
            raise RuntimeError("TABA selection certificate failed")

        report = QKUncertaintyTangentSelectionReport(
            batch_heads=batch * heads,
            query_rows=self._num_blocks,
            edges_per_row=self._degree,
            row_duplicate_violations=duplicates,
            selected_orientation_violations=0,
            max_orientation_violations_per_row=0,
            mean_selected_ray_cosine=0.0,
            mean_selected_qk_cosine=float(content_score.mean().item()),
            qk_read_for_selection=True,
            selector_count=1,
            protected_local_edges=expected_local,
            missing_local_edges=missing_local,
            mean_geometry_gate=0.0,
            max_geometry_gate=0.0,
            budget_min=int(row_degrees.min().item()),
            budget_max=max_degree,
            budget_total=int(row_degrees.sum().item()),
            mean_ray_motion=float(self._ray_motion.mean().item()),
            max_ray_motion=float(self._ray_motion.max().item()),
            trajectory_complexity_kind=self.trajectory_complexity_kind,
            mean_first_order_motion=float(self._first_order_motion.mean().item()),
        )
        return selected.to(torch.int32), counts, report


class MatrixLogTrajectoryAdaptiveBudgetAttentionCompiler(
    MatrixTrajectoryAdaptiveBudgetAttentionCompiler
):
    """Compress TABA motion outliers with the parameter-free log1p map."""

    name = "matrix_log_trajectory_adaptive_budget_attention_compiler"
    selection_standard = "log_trajectory_adaptive_fixed_total_budget_native_qk"
    cache_key_suffix = "log_trajectory_adaptive_budget_native_qk"

    @staticmethod
    def _motion_weights(ray_motion: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(ray_motion.dtype).eps
        normalized_motion = ray_motion / (ray_motion.mean() + eps)
        return 1.0 + torch.log1p(normalized_motion)


class MatrixResidualTrajectoryAdaptiveBudgetAttentionCompiler(
    MatrixLogTrajectoryAdaptiveBudgetAttentionCompiler
):
    """Keep 80 percent uniform coverage and redistribute only the residual."""

    name = "matrix_residual_trajectory_adaptive_budget_attention_compiler"
    selection_standard = "residual_log_trajectory_adaptive_budget_native_qk"
    cache_key_suffix = "residual_log_trajectory_adaptive_budget_native_qk"

    def _allocate_degrees(
        self, weights: torch.Tensor, total: int
    ) -> torch.Tensor:
        # ceil(4K/5) guarantees every query receives at least 80% of the
        # released uniform degree, including the 24-edge short layouts.
        base = math.ceil(4 * self._degree / 5)
        base_total = len(weights) * base
        if base_total > total:
            raise RuntimeError("TABA-Residual base exceeds the global budget")
        bonus = self._apportion(weights, total - base_total)
        result = bonus + base
        if int(result.sum().item()) != total or int(result.min().item()) < base:
            raise RuntimeError("TABA-Residual budget certificate failed")
        return result


class MatrixCurvatureTrajectoryAdaptiveBudgetAttentionCompiler(
    MatrixResidualTrajectoryAdaptiveBudgetAttentionCompiler
):
    """Allocate only the residual budget from camera-ray non-stationarity."""

    name = "matrix_curvature_trajectory_adaptive_budget_attention_compiler"
    selection_standard = "curvature_residual_trajectory_budget_native_qk"
    cache_key_suffix = "curvature_residual_trajectory_budget_native_qk"
    trajectory_complexity_kind = "ray_velocity_curvature"

    @staticmethod
    def _trajectory_complexity(section: torch.Tensor) -> torch.Tensor:
        if len(section) <= 2:
            return torch.zeros(
                section.shape[1], device=section.device, dtype=section.dtype
            )
        velocity = section[1:] - section[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        return torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=0)


class MatrixSectionConservativeCurvatureAttentionCompiler(
    MatrixCurvatureTrajectoryAdaptiveBudgetAttentionCompiler
):
    """Conserve attention mass in every temporal world-model section.

    Curvature redistributes only the residual query budget inside a temporal
    tile.  Every tile retains exactly ``S * K`` edges, preventing a turn or
    reversal from borrowing capacity from an otherwise coherent rollout
    section.  Key identities remain the native global Q/K winners and the LI
    local stencil is unchanged.  Thus this is a conservation law on compute,
    not another content/geometry score fusion.
    """

    name = "matrix_section_conservative_curvature_attention_compiler"
    selection_standard = "section_conservative_curvature_budget_native_qk"
    cache_key_suffix = "section_conservative_curvature_budget_native_qk"
    trajectory_complexity_kind = "section_conservative_ray_velocity_curvature"

    def __init__(self) -> None:
        super().__init__()
        self._section_budgets: torch.Tensor | None = None

    def _motion_weights_for_layout(
        self, ray_motion: torch.Tensor, spatial: int
    ) -> torch.Tensor:
        sections = ray_motion.reshape(-1, spatial)
        eps = torch.finfo(sections.dtype).eps
        section_mean = sections.mean(dim=-1, keepdim=True)
        return (1.0 + torch.log1p(sections / (section_mean + eps))).reshape(-1)

    def _allocate_degrees(
        self, weights: torch.Tensor, total: int
    ) -> torch.Tensor:
        if self._spatial_blocks is None:
            raise RuntimeError("section size is unavailable during allocation")
        spatial = self._spatial_blocks
        if len(weights) % spatial:
            raise RuntimeError("attention rows do not form complete sections")
        base = math.ceil(4 * self._degree / 5)
        residual_per_section = spatial * (self._degree - base)
        sections = weights.reshape(-1, spatial)
        bonuses = torch.stack(
            [self._apportion(section, residual_per_section) for section in sections]
        )
        result = (bonuses + base).reshape(-1)
        section_budgets = result.reshape(-1, spatial).sum(dim=-1)
        expected = spatial * self._degree
        if int(result.sum().item()) != total or not bool(
            torch.all(section_budgets == expected)
        ):
            raise RuntimeError("section-conservative budget certificate failed")
        self._section_budgets = section_budgets.detach()
        return result

    def select(self, *args, **kwargs):
        selected, counts, report = super().select(*args, **kwargs)
        if self._section_budgets is None:
            raise RuntimeError("section budget certificate disappeared")
        return selected, counts, replace(
            report,
            section_budget_min=int(self._section_budgets.min().item()),
            section_budget_max=int(self._section_budgets.max().item()),
        )


class MatrixCurvatureTemperedUncertaintyAttentionCompiler(
    MatrixSectionConservativeCurvatureAttentionCompiler
):
    """Spend conserved section budgets on uncertain live world queries.

    Camera-ray curvature controls how non-uniform a temporal section is
    allowed to become, while live Q-token dispersion identifies the query
    rows that need the residual edges.  Key identities are still selected by
    the unmodified global Q/K ranking.  This cleanly separates three roles:
    section conservation protects rollout continuity, trajectory curvature
    sets compute urgency, and model uncertainty places the compute.
    """

    name = "matrix_curvature_tempered_uncertainty_attention_compiler"
    selection_standard = (
        "section_conservative_curvature_tempered_query_uncertainty_native_qk"
    )
    cache_key_suffix = "curvature_tempered_query_uncertainty_native_qk"
    trajectory_complexity_kind = (
        "section_conservative_curvature_tempered_query_uncertainty"
    )

    def __init__(self) -> None:
        super().__init__()
        self._mean_query_uncertainty = 0.0
        self._max_query_uncertainty = 0.0

    @staticmethod
    def _apportion_section_matrix(
        weights: torch.Tensor, total_per_section: int
    ) -> torch.Tensor:
        normalized = weights.float().clamp_min(0.0)
        normalized = torch.where(
            normalized.sum(dim=-1, keepdim=True) > 0,
            normalized,
            torch.ones_like(normalized),
        )
        raw = (
            float(total_per_section)
            * normalized
            / normalized.sum(dim=-1, keepdim=True)
        )
        result = torch.floor(raw).to(torch.long)
        remainder = total_per_section - result.sum(dim=-1)
        fractions = raw - result
        order = torch.argsort(fractions, dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            weights.shape[-1], device=weights.device, dtype=torch.long
        )[None].expand_as(order)
        ranks.scatter_(-1, order, rank_values)
        result += (ranks < remainder[:, None]).to(result.dtype)
        return result

    def _runtime_degrees(self, q_blocks: torch.Tensor) -> torch.Tensor:
        if self._ray_curvature is None or self._spatial_blocks is None:
            raise RuntimeError("curvature-tempered state was not compiled")
        spatial = self._spatial_blocks
        curvature = self._ray_curvature.to(q_blocks.device).reshape(-1, spatial)
        eps = torch.finfo(q_blocks.float().dtype).eps

        # Token cancellation is a parameter-free uncertainty proxy: coherent
        # query tokens have norm(mean(q)) ~= mean(norm(q)), while conflicting
        # tokens cancel in the block mean.  Average over batch and heads so the
        # sparse layout remains shared by the released kernel.
        q_float = q_blocks.float()
        mean_norm = torch.linalg.vector_norm(q_float.mean(dim=-2), dim=-1)
        token_norm = torch.linalg.vector_norm(q_float, dim=-1).mean(dim=-1)
        uncertainty = (1.0 - mean_norm / token_norm.clamp_min(eps)).clamp(0.0, 1.0)
        uncertainty = uncertainty.mean(dim=(0, 1)).reshape(-1, spatial)
        uncertainty_scale = uncertainty / uncertainty.mean(
            dim=-1, keepdim=True
        ).clamp_min(eps)

        section_curvature = curvature.mean(dim=-1, keepdim=True)
        curvature_gate = torch.log1p(
            section_curvature / section_curvature.mean().clamp_min(eps)
        )
        weights = 1.0 + curvature_gate * uncertainty_scale
        base = math.ceil(4 * self._degree / 5)
        residual = spatial * (self._degree - base)
        bonuses = self._apportion_section_matrix(weights, residual)
        degrees = (bonuses + base).reshape(-1)
        section_budgets = degrees.reshape(-1, spatial).sum(dim=-1)
        expected = spatial * self._degree
        if not bool(torch.all(section_budgets == expected)):
            raise RuntimeError("curvature-tempered section budget drifted")
        self._section_budgets = section_budgets.detach()
        self._mean_query_uncertainty = float(uncertainty.mean().item())
        self._max_query_uncertainty = float(uncertainty.max().item())
        return degrees.detach()

    def select(self, *args, **kwargs):
        selected, counts, report = super().select(*args, **kwargs)
        return selected, counts, replace(
            report,
            mean_query_uncertainty=self._mean_query_uncertainty,
            max_query_uncertainty=self._max_query_uncertainty,
        )


class MatrixCurvatureBalancedReservoirAttentionCompiler(
    MatrixSectionConservativeCurvatureAttentionCompiler
):
    """Keep fixed query capacity and balance world-memory retrieval on turns.

    Each current query reads from two world-model reservoirs: persistent
    Memory and the current rollout.  Smooth motion uses the capacity prior
    implied by the two reservoir sizes.  Ray curvature continuously moves the
    prior toward an equal split, preventing either reservoir from monopolizing
    a query during turns or reversals.  Native Q/K independently ranks keys
    inside both reservoirs; the per-row K, total FLOPs and LI local stencil are
    unchanged.
    """

    name = "matrix_curvature_balanced_reservoir_attention_compiler"
    selection_standard = "curvature_balanced_dual_reservoir_native_qk"
    cache_key_suffix = "curvature_balanced_dual_reservoir_native_qk"
    trajectory_complexity_kind = "curvature_balanced_memory_current_reservoir"

    def __init__(self) -> None:
        super().__init__()
        self._memory_blocks = 0
        self._remote_quota: torch.Tensor | None = None

    @staticmethod
    def _balanced_remote_fraction(
        curvature: torch.Tensor, memory_fraction: float
    ) -> torch.Tensor:
        eps = torch.finfo(curvature.dtype).eps
        gate = curvature / (curvature + curvature.mean() + eps)
        return memory_fraction + gate * (0.5 - memory_fraction)

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if self._ray_curvature is None or self._spatial_blocks is None:
            raise RuntimeError("reservoir curvature was not compiled")
        spatial = self._spatial_blocks
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        memory_blocks = min(memory_blocks, self._num_blocks)
        self._memory_blocks = memory_blocks

        # Query capacity is deliberately uniform.  Curvature changes only how
        # that fixed capacity is divided between the two causal reservoirs.
        degrees = torch.full_like(self._ray_curvature, self._degree, dtype=torch.long)
        self._row_degrees = degrees.detach()
        self._section_budgets = degrees.reshape(-1, spatial).sum(dim=-1).detach()

        ids = torch.arange(self._num_blocks, device=degrees.device)
        current_query = ids >= memory_blocks
        if memory_blocks == 0 or memory_blocks == self._num_blocks:
            quota = torch.zeros_like(degrees)
        else:
            memory_fraction = float(memory_blocks) / float(self._num_blocks)
            fraction = self._balanced_remote_fraction(
                self._ray_curvature, memory_fraction
            )
            quota = torch.round(float(self._degree) * fraction).to(torch.long)

            _, th, tw = self._block_shape
            nh = math.ceil(int(kwargs["token_h"]) / th)
            nw = math.ceil(int(kwargs["token_w"]) / tw)
            time_ids = ids // spatial
            spatial_ids = ids % spatial
            h_ids = spatial_ids // nw
            w_ids = spatial_ids % nw
            local = (
                (time_ids[:, None] - time_ids[None, :]).abs()
                + (h_ids[:, None] - h_ids[None, :]).abs()
                + (w_ids[:, None] - w_ids[None, :]).abs()
            ) <= 1
            local_memory = local[:, :memory_blocks].sum(dim=-1).to(torch.long)
            local_current = local[:, memory_blocks:].sum(dim=-1).to(torch.long)
            quota = torch.maximum(quota, local_memory)
            quota = torch.minimum(quota, self._degree - local_current)
            quota = quota.clamp(0, min(self._degree, memory_blocks))
            quota = torch.maximum(
                quota,
                torch.full_like(quota, self._degree - (self._num_blocks - memory_blocks)),
            )
        quota = torch.where(current_query, quota, torch.full_like(quota, -1))
        self._remote_quota = quota.detach()
        return compiled, replace(report, selection_standard=self.selection_standard)

    def _select_by_budget(
        self,
        ranking: torch.Tensor,
        local: torch.Tensor,
        row_degrees: torch.Tensor,
    ) -> torch.Tensor:
        if self._remote_quota is None:
            raise RuntimeError("reservoir quotas were not compiled")
        if self._memory_blocks in {0, self._num_blocks}:
            return super()._select_by_budget(ranking, local, row_degrees)
        quota = self._remote_quota.to(ranking.device)
        current_quota = torch.where(quota >= 0, self._degree - quota, -1)
        must_keep = torch.zeros_like(ranking, dtype=torch.bool)

        max_memory = int(quota.clamp_min(0).max().item())
        if max_memory:
            memory_order = torch.argsort(
                ranking[..., : self._memory_blocks],
                dim=-1,
                descending=True,
                stable=True,
            )[..., :max_memory]
            valid = (
                torch.arange(max_memory, device=ranking.device)[None, None, None]
                < quota[None, None, :, None]
            )
            must_keep[..., : self._memory_blocks].scatter_(
                -1, memory_order, valid.expand_as(memory_order)
            )

        max_current = int(current_quota.clamp_min(0).max().item())
        if max_current:
            current_order = torch.argsort(
                ranking[..., self._memory_blocks :],
                dim=-1,
                descending=True,
                stable=True,
            )[..., :max_current]
            valid = (
                torch.arange(max_current, device=ranking.device)[None, None, None]
                < current_quota[None, None, :, None]
            )
            current_keep = must_keep[..., self._memory_blocks :]
            current_keep.scatter_(-1, current_order, valid.expand_as(current_order))
        constrained = ranking + must_keep.to(ranking.dtype) * 1e6
        return super()._select_by_budget(constrained, local, row_degrees)

    def select(self, *args, **kwargs):
        selected, counts, report = super().select(*args, **kwargs)
        if self._remote_quota is None:
            raise RuntimeError("reservoir quotas disappeared")
        quota = self._remote_quota.to(selected.device)
        current_rows = quota >= 0
        valid = (
            torch.arange(selected.shape[-1], device=selected.device)[None, None, None]
            < counts[..., None]
        )
        actual = ((selected < self._memory_blocks) & valid).sum(dim=-1)
        expected = quota[None, None].expand_as(actual)
        violations = int(torch.count_nonzero(
            (actual != expected) & current_rows[None, None]
        ).item())
        if violations:
            raise RuntimeError("curvature-balanced reservoir quota failed")
        active = quota[current_rows]
        return selected, counts, replace(
            report,
            remote_quota_min=int(active.min().item()) if active.numel() else 0,
            remote_quota_max=int(active.max().item()) if active.numel() else 0,
            remote_quota_violations=violations,
        )


class MatrixChronologicalWorldlineCurvatureAttentionCompiler(
    MatrixSectionConservativeCurvatureAttentionCompiler
):
    """Differentiate the causal camera timeline, never Memory retrieval order.

    Retrieved Memory atoms form an unordered support set, so taking finite
    differences in selector order creates fictitious motion.  This compiler
    sorts only the two causal anchors used for differentiation, appends the
    ordered current camera rollout, and computes time-normalized ray
    acceleration on that physical timeline.  Memory queries retain uniform K;
    current residual capacity is transported only along each spatial
    worldline, whose lifetime budget is exactly conserved.  Native global Q/K
    still selects every key.
    """

    name = "matrix_chronological_worldline_curvature_attention_compiler"
    selection_standard = "chronological_worldline_curvature_budget_native_qk"
    cache_key_suffix = "chronological_worldline_curvature_budget_native_qk"
    trajectory_complexity_kind = "chronological_time_normalized_worldline_curvature"

    def __init__(
        self,
        *,
        curvature_temporal_pooling: str = "mean",
        curvature_softmax_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if curvature_temporal_pooling not in {
            "mean", "softmax", "normalized_softmax"
        }:
            raise ValueError(
                "chronological curvature pooling must be mean, softmax, or "
                "normalized_softmax"
            )
        if curvature_softmax_temperature <= 0.0:
            raise ValueError("curvature softmax temperature must be positive")
        self.curvature_temporal_pooling = str(curvature_temporal_pooling)
        self.curvature_softmax_temperature = float(
            curvature_softmax_temperature
        )
        self._memory_budget_total = 0
        self._current_worldline_budgets: torch.Tensor | None = None
        self._profile_current_block_curvature: torch.Tensor | None = None
        self._profile_memory_blocks = 0
        self._profile_token_hw = (0, 0)
        self._profile_curvature_pooling: dict[str, object] = {
            "mode": self.curvature_temporal_pooling,
            "temperature": self.curvature_softmax_temperature,
            "rows": [],
        }

    def _pool_temporal_curvature(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Pool chronological frame curvature inside one attention time tile."""

        if values.ndim != 2 or not values.shape[0]:
            raise RuntimeError("temporal curvature pool requires T x S values")
        mean = values.mean(dim=0)
        if self.curvature_temporal_pooling == "mean":
            pooled = mean
            peak_weight = values.new_full(
                (values.shape[1],), 1.0 / float(values.shape[0])
            )
        else:
            logits = values
            if self.curvature_temporal_pooling == "normalized_softmax":
                eps = torch.finfo(values.dtype).eps
                logits = values / mean.clamp_min(eps)
            weights = torch.softmax(
                logits / self.curvature_softmax_temperature, dim=0
            )
            pooled = (weights * values).sum(dim=0)
            peak_weight = weights.max(dim=0).values
        delta = (pooled - mean).abs()
        return pooled, {
            "frames": int(values.shape[0]),
            "mean_abs_delta_from_mean": float(delta.mean().item()),
            "max_abs_delta_from_mean": float(delta.max().item()),
            "mean_peak_weight": float(peak_weight.mean().item()),
            "max_peak_weight": float(peak_weight.max().item()),
        }

    @staticmethod
    def _ordered_anchor_poses(active_atoms, device: torch.device):
        ordered = sorted(
            active_atoms, key=lambda atom: int(getattr(atom, "original_time_index"))
        )
        ordered = ordered[-2:]
        poses = [getattr(atom, "c2w").to(device=device) for atom in ordered]
        times = [float(getattr(atom, "original_time_index")) for atom in ordered]
        return poses, times

    @staticmethod
    def _time_normalized_curvature(
        ray_states: torch.Tensor, times: torch.Tensor, history: int
    ) -> torch.Tensor:
        current = len(ray_states) - history
        result = torch.zeros(
            current,
            ray_states.shape[1],
            device=ray_states.device,
            dtype=ray_states.dtype,
        )
        counts = torch.zeros(current, device=ray_states.device, dtype=ray_states.dtype)
        if len(ray_states) <= 2 or current <= 0:
            return result
        dt = (times[1:] - times[:-1]).clamp_min(1.0)
        velocity = (ray_states[1:] - ray_states[:-1]) / dt[:, None, None]
        mid_dt = 0.5 * (dt[1:] + dt[:-1])
        acceleration = (velocity[1:] - velocity[:-1]) / mid_dt[:, None, None]
        magnitude = torch.linalg.vector_norm(acceleration, dim=-1)
        for acceleration_index in range(len(magnitude)):
            center = acceleration_index + 1
            current_index = max(0, center - history)
            current_index = min(current - 1, current_index)
            result[current_index] += magnitude[acceleration_index]
            counts[current_index] += 1
        return result / counts[:, None].clamp_min(1.0)

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if self._spatial_blocks is None:
            raise RuntimeError("chronological curvature has no spatial layout")
        tt, th, tw = self._block_shape
        spatial = self._spatial_blocks
        device = kwargs["current_c2ws"].device
        anchor_poses, anchor_times = self._ordered_anchor_poses(
            kwargs["active_atoms"], device
        )
        current_c2ws = kwargs["current_c2ws"].float()
        if anchor_times:
            start_time = anchor_times[-1]
        else:
            start_time = 0.0
        current_times = [start_time + index + 1.0 for index in range(len(current_c2ws))]
        poses = anchor_poses + [pose for pose in current_c2ws]
        times = torch.tensor(anchor_times + current_times, device=device, dtype=torch.float32)
        pose_tensor = torch.stack(poses).float()
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(
            pose_tensor
        )
        rays = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            pose_tensor,
            token_h=int(kwargs["token_h"]),
            token_w=int(kwargs["token_w"]),
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        ).float().reshape(len(poses), spatial, 6)
        current_curvature = self._time_normalized_curvature(
            rays, times, len(anchor_poses)
        )

        memory_temporal = math.ceil(int(layout.memory_length) / tt)
        memory_blocks = min(memory_temporal * spatial, self._num_blocks)
        current_temporal = self._num_blocks // spatial - memory_temporal
        current_offset = memory_temporal * tt - int(layout.memory_length)
        current_rows = []
        pooling_rows = []
        for block in range(max(0, current_temporal)):
            low = current_offset + block * tt
            high = min(low + tt, len(current_curvature))
            if low < high:
                pooled, pooling_record = self._pool_temporal_curvature(
                    current_curvature[low:high]
                )
                current_rows.append(pooled)
                pooling_rows.append(
                    {
                        "temporal_row": int(block),
                        "current_low": int(low),
                        "current_high": int(high),
                        **pooling_record,
                    }
                )
            else:
                current_rows.append(torch.zeros(spatial, device=device))
                pooling_rows.append(
                    {
                        "temporal_row": int(block),
                        "current_low": int(low),
                        "current_high": int(high),
                        "frames": 0,
                        "mean_abs_delta_from_mean": 0.0,
                        "max_abs_delta_from_mean": 0.0,
                        "mean_peak_weight": 0.0,
                        "max_peak_weight": 0.0,
                    }
                )
        self._profile_curvature_pooling = {
            "mode": self.curvature_temporal_pooling,
            "temperature": self.curvature_softmax_temperature,
            "rows": pooling_rows,
        }

        degrees = torch.full(
            (self._num_blocks,), self._degree, device=device, dtype=torch.long
        )
        if current_rows:
            curvature = torch.stack(current_rows)  # T_current, S
            eps = torch.finfo(curvature.dtype).eps
            normalized = curvature / curvature.mean(dim=0, keepdim=True).clamp_min(eps)
            weights = 1.0 + torch.log1p(normalized)
            base = math.ceil(4 * self._degree / 5)
            residual_per_worldline = len(current_rows) * (self._degree - base)
            bonuses = MatrixCurvatureTemperedUncertaintyAttentionCompiler._apportion_section_matrix(
                weights.transpose(0, 1), residual_per_worldline
            ).transpose(0, 1)
            current_degrees = (bonuses + base).reshape(-1)
            degrees[memory_blocks : memory_blocks + len(current_degrees)] = current_degrees
            worldline_budgets = (bonuses + base).sum(dim=0)
        else:
            curvature = torch.empty(
                (0, spatial), device=device, dtype=current_curvature.dtype
            )
            worldline_budgets = torch.empty(0, device=device, dtype=torch.long)
        if int(degrees.sum().item()) != self._num_blocks * self._degree:
            raise RuntimeError("chronological curvature lost the global budget")
        self._row_degrees = degrees.detach()
        self._section_budgets = degrees.reshape(-1, spatial).sum(dim=-1).detach()
        self._ray_motion = torch.cat(
            [
                torch.zeros(memory_blocks, device=device),
                torch.stack(current_rows).reshape(-1) if current_rows else torch.empty(0, device=device),
            ]
        ).detach()
        self._memory_budget_total = memory_blocks * self._degree
        self._current_worldline_budgets = worldline_budgets.detach()
        self._profile_current_block_curvature = curvature.reshape(-1).detach()
        self._profile_memory_blocks = int(memory_blocks)
        self._profile_token_hw = (
            int(kwargs["token_h"]),
            int(kwargs["token_w"]),
        )
        return compiled, replace(report, selection_standard=self.selection_standard)

    def current_block_curvature_profile(self) -> dict[str, object]:
        """Expose the exact CWCA current-block geometry for diagnostics only."""

        if self._profile_current_block_curvature is None:
            raise RuntimeError("CWCA curvature was requested before compilation")
        return {
            "curvature": self._profile_current_block_curvature,
            "memory_blocks": self._profile_memory_blocks,
            "token_hw": self._profile_token_hw,
            "block_shape": self._block_shape,
            "num_blocks": self._num_blocks,
            "temporal_pooling": self._profile_curvature_pooling,
        }

    def select(self, *args, **kwargs):
        selected, counts, report = super().select(*args, **kwargs)
        budgets = self._current_worldline_budgets
        return selected, counts, replace(
            report,
            memory_budget_total=self._memory_budget_total,
            current_worldline_budget_min=(
                int(budgets.min().item()) if budgets is not None and budgets.numel() else 0
            ),
            current_worldline_budget_max=(
                int(budgets.max().item()) if budgets is not None and budgets.numel() else 0
            ),
        )


class MatrixCurvaturePersistenceTrajectoryAdaptiveBudgetAttentionCompiler(
    MatrixCurvatureTrajectoryAdaptiveBudgetAttentionCompiler
):
    """Use curvature for row budgets and first-order motion for local continuity.

    Curvature allocates the fixed residual edge budget exactly as in
    TABA-Curvature.  First-order motion never changes a key score and never
    removes a global candidate.  It only reserves part of each row's curvature
    bonus for the native-Q/K winners inside a one-step spatiotemporal tube.
    The remaining slots are native-Q/K winners over all keys, so the global
    edge count and the released LI local stencil stay unchanged.
    """

    name = "matrix_curvature_persistence_trajectory_adaptive_budget_attention_compiler"
    selection_standard = (
        "curvature_budget_motion_persistence_constrained_native_qk"
    )
    cache_key_suffix = "curvature_budget_motion_persistence_native_qk"
    trajectory_complexity_kind = "curvature_budget_first_order_local_persistence"

    def __init__(self) -> None:
        super().__init__()
        self._persistence_tube: torch.Tensor | None = None
        self._persistence_quota: torch.Tensor | None = None

    @staticmethod
    def _persistence_fraction(
        first_order_motion: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        eps = torch.finfo(first_order_motion.dtype).eps
        motion = torch.log1p(
            first_order_motion / (first_order_motion.mean() + eps)
        )
        turning = torch.log1p(curvature / (curvature.mean() + eps))
        fraction = motion / (motion + turning + eps)
        inactive = (first_order_motion <= eps) & (curvature <= eps)
        return torch.where(inactive, torch.zeros_like(fraction), fraction)

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if (
            self._row_degrees is None
            or self._first_order_motion is None
            or self._ray_curvature is None
        ):
            raise RuntimeError("TABA persistence state was not compiled")
        _, th, tw = self._block_shape
        token_h = int(kwargs["token_h"])
        token_w = int(kwargs["token_w"])
        nh = math.ceil(token_h / th)
        nw = math.ceil(token_w / tw)
        spatial = nh * nw
        ids = torch.arange(self._num_blocks, device=self._row_degrees.device)
        time_ids = ids // spatial
        spatial_ids = ids % spatial
        h_ids = spatial_ids // nw
        w_ids = spatial_ids % nw

        # A local world-tube is deliberately broader than the released LI
        # Manhattan-one stencil: adjacent temporal tiles may also move by one
        # spatial block.  Motion only sets a minimum Q/K-selected coverage of
        # this tube; it is not added to the Q/K score.
        tube = (
            (time_ids[:, None] - time_ids[None, :]).abs() <= 1
        ) & (
            (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
            <= 1
        )
        li_local = (
            (time_ids[:, None] - time_ids[None, :]).abs()
            + (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
        ) <= 1
        base = math.ceil(4 * self._degree / 5)
        bonus = (self._row_degrees - base).clamp_min(0)
        persistence = self._persistence_fraction(
            self._first_order_motion, self._ray_curvature
        )
        extra_local = torch.round(persistence * bonus.float()).to(torch.long)
        quota = li_local.sum(dim=-1).to(torch.long) + extra_local
        quota = torch.minimum(quota, tube.sum(dim=-1).to(torch.long))
        quota = torch.minimum(quota, self._row_degrees)
        if bool(torch.any(quota < li_local.sum(dim=-1))):
            raise RuntimeError("TABA persistence quota lost LI locality")
        self._persistence_tube = tube.detach()
        self._persistence_quota = quota.detach()
        return compiled, replace(report, selection_standard=self.selection_standard)

    def _select_by_budget(
        self,
        ranking: torch.Tensor,
        local: torch.Tensor,
        row_degrees: torch.Tensor,
    ) -> torch.Tensor:
        if self._persistence_tube is None or self._persistence_quota is None:
            raise RuntimeError("TABA persistence quotas were not compiled")
        tube = self._persistence_tube.to(ranking.device)
        quota = self._persistence_quota.to(ranking.device)
        max_quota = int(quota.max().item())
        tube_ranking = ranking.masked_fill(~tube[None, None], -torch.inf)
        tube_order = torch.argsort(
            tube_ranking, dim=-1, descending=True, stable=True
        )[..., :max_quota]
        quota_valid = (
            torch.arange(max_quota, device=ranking.device)[None, None, None, :]
            < quota[None, None, :, None]
        )
        must_keep = torch.zeros_like(ranking, dtype=torch.bool)
        must_keep.scatter_(-1, tube_order, quota_valid.expand_as(tube_order))
        constrained = ranking + must_keep.to(ranking.dtype) * 1e6
        return super()._select_by_budget(constrained, local, row_degrees)

    def select(self, *args, **kwargs):
        selected, counts, report = super().select(*args, **kwargs)
        if self._persistence_quota is None:
            raise RuntimeError("TABA persistence quotas disappeared")
        return selected, counts, replace(
            report,
            persistence_quota_min=int(self._persistence_quota.min().item()),
            persistence_quota_max=int(self._persistence_quota.max().item()),
        )


class MatrixMarginCalibratedActionWorldlineAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Resolve only uncertain A4 top-k decisions with action-worldline support.

    This is the old-path A4-M2 experiment.  It keeps A4's Q/K coherence gate,
    remote-memory transport, same-ray current worldlines, local LI stencil and
    one fixed-degree top-k.  The only changes are:

    * geometry is calibrated to each row's live Q/K top-k boundary, so it
      cannot easily overturn candidates far from that boundary; and
    * geometry trust is the rotation share of the active command families.

    Command-family occupancy is used instead of comparing raw mouse and
    keyboard magnitudes because Matrix encodes those controls in different
    units.  Pure rotation therefore retains geometry, pure translation falls
    back to Q/K, and a boundary window containing both families interpolates
    between them without a fitted coefficient.
    """

    name = "matrix_margin_calibrated_action_worldline_attention_compiler"
    selection_standard = (
        "local_qk_margin_calibrated_action_worldline_curvature_single_topk"
    )
    cache_key_suffix = "local_qk_margin_calibrated_action_worldline_curvature"

    def __init__(self) -> None:
        super().__init__()
        self._local_mask: torch.Tensor | None = None
        self._action_geometry_trust = 0.0
        self._last_mean_rank_margin_gate = 0.0
        self._last_max_rank_margin_gate = 0.0

    @staticmethod
    def _command_family_trust(current_action: torch.Tensor) -> float:
        command = MatrixCommandPhaseRayWorldlineAttentionCompiler._command_vector(
            current_action
        )
        eps = torch.finfo(torch.float32).eps
        rotation_active = bool(
            command.numel() >= 2
            and float(torch.linalg.vector_norm(command[:2]).item()) > eps
        )
        translation_active = False
        if command.numel() >= 4:
            translation_active = translation_active or abs(
                float((command[2] - command[3]).item())
            ) > eps
        if command.numel() >= 6:
            translation_active = translation_active or abs(
                float((command[5] - command[4]).item())
            ) > eps
        families = int(rotation_active) + int(translation_active)
        return float(rotation_active) / float(families) if families else 0.0

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        _, th, tw = self._block_shape
        height_blocks = math.ceil(int(kwargs["token_h"]) / th)
        width_blocks = math.ceil(int(kwargs["token_w"]) / tw)
        spatial = height_blocks * width_blocks
        ids = torch.arange(self._num_blocks, device=self._ray_features.device)
        time_ids = ids // spatial
        spatial_ids = ids % spatial
        h_ids = spatial_ids // width_blocks
        w_ids = spatial_ids % width_blocks
        self._local_mask = (
            (time_ids[:, None] - time_ids[None, :]).abs()
            + (h_ids[:, None] - h_ids[None, :]).abs()
            + (w_ids[:, None] - w_ids[None, :]).abs()
        ) <= 1
        local_counts = self._local_mask.sum(dim=-1)
        if int(local_counts.max().item()) >= self._degree:
            raise RuntimeError("A4-M2 requires at least one non-local top-k slot")
        self._action_geometry_trust = self._command_family_trust(
            kwargs["current_action"]
        )
        return compiled, report

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del q_content, k_content, temporal
        if self._local_mask is None:
            raise RuntimeError("A4-M2 selector has no compiled local stencil")
        local = self._local_mask.to(device=content_score.device)
        local_counts = local.sum(dim=-1, dtype=torch.long)
        remote_slots = self._degree - local_counts
        masked_content = content_score.masked_fill(
            local[None, None], -torch.inf
        )
        ordered = torch.sort(masked_content, dim=-1, descending=True).values
        cutoff_indices = (remote_slots - 1)[None, None, :, None].expand(
            content_score.shape[0], content_score.shape[1], -1, 1
        )
        cutoff = torch.gather(ordered, -1, cutoff_indices)

        distance = (content_score - cutoff).abs().masked_fill(
            local[None, None], torch.inf
        )
        row_scale = torch.median(distance, dim=-1, keepdim=True).values
        eps = torch.finfo(content_score.dtype).eps
        row_scale = torch.where(
            torch.isfinite(row_scale) & (row_scale > eps),
            row_scale,
            torch.full_like(row_scale, eps),
        )
        margin_gate = torch.exp(-distance / row_scale).masked_fill(
            local[None, None], 0.0
        )
        self._last_mean_rank_margin_gate = float(margin_gate.mean().item())
        self._last_max_rank_margin_gate = float(margin_gate.max().item())

        calibrated_geometry = (
            geometry_gate
            * tangent_score
            * margin_gate
            * row_scale
            * self._action_geometry_trust
        )
        return content_score + calibrated_geometry.to(content_score.dtype)

    def select(self, *args, **kwargs):
        selected, counts, report = super().select(*args, **kwargs)
        return selected, counts, replace(
            report,
            mean_rank_margin_gate=self._last_mean_rank_margin_gate,
            max_rank_margin_gate=self._last_max_rank_margin_gate,
            action_geometry_trust=self._action_geometry_trust,
        )


class MatrixWorldlineQKCurvatureWeightSweepAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Sweep the A4 content/geometry mixture with one fixed convex weight.

    The Q/K branch deliberately preserves released LI's model-dtype block
    pooling, cosine computation, local bias, and ``torch.topk``.  Geometry is
    added to that native score before the same single top-k.  Consequently the
    zero endpoint naturally degenerates to LI inside this selector rather than
    bypassing it.  The endpoints isolate native content and gated geometry
    while retaining the same LI local stencil and fixed sparse budget.
    """

    def __init__(self, geometry_weight: float) -> None:
        super().__init__()
        value = float(geometry_weight)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("A4 geometry weight must lie in [0, 1]")
        self.geometry_weight = value
        label = f"{int(round(100 * value)):03d}"
        self.name = f"matrix_worldline_qk_curvature_weight_{label}_attention_compiler"
        self.selection_standard = (
            f"local_qk_remote_and_worldline_curvature_weight_{label}_single_topk"
        )
        self.cache_key_suffix = (
            f"local_qk_remote_and_worldline_curvature_weight_{label}"
        )

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        del q_content, k_content, temporal
        geometry_score = (geometry_gate * tangent_score).to(content_score.dtype)
        if self.geometry_weight == 0.0:
            return content_score
        if self.geometry_weight == 1.0:
            return geometry_score
        return (
            (1.0 - self.geometry_weight) * content_score
            + self.geometry_weight * geometry_score
        ).to(content_score.dtype)

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        del q_content, k_content
        # Match released LI exactly: mean and normalize without an FP32
        # promotion.  In production these tensors are BF16, as are LI's
        # q_pool/k_pool tensors in generate_prefill_indices.
        q_native = F.normalize(q_blocks.mean(dim=-2), dim=-1)
        k_native = F.normalize(k_blocks.mean(dim=-2), dim=-1)
        return torch.matmul(q_native, k_native.transpose(-2, -1))

    def _select_indices(self, score: torch.Tensor) -> torch.Tensor:
        # Match released LI's tie/boundary semantics for every sweep weight.
        _, selected = torch.topk(score, k=self._degree, dim=-1)
        return selected


@dataclass(frozen=True)
class CausalFrontierExchangeReport:
    """Runtime certificate for one native-LI frontier exchange."""

    swapped_rows: int
    total_rows: int
    mean_content_cost: float
    mean_geometry_gain: float
    action_prefix_coherence: float
    max_support_symmetric_difference: int


class MatrixCausalActionTransportFrontierExchangeCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Correct at most one uncertain native-LI frontier edge per query.

    CAFE starts from released Light Interaction's model-dtype cosine top-k.
    Local edges and the first K-1 native non-local choices are immutable.  The
    weakest selected non-local edge may be exchanged with the strongest
    excluded edge only when curvature-gated, pose-transport geometry pays the
    row-normalized Q/K margin.  Thus every row differs from native LI by zero
    or one edge and there is still exactly one final sparse selector.

    The geometric certificate transports a key ray's canonical trajectory-
    scale point into the query camera ray.  It is active only for current
    queries and command-consistent action time.  Once a causal action prefix
    contains a different command direction, its coherence becomes zero for
    the rest of that sample, making mixed-action regions an exact LI fallback.
    """

    name = "matrix_causal_action_transport_frontier_exchange_compiler"
    selection_standard = "native_li_causal_action_transport_frontier_exchange"
    cache_key_suffix = "native_li_cafe"

    def __init__(self) -> None:
        super().__init__()
        self._transport_score: torch.Tensor | None = None
        self._action_prefix: list[torch.Tensor] = []
        self._action_prefix_coherence = 1.0

    def reset_action_history(self) -> None:
        self._action_prefix.clear()
        self._action_prefix_coherence = 1.0

    @staticmethod
    def _canonical_action_direction(current_action: torch.Tensor) -> torch.Tensor:
        command = MatrixCommandPhaseRayWorldlineAttentionCompiler._command_vector(
            current_action
        )
        pitch = command[0] if command.numel() >= 1 else command.new_zeros(())
        yaw = command[1] if command.numel() >= 2 else command.new_zeros(())
        forward = (
            command[2] - command[3]
            if command.numel() >= 4
            else command.new_zeros(())
        )
        lateral = (
            command[5] - command[4]
            if command.numel() >= 6
            else command.new_zeros(())
        )
        direction = torch.stack((pitch, yaw, forward, lateral)).float()
        norm = torch.linalg.vector_norm(direction)
        if float(norm.item()) <= torch.finfo(torch.float32).eps:
            raise RuntimeError("CAFE requires a non-zero Matrix command")
        return (direction / norm).detach().cpu()

    def _extend_action_prefix(self, current_action: torch.Tensor) -> float:
        direction = self._canonical_action_direction(current_action)
        self._action_prefix.append(direction)
        similarities = torch.stack(
            [torch.dot(direction, previous).clamp(0.0, 1.0) for previous in self._action_prefix]
        )
        # The minimum is a causal, parameter-free certificate: every command
        # in the observed prefix must agree before geometry can alter LI.
        self._action_prefix_coherence = float(similarities.min().item())
        return self._action_prefix_coherence

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        tt, th, tw = self._block_shape
        poses = MatrixReciprocalRayGeodesicAttentionCompiler._temporal_block_poses(
            kwargs["active_atoms"], kwargs["current_c2ws"], temporal_tile=tt
        )
        scale = MatrixReciprocalRayGeodesicAttentionCompiler._trajectory_scale(poses)
        rays = MatrixReciprocalRayGeodesicAttentionCompiler._ray_states(
            poses,
            token_h=int(kwargs["token_h"]),
            token_w=int(kwargs["token_w"]),
            block_h=th,
            block_w=tw,
            trajectory_scale=scale,
        ).float()
        origins = rays[:, :3]
        directions = F.normalize(rays[:, 3:], dim=-1)
        # A key ray contributes the world point one trajectory scale from its
        # camera.  Re-observing that point from the query implements a finite-
        # depth SE(3) ray transport without a fitted scene-depth constant.
        key_points = origins + directions
        transported = F.normalize(
            key_points[None, :, :] - origins[:, None, :], dim=-1
        )
        cosine = torch.einsum("id,ijd->ij", directions, transported)
        transport = (0.5 * (cosine + 1.0)).clamp(0.0, 1.0)

        spatial = math.ceil(int(kwargs["token_h"]) / th) * math.ceil(
            int(kwargs["token_w"]) / tw
        )
        memory_blocks = math.ceil(int(layout.memory_length) / tt) * spatial
        ids = torch.arange(self._num_blocks, device=transport.device)
        current_query = ids[:, None] >= memory_blocks
        if self._orientation_violations is None:
            raise RuntimeError("CAFE has no compiled action orientation")
        command_consistent = ~self._orientation_violations.to(transport.device)
        self._transport_score = (
            transport * current_query.to(transport.dtype)
            * command_consistent.to(transport.dtype)
        ).detach()
        self._extend_action_prefix(kwargs["current_action"])
        return compiled, report

    def frontier_exchange(
        self,
        content_score: torch.Tensor,
        curvature_gate: torch.Tensor,
        local: torch.Tensor,
        *,
        degree: int,
    ) -> tuple[torch.Tensor, CausalFrontierExchangeReport]:
        """Return native top-k with no more than one certified edge exchange."""
        if self._transport_score is None:
            raise RuntimeError("CAFE has no compiled action transport")
        if content_score.ndim != 4 or content_score.shape[-1] != content_score.shape[-2]:
            raise ValueError("CAFE requires square B,H,N,N content scores")
        if curvature_gate.shape != content_score.shape:
            raise ValueError("CAFE curvature gate differs from the Q/K score grid")
        if local.shape != content_score.shape[-2:]:
            raise ValueError("CAFE local stencil differs from the Q/K score grid")
        if not 0 < int(degree) < content_score.shape[-1]:
            raise ValueError("CAFE requires a non-trivial sparse degree")

        local4 = local.to(device=content_score.device)[None, None]
        native_rank = content_score + local4.to(content_score.dtype) * 1e6
        _, native = torch.topk(native_rank, k=int(degree), dim=-1)
        selected_local = torch.gather(
            local4.expand(*native.shape[:2], -1, -1), -1, native
        )
        if bool(torch.all(selected_local).item()):
            raise RuntimeError("CAFE frontier has no selected non-local edge")

        selected_content = torch.gather(content_score.float(), -1, native)
        weak_position = selected_content.masked_fill(selected_local, torch.inf).argmin(
            dim=-1, keepdim=True
        )
        weak_index = torch.gather(native, -1, weak_position)
        weak_content = torch.gather(selected_content, -1, weak_position)

        membership = torch.zeros_like(content_score, dtype=torch.bool)
        membership.scatter_(-1, native, True)
        excluded_content = content_score.float().masked_fill(membership | local4, -torch.inf)
        strong_content, strong_index = excluded_content.max(dim=-1, keepdim=True)
        if not bool(torch.isfinite(strong_content).all().item()):
            raise RuntimeError("CAFE frontier has no excluded non-local candidate")

        score32 = content_score.float()
        row_median = score32.median(dim=-1, keepdim=True).values
        row_mad = (score32 - row_median).abs().median(dim=-1, keepdim=True).values
        row_mad = row_mad.clamp_min(torch.finfo(torch.float32).eps)
        content_cost = (weak_content - strong_content).clamp_min(0.0) / row_mad

        geometry = curvature_gate.float() * self._transport_score.to(
            device=content_score.device
        )[None, None]
        geometry = geometry * float(self._action_prefix_coherence)
        weak_geometry = torch.gather(geometry, -1, weak_index)
        strong_geometry = torch.gather(geometry, -1, strong_index)
        geometry_gain = strong_geometry - weak_geometry
        exchange = geometry_gain > content_cost

        selected = native.clone()
        replacement = torch.where(exchange, strong_index, weak_index)
        selected.scatter_(-1, weak_position, replacement)
        selected = torch.sort(selected, dim=-1).values
        duplicates = torch.count_nonzero(selected[..., 1:] == selected[..., :-1])
        if int(duplicates.item()):
            raise RuntimeError("CAFE emitted duplicate sparse edges")
        chosen_local = torch.gather(
            local4.expand(*selected.shape[:2], -1, -1), -1, selected
        )
        expected_local = int(local.sum().item()) * selected.shape[0] * selected.shape[1]
        if int(chosen_local.sum().item()) != expected_local:
            raise RuntimeError("CAFE lost a native LI local edge")

        swapped = int(exchange.sum().item())
        report = CausalFrontierExchangeReport(
            swapped_rows=swapped,
            total_rows=int(exchange.numel()),
            mean_content_cost=float(content_cost.mean().item()),
            mean_geometry_gain=float(geometry_gain.mean().item()),
            action_prefix_coherence=float(self._action_prefix_coherence),
            max_support_symmetric_difference=2 if swapped else 0,
        )
        return selected, report


class MatrixTranslationWorldtubeQKCurvatureSupportAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Use a one-ray-cell worldtube only for keyboard translation.

    Translation produces spatial parallax, so its temporal support includes
    the same ray and the four Manhattan-one neighboring ray cells.  The radius
    is inherited from LI's protected local stencil rather than introduced as a
    fitted hyperparameter.  Mouse rotation keeps A4's exact same-ray support.
    """

    name = "matrix_translation_worldtube_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_translation_worldtube_curvature_single_topk"
    cache_key_suffix = "local_qk_translation_worldtube_curvature"

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if not str(report.command_axis_source).startswith("keyboard_"):
            return compiled, report
        if self._remote_transport_mask is None:
            raise RuntimeError("translation worldtube selector has no support mask")
        _, th, tw = self._block_shape
        nh = math.ceil(int(kwargs["token_h"]) / th)
        nw = math.ceil(int(kwargs["token_w"]) / tw)
        spatial = nh * nw
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        ids = torch.arange(self._num_blocks, device=self._remote_transport_mask.device)
        current = ids >= memory_blocks
        spatial_ids = ids % spatial
        h = spatial_ids // nw
        w = spatial_ids % nw
        ray_distance = (h[:, None] - h[None, :]).abs() + (
            w[:, None] - w[None, :]
        ).abs()
        current_worldtube = current[:, None] & current[None, :] & (ray_distance <= 1)
        self._remote_transport_mask = self._remote_transport_mask | current_worldtube
        return compiled, report


class MatrixActionFlowWorldlineQKCurvatureSupportAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Transport current worldlines along one command-induced optical-flow edge.

    A4's same-ray current support is retained.  For keyboard translation, each
    directed query/key temporal pair receives at most one additional spatial
    neighbor: lateral commands induce horizontal image flow, while
    forward/reverse commands induce radial flow about the image centre.  The
    displacement reverses with temporal direction.  This is a directed
    action-conditioned transport cone rather than A5's isotropic four-neighbor
    tube, and it introduces neither a second selector nor a fitted radius.
    """

    name = "matrix_action_flow_worldline_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_action_flow_worldline_curvature_single_topk"
    cache_key_suffix = "local_qk_action_flow_worldline_curvature"
    flow_time_radius: int | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if not str(report.command_axis_source).startswith("keyboard_"):
            return compiled, report
        if self._remote_transport_mask is None:
            raise RuntimeError("action-flow selector has no support mask")

        _, th, tw = self._block_shape
        nh = math.ceil(int(kwargs["token_h"]) / th)
        nw = math.ceil(int(kwargs["token_w"]) / tw)
        spatial = nh * nw
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        ids = torch.arange(self._num_blocks, device=self._remote_transport_mask.device)
        current = ids >= memory_blocks
        temporal_ids = ids // spatial
        spatial_ids = ids % spatial
        query_h = spatial_ids[:, None] // nw
        query_w = spatial_ids[:, None] % nw
        key_h = spatial_ids[None, :] // nw
        key_w = spatial_ids[None, :] % nw
        dt_sign = torch.sign(
            temporal_ids[None, :] - temporal_ids[:, None]
        ).to(torch.int64)

        command = kwargs["current_action"].detach().float()
        if command.ndim > 1:
            command = command.reshape(-1, command.shape[-1]).mean(dim=0)
        command = command.reshape(-1)
        forward = float((command[2] - command[3]).item()) if command.numel() >= 4 else 0.0
        lateral = float((command[5] - command[4]).item()) if command.numel() >= 6 else 0.0
        delta_h = torch.zeros_like(query_h)
        delta_w = torch.zeros_like(query_w)
        if abs(forward) > torch.finfo(torch.float32).eps:
            # Camera-forward image flow is radial-outward; reverse is inward.
            radial_h = 2 * query_h - (nh - 1)
            radial_w = 2 * query_w - (nw - 1)
            use_h = radial_h.abs() > radial_w.abs()
            flow_h = torch.where(use_h, torch.sign(radial_h), torch.zeros_like(radial_h))
            flow_w = torch.where(use_h, torch.zeros_like(radial_w), torch.sign(radial_w))
            direction = 1 if forward > 0.0 else -1
            delta_h = dt_sign * direction * flow_h
            delta_w = dt_sign * direction * flow_w
        elif abs(lateral) > torch.finfo(torch.float32).eps:
            # Camera-right translation moves the image to the left.
            direction = -1 if lateral > 0.0 else 1
            delta_w = dt_sign * direction

        target_h = query_h + delta_h
        target_w = query_w + delta_w
        directed_neighbor = (key_h == target_h) & (key_w == target_w)
        cross_time = temporal_ids[:, None] != temporal_ids[None, :]
        action_flow = (
            current[:, None]
            & current[None, :]
            & cross_time
            & directed_neighbor
        )
        if self.flow_time_radius is not None:
            action_flow = action_flow & (
                (temporal_ids[:, None] - temporal_ids[None, :]).abs()
                <= self.flow_time_radius
            )
        self._remote_transport_mask = self._remote_transport_mask | action_flow
        return compiled, report


class MatrixLocalActionFlowWorldlineQKCurvatureSupportAttentionCompiler(
    MatrixActionFlowWorldlineQKCurvatureSupportAttentionCompiler
):
    """Integrate command-induced ray transport one temporal block at a time.

    The DiT depth repeatedly composes these local Lie--Euler transport edges,
    so distant motion is reachable without a direct cross-time shortcut.  The
    unit step is the native attention temporal block and adds no new radius.
    """

    name = "matrix_local_action_flow_worldline_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_lie_euler_action_flow_curvature_single_topk"
    cache_key_suffix = "local_qk_lie_euler_action_flow_curvature"
    flow_time_radius = 1


class MatrixActionSecantRemoteQKCurvatureSupportAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Weight only remote-memory support by its measured action secant.

    The support topology is exactly A4: current-to-remote edges plus same-ray
    current worldlines.  For an existing current-to-remote edge, geometry is
    continuously scaled by the positive pose displacement per temporal step
    along the current command axis.  Row-max normalization is unit-free and
    parameter-free.  Current worldline support remains byte-identical to A4.
    """

    name = "matrix_action_secant_remote_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_action_secant_remote_curvature_single_topk"
    cache_key_suffix = "local_qk_action_secant_remote_curvature"

    def __init__(self) -> None:
        super().__init__()
        self._action_secant_weight: torch.Tensor | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        _, th, tw = self._block_shape
        spatial = math.ceil(int(kwargs["token_h"]) / th) * math.ceil(
            int(kwargs["token_w"]) / tw
        )
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        ids = torch.arange(self._num_blocks, device=self._ray_features.device)
        temporal_ids = ids // spatial
        axis, _, _ = MatrixCommandPhaseRayWorldlineAttentionCompiler._command_axis(
            kwargs["current_action"], kwargs["current_c2ws"]
        )
        axis = F.normalize(axis.to(self._ray_features), dim=0)
        longitudinal = self._ray_features @ axis
        dt = temporal_ids[None, :] - temporal_ids[:, None]
        secant = torch.where(
            dt != 0,
            (longitudinal[None, :] - longitudinal[:, None])
            * torch.sign(dt).to(longitudinal.dtype),
            torch.zeros_like(dt, dtype=longitudinal.dtype),
        ).clamp_min(0.0)
        remote = (ids[:, None] >= memory_blocks) & (ids[None, :] < memory_blocks)
        remote_max = torch.where(remote, secant, torch.zeros_like(secant)).amax(
            dim=-1, keepdim=True
        )
        normalized = secant / remote_max.clamp_min(torch.finfo(secant.dtype).eps)
        weight = torch.ones_like(normalized)
        self._action_secant_weight = torch.where(remote, normalized, weight).detach()
        return compiled, report

    def _tangent_similarity(
        self,
        ray_score: torch.Tensor,
        violations: torch.Tensor,
    ) -> torch.Tensor:
        support = super()._tangent_similarity(ray_score, violations)
        if self._action_secant_weight is None:
            raise RuntimeError("action-secant selector has no compiled weights")
        return support * self._action_secant_weight.to(
            device=ray_score.device, dtype=ray_score.dtype
        )[None, None]


class MatrixResidualActionSecantRemoteQKCurvatureSupportAttentionCompiler(
    MatrixActionSecantRemoteQKCurvatureSupportAttentionCompiler
):
    """Add action-secant evidence as a residual over A4 remote support.

    A8 replaced A4's remote support and showed that even weakly aligned memory
    geometry is needed for direction.  A9 therefore keeps the A4 identity path
    and adds the unit-free normalized action secant only on current-to-remote
    edges.  No edge is removed or downweighted.
    """

    name = "matrix_residual_action_secant_remote_qk_curvature_support_attention_compiler"
    selection_standard = "local_qk_residual_action_secant_remote_curvature_single_topk"
    cache_key_suffix = "local_qk_residual_action_secant_remote_curvature"

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        if self._action_secant_weight is None:
            raise RuntimeError("residual action-secant selector has no weights")
        _, th, tw = self._block_shape
        spatial = math.ceil(int(kwargs["token_h"]) / th) * math.ceil(
            int(kwargs["token_w"]) / tw
        )
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        ids = torch.arange(self._num_blocks, device=self._action_secant_weight.device)
        remote = (ids[:, None] >= memory_blocks) & (ids[None, :] < memory_blocks)
        self._action_secant_weight = torch.where(
            remote,
            1.0 + self._action_secant_weight,
            torch.ones_like(self._action_secant_weight),
        )
        return compiled, report


class MatrixWorldlineVelocityMassPreservingQKAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Redistribute A4 support by live DiT latent-worldline velocity.

    At each denoising layer, Q and K block means are differenced against the
    previous temporal block at the same camera-ray cell.  Velocity agreement
    becomes a positive transport likelihood.  The likelihood is normalized to
    unit mean over each row's existing A4 support, preserving total geometry
    mass while changing which worldline evidence wins the single top-k.
    """

    name = "matrix_worldline_velocity_mass_preserving_qk_attention_compiler"
    selection_standard = "local_qk_worldline_velocity_mass_preserving_single_topk"
    cache_key_suffix = "local_qk_worldline_velocity_mass_preserving"

    @staticmethod
    def _worldline_velocity(content: torch.Tensor, temporal: int) -> torch.Tensor:
        spatial = content.shape[-2] // temporal
        grid = content.reshape(*content.shape[:-2], temporal, spatial, content.shape[-1])
        velocity = torch.zeros_like(grid)
        velocity[..., 1:, :, :] = grid[..., 1:, :, :] - grid[..., :-1, :, :]
        return F.normalize(velocity, dim=-1).reshape_as(content)

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        q_velocity = self._worldline_velocity(q_content, temporal)
        k_velocity = self._worldline_velocity(k_content, temporal)
        agreement = 0.5 * (
            1.0 + torch.matmul(q_velocity, k_velocity.transpose(-2, -1))
        )
        spatial = q_content.shape[-2] // temporal
        temporal_ids = torch.arange(
            q_content.shape[-2], device=q_content.device
        ) // spatial
        valid = (temporal_ids[:, None] > 0) & (temporal_ids[None, :] > 0)
        active = (geometry_gate > 0.0) & (tangent_score > 0.0) & valid[None, None]
        mass = torch.where(active, agreement, torch.zeros_like(agreement))
        mean = mass.sum(dim=-1, keepdim=True) / active.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        multiplier = torch.where(
            active,
            agreement / mean.clamp_min(torch.finfo(agreement.dtype).eps),
            torch.ones_like(agreement),
        )
        return content_score + geometry_gate * tangent_score * multiplier


class MatrixLatentSpeedConcordanceQKAttentionCompiler(
    MatrixWorldlineVelocityMassPreservingQKAttentionCompiler
):
    """Match projection-invariant latent motion speed on the A4 graph.

    Q and K velocity directions live in different learned projection spaces.
    Their norms, normalized inside each projection, remain comparable motion
    saliency signals.  A symmetric min/max concordance redistributes A4's
    fixed geometry mass without adding edges or a learned scale.
    """

    name = "matrix_latent_speed_concordance_qk_attention_compiler"
    selection_standard = "local_qk_latent_speed_concordance_single_topk"
    cache_key_suffix = "local_qk_latent_speed_concordance"

    def _compose_score(
        self,
        content_score: torch.Tensor,
        geometry_gate: torch.Tensor,
        tangent_score: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
        *,
        temporal: int,
    ) -> torch.Tensor:
        q_speed = torch.linalg.vector_norm(
            self._worldline_velocity(q_content, temporal), dim=-1
        )
        k_speed = torch.linalg.vector_norm(
            self._worldline_velocity(k_content, temporal), dim=-1
        )
        eps = torch.finfo(q_speed.dtype).eps
        q_speed = q_speed / q_speed.mean(dim=-1, keepdim=True).clamp_min(eps)
        k_speed = k_speed / k_speed.mean(dim=-1, keepdim=True).clamp_min(eps)
        low = torch.minimum(q_speed[..., :, None], k_speed[..., None, :])
        high = torch.maximum(q_speed[..., :, None], k_speed[..., None, :])
        agreement = low / high.clamp_min(eps)
        spatial = q_content.shape[-2] // temporal
        temporal_ids = torch.arange(q_content.shape[-2], device=q_content.device) // spatial
        valid = (temporal_ids[:, None] > 0) & (temporal_ids[None, :] > 0)
        active = (geometry_gate > 0.0) & (tangent_score > 0.0) & valid[None, None]
        mass = torch.where(active, agreement, torch.zeros_like(agreement))
        mean = mass.sum(dim=-1, keepdim=True) / active.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        multiplier = torch.where(
            active,
            agreement / mean.clamp_min(eps),
            torch.ones_like(agreement),
        )
        return content_score + geometry_gate * tangent_score * multiplier


class MatrixSubtokenTransportQKAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Retain within-block world-model microstate correspondence.

    A4 ranks content using one mean vector per spatiotemporal block.  A12 adds
    an aligned subtoken transport kernel: corresponding latent patch slots in
    two blocks vote before averaging.  The block semantic and microstate
    kernels form an equal canonical product-space mean, followed by A4's one
    geometry/QK top-k.  No second selector or extra sparse budget is used.
    """

    name = "matrix_subtoken_transport_qk_attention_compiler"
    selection_standard = "local_qk_subtoken_transport_curvature_single_topk"
    cache_key_suffix = "local_qk_subtoken_transport_curvature"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_micro = F.normalize(q_blocks.float(), dim=-1)
        k_micro = F.normalize(k_blocks.float(), dim=-1)
        aligned = torch.einsum("...ntd,...mtd->...nm", q_micro, k_micro)
        aligned = aligned / q_micro.shape[-2]
        return 0.5 * (semantic + aligned)


class MatrixTimeReversalQuotientSubtokenQKAttentionCompiler(
    MatrixSubtokenTransportQKAttentionCompiler
):
    """Quotient subtoken transport by within-block temporal reversal.

    Return trajectories carry the same local world microstate in reverse time
    order.  Each block-pair kernel therefore takes the stronger of native and
    temporal-reversed slot correspondence before the single sparse top-k.
    Spatial patch coordinates are never reversed or paired by force.
    """

    name = "matrix_time_reversal_quotient_subtoken_qk_attention_compiler"
    selection_standard = "local_qk_time_reversal_quotient_subtoken_single_topk"
    cache_key_suffix = "local_qk_time_reversal_quotient_subtoken"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_micro = F.normalize(q_blocks.float(), dim=-1)
        k_micro = F.normalize(k_blocks.float(), dim=-1)
        tt, th, tw = self._block_shape
        if q_micro.shape[-2] != tt * th * tw:
            raise RuntimeError("subtoken count does not match compiled block shape")
        reversed_k = k_micro.reshape(
            *k_micro.shape[:-2], tt, th, tw, k_micro.shape[-1]
        ).flip(-4).reshape_as(k_micro)
        aligned = torch.einsum("...ntd,...mtd->...nm", q_micro, k_micro)
        reversed_time = torch.einsum(
            "...ntd,...mtd->...nm", q_micro, reversed_k
        )
        microstate = torch.maximum(aligned, reversed_time) / q_micro.shape[-2]
        return 0.5 * (semantic + microstate)


class MatrixActionParitySubtokenQKAttentionCompiler(
    MatrixTimeReversalQuotientSubtokenQKAttentionCompiler
):
    """Choose subtoken temporal parity from the compiled action orientation.

    Command-consistent edges keep native latent time order; edges crossing the
    action chart use temporal reversal.  Unlike A13's unconditional maximum,
    this cannot explain a forward edge with a spuriously reversed microstate.
    """

    name = "matrix_action_parity_subtoken_qk_attention_compiler"
    selection_standard = "local_qk_action_parity_subtoken_single_topk"
    cache_key_suffix = "local_qk_action_parity_subtoken"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        if self._orientation_violations is None:
            raise RuntimeError("action-parity selector has no orientation chart")
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_micro = F.normalize(q_blocks.float(), dim=-1)
        k_micro = F.normalize(k_blocks.float(), dim=-1)
        tt, th, tw = self._block_shape
        reversed_k = k_micro.reshape(
            *k_micro.shape[:-2], tt, th, tw, k_micro.shape[-1]
        ).flip(-4).reshape_as(k_micro)
        aligned = torch.einsum("...ntd,...mtd->...nm", q_micro, k_micro)
        reversed_time = torch.einsum(
            "...ntd,...mtd->...nm", q_micro, reversed_k
        )
        reverse = self._orientation_violations.to(q_micro.device)[None, None]
        microstate = torch.where(reverse, reversed_time, aligned) / q_micro.shape[-2]
        return 0.5 * (semantic + microstate)


class MatrixCausalRoleParitySubtokenQKAttentionCompiler(
    MatrixTimeReversalQuotientSubtokenQKAttentionCompiler
):
    """Apply action parity only to causal current-to-memory retrieval.

    Remote retrieval uses the command chart to select native or reversed
    microtime.  Current-current worldlines retain A13's reversal quotient,
    which empirically preserves stability and locality.  The distinction is a
    causal world-model role, not an additional selector.
    """

    name = "matrix_causal_role_parity_subtoken_qk_attention_compiler"
    selection_standard = "local_qk_causal_role_parity_subtoken_single_topk"
    cache_key_suffix = "local_qk_causal_role_parity_subtoken"

    def __init__(self) -> None:
        super().__init__()
        self._causal_remote_mask: torch.Tensor | None = None

    def compile(self, layout, **kwargs):
        compiled, report = super().compile(layout, **kwargs)
        _, th, tw = self._block_shape
        spatial = math.ceil(int(kwargs["token_h"]) / th) * math.ceil(
            int(kwargs["token_w"]) / tw
        )
        memory_blocks = math.ceil(
            int(layout.memory_length) / self._block_shape[0]
        ) * spatial
        ids = torch.arange(self._num_blocks, device=self._ray_features.device)
        self._causal_remote_mask = (ids[:, None] >= memory_blocks) & (
            ids[None, :] < memory_blocks
        )
        return compiled, report

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        if self._orientation_violations is None or self._causal_remote_mask is None:
            raise RuntimeError("causal-role selector has no compiled chart")
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_micro = F.normalize(q_blocks.float(), dim=-1)
        k_micro = F.normalize(k_blocks.float(), dim=-1)
        tt, th, tw = self._block_shape
        reversed_k = k_micro.reshape(
            *k_micro.shape[:-2], tt, th, tw, k_micro.shape[-1]
        ).flip(-4).reshape_as(k_micro)
        aligned = torch.einsum("...ntd,...mtd->...nm", q_micro, k_micro)
        reversed_time = torch.einsum(
            "...ntd,...mtd->...nm", q_micro, reversed_k
        )
        quotient = torch.maximum(aligned, reversed_time)
        parity = torch.where(
            self._orientation_violations.to(q_micro.device)[None, None],
            reversed_time,
            aligned,
        )
        microstate = torch.where(
            self._causal_remote_mask.to(q_micro.device)[None, None],
            parity,
            quotient,
        ) / q_micro.shape[-2]
        return 0.5 * (semantic + microstate)


class MatrixSoftTimeReversalOrbitSubtokenQKAttentionCompiler(
    MatrixTimeReversalQuotientSubtokenQKAttentionCompiler
):
    """Use the smooth canonical kernel on the temporal-reversal orbit.

    Native and reversed block microtime are the two elements of Z2.  Their
    log-mean-exp is a parameter-free smooth orbit quotient: it preserves
    reversal equivalence without A13's brittle winner-take-all branch.
    """

    name = "matrix_soft_time_reversal_orbit_subtoken_qk_attention_compiler"
    selection_standard = "local_qk_soft_time_reversal_orbit_subtoken_single_topk"
    cache_key_suffix = "local_qk_soft_time_reversal_orbit_subtoken"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_micro = F.normalize(q_blocks.float(), dim=-1)
        k_micro = F.normalize(k_blocks.float(), dim=-1)
        tt, th, tw = self._block_shape
        reversed_k = k_micro.reshape(
            *k_micro.shape[:-2], tt, th, tw, k_micro.shape[-1]
        ).flip(-4).reshape_as(k_micro)
        scale = q_micro.shape[-2]
        aligned = torch.einsum(
            "...ntd,...mtd->...nm", q_micro, k_micro
        ) / scale
        reversed_time = torch.einsum(
            "...ntd,...mtd->...nm", q_micro, reversed_k
        ) / scale
        orbit = torch.logaddexp(aligned, reversed_time) - math.log(2.0)
        return 0.5 * (semantic + orbit)


class MatrixTemporalPhaseOrbitSubtokenQKAttentionCompiler(
    MatrixTimeReversalQuotientSubtokenQKAttentionCompiler
):
    """Quotient microstate matching by the latent block's temporal phase.

    A world-model block can encode the same short motion with a one-slot phase
    offset because denoising time and video time are not synchronized.  The
    cyclic temporal group is therefore marginalized with a parameter-free
    log-mean-exp before the one geometry/QK top-k.  Spatial slots remain fixed,
    so the selector gains timing tolerance without discarding scene layout.
    """

    name = "matrix_temporal_phase_orbit_subtoken_qk_attention_compiler"
    selection_standard = "local_qk_temporal_phase_orbit_subtoken_single_topk"
    cache_key_suffix = "local_qk_temporal_phase_orbit_subtoken"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_micro = F.normalize(q_blocks.float(), dim=-1)
        k_micro = F.normalize(k_blocks.float(), dim=-1)
        tt, th, tw = self._block_shape
        k_grid = k_micro.reshape(
            *k_micro.shape[:-2], tt, th, tw, k_micro.shape[-1]
        )
        phase_scores = []
        scale = q_micro.shape[-2]
        for shift in range(tt):
            shifted = torch.roll(k_grid, shifts=shift, dims=-4).reshape_as(k_micro)
            phase_scores.append(
                torch.einsum("...ntd,...mtd->...nm", q_micro, shifted) / scale
            )
        phases = torch.stack(phase_scores, dim=-1)
        orbit = torch.logsumexp(phases, dim=-1) - math.log(float(tt))
        return 0.5 * (semantic + orbit)


class MatrixCoarseDetailFactorizedSubtokenQKAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Factor sparse retrieval into scene identity and latent detail evidence.

    The block mean carries coarse world state.  Mean-free subtokens carry the
    local appearance/motion residual that the decoder can expose.  Residual
    correspondence is admitted in proportion to its parameter-free energy
    share, then geometry and Q/K evidence still meet in the same single top-k.
    This keeps A4's coarse semantics while letting genuine detail disambiguate
    remote memories instead of giving every subtoken match equal authority.
    """

    name = "matrix_coarse_detail_factorized_subtoken_qk_attention_compiler"
    selection_standard = "local_qk_coarse_detail_factorized_subtoken_single_topk"
    cache_key_suffix = "local_qk_coarse_detail_factorized_subtoken"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        q_float = q_blocks.float()
        k_float = k_blocks.float()
        q_residual = q_float - q_float.mean(dim=-2, keepdim=True)
        k_residual = k_float - k_float.mean(dim=-2, keepdim=True)
        q_detail = F.normalize(q_residual, dim=-1)
        k_detail = F.normalize(k_residual, dim=-1)
        detail = torch.einsum(
            "...ntd,...mtd->...nm", q_detail, k_detail
        ) / q_detail.shape[-2]
        eps = torch.finfo(q_float.dtype).eps
        q_ratio = torch.linalg.vector_norm(q_residual, dim=-1).mean(dim=-1)
        q_ratio = q_ratio / torch.linalg.vector_norm(q_float, dim=-1).mean(
            dim=-1
        ).clamp_min(eps)
        k_ratio = torch.linalg.vector_norm(k_residual, dim=-1).mean(dim=-1)
        k_ratio = k_ratio / torch.linalg.vector_norm(k_float, dim=-1).mean(
            dim=-1
        ).clamp_min(eps)
        detail_gate = (q_ratio[..., :, None] * k_ratio[..., None, :]).clamp(0.0, 1.0)
        return semantic + detail_gate * detail


class MatrixObjectPersistencePMIQKAttentionCompiler(
    MatrixWorldlineQKCurvatureSupportAttentionCompiler
):
    """Rank remote evidence by object-specific rather than attention-sink fit.

    Persistent backgrounds can be similar to every query and become Q/K hubs,
    crowding moving objects out of a fixed sparse budget.  Subtracting each
    key's log-mean-exp evidence marginal yields a pointwise-mutual-information
    score: an edge is strong only when it is unusually explanatory for this
    query.  The correction is parameter free and is folded into A4's sole
    geometry/QK selector, with the released local stencil still protected.
    """

    name = "matrix_object_persistence_pmi_qk_attention_compiler"
    selection_standard = "local_qk_object_persistence_pmi_single_topk"
    cache_key_suffix = "local_qk_object_persistence_pmi"

    def _content_score(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        q_content: torch.Tensor,
        k_content: torch.Tensor,
    ) -> torch.Tensor:
        del q_blocks, k_blocks
        semantic = torch.matmul(q_content, k_content.transpose(-2, -1))
        key_marginal = torch.logsumexp(semantic, dim=-2, keepdim=True)
        key_marginal = key_marginal - math.log(float(semantic.shape[-2]))
        return semantic - key_marginal


__all__ = [
    "CausalFrontierExchangeReport",
    "MatrixCausalActionTransportFrontierExchangeCompiler",
    "MatrixQKCurvatureSupportAttentionCompiler",
    "MatrixRemoteQKCurvatureSupportAttentionCompiler",
    "MatrixWorldlineQKCurvatureSupportAttentionCompiler",
    "MatrixCausalWorldlineProductKernelAttentionCompiler",
    "MatrixActionCausalFeasibleQKAttentionCompiler",
    "MatrixWorldRayTransportQKAttentionCompiler",
    "MatrixTrajectoryAdaptiveBudgetAttentionCompiler",
    "MatrixLogTrajectoryAdaptiveBudgetAttentionCompiler",
    "MatrixResidualTrajectoryAdaptiveBudgetAttentionCompiler",
    "MatrixCurvatureTrajectoryAdaptiveBudgetAttentionCompiler",
    "MatrixTranslationWorldtubeQKCurvatureSupportAttentionCompiler",
    "MatrixActionFlowWorldlineQKCurvatureSupportAttentionCompiler",
    "MatrixLocalActionFlowWorldlineQKCurvatureSupportAttentionCompiler",
    "MatrixActionSecantRemoteQKCurvatureSupportAttentionCompiler",
    "MatrixResidualActionSecantRemoteQKCurvatureSupportAttentionCompiler",
    "MatrixWorldlineVelocityMassPreservingQKAttentionCompiler",
    "MatrixLatentSpeedConcordanceQKAttentionCompiler",
    "MatrixSubtokenTransportQKAttentionCompiler",
    "MatrixTimeReversalQuotientSubtokenQKAttentionCompiler",
    "MatrixActionParitySubtokenQKAttentionCompiler",
    "MatrixCausalRoleParitySubtokenQKAttentionCompiler",
    "MatrixSoftTimeReversalOrbitSubtokenQKAttentionCompiler",
    "MatrixTemporalPhaseOrbitSubtokenQKAttentionCompiler",
    "MatrixCoarseDetailFactorizedSubtokenQKAttentionCompiler",
    "MatrixObjectPersistencePMIQKAttentionCompiler",
    "MatrixQKUncertaintyTangentAttentionCompiler",
    "QKUncertaintyTangentSelectionReport",
]
