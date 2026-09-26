"""One-shot geometry/content selection on reciprocal C2 attention orbits."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch
import torch.nn.functional as F

from .reciprocal_degree_attention import MatrixReciprocalDegreeAttentionCompiler


@dataclass(frozen=True)
class ReciprocalJointAttentionCompileReport:
    query_rows: int
    query_orbits: int
    edges_per_row: int
    native_row_duplicate_violations: int
    activation_residual_fill: bool
    geometry_guard_ratio: float
    qk_read_for_selection: bool
    selector_count: int
    involution: str


@dataclass(frozen=True)
class ReciprocalJointAttentionSelectionReport:
    batch_heads: int
    query_rows: int
    query_orbits: int
    edges_per_row: int
    frechet_optimality_violations: int
    equivariance_violations: int
    row_duplicate_violations: int
    intersection_edges: int
    content_resolved_boundary_edges: int
    qk_read_for_selection: bool
    selector_count: int


def _block_involution(
    num_blocks: int,
    *,
    temporal_blocks: int,
    spatial_blocks: int,
) -> list[int]:
    if num_blocks != temporal_blocks * spatial_blocks:
        raise ValueError("block grid does not match the attention row count")
    mate = [
        MatrixReciprocalDegreeAttentionCompiler._involution(
            block,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        for block in range(num_blocks)
    ]
    if any(mate[mate[block]] != block or mate[block] == block for block in range(num_blocks)):
        raise ValueError("query/target mapping is not a free C2 involution")
    return mate


def _tile_visual_tensor(
    tensor: torch.Tensor,
    *,
    temporal: int,
    height: int,
    width: int,
    block_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Match Matrix's visual tiling while keeping this method upstream-free."""

    if tensor.ndim != 4:
        raise ValueError("joint selector expects a B,H,L,D visual tensor")
    batch, heads, length, channels = tensor.shape
    if length != temporal * height * width:
        raise ValueError("visual token length does not match the latent grid")
    tt, th, tw = block_shape
    pad_t = (-temporal) % tt
    pad_h = (-height) % th
    pad_w = (-width) % tw
    tensor = tensor.reshape(batch, heads, temporal, height, width, channels)
    if pad_t or pad_h or pad_w:
        padded = tensor.new_zeros(
            batch,
            heads,
            temporal + pad_t,
            height + pad_h,
            width + pad_w,
            channels,
        )
        padded[:, :, :temporal, :height, :width].copy_(tensor)
        tensor = padded
    nt = tensor.shape[2] // tt
    nh = tensor.shape[3] // th
    nw = tensor.shape[4] // tw
    return (
        tensor.reshape(batch, heads, nt, tt, nh, th, nw, tw, channels)
        .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
        .contiguous()
        .reshape(batch, heads, nt * nh * nw, tt * th * tw, channels)
    )


class MatrixReciprocalJointAttentionCompiler:
    """Prepare one C2 geometry/content sparse-block selector.

    For a reciprocal query orbit ``(q, gq)``, the two native geometry rows are
    pulled into one quotient chart.  At runtime a *single* lexicographic
    optimization chooses the complete fixed-degree section:

    1. minimize total Hamming distance to the two geometry rows;
    2. among those exact minimizers, minimize the worst reciprocal Q/K rank;
    3. break remaining ties by total Q/K rank, paired geometry rank, and the
       immutable block id.

    The first objective makes geometry a hard, auditable feasible set.  The
    second reads current activations without a tuned geometry/content weight.
    There is no mandatory prefix followed by a residual fill: all edges are
    emitted by the same quotient argmin and the return row is its C2 lift.
    """

    name = "matrix_reciprocal_joint_attention_compiler"

    def compile(
        self,
        layout: Any,
        *,
        token_h: int,
        token_w: int,
    ) -> tuple[Any, ReciprocalJointAttentionCompileReport]:
        if bool(getattr(layout, "protected_current")):
            raise ValueError("joint reciprocal selection requires aggressive C3 layout")
        tt, th, tw = tuple(int(value) for value in layout.block_shape)
        temporal = int(layout.memory_length) + int(layout.current_length)
        temporal_blocks = math.ceil(temporal / tt)
        height_blocks = math.ceil(int(token_h) / th)
        width_blocks = math.ceil(int(token_w) / tw)
        spatial_blocks = height_blocks * width_blocks
        num_blocks = temporal_blocks * spatial_blocks
        if tuple(layout.indices.shape[:3]) != (1, 1, num_blocks):
            raise RuntimeError("layout dimensions do not match its block geometry")
        counts = layout.counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise RuntimeError("joint reciprocal selection requires uniform degree")
        degree = int(counts[0].item())
        if degree <= 0:
            raise RuntimeError("joint reciprocal selection requires positive degree")
        mate = _block_involution(
            num_blocks,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        duplicate_violations = 0
        for query in range(num_blocks):
            row = layout.indices[0, 0, query, :degree].detach().cpu().tolist()
            duplicate_violations += int(len(set(int(value) for value in row)) != degree)
        if duplicate_violations:
            raise RuntimeError("native geometry contains duplicate sparse blocks")
        report = ReciprocalJointAttentionCompileReport(
            query_rows=num_blocks,
            query_orbits=sum(block < mate[block] for block in range(num_blocks)),
            edges_per_row=degree,
            native_row_duplicate_violations=duplicate_violations,
            activation_residual_fill=False,
            geometry_guard_ratio=1.0,
            qk_read_for_selection=True,
            selector_count=1,
            involution="same_spatial_temporal_reversal_with_mid_slab_adjacent_pairing",
        )
        return (
            replace(
                layout,
                activation_residual_fill=False,
                geometry_guard_ratio=1.0,
                cache_key=tuple(layout.cache_key)
                + ("reciprocal_joint_frechet_qk_c2",),
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
    ) -> tuple[torch.Tensor, torch.Tensor, ReciprocalJointAttentionSelectionReport]:
        """Select every sparse row once from current Q/K and paired geometry."""

        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("joint selector requires equal B,H,L,D Q/K tensors")
        if geometry_indices.shape[0:2] != (1, 1):
            raise ValueError("joint selector expects one shared geometry layout")
        height, width = (int(value) for value in latent_hw)
        tokens_per_frame = height * width
        if tokens_per_frame <= 0 or q.shape[2] % tokens_per_frame:
            raise ValueError("Q/K tokens do not define an integral temporal grid")
        temporal = q.shape[2] // tokens_per_frame
        tt, th, tw = (int(value) for value in block_shape)
        temporal_blocks = math.ceil(temporal / tt)
        spatial_blocks = math.ceil(height / th) * math.ceil(width / tw)
        num_blocks = temporal_blocks * spatial_blocks
        if geometry_indices.shape[2] != num_blocks:
            raise ValueError("geometry query count does not match tiled Q/K")
        counts = geometry_counts[0, 0].detach().to(device="cpu", dtype=torch.int64)
        if counts.numel() != num_blocks or not torch.all(counts == counts[0]):
            raise ValueError("joint selector requires a uniform geometry degree")
        degree = int(counts[0].item())
        if degree <= 0 or degree > num_blocks:
            raise ValueError("invalid joint sparse degree")

        q_blocks = _tile_visual_tensor(
            q,
            temporal=temporal,
            height=height,
            width=width,
            block_shape=(tt, th, tw),
        )
        k_blocks = _tile_visual_tensor(
            k,
            temporal=temporal,
            height=height,
            width=width,
            block_shape=(tt, th, tw),
        )
        q_pool = F.normalize(q_blocks.mean(dim=-2).float(), dim=-1)
        k_pool = F.normalize(k_blocks.mean(dim=-2).float(), dim=-1)
        similarity = torch.matmul(q_pool, k_pool.transpose(-2, -1))

        mate_list = _block_involution(
            num_blocks,
            temporal_blocks=temporal_blocks,
            spatial_blocks=spatial_blocks,
        )
        representatives = [
            block for block in range(num_blocks) if block < mate_list[block]
        ]
        return_queries = [mate_list[block] for block in representatives]
        device = q.device
        mate = torch.tensor(mate_list, device=device, dtype=torch.long)
        rep = torch.tensor(representatives, device=device, dtype=torch.long)
        ret = torch.tensor(return_queries, device=device, dtype=torch.long)

        # Rank both reciprocal directions in the same quotient chart.  Stable
        # sorting makes block id the deterministic last content tie-break.
        rep_similarity = similarity.index_select(2, rep)
        ret_similarity = similarity.index_select(2, ret).index_select(3, mate)
        rep_order = torch.argsort(rep_similarity, dim=-1, descending=True, stable=True)
        ret_order = torch.argsort(ret_similarity, dim=-1, descending=True, stable=True)
        rank_shape = rep_similarity.shape
        ordinal = torch.arange(num_blocks, device=device, dtype=torch.int64)
        ordinal = ordinal.view(1, 1, 1, -1).expand(rank_shape)
        rep_rank = torch.empty(rank_shape, device=device, dtype=torch.int64)
        ret_rank = torch.empty(rank_shape, device=device, dtype=torch.int64)
        rep_rank.scatter_(-1, rep_order, ordinal)
        ret_rank.scatter_(-1, ret_order, ordinal)
        content_worst = torch.maximum(rep_rank, ret_rank)
        content_sum = rep_rank + ret_rank

        native = geometry_indices[0, 0, :, :degree].detach().to(
            device=device, dtype=torch.long
        )
        left = native.index_select(0, rep)
        pulled_return = mate[native.index_select(0, ret)]
        orbit_count = len(representatives)
        membership = torch.zeros(
            orbit_count, num_blocks, device=device, dtype=torch.int64
        )
        membership.scatter_add_(
            1, left, torch.ones_like(left, dtype=torch.int64)
        )
        membership.scatter_add_(
            1, pulled_return, torch.ones_like(pulled_return, dtype=torch.int64)
        )
        missing_rank = degree + num_blocks
        geometry_left_rank = torch.full_like(membership, missing_rank)
        geometry_return_rank = torch.full_like(membership, missing_rank)
        local_rank = torch.arange(degree, device=device, dtype=torch.int64)
        local_rank = local_rank.view(1, -1).expand(orbit_count, -1)
        geometry_left_rank.scatter_(1, left, local_rank)
        geometry_return_rank.scatter_(1, pulled_return, local_rank)
        geometry_min = torch.minimum(geometry_left_rank, geometry_return_rank)
        geometry_max = torch.maximum(geometry_left_rank, geometry_return_rank)

        # Integer mixed-radix encoding is an exact lexicographic objective,
        # not a continuous weighted blend.  Membership is the Hamming term;
        # reciprocal content ranks resolve its degenerate boundary once.
        membership_term = (2 - membership).view(1, 1, orbit_count, num_blocks)
        geometry_min = geometry_min.view(1, 1, orbit_count, num_blocks)
        geometry_max = geometry_max.view(1, 1, orbit_count, num_blocks)
        block_id = torch.arange(num_blocks, device=device, dtype=torch.int64)
        block_id = block_id.view(1, 1, 1, num_blocks)
        key = membership_term
        key = key * num_blocks + content_worst
        key = key * (2 * num_blocks) + content_sum
        key = key * (missing_rank + 1) + geometry_min
        key = key * (missing_rank + 1) + geometry_max
        key = key * num_blocks + block_id
        section = torch.argsort(key, dim=-1, stable=True)[..., :degree]
        section = torch.sort(section, dim=-1).values

        batch, heads = q.shape[:2]
        selected = torch.empty(
            batch,
            heads,
            num_blocks,
            degree,
            device=device,
            dtype=torch.int32,
        )
        selected[:, :, rep, :] = section.to(torch.int32)
        selected[:, :, ret, :] = mate[section].to(torch.int32)
        selected_counts = torch.full(
            (batch, heads, num_blocks),
            degree,
            device=device,
            dtype=torch.int32,
        )

        selected_membership = membership.view(1, 1, orbit_count, num_blocks).expand(
            batch, heads, -1, -1
        ).gather(-1, section)
        frechet_violations = int(torch.count_nonzero(selected_membership == 0).item())
        duplicates = int(
            torch.count_nonzero(section[..., 1:] == section[..., :-1]).item()
        )
        lifted = mate[section]
        return_rows = selected[:, :, ret, :].to(torch.long)
        equivariance_violations = int(torch.count_nonzero(return_rows != lifted).item())
        if frechet_violations or duplicates or equivariance_violations:
            raise RuntimeError("joint reciprocal attention certificate failed")
        intersection_edges = int(torch.count_nonzero(selected_membership == 2).item())
        boundary_edges = int(torch.count_nonzero(selected_membership == 1).item())
        report = ReciprocalJointAttentionSelectionReport(
            batch_heads=batch * heads,
            query_rows=num_blocks,
            query_orbits=orbit_count,
            edges_per_row=degree,
            frechet_optimality_violations=frechet_violations,
            equivariance_violations=equivariance_violations,
            row_duplicate_violations=duplicates,
            intersection_edges=intersection_edges,
            content_resolved_boundary_edges=boundary_edges,
            qk_read_for_selection=True,
            selector_count=1,
        )
        return selected, selected_counts, report


__all__ = [
    "MatrixReciprocalJointAttentionCompiler",
    "ReciprocalJointAttentionCompileReport",
    "ReciprocalJointAttentionSelectionReport",
]
