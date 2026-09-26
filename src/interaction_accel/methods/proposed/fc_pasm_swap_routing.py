"""FC-PASM-aware fixed-budget refinement of compact sparse-attention routes.

The router is deliberately downstream of CWCA's budget allocator.  It never
changes a row degree: pooled-Q/K Top-K is the initialization and a bounded
coordinate swap may only exchange one retained key for one candidate key.
The objective uses the standard single-block omission proxy and the detached
per-frequency transport of the preceding layer's FC-PASM reconstruction.
Missing transport evidence is exactly the g=0 (ordinary linear propagation)
case.  General Scheme A may exchange any non-local KV; the explicitly named
historical-only ablation restricts both sides of the exchange to Memory KV.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FCPASMSwapRoutingConfig:
    candidate_multiplier: int = 2
    num_swap_rounds: int = 1
    swap_eps: float = 1e-6
    lambda_anchor: float = 1.0
    lambda_reconstruction: float = 1.0
    sketch_groups: int = 16
    require_fc_transport: bool = False
    min_transport_affinity: float = 0.0
    min_pair_transport_affinity: float = 0.0
    max_swaps_per_call: int = 0
    active_layers: tuple[int, ...] = ()
    historical_only: bool = False
    same_bank_only: bool = False

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 1:
            raise ValueError("candidate multiplier must be positive")
        if self.num_swap_rounds not in {0, 1, 2}:
            raise ValueError("FC-PASM routing supports zero to two swap rounds")
        if not math.isfinite(self.swap_eps) or self.swap_eps < 0.0:
            raise ValueError("swap epsilon must be finite and non-negative")
        if self.lambda_anchor < 0.0 or self.lambda_reconstruction < 0.0:
            raise ValueError("routing objective weights must be non-negative")
        if self.lambda_anchor + self.lambda_reconstruction <= 0.0:
            raise ValueError("routing objective must contain a positive term")
        if self.sketch_groups < 2 or self.sketch_groups % 2:
            raise ValueError("routing sketch groups must be a positive even number")
        if not 0.0 <= self.min_transport_affinity <= 1.0:
            raise ValueError("minimum FC transport affinity must be in [0, 1]")
        if not 0.0 <= self.min_pair_transport_affinity <= 1.0:
            raise ValueError("minimum pair FC transport affinity must be in [0, 1]")
        if self.max_swaps_per_call < 0:
            raise ValueError("maximum swaps per call must be non-negative")
        if any(layer < 0 for layer in self.active_layers):
            raise ValueError("routing active layers must be non-negative")
        if self.historical_only and self.same_bank_only:
            raise ValueError("historical-only routing already fixes the KV bank")


def estimate_block_omission_error(
    probability: torch.Tensor,
    pooled_value: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    sketch_groups: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the attachment's omission proxy for candidate KV blocks.

    Shapes are ``probability=[B,H,Q,K]``, ``pooled_value=[B,H,K,D]`` and
    ``candidate_indices=[B,H,Q,L]``.  The final D-axis is deterministically
    averaged into a small even-dimensional sketch; no attention forward or
    dense-attention oracle is evaluated.
    """

    if probability.ndim != 4 or pooled_value.ndim != 4:
        raise ValueError("omission proxy expects BHQK probability and BHKD values")
    if candidate_indices.ndim != 4:
        raise ValueError("candidate indices must have shape BHQL")
    if probability.shape[:2] != pooled_value.shape[:2]:
        raise ValueError("probability/value batch-head layouts differ")
    if probability.shape[-1] != pooled_value.shape[-2]:
        raise ValueError("probability key count differs from pooled values")
    if candidate_indices.shape[:3] != probability.shape[:3]:
        raise ValueError("candidate/query layouts differ")
    if pooled_value.shape[-1] % sketch_groups:
        raise ValueError("value width must be divisible by sketch groups")
    output = torch.matmul(probability, pooled_value.float())
    gather_index = candidate_indices[..., None].expand(
        *candidate_indices.shape, pooled_value.shape[-1]
    )
    expanded_value = pooled_value.float()[:, :, None].expand(
        -1, -1, probability.shape[-2], -1, -1
    )
    candidate_value = torch.gather(expanded_value, 3, gather_index)
    candidate_probability = torch.gather(
        probability, -1, candidate_indices
    ).float()
    scale = candidate_probability / (1.0 - candidate_probability + eps)
    proxy = scale[..., None] * (output[..., None, :] - candidate_value)
    group_width = pooled_value.shape[-1] // sketch_groups
    return proxy.reshape(*proxy.shape[:-1], sketch_groups, group_width).mean(-1)


class FCPASMLocalCoordinateSwapRouter:
    """Refine compact pooled-Q/K routes without changing CWCA row budgets."""

    def __init__(self, config: FCPASMSwapRoutingConfig) -> None:
        self.config = config
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _transport_matrix(
        transport: dict[str, Any] | None,
        pair: tuple[int, int],
        *,
        spatial_shape: tuple[int, int],
        device: torch.device,
        exponent: float,
    ) -> tuple[torch.Tensor, bool]:
        """Return the detached FC-PASM spatial operator on the attention grid.

        The endpoint cache retains per-overlap-window, per-frequency ``theta``
        and ``g`` from layer ``l-1``.  We apply that operator to basis maps with
        the same Hann analysis/OLA convention as FC-PASM, then restrict it back
        to the sparse-attention block grid.  Subtracting the zero-phase round
        trip makes exponent zero exactly the identity despite up/downsampling.
        """

        height, width = spatial_shape
        spatial = height * width
        identity = torch.eye(spatial, device=device, dtype=torch.float32)
        if transport is None:
            return identity, False
        entry = transport.get("pairs", {}).get(pair)
        if entry is None:
            return identity, False
        theta = entry.get("theta")
        affinity = entry.get("affinity")
        if (
            not isinstance(theta, torch.Tensor)
            or not isinstance(affinity, torch.Tensor)
            or theta.ndim != 3
            or affinity.shape != theta.shape
        ):
            return identity, False
        cache = entry.setdefault("_attention_operator_cache", {})
        cache_key = (height, width, round(float(exponent), 12), str(device))
        cached = cache.get(cache_key)
        if isinstance(cached, torch.Tensor):
            return cached.to(device), True

        output_h, output_w = tuple(int(v) for v in transport["output_shape"])
        transport_h, transport_w = tuple(
            int(v) for v in transport["transport_shape"]
        )
        window_h, window_w = tuple(int(v) for v in transport["window_shape"])
        stride_h, stride_w = tuple(int(v) for v in transport["stride"])
        pad_h, pad_w = tuple(int(v) for v in transport["padding"])
        tile_h, tile_w = tuple(int(v) for v in transport["tile_grid"])
        patches = tile_h * tile_w
        if theta.shape[0] != patches:
            raise RuntimeError("FC routing frequency tiles do not match the OLA grid")

        basis = identity.reshape(spatial, 1, height, width)
        full = F.interpolate(
            basis, size=(output_h, output_w), mode="bilinear", align_corners=False
        )
        padded = F.pad(full, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        columns = F.unfold(
            padded,
            kernel_size=(window_h, window_w),
            stride=(stride_h, stride_w),
        )
        window_y = torch.hann_window(
            window_h, periodic=False, device=device, dtype=torch.float32
        )
        window_x = torch.hann_window(
            window_w, periodic=False, device=device, dtype=torch.float32
        )
        window = (window_y[:, None] * window_x[None, :]).clamp_min(0.0).sqrt()
        tiles = columns.transpose(1, 2).reshape(
            spatial, patches, 1, window_h, window_w
        ) * window
        if (transport_h, transport_w) != (window_h, window_w):
            tiles = F.avg_pool2d(
                tiles.reshape(spatial * patches, 1, window_h, window_w),
                kernel_size=(2, 2),
                stride=(2, 2),
                ceil_mode=True,
                count_include_pad=False,
            ).reshape(spatial, patches, 1, transport_h, transport_w)
        spectrum = torch.fft.rfft2(tiles, dim=(-2, -1))
        theta = theta.to(device=device, dtype=torch.float32)
        affinity = affinity.to(device=device, dtype=torch.float32)
        if theta.shape[-2:] != spectrum.shape[-2:]:
            raise RuntimeError("FC routing frequency operator has an invalid shape")
        phase = torch.polar(
            torch.ones_like(theta), float(exponent) * affinity * theta
        )
        transformed = torch.fft.irfft2(
            spectrum * phase[None, :, None],
            s=(transport_h, transport_w),
            dim=(-2, -1),
        )
        baseline = torch.fft.irfft2(
            spectrum, s=(transport_h, transport_w), dim=(-2, -1)
        )
        if (transport_h, transport_w) != (window_h, window_w):
            transformed = F.interpolate(
                transformed.reshape(spatial * patches, 1, transport_h, transport_w),
                size=(window_h, window_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(spatial, patches, 1, window_h, window_w)
            baseline = F.interpolate(
                baseline.reshape(spatial * patches, 1, transport_h, transport_w),
                size=(window_h, window_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(spatial, patches, 1, window_h, window_w)

        def overlap_add(value: torch.Tensor) -> torch.Tensor:
            weighted = value * window
            folded = F.fold(
                weighted.reshape(spatial, patches, -1).transpose(1, 2),
                output_size=(output_h + 2 * pad_h, output_w + 2 * pad_w),
                kernel_size=(window_h, window_w),
                stride=(stride_h, stride_w),
            )
            weight_columns = (window * window).reshape(1, -1, 1).expand(
                1, window_h * window_w, patches
            )
            normalization = F.fold(
                weight_columns,
                output_size=(output_h + 2 * pad_h, output_w + 2 * pad_w),
                kernel_size=(window_h, window_w),
                stride=(stride_h, stride_w),
            )
            cropped = folded[:, :, pad_h : pad_h + output_h, pad_w : pad_w + output_w]
            norm = normalization[
                :, :, pad_h : pad_h + output_h, pad_w : pad_w + output_w
            ].clamp_min(1e-8)
            return cropped / norm

        transformed_full = overlap_add(transformed)
        baseline_full = overlap_add(baseline)
        block_h = math.ceil(output_h / height)
        block_w = math.ceil(output_w / width)
        delta = F.avg_pool2d(
            transformed_full - baseline_full,
            kernel_size=(block_h, block_w),
            stride=(block_h, block_w),
            ceil_mode=True,
            count_include_pad=False,
        )
        if delta.shape[-2:] != (height, width):
            raise RuntimeError("FC routing operator did not return the attention grid")
        # Rows above enumerate input basis coordinates; transpose to the usual
        # [output,input] linear-operator convention.
        operator = (identity + delta.reshape(spatial, spatial)).transpose(0, 1)
        operator = operator.detach()
        cache[cache_key] = operator
        return operator, True

    @staticmethod
    def _apply_transport_maps(
        transport: dict[str, Any] | None,
        pair: tuple[int, int],
        maps: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        exponent: float,
    ) -> tuple[torch.Tensor, bool]:
        """Apply the detached FC operator without materializing its matrix.

        ``maps`` is ``[N,G,Hb,Wb]`` on the sparse-attention grid.  The math is
        identical to ``_transport_matrix``: bilinear lift, overlapping Hann
        analysis, frequency-domain phase transport, OLA, and block pooling.
        This path scales with the number of actual error/candidate maps rather
        than with all ``Hb*Wb`` unit bases.
        """

        if maps.ndim != 4:
            raise ValueError("FC matrix-free transport expects NGHW maps")
        height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
        if tuple(int(v) for v in maps.shape[-2:]) != (height, width):
            raise ValueError("FC matrix-free map does not match attention grid")
        if transport is None:
            return maps, False
        entry = transport.get("pairs", {}).get(pair)
        if not isinstance(entry, dict):
            return maps, False
        theta = entry.get("theta")
        affinity = entry.get("affinity")
        if (
            not isinstance(theta, torch.Tensor)
            or not isinstance(affinity, torch.Tensor)
            or theta.ndim != 3
            or affinity.shape != theta.shape
        ):
            return maps, False

        output_h, output_w = tuple(int(v) for v in transport["output_shape"])
        transport_h, transport_w = tuple(
            int(v) for v in transport["transport_shape"]
        )
        window_h, window_w = tuple(int(v) for v in transport["window_shape"])
        stride_h, stride_w = tuple(int(v) for v in transport["stride"])
        pad_h, pad_w = tuple(int(v) for v in transport["padding"])
        tile_h, tile_w = tuple(int(v) for v in transport["tile_grid"])
        patches = tile_h * tile_w
        if int(theta.shape[0]) != patches:
            raise RuntimeError("FC matrix-free tiles do not match OLA grid")

        maps = maps.detach().float()
        count, groups = int(maps.shape[0]), int(maps.shape[1])
        full = F.interpolate(
            maps, size=(output_h, output_w), mode="bilinear", align_corners=False
        )
        padded = F.pad(full, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        window_y = torch.hann_window(
            window_h, periodic=False, device=maps.device, dtype=torch.float32
        )
        window_x = torch.hann_window(
            window_w, periodic=False, device=maps.device, dtype=torch.float32
        )
        window = (window_y[:, None] * window_x[None, :]).clamp_min(0.0).sqrt()
        columns = F.unfold(
            padded, kernel_size=(window_h, window_w), stride=(stride_h, stride_w)
        )
        tiles = columns.transpose(1, 2).reshape(
            count, patches, groups, window_h, window_w
        ) * window
        if (transport_h, transport_w) != (window_h, window_w):
            tiles = F.avg_pool2d(
                tiles.reshape(count * patches, groups, window_h, window_w),
                kernel_size=(2, 2),
                stride=(2, 2),
                ceil_mode=True,
                count_include_pad=False,
            ).reshape(count, patches, groups, transport_h, transport_w)
        spectrum = torch.fft.rfft2(tiles, dim=(-2, -1))
        theta = theta.to(device=maps.device, dtype=torch.float32)
        affinity = affinity.to(device=maps.device, dtype=torch.float32)
        if theta.shape[-2:] != spectrum.shape[-2:]:
            raise RuntimeError("FC matrix-free spectrum shape mismatch")
        phase = torch.polar(
            torch.ones_like(theta), float(exponent) * affinity * theta
        )
        transformed = torch.fft.irfft2(
            spectrum * phase[None, :, None],
            s=(transport_h, transport_w),
            dim=(-2, -1),
        )
        baseline = torch.fft.irfft2(
            spectrum, s=(transport_h, transport_w), dim=(-2, -1)
        )
        if (transport_h, transport_w) != (window_h, window_w):
            transformed = F.interpolate(
                transformed.reshape(count * patches, groups, transport_h, transport_w),
                size=(window_h, window_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(count, patches, groups, window_h, window_w)
            baseline = F.interpolate(
                baseline.reshape(count * patches, groups, transport_h, transport_w),
                size=(window_h, window_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(count, patches, groups, window_h, window_w)

        def overlap_add(value: torch.Tensor) -> torch.Tensor:
            weighted = value * window
            folded = F.fold(
                weighted.reshape(count, patches, -1).transpose(1, 2),
                output_size=(output_h + 2 * pad_h, output_w + 2 * pad_w),
                kernel_size=(window_h, window_w),
                stride=(stride_h, stride_w),
            )
            weight_columns = (window * window).reshape(1, -1).transpose(0, 1)
            weight_columns = weight_columns.expand(-1, patches).unsqueeze(0)
            normalization = F.fold(
                weight_columns,
                output_size=(output_h + 2 * pad_h, output_w + 2 * pad_w),
                kernel_size=(window_h, window_w),
                stride=(stride_h, stride_w),
            )
            cropped = folded[:, :, pad_h : pad_h + output_h, pad_w : pad_w + output_w]
            norm = normalization[
                :, :, pad_h : pad_h + output_h, pad_w : pad_w + output_w
            ].clamp_min(1e-8)
            return cropped / norm

        transformed_full = overlap_add(transformed)
        baseline_full = overlap_add(baseline)
        block_h = math.ceil(output_h / height)
        block_w = math.ceil(output_w / width)
        delta = F.avg_pool2d(
            transformed_full - baseline_full,
            kernel_size=(block_h, block_w),
            stride=(block_h, block_w),
            ceil_mode=True,
            count_include_pad=False,
        )
        if delta.shape[-2:] != (height, width):
            raise RuntimeError("FC matrix-free transport returned wrong grid")
        return maps + delta, True

    @staticmethod
    def _objective_terms(
        error: torch.Tensor,
        *,
        temporal_cells: int,
        spatial: int,
        exact_frames: tuple[int, ...],
        temporal_group_size: int,
        current_boundary: int,
        transport: dict[str, Any] | None,
        lambda_anchor: float,
        lambda_reconstruction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int]:
        """Build exact coordinate-quadratic terms for the coupled objective."""

        batch, heads, rows, sketch = error.shape
        if rows != temporal_cells * spatial:
            raise RuntimeError("routing error rows do not match Current cells")
        view = error.reshape(batch, heads, temporal_cells, spatial, sketch)
        # ``exact_frames`` contains Current-local frame ids, not a compact
        # anchor ordinal.  The pooled attention rows are laid out by the
        # actual temporal block containing each frame.  Using the anchor-list
        # ordinal here silently mis-assigned sparse Exact anchors whenever a
        # frame was skipped (for example 0,2,4,5 became cells 0,0,1,1).
        frame_to_cell = {
            int(frame): min(
                int(frame) // int(temporal_group_size), temporal_cells - 1
            )
            for frame in exact_frames
        }
        anchor_count = error.new_zeros(temporal_cells)
        for frame in exact_frames:
            anchor_count[frame_to_cell[int(frame)]] += 1.0
        coefficient = anchor_count[:, None].expand(-1, spatial).clone() * lambda_anchor
        linear = torch.zeros_like(view)
        reconstruction_terms = 0
        cache_used = False
        for left, right in zip(exact_frames, exact_frames[1:]):
            if (left < current_boundary) != (right < current_boundary):
                continue
            skipped = tuple(range(left + 1, right))
            if not skipped:
                continue
            left_cell = frame_to_cell[left]
            right_cell = frame_to_cell[right]
            entry_found = False
            left_error = view[:, :, left_cell]
            right_error = view[:, :, right_cell]
            for target in skipped:
                alpha = float(target - left) / float(right - left)
                left_matrix, left_found = (
                    FCPASMLocalCoordinateSwapRouter._transport_matrix(
                        transport,
                        (int(left), int(right)),
                        spatial_shape=(
                            int(transport.get("attention_spatial_h", 1))
                            if transport is not None else 1,
                            int(transport.get("attention_spatial_w", spatial))
                            if transport is not None else spatial,
                        ),
                        device=error.device,
                        exponent=alpha,
                    )
                )
                right_matrix, right_found = (
                    FCPASMLocalCoordinateSwapRouter._transport_matrix(
                        transport,
                        (int(left), int(right)),
                        spatial_shape=(
                            int(transport.get("attention_spatial_h", 1))
                            if transport is not None else 1,
                            int(transport.get("attention_spatial_w", spatial))
                            if transport is not None else spatial,
                        ),
                        device=error.device,
                        exponent=-(1.0 - alpha),
                    )
                )
                entry_found = entry_found or left_found or right_found
                left_operator = float(1.0 - alpha) * left_matrix
                right_operator = float(alpha) * right_matrix
                if left_cell == right_cell:
                    combined = left_operator + right_operator
                    hessian = lambda_reconstruction * (
                        combined.transpose(0, 1) @ combined
                    )
                    diagonal = torch.diagonal(hessian)
                    coefficient[left_cell] += diagonal
                    context = torch.einsum(
                        "st,bhtg->bhsg", hessian, left_error
                    )
                    linear[:, :, left_cell] += (
                        context - diagonal[None, None, :, None] * left_error
                    )
                else:
                    h_left = lambda_reconstruction * (
                        left_operator.transpose(0, 1) @ left_operator
                    )
                    h_right = lambda_reconstruction * (
                        right_operator.transpose(0, 1) @ right_operator
                    )
                    h_cross = lambda_reconstruction * (
                        left_operator.transpose(0, 1) @ right_operator
                    )
                    diagonal_left = torch.diagonal(h_left)
                    diagonal_right = torch.diagonal(h_right)
                    coefficient[left_cell] += diagonal_left
                    coefficient[right_cell] += diagonal_right
                    left_context = torch.einsum(
                        "st,bhtg->bhsg", h_left, left_error
                    )
                    right_context = torch.einsum(
                        "st,bhtg->bhsg", h_right, right_error
                    )
                    linear[:, :, left_cell] += (
                        left_context
                        - diagonal_left[None, None, :, None] * left_error
                        + torch.einsum("st,bhtg->bhsg", h_cross, right_error)
                    )
                    linear[:, :, right_cell] += (
                        right_context
                        - diagonal_right[None, None, :, None] * right_error
                        + torch.einsum(
                            "st,bhtg->bhsg", h_cross.transpose(0, 1), left_error
                        )
                    )
                reconstruction_terms += 1
            cache_used = cache_used or entry_found
        return (
            coefficient.reshape(1, 1, rows, 1),
            linear.reshape(batch, heads, rows, sketch),
            view,
            cache_used,
            reconstruction_terms,
        )

    @staticmethod
    def _objective_value(
        error: torch.Tensor,
        *,
        temporal_cells: int,
        spatial: int,
        exact_frames: tuple[int, ...],
        temporal_group_size: int,
        current_boundary: int,
        transport: dict[str, Any] | None,
        lambda_anchor: float,
        lambda_reconstruction: float,
    ) -> torch.Tensor:
        """Evaluate every unary/edge exactly once for validation and logging."""

        view = error.reshape(*error.shape[:2], temporal_cells, spatial, error.shape[-1])
        # Match the real packed Current block layout; do not use the ordinal
        # of an Exact anchor in the (possibly sparse) anchor list.
        frame_to_cell = {
            int(frame): min(
                int(frame) // int(temporal_group_size), temporal_cells - 1
            )
            for frame in exact_frames
        }
        anchor_count = error.new_zeros(temporal_cells)
        for frame in exact_frames:
            anchor_count[frame_to_cell[int(frame)]] += 1.0
        value = lambda_anchor * (
            view.square() * anchor_count[None, None, :, None, None]
        ).sum()
        grid_shape = (
            int(transport.get("attention_spatial_h", 1))
            if transport is not None else 1,
            int(transport.get("attention_spatial_w", spatial))
            if transport is not None else spatial,
        )
        for left, right in zip(exact_frames, exact_frames[1:]):
            if (left < current_boundary) != (right < current_boundary):
                continue
            left_error = view[:, :, frame_to_cell[left]]
            right_error = view[:, :, frame_to_cell[right]]
            for target in range(left + 1, right):
                alpha = float(target - left) / float(right - left)
                left_matrix, _ = FCPASMLocalCoordinateSwapRouter._transport_matrix(
                    transport,
                    (int(left), int(right)),
                    spatial_shape=grid_shape,
                    device=error.device,
                    exponent=alpha,
                )
                right_matrix, _ = FCPASMLocalCoordinateSwapRouter._transport_matrix(
                    transport,
                    (int(left), int(right)),
                    spatial_shape=grid_shape,
                    device=error.device,
                    exponent=-(1.0 - alpha),
                )
                propagated = (
                    float(1.0 - alpha)
                    * torch.einsum("st,bhtg->bhsg", left_matrix, left_error)
                    + float(alpha)
                    * torch.einsum("st,bhtg->bhsg", right_matrix, right_error)
                )
                value = value + lambda_reconstruction * propagated.square().sum()
        return value

    def refine(
        self,
        *,
        score: torch.Tensor,
        pooled_value: torch.Tensor,
        selected: torch.Tensor,
        counts: torch.Tensor,
        local: torch.Tensor,
        memory_blocks: int,
        spatial_shape: tuple[int, int],
        exact_current_frames: tuple[int, ...],
        temporal_group_size: int,
        current_boundary: int,
        transport: dict[str, Any] | None,
        layer_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run zero, one, or two vectorized fixed-degree swap rounds."""

        if score.ndim != 4 or pooled_value.ndim != 4:
            raise ValueError("routing requires BHQK score and BHKD pooled values")
        if score.shape[:2] != pooled_value.shape[:2] or score.shape[-1] != pooled_value.shape[-2]:
            raise ValueError("routing score/value layouts differ")
        if selected.shape[:3] != score.shape[:3] or counts.shape != score.shape[:3]:
            raise ValueError("routing selection/count layouts differ")
        if local.shape != score.shape[-2:]:
            raise ValueError("routing local stencil has invalid shape")
        # Routing is a discrete inference-time decision.  Make its no-autograd
        # contract explicit instead of depending on every caller's context.
        score = score.detach()
        pooled_value = pooled_value.detach()
        baseline = selected.clone()
        if self.config.active_layers and layer_index not in self.config.active_layers:
            record = {
                "layer_index": int(layer_index),
                "status": "independent_topk_fallback",
                "fallback_reason": "layer_not_enabled",
                "swap_count": 0,
                "fixed_budget": True,
            }
            self._records.append(record)
            return baseline, record
        num_blocks = score.shape[-1]
        current_rows = score.shape[-2] - int(memory_blocks)
        spatial = int(spatial_shape[0] * spatial_shape[1])
        if current_rows <= 0 or current_rows % spatial:
            record = {
                "layer_index": int(layer_index),
                "status": "independent_topk_fallback",
                "fallback_reason": "no_rectangular_current_rows",
                "swap_count": 0,
                "fc_cache_used": False,
                "fixed_budget": True,
            }
            self._records.append(record)
            return baseline, record
        temporal_cells = current_rows // spatial
        if tuple(sorted(set(exact_current_frames))) != tuple(exact_current_frames):
            raise ValueError("routing Exact Current frames must be unique and chronological")
        has_skipped_interval = any(
            right - left > 1
            and (left < current_boundary) == (right < current_boundary)
            for left, right in zip(exact_current_frames, exact_current_frames[1:])
        )
        if (
            self.config.num_swap_rounds == 0
            or len(exact_current_frames) < 2
            or not has_skipped_interval
        ):
            record = {
                "layer_index": int(layer_index),
                "status": "independent_topk_fallback",
                "fallback_reason": (
                    "zero_swap_rounds"
                    if self.config.num_swap_rounds == 0
                    else "fewer_than_two_exact_anchors"
                    if len(exact_current_frames) < 2
                    else "no_skipped_frames"
                ),
                "swap_count": 0,
                "fc_cache_used": False,
                "fixed_budget": True,
            }
            self._records.append(record)
            return baseline, record

        # A missing previous-layer endpoint transport is a legitimate g=0
        # fallback in the original candidate.  The high-precision child can
        # instead retain the native pooled-Q/K Top-K exactly.  Test this before
        # materialising probabilities, omission proxies, or 2K candidates so
        # the fallback is both numerically exact and essentially free.
        transport_pairs = transport.get("pairs", {}) if isinstance(transport, dict) else {}
        matching_entries = [
            transport_pairs[(int(left), int(right))]
            for left, right in zip(exact_current_frames, exact_current_frames[1:])
            if right - left > 1
            and (left < current_boundary) == (right < current_boundary)
            and (int(left), int(right)) in transport_pairs
            and isinstance(transport_pairs[(int(left), int(right))], dict)
            and isinstance(
                transport_pairs[(int(left), int(right))].get("theta"),
                torch.Tensor,
            )
        ]
        has_matching_transport = bool(matching_entries)
        matching_transport_affinity = max(
            (
                float(entry.get("mean_affinity", 0.0).detach().float().item())
                if isinstance(entry.get("mean_affinity"), torch.Tensor)
                else float(entry.get("mean_affinity", 0.0))
            )
            for entry in matching_entries
        ) if matching_entries else 0.0
        transport_is_strong = (
            has_matching_transport
            and matching_transport_affinity >= self.config.min_transport_affinity
        )
        if self.config.require_fc_transport and not transport_is_strong:
            record = {
                "layer_index": int(layer_index),
                "status": "independent_topk_fallback",
                "fallback_reason": (
                    "weak_previous_layer_fc_transport"
                    if has_matching_transport
                    else "missing_previous_layer_fc_transport"
                ),
                "matching_transport_affinity": matching_transport_affinity,
                "swap_count": 0,
                "fc_cache_used": False,
                "g0_fallback": True,
                "fixed_budget": True,
            }
            self._records.append(record)
            return baseline, record

        # Optional high-precision admission at the temporal-cell level.  The
        # original ``min_transport_affinity`` is a call-level prerequisite;
        # this stricter child additionally prevents a swap on a Current cell
        # whose enclosing Exact endpoint pair has weak shared-phase evidence.
        # A zero threshold is deliberately a no-op for all existing variants.
        pair_cell_gate = torch.ones(
            temporal_cells, dtype=torch.bool, device=score.device
        )
        pair_gate_cells = 0
        if self.config.min_pair_transport_affinity > 0.0:
            pair_cell_gate.zero_()
            for left, right in zip(exact_current_frames, exact_current_frames[1:]):
                if right - left <= 1 or (left < current_boundary) != (right < current_boundary):
                    continue
                entry = transport_pairs.get((int(left), int(right)))
                affinity = (
                    float(entry.get("mean_affinity", 0.0).detach().float().item())
                    if isinstance(entry, dict)
                    and isinstance(entry.get("mean_affinity"), torch.Tensor)
                    else float(entry.get("mean_affinity", 0.0))
                    if isinstance(entry, dict)
                    else 0.0
                )
                if affinity < self.config.min_pair_transport_affinity:
                    continue
                for target in range(left + 1, right):
                    cell = min(
                        int(target) // int(temporal_group_size),
                        temporal_cells - 1,
                    )
                    if not bool(pair_cell_gate[cell]):
                        pair_gate_cells += 1
                    pair_cell_gate[cell] = True
            if not bool(pair_cell_gate.any()):
                record = {
                    "layer_index": int(layer_index),
                    "status": "independent_topk_fallback",
                    "fallback_reason": "no_strong_transport_pair_cells",
                    "matching_transport_affinity": matching_transport_affinity,
                    "min_pair_transport_affinity": self.config.min_pair_transport_affinity,
                    "pair_gate_cells": 0,
                    "swap_count": 0,
                    "fc_cache_used": has_matching_transport,
                    "fixed_budget": True,
                }
                self._records.append(record)
                return baseline, record
        pair_row_gate = pair_cell_gate.repeat_interleave(spatial)

        valid_selected = (
            torch.arange(selected.shape[-1], device=score.device)[None, None, None]
            < counts[..., None]
        )
        historical_selected = valid_selected & (selected < memory_blocks)
        historical_counts = historical_selected.sum(dim=-1)
        swappable_selected = (
            historical_selected if self.config.historical_only else valid_selected
        )
        swappable_counts = swappable_selected.sum(dim=-1)
        maximum_swappable_k = int(swappable_counts.max().item())
        if maximum_swappable_k == 0:
            record = {
                "layer_index": int(layer_index),
                "status": "independent_topk_fallback",
                "fallback_reason": "no_swappable_kv_selected",
                "swap_count": 0,
                "fixed_budget": True,
            }
            self._records.append(record)
            return baseline, record
        maximum_k = int(counts.max().item())
        candidate_blocks = memory_blocks if self.config.historical_only else num_blocks
        maximum_l = min(
            candidate_blocks,
            self.config.candidate_multiplier * maximum_swappable_k,
        )
        ranking = (
            score[..., :candidate_blocks]
            + local[:, :candidate_blocks][None, None].to(score.dtype) * 1e6
        )
        candidates = torch.topk(ranking, k=maximum_l, dim=-1).indices
        probability = torch.softmax(score.float(), dim=-1)
        proxy = estimate_block_omission_error(
            probability,
            pooled_value,
            candidates,
            sketch_groups=self.config.sketch_groups,
        )
        candidate_rank = torch.arange(maximum_l, device=score.device)
        candidate_limit = torch.minimum(
            swappable_counts.to(torch.long) * self.config.candidate_multiplier,
            torch.full_like(swappable_counts, candidate_blocks),
        )
        candidate_valid = candidate_rank[None, None, None] < candidate_limit[..., None]

        def membership(current: torch.Tensor) -> torch.Tensor:
            valid_selected = (
                torch.arange(current.shape[-1], device=score.device)[None, None, None]
                < counts[..., None]
            )
            return (
                (candidates[..., None] == current[..., None, :])
                & valid_selected[..., None, :]
            ).any(dim=-1)

        retained = membership(selected)
        error = (proxy * (candidate_valid & ~retained)[..., None]).sum(dim=-2)
        current_error = error[..., memory_blocks:, :]
        cache_used = has_matching_transport
        reconstruction_terms = sum(
            max(0, int(right) - int(left) - 1)
            for left, right in zip(exact_current_frames, exact_current_frames[1:])
            if (left < current_boundary) == (right < current_boundary)
        )
        objective_before = self._objective_value(
            current_error,
            temporal_cells=temporal_cells,
            spatial=spatial,
            exact_frames=tuple(int(value) for value in exact_current_frames),
            temporal_group_size=int(temporal_group_size),
            current_boundary=int(current_boundary),
            transport=transport,
            lambda_anchor=self.config.lambda_anchor,
            lambda_reconstruction=self.config.lambda_reconstruction,
        )
        swaps = 0
        improvements: list[torch.Tensor] = []
        row_offset = memory_blocks
        for _round in range(self.config.num_swap_rounds):
            # A previous round changes the fixed neighbor context of adjacent
            # temporal cells.  Rebuild only these cheap quadratic coefficients
            # before evaluating the next coordinate round.
            coefficient, linear, _, round_cache_used, _ = self._objective_terms(
                current_error,
                temporal_cells=temporal_cells,
                spatial=spatial,
                exact_frames=tuple(int(value) for value in exact_current_frames),
                temporal_group_size=int(temporal_group_size),
                current_boundary=int(current_boundary),
                transport=transport,
                lambda_anchor=self.config.lambda_anchor,
                lambda_reconstruction=self.config.lambda_reconstruction,
            )
            cache_used = cache_used or round_cache_used
            retained = membership(selected)
            valid_selected = (
                torch.arange(selected.shape[-1], device=score.device)[None, None, None]
                < counts[..., None]
            )
            selected_local = torch.gather(
                local[None, None].expand(*selected.shape[:2], -1, -1), -1, selected
            )
            removable = valid_selected & ~selected_local
            if self.config.historical_only:
                removable = removable & (selected < memory_blocks)
            addable = candidate_valid & ~retained
            selected_candidate_match = candidates[..., None] == selected[..., None, :]
            remove_proxy = torch.einsum(
                "bhqls,bhqlk->bhqks", proxy, selected_candidate_match.to(proxy.dtype)
            )
            remove_proxy = remove_proxy[..., row_offset:, :, :]
            add_proxy = proxy[..., row_offset:, :, :]
            base_error = current_error
            delta = remove_proxy[..., :, None, :] - add_proxy[..., None, :, :]
            a = coefficient[..., None, None]
            b = linear[..., None, None, :]
            improvement_delta = (
                a * (2.0 * base_error[..., None, None, :] * delta + delta.square())
                + 2.0 * b * delta
            ).sum(dim=-1)
            valid_pair = (
                removable[..., row_offset:, :, None]
                & addable[..., row_offset:, None, :]
            )
            if self.config.min_pair_transport_affinity > 0.0:
                valid_pair = valid_pair & pair_row_gate[None, None, :, None, None]
            if self.config.same_bank_only:
                removed_is_memory = (
                    selected[..., row_offset:, :, None] < memory_blocks
                )
                added_is_memory = (
                    candidates[..., row_offset:, None, :] < memory_blocks
                )
                valid_pair = valid_pair & (removed_is_memory == added_is_memory)
            improvement_delta = improvement_delta.masked_fill(~valid_pair, torch.inf)
            flat = improvement_delta.flatten(-2)
            best_delta, best_flat = flat.min(dim=-1)
            # A spectral OLA operator couples spatial coordinates.  A true
            # coordinate-descent round must therefore accept one coordinate at
            # a time; parallel row swaps omit delta_i^T H_ij delta_j terms.
            flat_best = best_delta.reshape(-1)
            strongest = torch.argmin(flat_best)
            should_swap = torch.zeros_like(best_delta, dtype=torch.bool)
            if flat_best[strongest] < -self.config.swap_eps:
                should_swap.reshape(-1)[strongest] = True
            if self.config.max_swaps_per_call and swaps >= self.config.max_swaps_per_call:
                break
            if not bool(should_swap.any()):
                break
            remove_slot = torch.div(best_flat, maximum_l, rounding_mode="floor")
            add_slot = best_flat.remainder(maximum_l)
            selected_current = selected[..., row_offset:, :]
            added_key = torch.gather(
                candidates[..., row_offset:, :], -1, add_slot[..., None]
            ).squeeze(-1)
            selected_current.scatter_(
                -1,
                remove_slot[..., None],
                torch.where(
                    should_swap, added_key, torch.gather(
                        selected_current, -1, remove_slot[..., None]
                    ).squeeze(-1),
                )[..., None],
            )
            chosen_remove = torch.gather(
                remove_proxy, -2, remove_slot[..., None, None].expand(
                    *remove_slot.shape, 1, remove_proxy.shape[-1]
                )
            ).squeeze(-2)
            chosen_add = torch.gather(
                add_proxy, -2, add_slot[..., None, None].expand(
                    *add_slot.shape, 1, add_proxy.shape[-1]
                )
            ).squeeze(-2)
            current_error = current_error + should_swap[..., None] * (
                chosen_remove - chosen_add
            )
            swaps += int(should_swap.sum().item())
            improvements.append((-best_delta[should_swap]).sum().detach())

        valid_final = (
            torch.arange(selected.shape[-1], device=selected.device)[None, None, None]
            < counts[..., None]
        )
        final_historical_counts = ((selected < memory_blocks) & valid_final).sum(-1)
        if self.config.same_bank_only and not torch.equal(
            historical_counts, final_historical_counts
        ):
            raise RuntimeError("same-bank FC-PASM routing changed Memory KV counts")
        if self.config.historical_only and not torch.equal(
            final_historical_counts, historical_counts
        ):
            raise RuntimeError("FC-PASM routing changed a historical-KV row budget")
        baseline_nonhistorical = torch.where(
            valid_final & (baseline >= memory_blocks), baseline, num_blocks
        ).sort(dim=-1).values
        final_nonhistorical = torch.where(
            valid_final & (selected >= memory_blocks), selected, num_blocks
        ).sort(dim=-1).values
        if self.config.historical_only and not torch.equal(
            final_nonhistorical, baseline_nonhistorical
        ):
            raise RuntimeError("FC-PASM routing modified Current-KV support")
        duplicate_matrix = selected[..., :, None] == selected[..., None, :]
        valid_pairs = valid_final[..., :, None] & valid_final[..., None, :]
        upper = torch.triu(
            torch.ones(
                selected.shape[-1], selected.shape[-1],
                dtype=torch.bool, device=selected.device,
            ),
            diagonal=1,
        )
        duplicates = duplicate_matrix & valid_pairs & upper
        if bool(duplicates.any()):
            raise RuntimeError("FC-PASM routing emitted duplicate keys")
        final_local = torch.gather(
            local[None, None].expand(*selected.shape[:2], -1, -1), -1, selected
        ) & valid_final
        expected_local = int(local.sum().item()) * selected.shape[0] * selected.shape[1]
        if int(final_local.sum().item()) != expected_local:
            raise RuntimeError("FC-PASM routing removed a protected local edge")
        objective_after = self._objective_value(
            current_error,
            temporal_cells=temporal_cells,
            spatial=spatial,
            exact_frames=tuple(int(value) for value in exact_current_frames),
            temporal_group_size=int(temporal_group_size),
            current_boundary=int(current_boundary),
            transport=transport,
            lambda_anchor=self.config.lambda_anchor,
            lambda_reconstruction=self.config.lambda_reconstruction,
        )
        tolerance = 1e-5 * objective_before.abs().clamp_min(1.0)
        if bool(objective_after > objective_before + tolerance):
            raise RuntimeError("FC-PASM swap increased its refinement objective")
        record = {
            "layer_index": int(layer_index),
            "status": "refined",
            "candidate_multiplier": self.config.candidate_multiplier,
            "candidate_max": maximum_l,
            "swap_rounds": self.config.num_swap_rounds,
            "max_swaps_per_call": self.config.max_swaps_per_call,
            "swap_count": swaps,
            "objective_before": objective_before.detach(),
            "objective_after": objective_after.detach(),
            "objective_improvement": (objective_before - objective_after).detach(),
            "average_swap_improvement": (
                torch.stack(improvements).sum() / float(swaps)
                if swaps and improvements
                else objective_before.new_zeros(())
            ),
            "anchor_term_weight": self.config.lambda_anchor,
            "reconstruction_term_weight": self.config.lambda_reconstruction,
            "reconstruction_terms": reconstruction_terms,
            "fc_cache_used": cache_used,
            "matching_transport_affinity": matching_transport_affinity,
            "min_pair_transport_affinity": self.config.min_pair_transport_affinity,
            "pair_gate_cells": pair_gate_cells,
            "g0_fallback": not cache_used,
            "fixed_budget": True,
            "historical_only": self.config.historical_only,
            "same_bank_only": self.config.same_bank_only,
            "historical_budget_before": int(historical_counts.sum().item()),
            "historical_budget_after": int(
                final_historical_counts.sum().item()
            ),
            "current_kv_support_unchanged": (
                bool(torch.equal(final_nonhistorical, baseline_nonhistorical))
            ),
            "minimum_k": int(counts.min().item()),
            "maximum_k": maximum_k,
            "budget_before": int(counts.sum().item()),
            "budget_after": int(counts.sum().item()),
            "unique_indices": True,
            "local_support_preserved": True,
        }
        self._records.append(record)
        return selected, record

    def summary(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for raw in self._records:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(value, torch.Tensor):
                    cpu = value.detach().float().cpu()
                    row[key] = (
                        float(cpu.item())
                        if cpu.numel() == 1
                        else [float(item) for item in cpu.flatten().tolist()]
                    )
                else:
                    row[key] = value
            records.append(row)
        return {
            "routing_mode": "fc_pasm_swap",
            "config": {
                "candidate_multiplier": self.config.candidate_multiplier,
                "num_swap_rounds": self.config.num_swap_rounds,
                "swap_eps": self.config.swap_eps,
                "lambda_anchor": self.config.lambda_anchor,
                "lambda_reconstruction": self.config.lambda_reconstruction,
                "sketch_groups": self.config.sketch_groups,
                "require_fc_transport": self.config.require_fc_transport,
                "min_transport_affinity": self.config.min_transport_affinity,
                "min_pair_transport_affinity": self.config.min_pair_transport_affinity,
                "max_swaps_per_call": self.config.max_swaps_per_call,
                "active_layers": list(self.config.active_layers),
                "historical_only": self.config.historical_only,
                "same_bank_only": self.config.same_bank_only,
            },
            "records": records,
        }

    def reset(self) -> None:
        self._records.clear()
