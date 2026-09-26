"""Direct two-anchor residual contraction for Matrix FrameWeave.

KDA candidate ``c1_direct_pair_writer``.  The caller supplies the exact target
frames, active-anchor slots and already-computed control-barycentric weights.
This module does not select frames or calculate control similarity.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _direct_control_barycentric_pair_writer(
        active_ptr,
        weights_ptr,
        source_pairs_ptr,
        target_frames_ptr,
        output_ptr,
        spatial_channels: tl.constexpr,
        BLOCK: tl.constexpr,
        ARITHMETIC_MODE: tl.constexpr,
    ):
        target_row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < spatial_channels
        left_slot = tl.load(source_pairs_ptr + target_row * 2).to(tl.int64)
        right_slot = tl.load(source_pairs_ptr + target_row * 2 + 1).to(tl.int64)
        target_frame = tl.load(target_frames_ptr + target_row).to(tl.int64)
        left = tl.load(
            active_ptr + left_slot * spatial_channels + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        right = tl.load(
            active_ptr + right_slot * spatial_channels + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        weight_left = tl.load(weights_ptr + target_row * 2).to(tl.float32)
        weight_right = tl.load(weights_ptr + target_row * 2 + 1).to(tl.float32)
        if ARITHMETIC_MODE == 1:
            left_term = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [left, weight_left],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            right_term = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [right, weight_right],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            mixed = tl.inline_asm_elementwise(
                "add.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [left_term, right_term],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif ARITHMETIC_MODE == 2:
            left_term = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [left, weight_left],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            mixed = tl.inline_asm_elementwise(
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                [right, weight_right, left_term],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        elif ARITHMETIC_MODE == 3:
            right_term = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                [right, weight_right],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            mixed = tl.inline_asm_elementwise(
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                [left, weight_left, right_term],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            mixed = left * weight_left + right * weight_right
        tl.store(
            output_ptr + target_frame * spatial_channels + offsets,
            mixed,
            mask=valid,
        )


def _validate(
    active_residual: torch.Tensor,
    weights: torch.Tensor,
    source_pairs: torch.Tensor,
    target_frames: torch.Tensor,
    output: torch.Tensor,
) -> tuple[int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for the direct pair writer")
    if not active_residual.is_cuda:
        raise ValueError("direct pair writer requires CUDA tensors")
    if active_residual.dtype != torch.float32 or not active_residual.is_contiguous():
        raise ValueError("active residual must be contiguous FP32")
    if active_residual.ndim != 4 or active_residual.shape[0] != 1:
        raise ValueError("active residual must be [1,A,S,C]")
    batch, active_count, spatial, channels = active_residual.shape
    target_count = int(target_frames.numel())
    if weights.shape != (batch, target_count, 2) or weights.dtype != torch.float32:
        raise ValueError("weights must be contiguous FP32 [1,N,2]")
    if not weights.is_contiguous():
        raise ValueError("weights must be contiguous")
    if source_pairs.shape != (target_count, 2) or source_pairs.dtype != torch.int32:
        raise ValueError("source pairs must be int32 [N,2]")
    if source_pairs.device != active_residual.device or not source_pairs.is_contiguous():
        raise ValueError("source pairs must be contiguous on the input device")
    for name, value in (("target frames", target_frames),):
        if value.shape != (target_count,) or value.dtype != torch.int32:
            raise ValueError(f"{name} must be int32 [N]")
        if value.device != active_residual.device or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the input device")
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError("output must be [1,T,S,C]")
    if output.shape[2:] != (spatial, channels):
        raise ValueError("output spatial/channel shape mismatch")
    if output.dtype != torch.float32 or output.device != active_residual.device:
        raise ValueError("output must be FP32 on the input device")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if target_count <= 0 or active_count not in (4, 5, 6):
        raise ValueError(
            "production candidate requires 4/5/6 Current anchors and targets"
        )
    return target_count, spatial, channels


def direct_control_barycentric_pair_write(
    active_residual: torch.Tensor,
    weights: torch.Tensor,
    source_pairs: torch.Tensor,
    target_frames: torch.Tensor,
    output: torch.Tensor,
    *,
    block: int = 256,
    arithmetic_mode: str = "fused",
) -> torch.Tensor:
    target_count, spatial, channels = _validate(
        active_residual,
        weights,
        source_pairs,
        target_frames,
        output,
    )
    if block not in (128, 256, 512, 1024):
        raise ValueError("block must be one of 128,256,512,1024")
    arithmetic_modes = {
        "fused": 0,
        "separate": 1,
        "left_then_right_fma": 2,
        "right_then_left_fma": 3,
    }
    if arithmetic_mode not in arithmetic_modes:
        raise ValueError(f"unknown arithmetic mode: {arithmetic_mode}")
    spatial_channels = spatial * channels
    grid = (target_count, triton.cdiv(spatial_channels, block))
    _direct_control_barycentric_pair_writer[grid](
        active_residual,
        weights,
        source_pairs,
        target_frames,
        output,
        spatial_channels=spatial_channels,
        BLOCK=block,
        ARITHMETIC_MODE=arithmetic_modes[arithmetic_mode],
        num_warps=4 if block <= 256 else 8,
    )
    return output
