"""Fused compact-frame gather and first AdaLN boundary for Matrix q0-mod5.

This is an isolated Kernel Design Agents candidate.  It does not change the
frame schedule, sparse-attention support, or any cache/memory decision.  The
kernel consumes an already-compiled chronological active-frame layout and
produces exactly the state needed by the unchanged remainder of a woven block:

* the active hidden state used by the attention residual;
* the input-dtype FP32-LayerNorm + adaptive shift/scale input to QKV; and
* FP32 adaptive gates 2--5 used after attention and by the FFN.

Gates 0/1 are consumed in-register and deliberately not materialized.  Unknown
shapes fail closed unless the caller explicitly requests the reference
fallback.  This file contains no production hook; promotion requires replay
and complete-block timing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the reference fallback
    triton = None
    tl = None


@dataclass(frozen=True)
class CompiledActiveFrameLayout:
    """Immutable chronological compact-frame layout for one emitted schedule."""

    total_frames: int
    spatial_tokens: int
    active_frames: tuple[int, ...]
    active_frames_device: torch.Tensor

    @property
    def active_tokens(self) -> int:
        return len(self.active_frames) * self.spatial_tokens


@dataclass
class GatherAdaLN1Workspace:
    """Caller-owned output buffers used by the steady-state candidate."""

    active_x: torch.Tensor
    normalized: torch.Tensor
    residual_gates: torch.Tensor


@dataclass
class NativeLNGatherAdaLN1Workspace:
    """Preallocated outputs around the intentionally native LayerNorm path."""

    active_x: torch.Tensor
    shift_scale_gates: torch.Tensor
    residual_gates: torch.Tensor


def compile_active_frame_layout(
    active_frames: Sequence[int],
    *,
    total_frames: int,
    spatial_tokens: int,
    device: torch.device | str,
) -> CompiledActiveFrameLayout:
    """Compile an exact frame layout without constructing per-token indices."""

    frames = tuple(int(frame) for frame in active_frames)
    if total_frames <= 0 or spatial_tokens <= 0:
        raise ValueError("total_frames and spatial_tokens must be positive")
    if not frames:
        raise ValueError("the active-frame layout may not be empty")
    if tuple(sorted(set(frames))) != frames:
        raise ValueError("active frame IDs must be unique and chronological")
    if frames[0] < 0 or frames[-1] >= total_frames:
        raise ValueError("active frame ID lies outside the temporal grid")
    frame_tensor = torch.tensor(frames, device=device, dtype=torch.int32)
    return CompiledActiveFrameLayout(
        total_frames=int(total_frames),
        spatial_tokens=int(spatial_tokens),
        active_frames=frames,
        active_frames_device=frame_tensor,
    )


def allocate_gather_adaln1_workspace(
    layout: CompiledActiveFrameLayout,
    *,
    channels: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> GatherAdaLN1Workspace:
    """Allocate the three outputs once for steady-state replay."""

    if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError("active hidden state must use FP32, BF16, or FP16")
    shape = (1, layout.active_tokens, int(channels))
    return GatherAdaLN1Workspace(
        active_x=torch.empty(shape, device=device, dtype=dtype),
        normalized=torch.empty(shape, device=device, dtype=dtype),
        residual_gates=torch.empty(
            (1, layout.active_tokens, 4, int(channels)),
            device=device,
            dtype=torch.float32,
        ),
    )


def allocate_native_ln_gather_adaln1_workspace(
    layout: CompiledActiveFrameLayout,
    *,
    channels: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> NativeLNGatherAdaLN1Workspace:
    """Allocate c1a gather/gate outputs; native LN owns its exact temporaries."""

    if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError("active hidden state must use FP32, BF16, or FP16")
    shape = (1, layout.active_tokens, int(channels))
    return NativeLNGatherAdaLN1Workspace(
        active_x=torch.empty(shape, device=device, dtype=dtype),
        shift_scale_gates=torch.empty(
            (1, layout.active_tokens, 2, int(channels)),
            device=device,
            dtype=torch.float32,
        ),
        residual_gates=torch.empty(
            (1, layout.active_tokens, 4, int(channels)),
            device=device,
            dtype=torch.float32,
        ),
    )


def _compact_token_indices(
    layout: CompiledActiveFrameLayout, *, device: torch.device
) -> torch.Tensor:
    space = torch.arange(layout.spatial_tokens, device=device, dtype=torch.long)
    frames = layout.active_frames_device.to(device=device, dtype=torch.long)
    return (frames[:, None] * layout.spatial_tokens + space[None, :]).reshape(-1)


def reference_gather_adaln1(
    x: torch.Tensor,
    embedding: torch.Tensor,
    modulation: torch.Tensor,
    layout: CompiledActiveFrameLayout,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Literal PyTorch equivalent of the current woven-block ingress."""

    _validate_inputs(x, embedding, modulation, layout, None, require_cuda=False)
    compact_indices = _compact_token_indices(layout, device=x.device)
    active_x = x[:, compact_indices]
    active_embedding = embedding[:, compact_indices]
    # Match Matrix WanAttentionBlock and MatrixCurvaturePhaseFrameWeave: the
    # six-way FP32 addition happens before chunking, LayerNorm is FP32 then
    # returned to the hidden dtype before FP32 adaptive modulation.  The
    # production runner uses FP32 hidden state, making this dtype round-trip an
    # identity; BF16/FP16 stress cases still round explicitly.
    gates = (modulation.unsqueeze(0) + active_embedding).chunk(6, dim=2)
    layer_norm = torch.nn.functional.layer_norm(
        active_x.float(), (active_x.shape[-1],), None, None, float(eps)
    ).to(active_x.dtype)
    normalized = (
        layer_norm.float() * (1.0 + gates[1].squeeze(2))
        + gates[0].squeeze(2)
    ).to(active_x.dtype)
    residual_gates = torch.cat(gates[2:6], dim=2).contiguous()
    return active_x, normalized, residual_gates


if triton is not None:

    @triton.jit
    def _fused_gather_adaln1_kernel(
        x_ptr,
        embedding_ptr,
        modulation_ptr,
        active_frames_ptr,
        active_x_ptr,
        normalized_ptr,
        residual_gates_ptr,
        spatial_tokens: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        compact_row = tl.program_id(0)
        active_frame_slot = compact_row // spatial_tokens
        within_frame = compact_row - active_frame_slot * spatial_tokens
        source_frame = tl.load(active_frames_ptr + active_frame_slot).to(tl.int64)
        source_row = source_frame * spatial_tokens + within_frame

        channel = tl.arange(0, BLOCK_C)
        channel_mask = channel < channels
        x_offset = source_row * channels + channel
        output_offset = compact_row * channels + channel
        x_value = tl.load(x_ptr + x_offset, mask=channel_mask, other=0.0)
        x_fp32 = x_value.to(tl.float32)
        tl.store(active_x_ptr + output_offset, x_value, mask=channel_mask)

        # WanLayerNorm uses FP32 LayerNorm with no affine parameters, then casts
        # to the hidden dtype before the FP32 shift/scale operation.
        mean = tl.sum(x_fp32, axis=0) / channels
        centered = tl.where(channel_mask, x_fp32 - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / channels
        norm_fp32 = centered * tl.rsqrt(variance + eps)
        # Make the reference's intermediate input-dtype conversion explicit;
        # it is an identity for production FP32 and a rounding step for BF16/FP16.
        norm_rounded = norm_fp32.to(x_ptr.type.element_ty).to(tl.float32)

        embedding_row = source_row * (6 * channels)
        mod0 = tl.load(
            modulation_ptr + channel, mask=channel_mask, other=0.0
        ).to(tl.float32)
        mod1 = tl.load(
            modulation_ptr + channels + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        emb0 = tl.load(
            embedding_ptr + embedding_row + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        emb1 = tl.load(
            embedding_ptr + embedding_row + channels + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        normalized = norm_rounded * (1.0 + emb1 + mod1) + emb0 + mod0
        tl.store(normalized_ptr + output_offset, normalized, mask=channel_mask)

        # The unchanged downstream block needs gates 2--5.  Store only those
        # four FP32 tensors; gates 0/1 die in-register at this boundary.
        for source_gate in range(2, 6):
            gate = (
                tl.load(
                    embedding_ptr
                    + embedding_row
                    + source_gate * channels
                    + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                + tl.load(
                    modulation_ptr + source_gate * channels + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            )
            gate_offset = (
                compact_row * (4 * channels)
                + (source_gate - 2) * channels
                + channel
            )
            tl.store(
                residual_gates_ptr + gate_offset, gate, mask=channel_mask
            )


    @triton.jit
    def _gather_active_x_and_all_gates_kernel(
        x_ptr,
        embedding_ptr,
        modulation_ptr,
        active_frames_ptr,
        active_x_ptr,
        shift_scale_gates_ptr,
        residual_gates_ptr,
        spatial_tokens: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """c1a: fuse only index/gather/add operations with no reduction."""

        compact_row = tl.program_id(0)
        active_frame_slot = compact_row // spatial_tokens
        within_frame = compact_row - active_frame_slot * spatial_tokens
        source_frame = tl.load(active_frames_ptr + active_frame_slot).to(tl.int64)
        source_row = source_frame * spatial_tokens + within_frame
        channel = tl.arange(0, BLOCK_C)
        channel_mask = channel < channels

        source_x_offset = source_row * channels + channel
        compact_x_offset = compact_row * channels + channel
        active_x = tl.load(
            x_ptr + source_x_offset, mask=channel_mask, other=0.0
        )
        tl.store(
            active_x_ptr + compact_x_offset, active_x, mask=channel_mask
        )

        embedding_row = source_row * (6 * channels)
        for source_gate in range(0, 6):
            # This is the same single FP32 addition as
            # `block.modulation.unsqueeze(0) + e_active`; it contains no fused
            # multiply/add or reduction whose rounding could differ.
            gate = (
                tl.load(
                    modulation_ptr + source_gate * channels + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
                + tl.load(
                    embedding_ptr
                    + embedding_row
                    + source_gate * channels
                    + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            )
            if source_gate < 2:
                destination = (
                    compact_row * (2 * channels)
                    + source_gate * channels
                    + channel
                )
                tl.store(
                    shift_scale_gates_ptr + destination,
                    gate,
                    mask=channel_mask,
                )
            else:
                destination = (
                    compact_row * (4 * channels)
                    + (source_gate - 2) * channels
                    + channel
                )
                tl.store(
                    residual_gates_ptr + destination,
                    gate,
                    mask=channel_mask,
                )


def _validate_inputs(
    x: torch.Tensor,
    embedding: torch.Tensor,
    modulation: torch.Tensor,
    layout: CompiledActiveFrameLayout,
    workspace: GatherAdaLN1Workspace | None,
    *,
    require_cuda: bool,
) -> None:
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError("candidate requires x shaped [1, full_tokens, channels]")
    if embedding.shape != (1, x.shape[1], 6, x.shape[2]):
        raise ValueError("embedding must be contiguous FP32 [1, tokens, 6, C]")
    if modulation.shape != (1, 6, x.shape[2]):
        raise ValueError("modulation must be shaped [1, 6, C]")
    if embedding.dtype != torch.float32:
        raise ValueError("Matrix timestep embedding must remain FP32")
    if x.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError("Matrix hidden state must be FP32, BF16, or FP16")
    if modulation.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError("modulation must use a floating Matrix model dtype")
    if embedding.device != x.device or modulation.device != x.device:
        raise ValueError("x/embedding/modulation must share a device")
    if not (x.is_contiguous() and embedding.is_contiguous() and modulation.is_contiguous()):
        raise ValueError("candidate requires contiguous x/embedding/modulation")
    if x.shape[1] != layout.total_frames * layout.spatial_tokens:
        raise ValueError("layout temporal/spatial product differs from x tokens")
    if layout.active_frames_device.device != x.device:
        raise ValueError("compiled layout and input must share a device")
    if require_cuda and not x.is_cuda:
        raise RuntimeError("fused candidate requires CUDA")
    if workspace is not None:
        expected = (1, layout.active_tokens, x.shape[2])
        if workspace.active_x.shape != expected or workspace.active_x.dtype != x.dtype:
            raise ValueError("active_x workspace has the wrong shape or dtype")
        if workspace.normalized.shape != expected or workspace.normalized.dtype != x.dtype:
            raise ValueError("normalized workspace has the wrong shape or dtype")
        if workspace.residual_gates.shape != (1, layout.active_tokens, 4, x.shape[2]):
            raise ValueError("residual-gate workspace has the wrong shape")
        if workspace.residual_gates.dtype != torch.float32:
            raise ValueError("residual-gate workspace must be FP32")
        buffers = (
            workspace.active_x,
            workspace.normalized,
            workspace.residual_gates,
        )
        if any(tensor.device != x.device or not tensor.is_contiguous() for tensor in buffers):
            raise ValueError("workspace buffers must be contiguous and colocated")


def fused_gather_adaln1(
    x: torch.Tensor,
    embedding: torch.Tensor,
    modulation: torch.Tensor,
    layout: CompiledActiveFrameLayout,
    *,
    workspace: GatherAdaLN1Workspace | None = None,
    eps: float = 1e-6,
    allow_reference_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run c1, failing closed unless an explicit fallback was authorized."""

    supported = (
        triton is not None
        and x.is_cuda
        and x.shape[0] == 1
        and 0 < x.shape[-1] <= 65536
    )
    if not supported:
        if allow_reference_fallback:
            return reference_gather_adaln1(
                x, embedding, modulation, layout, eps=eps
            )
        raise RuntimeError("no validated Triton CUDA c1 path for these inputs")
    _validate_inputs(x, embedding, modulation, layout, workspace, require_cuda=True)
    if workspace is None:
        workspace = allocate_gather_adaln1_workspace(
            layout, channels=x.shape[-1], dtype=x.dtype, device=x.device
        )
    block_channels = triton.next_power_of_2(x.shape[-1])
    _fused_gather_adaln1_kernel[(layout.active_tokens,)](
        x,
        embedding,
        modulation,
        layout.active_frames_device,
        workspace.active_x,
        workspace.normalized,
        workspace.residual_gates,
        spatial_tokens=layout.spatial_tokens,
        channels=x.shape[-1],
        eps=float(eps),
        BLOCK_C=block_channels,
        num_warps=8,
        num_stages=1,
    )
    return workspace.active_x, workspace.normalized, workspace.residual_gates


def native_ln_gather_adaln1(
    x: torch.Tensor,
    embedding: torch.Tensor,
    modulation: torch.Tensor,
    layout: CompiledActiveFrameLayout,
    *,
    workspace: NativeLNGatherAdaLN1Workspace | None = None,
    eps: float = 1e-6,
    allow_reference_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run c1a with native Wan/PyTorch LayerNorm and modulation arithmetic.

    Triton performs only chronological index resolution, active-x gather, and
    the six independent FP32 embedding/modulation additions.  In particular,
    neither the LayerNorm reduction nor the post-LN multiply/add is fused, so
    their input-dtype round-trip follows the canonical Python path.
    """

    supported = (
        triton is not None
        and x.is_cuda
        and x.shape[0] == 1
        and 0 < x.shape[-1] <= 65536
    )
    if not supported:
        if allow_reference_fallback:
            return reference_gather_adaln1(
                x, embedding, modulation, layout, eps=eps
            )
        raise RuntimeError("no validated Triton CUDA c1a path for these inputs")
    _validate_inputs(x, embedding, modulation, layout, None, require_cuda=True)
    if workspace is None:
        workspace = allocate_native_ln_gather_adaln1_workspace(
            layout, channels=x.shape[-1], dtype=x.dtype, device=x.device
        )
    expected = (1, layout.active_tokens, x.shape[-1])
    if workspace.active_x.shape != expected or workspace.active_x.dtype != x.dtype:
        raise ValueError("c1a active_x workspace has the wrong shape or dtype")
    if workspace.shift_scale_gates.shape != (
        1,
        layout.active_tokens,
        2,
        x.shape[-1],
    ):
        raise ValueError("c1a shift/scale workspace has the wrong shape")
    if workspace.residual_gates.shape != (
        1,
        layout.active_tokens,
        4,
        x.shape[-1],
    ):
        raise ValueError("c1a residual-gate workspace has the wrong shape")
    if (
        workspace.shift_scale_gates.dtype != torch.float32
        or workspace.residual_gates.dtype != torch.float32
    ):
        raise ValueError("c1a gate workspaces must be FP32")
    buffers = (
        workspace.active_x,
        workspace.shift_scale_gates,
        workspace.residual_gates,
    )
    if any(
        tensor.device != x.device or not tensor.is_contiguous()
        for tensor in buffers
    ):
        raise ValueError("c1a workspace buffers must be contiguous and colocated")

    block_channels = triton.next_power_of_2(x.shape[-1])
    _gather_active_x_and_all_gates_kernel[(layout.active_tokens,)](
        x,
        embedding,
        modulation,
        layout.active_frames_device,
        workspace.active_x,
        workspace.shift_scale_gates,
        workspace.residual_gates,
        spatial_tokens=layout.spatial_tokens,
        channels=x.shape[-1],
        BLOCK_C=block_channels,
        num_warps=8,
        num_stages=1,
    )

    # Keep these operations intentionally separate and identical to
    # WanLayerNorm + MatrixCurvaturePhaseFrameWeave.  The output allocation is
    # owned by native PyTorch; "preallocated" c1a timing applies only to the
    # gather/gate workspaces and is labelled accordingly by the harness.
    rounded_layer_norm = torch.nn.functional.layer_norm(
        workspace.active_x.float(),
        (x.shape[-1],),
        None,
        None,
        float(eps),
    ).to(x.dtype)
    normalized = (
        rounded_layer_norm.float()
        * (1.0 + workspace.shift_scale_gates[:, :, 1, :])
        + workspace.shift_scale_gates[:, :, 0, :]
    ).to(x.dtype)
    return workspace.active_x, normalized, workspace.residual_gates


__all__ = [
    "CompiledActiveFrameLayout",
    "GatherAdaLN1Workspace",
    "NativeLNGatherAdaLN1Workspace",
    "allocate_gather_adaln1_workspace",
    "allocate_native_ln_gather_adaln1_workspace",
    "compile_active_frame_layout",
    "fused_gather_adaln1",
    "native_ln_gather_adaln1",
    "reference_gather_adaln1",
]
