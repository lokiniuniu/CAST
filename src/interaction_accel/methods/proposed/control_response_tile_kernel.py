"""Triton kernels for fused control response and CWCA tile reduction."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the PyTorch fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _camera_response_partials(
        after_ptr, x_ptr, y_ptr, e_ptr, modulation_ptr,
        partial_sum_ptr, partial_count_ptr,
        total_t: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
        memory: tl.constexpr, current_offset: tl.constexpr,
        temporal_blocks: tl.constexpr, spatial_h: tl.constexpr,
        spatial_w: tl.constexpr, tile_t: tl.constexpr,
        tile_h: tl.constexpr, tile_w: tl.constexpr,
        channels: tl.constexpr, chunks_per_tile: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        chunk_id = tl.program_id(1)
        spatial = spatial_h * spatial_w
        time_block = tile_id // spatial
        spatial_id = tile_id - time_block * spatial
        block_h = spatial_id // spatial_w
        block_w = spatial_id - block_h * spatial_w

        slots = chunk_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        local_plane = tile_h * tile_w
        dt = slots // local_plane
        local = slots - dt * local_plane
        dh = local // tile_w
        dw = local - dh * tile_w
        current_t = current_offset + time_block * tile_t + dt
        frame = memory + current_t
        h = block_h * tile_h + dh
        w = block_w * tile_w + dw
        valid = (
            (dt < tile_t) & (frame < total_t) & (h < height) & (w < width)
        )
        token = (frame * height + h) * width + w

        numerator = tl.zeros((BLOCK_TOKENS,), tl.float32)
        denominator = tl.zeros((BLOCK_TOKENS,), tl.float32)
        for channel_start in range(0, channels, BLOCK_C):
            channel = channel_start + tl.arange(0, BLOCK_C)
            channel_valid = channel < channels
            offsets = token[:, None] * channels + channel[None, :]
            mask = valid[:, None] & channel_valid[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            after = tl.load(after_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            e_offsets = token[:, None] * (6 * channels) + 2 * channels + channel[None, :]
            e2 = tl.load(e_ptr + e_offsets, mask=mask, other=0.0).to(tl.float32)
            mod = tl.load(
                modulation_ptr + 2 * channels + channel[None, :],
                mask=channel_valid[None, :], other=0.0,
            ).to(tl.float32)
            before = x + y * (e2 + mod)
            numerator += tl.sum((after - before) * (after - before), axis=1)
            denominator += tl.sum(before * before, axis=1)
        eps = 1.1920928955078125e-7
        response = tl.sqrt(numerator) / (tl.sqrt(denominator) + eps)
        partial_offset = tile_id * chunks_per_tile + chunk_id
        tl.store(partial_sum_ptr + partial_offset, tl.sum(tl.where(valid, response, 0.0)))
        tl.store(partial_count_ptr + partial_offset, tl.sum(valid.to(tl.float32)))


    @triton.jit
    def _action_response_partials(
        after_ptr, before_ptr, partial_sum_ptr, partial_count_ptr,
        total_t: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
        memory: tl.constexpr, current_offset: tl.constexpr,
        spatial_h: tl.constexpr, spatial_w: tl.constexpr,
        tile_t: tl.constexpr, tile_h: tl.constexpr, tile_w: tl.constexpr,
        channels: tl.constexpr, chunks_per_tile: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        chunk_id = tl.program_id(1)
        spatial = spatial_h * spatial_w
        time_block = tile_id // spatial
        spatial_id = tile_id - time_block * spatial
        block_h = spatial_id // spatial_w
        block_w = spatial_id - block_h * spatial_w
        slots = chunk_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        local_plane = tile_h * tile_w
        dt = slots // local_plane
        local = slots - dt * local_plane
        dh = local // tile_w
        dw = local - dh * tile_w
        current_t = current_offset + time_block * tile_t + dt
        frame = memory + current_t
        h = block_h * tile_h + dh
        w = block_w * tile_w + dw
        valid = (
            (dt < tile_t) & (frame < total_t) & (h < height) & (w < width)
        )
        token = (frame * height + h) * width + w
        numerator = tl.zeros((BLOCK_TOKENS,), tl.float32)
        denominator = tl.zeros((BLOCK_TOKENS,), tl.float32)
        for channel_start in range(0, channels, BLOCK_C):
            channel = channel_start + tl.arange(0, BLOCK_C)
            channel_valid = channel < channels
            offsets = token[:, None] * channels + channel[None, :]
            mask = valid[:, None] & channel_valid[None, :]
            after = tl.load(after_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            before = tl.load(before_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            numerator += tl.sum((after - before) * (after - before), axis=1)
            denominator += tl.sum(before * before, axis=1)
        eps = 1.1920928955078125e-7
        response = tl.sqrt(numerator) / (tl.sqrt(denominator) + eps)
        partial_offset = tile_id * chunks_per_tile + chunk_id
        tl.store(partial_sum_ptr + partial_offset, tl.sum(tl.where(valid, response, 0.0)))
        tl.store(partial_count_ptr + partial_offset, tl.sum(valid.to(tl.float32)))


    @triton.jit
    def _finish_tile_means(
        partial_sum_ptr, partial_count_ptr, output_ptr,
        chunks_per_tile: tl.constexpr, BLOCK_CHUNKS: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        chunk = tl.arange(0, BLOCK_CHUNKS)
        mask = chunk < chunks_per_tile
        offsets = tile_id * chunks_per_tile + chunk
        total = tl.sum(tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0))
        count = tl.sum(tl.load(partial_count_ptr + offsets, mask=mask, other=0.0))
        tl.store(output_ptr + tile_id, total / tl.maximum(count, 1.0))


def _layout(
    tensor: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    memory: int,
    block_shape: tuple[int, int, int],
    temporal_blocks: int,
) -> tuple[int, ...]:
    if triton is None or not tensor.is_cuda or tensor.shape[0] != 1:
        raise RuntimeError("fused response kernel requires Triton CUDA batch=1")
    if not tensor.is_contiguous():
        raise RuntimeError("fused response kernel requires contiguous tensors")
    total_t, height, width = grid
    tile_t, tile_h, tile_w = block_shape
    spatial_h = math.ceil(height / tile_h)
    spatial_w = math.ceil(width / tile_w)
    current_offset = math.ceil(memory / tile_t) * tile_t - memory
    # Four tokens x 256 channels keeps each reduction program near 1K lanes;
    # this avoids fully unrolling 160 channel iterations at the Matrix width.
    chunks = math.ceil(tile_t * tile_h * tile_w / 4)
    tiles = temporal_blocks * spatial_h * spatial_w
    return (
        total_t, height, width, tile_t, tile_h, tile_w, spatial_h,
        spatial_w, current_offset, chunks, tiles, tensor.shape[-1],
    )


def _finish(partials: torch.Tensor, counts: torch.Tensor, tiles: int, chunks: int):
    output = torch.empty(tiles, device=partials.device, dtype=torch.float32)
    _finish_tile_means[(tiles,)](
        partials, counts, output,
        chunks_per_tile=chunks,
        BLOCK_CHUNKS=triton.next_power_of_2(chunks),
    )
    return output


def fused_camera_response_tiles(
    after: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    e: torch.Tensor,
    modulation: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    memory: int,
    block_shape: tuple[int, int, int],
    temporal_blocks: int,
) -> torch.Tensor:
    layout = _layout(
        after, grid=grid, memory=memory, block_shape=block_shape,
        temporal_blocks=temporal_blocks,
    )
    (total_t, height, width, tt, th, tw, sh, sw, offset, chunks, tiles, channels) = layout
    partials = torch.empty((tiles, chunks), device=after.device, dtype=torch.float32)
    counts = torch.empty_like(partials)
    _camera_response_partials[(tiles, chunks)](
        after, x, y, e, modulation, partials, counts,
        total_t=total_t, height=height, width=width, memory=memory,
        current_offset=offset, temporal_blocks=temporal_blocks,
        spatial_h=sh, spatial_w=sw, tile_t=tt, tile_h=th, tile_w=tw,
        channels=channels, chunks_per_tile=chunks, BLOCK_TOKENS=4, BLOCK_C=256,
        num_warps=4,
    )
    return _finish(partials, counts, tiles, chunks).reshape(temporal_blocks, sh * sw)


def fused_action_response_tiles(
    after: torch.Tensor,
    before: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    memory: int,
    block_shape: tuple[int, int, int],
    temporal_blocks: int,
) -> torch.Tensor:
    layout = _layout(
        after, grid=grid, memory=memory, block_shape=block_shape,
        temporal_blocks=temporal_blocks,
    )
    (total_t, height, width, tt, th, tw, sh, sw, offset, chunks, tiles, channels) = layout
    partials = torch.empty((tiles, chunks), device=after.device, dtype=torch.float32)
    counts = torch.empty_like(partials)
    _action_response_partials[(tiles, chunks)](
        after, before, partials, counts,
        total_t=total_t, height=height, width=width, memory=memory,
        current_offset=offset, spatial_h=sh, spatial_w=sw,
        tile_t=tt, tile_h=th, tile_w=tw, channels=channels,
        chunks_per_tile=chunks, BLOCK_TOKENS=4, BLOCK_C=256, num_warps=4,
    )
    return _finish(partials, counts, tiles, chunks).reshape(temporal_blocks, sh * sw)
