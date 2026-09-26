"""Triton primitives for the frozen Matrix FC-PASM arithmetic.

These kernels are opt-in candidates.  The default FC-R implementation remains
the PyTorch reference until production-shape numerical and latency gates pass.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fc_phat_peak_subpixel(
    correlation_ptr,
    displacement_ptr,
    peak_confidence_ptr,
    spatial: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse tiny PHAT argmax, periodic neighbours and parabola fit."""

    patch = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < spatial
    values = tl.load(
        correlation_ptr + patch * spatial + offsets,
        mask=valid,
        other=-float("inf"),
    )
    flat_index = tl.argmax(values, axis=0, tie_break_left=True)
    center = tl.max(values, axis=0)
    peak_y = flat_index // width
    peak_x = flat_index - peak_y * width
    x_left_index = peak_y * width + (peak_x + width - 1) % width
    x_right_index = peak_y * width + (peak_x + 1) % width
    y_up_index = ((peak_y + spatial // width - 1) % (spatial // width)) * width + peak_x
    y_down_index = ((peak_y + 1) % (spatial // width)) * width + peak_x
    base = correlation_ptr + patch * spatial
    x_left = tl.load(base + x_left_index)
    x_right = tl.load(base + x_right_index)
    y_up = tl.load(base + y_up_index)
    y_down = tl.load(base + y_down_index)

    denom_x = x_left - 2.0 * center + x_right
    denom_y = y_up - 2.0 * center + y_down
    offset_x = tl.where(
        tl.abs(denom_x) > 1e-8,
        tl.maximum(-0.5, tl.minimum(0.5, 0.5 * (x_left - x_right) / denom_x)),
        0.0,
    )
    offset_y = tl.where(
        tl.abs(denom_y) > 1e-8,
        tl.maximum(-0.5, tl.minimum(0.5, 0.5 * (y_up - y_down) / denom_y)),
        0.0,
    )
    signed_x = tl.where(peak_x >= (width + 1) // 2, peak_x - width, peak_x)
    height = spatial // width
    signed_y = tl.where(peak_y >= (height + 1) // 2, peak_y - height, peak_y)
    mean_abs = tl.sum(tl.where(valid, tl.abs(values), 0.0), axis=0) / spatial
    tl.store(displacement_ptr + patch * 2, signed_x.to(tl.float32) + offset_x)
    tl.store(displacement_ptr + patch * 2 + 1, signed_y.to(tl.float32) + offset_y)
    tl.store(peak_confidence_ptr + patch, center / tl.maximum(mean_abs, 1e-8))


def fc_phat_peak_subpixel(
    correlation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused production-shape PHAT peak/subpixel post-processing."""

    if (
        correlation.device.type != "cuda"
        or correlation.dtype != torch.float32
        or correlation.ndim != 3
        or not correlation.is_contiguous()
        or tuple(correlation.shape[1:]) not in {(11, 10), (6, 4)}
    ):
        raise RuntimeError(
            "FC PHAT postprocess requires contiguous CUDA float32 [P,11,10] or [P,6,4]"
        )
    patches, height, width = correlation.shape
    displacement = torch.empty(
        (patches, 2), device=correlation.device, dtype=torch.float32
    )
    peak_confidence = torch.empty(
        (patches,), device=correlation.device, dtype=torch.float32
    )
    _fc_phat_peak_subpixel[(patches,)](
        correlation,
        displacement,
        peak_confidence,
        spatial=height * width,
        width=width,
        BLOCK=128 if height * width > 64 else 32,
        num_warps=4,
        num_stages=1,
    )
    return displacement, peak_confidence


@triton.jit
def _fc_ramp_confidence(
    unit_ptr,
    displacement_ptr,
    weight_ptr,
    denominator_ptr,
    output_ptr,
    height: tl.constexpr,
    width_freq: tl.constexpr,
    spatial_width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    spatial = height * width_freq
    valid = offsets < spatial
    y = offsets // width_freq
    x = offsets - y * width_freq
    signed_y = tl.where(y <= height // 2, y, y - height).to(tl.float32)
    dx = tl.load(displacement_ptr + tile * 2)
    dy = tl.load(displacement_ptr + tile * 2 + 1)
    angle = -2.0 * 3.141592653589793 * (
        dy * signed_y / height + dx * x.to(tl.float32) / spatial_width
    )
    cosine = libdevice.cos(angle)
    sine = libdevice.sin(angle)
    complex_offset = (tile * spatial + offsets) * 2
    unit_real = tl.load(unit_ptr + complex_offset, mask=valid, other=0.0)
    unit_imag = tl.load(unit_ptr + complex_offset + 1, mask=valid, other=0.0)
    agreement = tl.maximum(
        0.0, tl.minimum(1.0, unit_real * cosine + unit_imag * sine)
    )
    weight = tl.load(weight_ptr + tile * spatial + offsets, mask=valid, other=0.0)
    numerator = tl.sum(agreement * weight, axis=0)
    denominator = tl.load(denominator_ptr + tile)
    result = tl.maximum(0.0, tl.minimum(1.0, numerator / denominator))
    tl.store(output_ptr + tile, result)


def fc_ramp_confidence(
    unit: torch.Tensor,
    displacement: torch.Tensor,
    confidence_weight: torch.Tensor,
    denominator: torch.Tensor,
    *,
    spatial_width: int,
) -> torch.Tensor:
    """Fuse rigid phase-ramp construction and weighted agreement."""

    if (
        unit.device.type != "cuda"
        or unit.dtype != torch.complex64
        or unit.ndim != 3
        or not unit.is_contiguous()
        or confidence_weight.shape != unit.shape
        or confidence_weight.dtype != torch.float32
        or not confidence_weight.is_contiguous()
        or displacement.shape != (unit.shape[0], 2)
        or displacement.dtype != torch.float32
        or denominator.shape != (unit.shape[0],)
        or denominator.dtype != torch.float32
        or any(
            value.device != unit.device
            for value in (confidence_weight, displacement, denominator)
        )
    ):
        raise RuntimeError("FC ramp confidence received an unsupported layout")
    tiles, height, width_freq = unit.shape
    if (height, width_freq, int(spatial_width)) not in {(11, 6, 10), (6, 3, 4)}:
        raise RuntimeError("FC ramp confidence received an unsupported spectrum")
    output = torch.empty((tiles,), device=unit.device, dtype=torch.float32)
    _fc_ramp_confidence[(tiles,)](
        torch.view_as_real(unit),
        displacement,
        confidence_weight,
        denominator,
        output,
        height=height,
        width_freq=width_freq,
        spatial_width=int(spatial_width),
        BLOCK=128 if height * width_freq > 32 else 32,
        num_warps=4,
        num_stages=1,
    )
    return output


@triton.jit
def _fc_cross_channel_reduce(
    left_ptr,
    right_ptr,
    shared_real_ptr,
    shared_imag_ptr,
    energy_ptr,
    channels: tl.constexpr,
    spatial_freq: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    output_index = tl.program_id(0)
    tile = output_index // spatial_freq
    frequency = output_index - tile * spatial_freq
    channel_offsets = tl.arange(0, BLOCK_C)
    real_sum = tl.zeros((BLOCK_C,), tl.float32)
    imag_sum = tl.zeros((BLOCK_C,), tl.float32)
    energy_sum = tl.zeros((BLOCK_C,), tl.float32)
    for channel_start in range(0, channels, BLOCK_C):
        channel = channel_start + channel_offsets
        mask = channel < channels
        complex_index = (
            (tile * channels + channel) * spatial_freq + frequency
        ) * 2
        left_real = tl.load(left_ptr + complex_index, mask=mask, other=0.0)
        left_imag = tl.load(left_ptr + complex_index + 1, mask=mask, other=0.0)
        right_real = tl.load(right_ptr + complex_index, mask=mask, other=0.0)
        right_imag = tl.load(right_ptr + complex_index + 1, mask=mask, other=0.0)
        cross_real = right_real * left_real + right_imag * left_imag
        cross_imag = right_imag * left_real - right_real * left_imag
        real_sum += cross_real
        imag_sum += cross_imag
        energy_sum += tl.sqrt(
            (right_real * right_real + right_imag * right_imag)
            * (left_real * left_real + left_imag * left_imag)
        )
    tl.store(shared_real_ptr + output_index, tl.sum(real_sum, axis=0))
    tl.store(shared_imag_ptr + output_index, tl.sum(imag_sum, axis=0))
    tl.store(energy_ptr + output_index, tl.sum(energy_sum, axis=0))


def fc_cross_channel_reduce(
    left_spectrum: torch.Tensor,
    right_spectrum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-summed cross spectrum and summed cross magnitude.

    Inputs must be contiguous complex64 ``[P,C,H,Wf]`` tensors.  No fallback
    is hidden in this wrapper: unsupported layouts fail closed.
    """

    if (
        left_spectrum.device.type != "cuda"
        or right_spectrum.device != left_spectrum.device
        or left_spectrum.dtype != torch.complex64
        or right_spectrum.dtype != torch.complex64
        or left_spectrum.ndim != 4
        or right_spectrum.shape != left_spectrum.shape
        or not left_spectrum.is_contiguous()
        or not right_spectrum.is_contiguous()
    ):
        raise RuntimeError(
            "FC cross reduction requires matching contiguous CUDA complex64 [P,C,H,Wf]"
        )
    tiles, channels, height, width_freq = left_spectrum.shape
    if channels != 3072 or (height, width_freq) not in {(11, 6), (6, 3)}:
        raise RuntimeError(
            "FC cross reduction only supports Matrix C3072 and validated spectra"
        )
    shape = (tiles, height, width_freq)
    shared_real = torch.empty(shape, device=left_spectrum.device, dtype=torch.float32)
    shared_imag = torch.empty_like(shared_real)
    energy = torch.empty_like(shared_real)
    left_real_view = torch.view_as_real(left_spectrum)
    right_real_view = torch.view_as_real(right_spectrum)
    grid = (tiles * height * width_freq,)
    _fc_cross_channel_reduce[grid](
        left_real_view,
        right_real_view,
        shared_real,
        shared_imag,
        energy,
        channels=channels,
        spatial_freq=height * width_freq,
        BLOCK_C=1024,
        num_warps=8,
        num_stages=2,
    )
    return torch.complex(shared_real, shared_imag), energy


@triton.jit
def _fc_phase_mix(
    left_ptr,
    right_ptr,
    theta_ptr,
    affinity_ptr,
    output_ptr,
    alpha,
    elements: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width_freq: tl.constexpr,
    spatial_freq: tl.constexpr,
    stride_lp,
    stride_lc,
    stride_lh,
    stride_lw,
    stride_rp,
    stride_rc,
    stride_rh,
    stride_rw,
    ACCURATE_MATH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    frequency = offsets % spatial_freq
    tile = offsets // (channels * spatial_freq)
    channel = (offsets // spatial_freq) % channels
    frequency_y = frequency // width_freq
    frequency_x = frequency - frequency_y * width_freq
    control_offset = tile * spatial_freq + frequency
    theta = tl.load(theta_ptr + control_offset, mask=mask, other=0.0)
    affinity = tl.load(affinity_ptr + control_offset, mask=mask, other=0.0)
    left_angle = affinity * alpha * theta
    right_angle = -affinity * (1.0 - alpha) * theta
    if ACCURATE_MATH:
        # PyTorch's CUDA complex-polar path uses libdevice-quality
        # transcendental functions.  Triton's default tl.sin/tl.cos lower to
        # fast approximate instructions; their small phase error is amplified
        # by recurrent Matrix rollouts.  Keep this opt-in so the existing fast
        # candidate remains an independent ablation.
        left_cos = libdevice.cos(left_angle)
        left_sin = libdevice.sin(left_angle)
        right_cos = libdevice.cos(right_angle)
        right_sin = libdevice.sin(right_angle)
    else:
        left_cos = tl.cos(left_angle)
        left_sin = tl.sin(left_angle)
        right_cos = tl.cos(right_angle)
        right_sin = tl.sin(right_angle)
    left_offsets = (
        tile * stride_lp
        + channel * stride_lc
        + frequency_y * stride_lh
        + frequency_x * stride_lw
    )
    right_offsets = (
        tile * stride_rp
        + channel * stride_rc
        + frequency_y * stride_rh
        + frequency_x * stride_rw
    )
    left_real = tl.load(left_ptr + left_offsets, mask=mask, other=0.0)
    left_imag = tl.load(left_ptr + left_offsets + 1, mask=mask, other=0.0)
    right_real = tl.load(right_ptr + right_offsets, mask=mask, other=0.0)
    right_imag = tl.load(right_ptr + right_offsets + 1, mask=mask, other=0.0)
    one_minus_alpha = 1.0 - alpha
    output_real = one_minus_alpha * (
        left_real * left_cos - left_imag * left_sin
    ) + alpha * (right_real * right_cos - right_imag * right_sin)
    output_imag = one_minus_alpha * (
        left_real * left_sin + left_imag * left_cos
    ) + alpha * (right_real * right_sin + right_imag * right_cos)
    output_offsets = offsets * 2
    tl.store(output_ptr + output_offsets, output_real, mask=mask)
    tl.store(output_ptr + output_offsets + 1, output_imag, mask=mask)


def fc_phase_mix(
    left_spectrum: torch.Tensor,
    right_spectrum: torch.Tensor,
    theta: torch.Tensor,
    affinity: torch.Tensor,
    alpha: float,
    *,
    accurate_math: bool = False,
) -> torch.Tensor:
    """Fuse the two phase rotations, complex products, scales, and add."""

    if (
        left_spectrum.device.type != "cuda"
        or right_spectrum.device != left_spectrum.device
        or left_spectrum.dtype != torch.complex64
        or right_spectrum.dtype != torch.complex64
        or left_spectrum.ndim != 4
        or right_spectrum.shape != left_spectrum.shape
    ):
        raise RuntimeError(
            "FC phase mix requires matching CUDA complex64 [P,C,H,Wf]"
        )
    tiles, channels, height, width_freq = left_spectrum.shape
    if (
        channels != 3072
        or theta.shape != (tiles, height, width_freq)
        or affinity.shape != theta.shape
        or theta.dtype != torch.float32
        or affinity.dtype != torch.float32
        or theta.device != left_spectrum.device
        or affinity.device != left_spectrum.device
        or not theta.is_contiguous()
        or not affinity.is_contiguous()
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise RuntimeError("FC phase mix metadata/layout contract failed")
    output = torch.empty_like(left_spectrum)
    if not output.is_contiguous():
        output = torch.empty(
            left_spectrum.shape,
            device=left_spectrum.device,
            dtype=left_spectrum.dtype,
        )
    elements = left_spectrum.numel()
    left_real = torch.view_as_real(left_spectrum)
    right_real = torch.view_as_real(right_spectrum)
    _fc_phase_mix[(triton.cdiv(elements, 256),)](
        left_real,
        right_real,
        theta,
        affinity,
        torch.view_as_real(output),
        alpha=float(alpha),
        elements=elements,
        channels=channels,
        height=height,
        width_freq=width_freq,
        spatial_freq=height * width_freq,
        stride_lp=left_real.stride(0),
        stride_lc=left_real.stride(1),
        stride_lh=left_real.stride(2),
        stride_lw=left_real.stride(3),
        stride_rp=right_real.stride(0),
        stride_rc=right_real.stride(1),
        stride_rh=right_real.stride(2),
        stride_rw=right_real.stride(3),
        ACCURATE_MATH=bool(accurate_math),
        BLOCK=256,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _fc_phase_mix_batched(
    anchors_ptr,
    left_index_ptr,
    right_index_ptr,
    alpha_ptr,
    theta_ptr,
    affinity_ptr,
    output_ptr,
    elements,
    tiles: tl.constexpr,
    channels: tl.constexpr,
    spatial_freq: tl.constexpr,
    width_freq: tl.constexpr,
    anchor_stride_exact,
    anchor_stride_tile,
    anchor_stride_channel,
    anchor_stride_height,
    anchor_stride_width,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    frequency = offsets % spatial_freq
    tile = (offsets // (channels * spatial_freq)) % tiles
    target = offsets // (tiles * channels * spatial_freq)
    channel = (offsets // spatial_freq) % channels
    control_offset = (target * tiles + tile) * spatial_freq + frequency
    alpha = tl.load(alpha_ptr + target, mask=mask, other=0.0)
    theta = tl.load(theta_ptr + control_offset, mask=mask, other=0.0)
    affinity = tl.load(affinity_ptr + control_offset, mask=mask, other=0.0)
    left_anchor = tl.load(left_index_ptr + target, mask=mask, other=0)
    right_anchor = tl.load(right_index_ptr + target, mask=mask, other=0)
    freq_y = frequency // width_freq
    freq_x = frequency - freq_y * width_freq
    within_anchor = (
        tile * anchor_stride_tile
        + channel * anchor_stride_channel
        + freq_y * anchor_stride_height
        + freq_x * anchor_stride_width
    )
    left_offset = (left_anchor * anchor_stride_exact + within_anchor) * 2
    right_offset = (right_anchor * anchor_stride_exact + within_anchor) * 2
    left_real = tl.load(anchors_ptr + left_offset, mask=mask, other=0.0)
    left_imag = tl.load(anchors_ptr + left_offset + 1, mask=mask, other=0.0)
    right_real = tl.load(anchors_ptr + right_offset, mask=mask, other=0.0)
    right_imag = tl.load(anchors_ptr + right_offset + 1, mask=mask, other=0.0)
    left_angle = affinity * alpha * theta
    right_angle = -affinity * (1.0 - alpha) * theta
    left_cos, left_sin = tl.cos(left_angle), tl.sin(left_angle)
    right_cos, right_sin = tl.cos(right_angle), tl.sin(right_angle)
    output_real = (1.0 - alpha) * (
        left_real * left_cos - left_imag * left_sin
    ) + alpha * (right_real * right_cos - right_imag * right_sin)
    output_imag = (1.0 - alpha) * (
        left_real * left_sin + left_imag * left_cos
    ) + alpha * (right_real * right_sin + right_imag * right_cos)
    output_offset = offsets * 2
    tl.store(output_ptr + output_offset, output_real, mask=mask)
    tl.store(output_ptr + output_offset + 1, output_imag, mask=mask)


def fc_phase_mix_batched(
    anchor_spectra: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    alphas: torch.Tensor,
    theta: torch.Tensor,
    affinity: torch.Tensor,
    *,
    block_size: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fuse every target mix into one launch and read anchors directly."""

    if (
        anchor_spectra.device.type != "cuda"
        or anchor_spectra.dtype != torch.complex64
        or anchor_spectra.ndim != 5
    ):
        raise RuntimeError("batched FC mix requires CUDA complex64 [E,P,C,H,Wf]")
    exact, tiles, channels, height, width_freq = anchor_spectra.shape
    targets = int(alphas.numel())
    expected_control = (targets, tiles, height, width_freq)
    if (
        channels != 3072
        or left_indices.shape != (targets,)
        or right_indices.shape != (targets,)
        or alphas.shape != (targets,)
        or theta.shape != expected_control
        or affinity.shape != expected_control
        or left_indices.dtype not in {torch.int32, torch.int64}
        or right_indices.dtype != left_indices.dtype
        or alphas.dtype != torch.float32
        or theta.dtype != torch.float32
        or affinity.dtype != torch.float32
        or any(
            tensor.device != anchor_spectra.device
            for tensor in (left_indices, right_indices, alphas, theta, affinity)
        )
    ):
        raise RuntimeError("batched FC mix metadata contract failed")
    if block_size not in {256, 512, 1024} or num_warps not in {4, 8}:
        raise RuntimeError("unsupported batched FC mix launch configuration")
    output = torch.empty(
        expected_control[:2] + (channels, height, width_freq),
        device=anchor_spectra.device,
        dtype=torch.complex64,
    )
    elements = output.numel()
    anchor_strides = tuple(int(value) for value in anchor_spectra.stride())
    if any(value <= 0 for value in anchor_strides):
        raise RuntimeError("batched FC mix requires positive anchor strides")
    _fc_phase_mix_batched[(triton.cdiv(elements, block_size),)](
        torch.view_as_real(anchor_spectra),
        left_indices,
        right_indices,
        alphas,
        theta,
        affinity,
        torch.view_as_real(output),
        elements=elements,
        tiles=tiles,
        channels=channels,
        spatial_freq=height * width_freq,
        width_freq=width_freq,
        anchor_stride_exact=anchor_strides[0],
        anchor_stride_tile=anchor_strides[1],
        anchor_stride_channel=anchor_strides[2],
        anchor_stride_height=anchor_strides[3],
        anchor_stride_width=anchor_strides[4],
        BLOCK=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


@triton.jit
def _fc_phase_mix_batched_gathered(
    left_ptr,
    right_ptr,
    theta_ptr,
    affinity_ptr,
    alpha_ptr,
    output_ptr,
    elements,
    tiles: tl.constexpr,
    channels: tl.constexpr,
    spatial_freq: tl.constexpr,
    target_stride,
    BLOCK: tl.constexpr,
):
    """Mix target-major contiguous endpoint spectra.

    The older batched candidate dereferenced the anchor bank with a different
    random anchor index for every target.  This variant deliberately performs
    the two target gathers in the wrapper and lets the kernel stream through
    contiguous [target, tile, channel, frequency] rows.  The arithmetic is
    identical to ``_fc_phase_mix``; only the layout and launch schedule differ.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    frequency = offsets % spatial_freq
    tile = (offsets // (channels * spatial_freq)) % tiles
    target = offsets // (tiles * channels * spatial_freq)
    # Width is not needed for pointer arithmetic: the within-target spectrum
    # is flattened in tile/channel/frequency order.
    channel = (offsets // spatial_freq) % channels
    within = (tile * channels + channel) * spatial_freq + frequency
    control_offset = (target * tiles + tile) * spatial_freq + frequency
    alpha = tl.load(alpha_ptr + target, mask=mask, other=0.0)
    theta = tl.load(theta_ptr + control_offset, mask=mask, other=0.0)
    affinity = tl.load(affinity_ptr + control_offset, mask=mask, other=0.0)
    left_offset = target * target_stride + within * 2
    right_offset = target * target_stride + within * 2
    left_real = tl.load(left_ptr + left_offset, mask=mask, other=0.0)
    left_imag = tl.load(left_ptr + left_offset + 1, mask=mask, other=0.0)
    right_real = tl.load(right_ptr + right_offset, mask=mask, other=0.0)
    right_imag = tl.load(right_ptr + right_offset + 1, mask=mask, other=0.0)
    left_angle = affinity * alpha * theta
    right_angle = -affinity * (1.0 - alpha) * theta
    left_cos, left_sin = tl.cos(left_angle), tl.sin(left_angle)
    right_cos, right_sin = tl.cos(right_angle), tl.sin(right_angle)
    output_real = (1.0 - alpha) * (
        left_real * left_cos - left_imag * left_sin
    ) + alpha * (right_real * right_cos - right_imag * right_sin)
    output_imag = (1.0 - alpha) * (
        left_real * left_sin + left_imag * left_cos
    ) + alpha * (right_real * right_sin + right_imag * right_cos)
    output_offset = offsets * 2
    tl.store(output_ptr + output_offset, output_real, mask=mask)
    tl.store(output_ptr + output_offset + 1, output_imag, mask=mask)


def fc_phase_mix_batched_gathered(
    anchor_spectra: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    alphas: torch.Tensor,
    theta: torch.Tensor,
    affinity: torch.Tensor,
) -> torch.Tensor:
    """Gather endpoint spectra once, then mix all targets in one launch.

    This is an opt-in implementation candidate.  It intentionally keeps the
    original anchor bank untouched and returns the same target-major complex
    layout as ``fc_phase_mix_batched``.
    """
    if (
        anchor_spectra.device.type != "cuda"
        or anchor_spectra.dtype != torch.complex64
        or anchor_spectra.ndim != 5
        or not anchor_spectra.is_contiguous()
    ):
        raise RuntimeError("gathered FC mix requires contiguous [E,P,C,H,Wf]")
    exact, tiles, channels, height, width_freq = anchor_spectra.shape
    targets = int(alphas.numel())
    expected_control = (targets, tiles, height, width_freq)
    if (
        channels != 3072
        or left_indices.shape != (targets,)
        or right_indices.shape != (targets,)
        or alphas.shape != (targets,)
        or theta.shape != expected_control
        or affinity.shape != expected_control
        or left_indices.dtype not in {torch.int32, torch.int64}
        or right_indices.dtype != left_indices.dtype
        or alphas.dtype != torch.float32
        or theta.dtype != torch.float32
        or affinity.dtype != torch.float32
        or any(
            tensor.device != anchor_spectra.device
            for tensor in (left_indices, right_indices, alphas, theta, affinity)
        )
    ):
        raise RuntimeError("gathered FC mix metadata contract failed")
    left = anchor_spectra.index_select(0, left_indices).contiguous()
    right = anchor_spectra.index_select(0, right_indices).contiguous()
    output = torch.empty_like(left)
    elements = output.numel()
    target_stride = tiles * channels * height * width_freq * 2
    # A large flat spectrum has very little control divergence.  Use a
    # 1024-element block to reduce the launch grid (the original per-target
    # kernel used 256 and paid that launch overhead once per target).
    _fc_phase_mix_batched_gathered[(triton.cdiv(elements, 1024),)](
        torch.view_as_real(left),
        torch.view_as_real(right),
        theta,
        affinity,
        alphas,
        torch.view_as_real(output),
        elements=elements,
        tiles=tiles,
        channels=channels,
        spatial_freq=height * width_freq,
        target_stride=target_stride,
        BLOCK=1024,
        num_warps=8,
        num_stages=2,
    )
    return output


@triton.jit
def _fc_extract_overlap_tiles(
    input_ptr,
    window_ptr,
    output_ptr,
    elements,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    grid_w: tl.constexpr,
    patches: tl.constexpr,
    window_h: tl.constexpr,
    window_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    in_stride_e,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_e,
    out_stride_p,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK: tl.constexpr,
):
    """Fused reflect-pad/unfold/window for Matrix FC-PASM tiles.

    Output is contiguous [E, P, C, window_h, window_w].  The index formula
    matches ``F.pad(..., mode='reflect')`` followed by ``F.unfold`` and the
    row-major patch reshape used by the reference implementation.  Only
    address generation and multiplication by the fixed analysis window are
    fused; FFT and phase arithmetic remain unchanged.
    """
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    x = offsets % window_w
    q = offsets // window_w
    y = q % window_h
    q = q // window_h
    channel = q % channels
    q = q // channels
    patch = q % patches
    example = q // patches
    patch_y = patch // grid_w
    patch_x = patch - patch_y * grid_w
    padded_y = patch_y * stride_h + y
    padded_x = patch_x * stride_w + x
    reflected_y = tl.abs(padded_y - pad_h)
    reflected_y = tl.where(
        reflected_y >= height,
        2 * height - reflected_y - 2,
        reflected_y,
    )
    reflected_x = tl.abs(padded_x - pad_w)
    reflected_x = tl.where(
        reflected_x >= width,
        2 * width - reflected_x - 2,
        reflected_x,
    )
    input_offset = (
        example * in_stride_e
        + channel * in_stride_c
        + reflected_y * in_stride_h
        + reflected_x * in_stride_w
    )
    value = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    window = tl.load(window_ptr + y * window_w + x, mask=mask, other=0.0)
    output_offset = (
        example * out_stride_e
        + patch * out_stride_p
        + channel * out_stride_c
        + y * out_stride_h
        + x * out_stride_w
    )
    tl.store(output_ptr + output_offset, value * window, mask=mask)


def fc_extract_overlap_tiles(
    value: torch.Tensor,
    analysis_window: torch.Tensor,
    *,
    window_shape: tuple[int, int] = (11, 10),
    stride: tuple[int, int] = (8, 8),
    padding: tuple[int, int] = (4, 5),
) -> torch.Tensor:
    """Return overlap tiles with the reference reflect/window semantics."""
    if (
        value.device.type != "cuda"
        or value.dtype != torch.float32
        or value.ndim != 4
        or analysis_window.device != value.device
        or analysis_window.dtype != torch.float32
        or analysis_window.shape != window_shape
    ):
        raise RuntimeError("FC tile extraction requires CUDA float32 [E,C,H,W]")
    examples, channels, height, width = (int(v) for v in value.shape)
    window_h, window_w = (int(v) for v in window_shape)
    stride_h, stride_w = (int(v) for v in stride)
    pad_h, pad_w = (int(v) for v in padding)
    if channels != 3072 or (height, width) != (22, 40):
        raise RuntimeError("FC tile extraction only supports production Matrix geometry")
    grid_h = (height + 2 * pad_h - window_h) // stride_h + 1
    grid_w = (width + 2 * pad_w - window_w) // stride_w + 1
    patches = grid_h * grid_w
    # F.unfold returns [E,C*H*W,P]; the reference then transposes P forward
    # and reshapes without materialising it.  Preserve that exact physical
    # layout so cuFFT sees the same batch strides and therefore chooses the
    # same numerical path as the frozen implementation.
    output = torch.empty_strided(
        (examples, patches, channels, window_h, window_w),
        (
            channels * window_h * window_w * patches,
            1,
            window_h * window_w * patches,
            window_w * patches,
            patches,
        ),
        device=value.device,
        dtype=torch.float32,
    )
    elements = output.numel()
    _fc_extract_overlap_tiles[(triton.cdiv(elements, 256),)](
        value,
        analysis_window,
        output,
        elements=elements,
        channels=channels,
        height=height,
        width=width,
        grid_w=grid_w,
        patches=patches,
        window_h=window_h,
        window_w=window_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        in_stride_e=value.stride(0),
        in_stride_c=value.stride(1),
        in_stride_h=value.stride(2),
        in_stride_w=value.stride(3),
        out_stride_e=output.stride(0),
        out_stride_p=output.stride(1),
        out_stride_c=output.stride(2),
        out_stride_h=output.stride(3),
        out_stride_w=output.stride(4),
        BLOCK=256,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _fc_unphase_overlap_tiles(
    tiles_ptr,
    synthesis_ptr,
    normalization_ptr,
    output_ptr,
    elements,
    patches: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    grid_h: tl.constexpr,
    grid_w: tl.constexpr,
    window_h: tl.constexpr,
    window_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    normalization_stride_h,
    normalization_stride_w,
    tile_stride_e,
    tile_stride_p,
    tile_stride_c,
    tile_stride_h,
    tile_stride_w,
    BLOCK: tl.constexpr,
):
    """Direct normalized overlap-add for the fixed FC-PASM geometry."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    # Derive coordinates from a clamped offset.  This makes every pointer
    # below valid even for the inactive lanes in the final program; those
    # lanes are still suppressed by the final store mask.
    safe_offsets = tl.where(mask, offsets, 0)
    x = safe_offsets % width
    q = safe_offsets // width
    y = q % height
    q = q // height
    channel = q % channels
    example = q // channels
    accumulated = tl.zeros((BLOCK,), dtype=tl.float32)
    for patch_y in range(grid_h):
        local_y = y + pad_h - patch_y * stride_h
        valid_y = (local_y >= 0) & (local_y < window_h)
        for patch_x in range(grid_w):
            local_x = x + pad_w - patch_x * stride_w
            valid = mask & valid_y & (local_x >= 0) & (local_x < window_w)
            # Keep pointer arithmetic in-bounds even for masked lanes.  Triton
            # normally guarantees that a masked load is not dereferenced, but
            # constructing negative offsets can still lead to undefined
            # address arithmetic on some backends (and was observed as inf in
            # the first production-shape smoke).  The clamped coordinates are
            # never consumed for invalid lanes because both loads remain
            # masked.
            safe_y = tl.where(valid_y, local_y, 0)
            safe_x = tl.where((local_x >= 0) & (local_x < window_w), local_x, 0)
            patch = patch_y * grid_w + patch_x
            tile_offset = (
                example * tile_stride_e
                + patch * tile_stride_p
                + channel * tile_stride_c
                + safe_y * tile_stride_h
                + safe_x * tile_stride_w
            )
            window_offset = safe_y * window_w + safe_x
            # All addresses are valid after the coordinate clamps above.  Use
            # an arithmetic validity mask instead of masked loads: this avoids
            # backend-specific behavior for masked negative/edge addresses in
            # the nested overlap loops while preserving exact zero contribution
            # for non-overlapping patches.
            tile = tl.where(valid, tl.load(tiles_ptr + tile_offset), 0.0)
            synthesis = tl.where(
                valid, tl.load(synthesis_ptr + window_offset), 0.0
            )
            accumulated += tile * synthesis
    normalization = tl.load(
        normalization_ptr
        + y * normalization_stride_h
        + x * normalization_stride_w,
        mask=mask,
        other=1.0,
    )
    tl.store(output_ptr + offsets, accumulated / normalization, mask=mask)


def fc_unphase_overlap_tiles(
    tiles: torch.Tensor,
    synthesis_window: torch.Tensor,
    synthesis_normalization: torch.Tensor,
    *,
    output_shape: tuple[int, int] = (22, 40),
    window_shape: tuple[int, int] = (11, 10),
    stride: tuple[int, int] = (8, 8),
    padding: tuple[int, int] = (4, 5),
) -> torch.Tensor:
    """Direct Triton replacement for the production overlap-add path."""
    if (
        tiles.device.type != "cuda"
        or tiles.dtype != torch.float32
        or tiles.ndim != 5
        or synthesis_window.device != tiles.device
        or synthesis_window.dtype != torch.float32
        or synthesis_window.shape != window_shape
        or synthesis_normalization.device != tiles.device
        or synthesis_normalization.dtype != torch.float32
        or synthesis_normalization.shape[-2:] != output_shape
    ):
        raise RuntimeError("FC OLA kernel layout contract failed")
    examples, patches, channels, window_h, window_w = (
        int(v) for v in tiles.shape
    )
    height, width = (int(v) for v in output_shape)
    if (channels, window_h, window_w) != (3072, *window_shape):
        raise RuntimeError("FC OLA kernel only supports production channels/window")
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    grid_h = (height + 2 * pad_h - window_h) // stride_h + 1
    grid_w = (width + 2 * pad_w - window_w) // stride_w + 1
    if patches != grid_h * grid_w:
        raise RuntimeError("FC OLA kernel patch count mismatch")
    normalization = synthesis_normalization.reshape(height, width)
    output = torch.empty(
        (examples, channels, height, width),
        device=tiles.device,
        dtype=torch.float32,
    )
    elements = output.numel()
    _fc_unphase_overlap_tiles[(triton.cdiv(elements, 256),)](
        tiles,
        synthesis_window,
        normalization,
        output,
        elements=elements,
        patches=patches,
        channels=channels,
        height=height,
        width=width,
        grid_h=grid_h,
        grid_w=grid_w,
        window_h=window_h,
        window_w=window_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        normalization_stride_h=normalization.stride(0),
        normalization_stride_w=normalization.stride(1),
        tile_stride_e=tiles.stride(0),
        tile_stride_p=tiles.stride(1),
        tile_stride_c=tiles.stride(2),
        tile_stride_h=tiles.stride(3),
        tile_stride_w=tiles.stride(4),
        BLOCK=256,
        num_warps=4,
        num_stages=2,
    )
    return output
