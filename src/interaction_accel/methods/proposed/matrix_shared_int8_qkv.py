"""Exact shared-input quantization for Matrix compact Q/K/V projections.

The upstream Matrix ``Int8Linear`` quantizes the same activation independently
for q, k, and v.  This helper preserves its quantizer, GEMM, bias, output dtype,
and rounding order while reusing the deterministic input quantization once.
It is intentionally restricted to the three homogeneous upstream Int8Linear
modules and fails closed for every other configuration.
"""

from __future__ import annotations

import importlib
from types import MethodType
from typing import Any

import torch


def _project_from_shared_quantization(
    module: Any,
    x_int8: torch.Tensor,
    x_scales: torch.Tensor,
    x_shape: torch.Size,
    output_dtype: torch.dtype,
    kernels: Any,
) -> torch.Tensor:
    output = kernels.int8_gemm_triton(
        x_int8,
        module.weight_int8,
        x_scales.view(-1),
        module.weight_scales.view(-1),
        output_dtype=torch.bfloat16,
    )
    output = output.view(*x_shape[:-1], int(module.out_features))
    if module.bias is not None:
        output = output + module.bias.to(output.dtype)
    return output.to(output_dtype)


def shared_int8_qkv(
    q_module: Any,
    k_module: Any,
    v_module: Any,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return byte-equivalent upstream q/k/v projections with one quantize."""

    modules = (q_module, k_module, v_module)
    required = (
        "in_features",
        "out_features",
        "weight_int8",
        "weight_scales",
        "bias",
    )
    if x.device.type != "cuda" or not x.is_contiguous():
        raise RuntimeError("shared Matrix QKV requires contiguous CUDA input")
    if x.dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise RuntimeError("shared Matrix QKV received an unsupported input dtype")
    if any(not all(hasattr(module, name) for name in required) for module in modules):
        raise RuntimeError("shared Matrix QKV requires upstream Int8Linear modules")
    in_features = int(q_module.in_features)
    if any(int(module.in_features) != in_features for module in modules):
        raise RuntimeError("shared Matrix QKV input dimensions disagree")
    if int(x.shape[-1]) != in_features or in_features < 16:
        raise RuntimeError("shared Matrix QKV activation shape is unsupported")
    if any(module.weight_int8.dtype != torch.int8 for module in modules):
        raise RuntimeError("shared Matrix QKV weights must be int8")

    upstream = importlib.import_module(type(q_module).__module__)
    get_kernels = getattr(upstream, "_get_triton_kernels", None)
    kernels = get_kernels() if callable(get_kernels) else None
    if kernels is None:
        raise RuntimeError("shared Matrix QKV requires upstream Triton kernels")

    # Match Int8Linear.forward exactly: one contiguous clone followed by the
    # unchanged upstream quantizer.  The original three invocations produce
    # identical quantized activations, so sharing them changes no arithmetic.
    x_shape = x.shape
    x_flat = x.reshape(-1, x_shape[-1]).clone()
    x_int8, x_scales = kernels.int8_quantize_triton(x_flat)
    return tuple(
        _project_from_shared_quantization(
            module,
            x_int8,
            x_scales,
            x_shape,
            x.dtype,
            kernels,
        )
        for module in modules
    )


class SharedInt8QKVForwardCoordinator:
    """Preserve native self-attention while sharing its q/k/v quantization."""

    def __init__(self, self_attention: Any) -> None:
        self.module = self_attention
        self.modules = (self_attention.q, self_attention.k, self_attention.v)
        self.native_forwards = tuple(module.forward for module in self.modules)
        self._input: torch.Tensor | None = None
        self._outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._next = 0

    def install(self) -> None:
        if self._next or self._outputs is not None:
            raise RuntimeError("cannot install an active shared QKV coordinator")
        coordinator = self
        for projection_index, module in enumerate(self.modules):
            def wrapped(
                _module: Any,
                x: torch.Tensor,
                _projection_index: int = projection_index,
            ) -> torch.Tensor:
                return coordinator.project(_projection_index, x)

            module.forward = MethodType(wrapped, module)

    def uninstall(self) -> None:
        for module, native in zip(self.modules, self.native_forwards):
            module.forward = native
        self._input = None
        self._outputs = None
        self._next = 0

    def project(self, projection_index: int, x: torch.Tensor) -> torch.Tensor:
        if projection_index == 0:
            if self._outputs is not None:
                raise RuntimeError("overlapping native self-attention QKV calls")
            self._input = x
            self._outputs = shared_int8_qkv(*self.modules, x.contiguous())
            self._next = 1
            return self._outputs[0]
        if (
            self._outputs is None
            or self._input is not x
            or projection_index != self._next
        ):
            raise RuntimeError("native Matrix self-attention did not call q/k/v in order")
        output = self._outputs[projection_index]
        self._next += 1
        if projection_index == 2:
            self._input = None
            self._outputs = None
            self._next = 0
        return output
