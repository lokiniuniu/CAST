"""Opt-in ahead-of-time warmup for Matrix's persistent serial VAE decoder.

The released pipeline runs the first clip with the native decoder and lazily
compiles the decoder at the second clip.  This helper preserves that dispatch
order for real samples while compiling the recurrent 4/4/2 segment shapes
during persistent-worker setup.  Dummy state is isolated from every real VAE
cache and no generated tensor is reused.
"""

from __future__ import annotations

import gc
import contextlib
import time
from typing import Any

import torch


def _module_state_signature(module: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    rows = []
    for kind, values in (("parameter", module.named_parameters()),
                         ("buffer", module.named_buffers())):
        for name, value in values:
            rows.append(
                (
                    kind,
                    name,
                    tuple(value.shape),
                    tuple(value.stride()),
                    str(value.dtype),
                    str(value.device),
                    int(value.data_ptr()),
                    int(value._version),
                )
            )
    return tuple(rows)


class MatrixVAEDecoderAOTWarmup:
    """Precompile the recurrent decoder without changing real-call semantics."""

    def __init__(
        self,
        pipeline: Any,
        *,
        output_height: int,
        output_width: int,
        first_latent_frames: int,
        recurrent_latent_frames: int,
        segment_size: int,
        warmup_no_grad: bool = False,
        compile_mode: str | None = None,
    ) -> None:
        if pipeline.use_async_vae:
            raise RuntimeError("VAE AOT warmup requires serial VAE execution")
        if pipeline.rank != 0 or pipeline.vae is None:
            raise RuntimeError("VAE AOT warmup requires the rank-0 VAE")
        if min(
            output_height,
            output_width,
            first_latent_frames,
            recurrent_latent_frames,
            segment_size,
        ) <= 0:
            raise ValueError("VAE AOT warmup geometry must be positive")
        if recurrent_latent_frames % segment_size == 0:
            raise ValueError(
                "warmup must cover a recurrent tail segment as well as full segments"
            )
        stride_t, stride_h, stride_w = (
            int(value) for value in pipeline.vae_stride
        )
        if output_height % stride_h or output_width % stride_w:
            raise ValueError("output geometry is not aligned to the VAE stride")
        latent_h = output_height // stride_h
        latent_w = output_width // stride_w
        vae = pipeline.vae
        model = vae.model
        native_decoder = model.decoder
        if hasattr(native_decoder, "_is_compiled"):
            raise RuntimeError("VAE decoder was compiled before AOT warmup")
        latent_channels = int(getattr(model, "z_dim", 0))
        if latent_channels <= 0:
            raise RuntimeError("VAE decoder does not expose a latent width")

        self.pipeline = pipeline
        self.vae = vae
        self.native_stream_decode = vae.stream_decode
        self.native_decoder = native_decoder
        self.compiled_decoder: torch.nn.Module | None = None
        self.real_stream_calls = 0
        self.real_native_calls = 0
        self.real_compiled_calls = 0
        self.real_call_records: list[dict[str, Any]] = []
        self.report: dict[str, Any] = {
            "enabled": True,
            "status": "running",
            "output_mutation": False,
            "real_dispatch_contract": (
                "first_worker_sample_clip0_native_then_all_calls_compiled"
            ),
            "dummy_cache_isolated": True,
            "compile_cost_in_generation_timer": False,
            "setup_cost_reported_separately": True,
            "segment_size": int(segment_size),
            "warmup_grad_enabled": not bool(warmup_no_grad),
            "compile_mode": compile_mode or "default",
            "first_latent_shape": [
                1, latent_channels, int(first_latent_frames), latent_h, latent_w
            ],
            "recurrent_latent_shape": [
                1,
                latent_channels,
                int(recurrent_latent_frames),
                latent_h,
                latent_w,
            ],
            "vae_stride": [stride_t, stride_h, stride_w],
            "real_call_records": self.real_call_records,
        }

        device = torch.device(vae.device)
        dtype = vae.dtype
        state_before = _module_state_signature(model)
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone()
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        setup_started = time.perf_counter()

        dummy_cache: list[Any] = [None for _ in range(34)]
        dummy_first = torch.zeros(
            self.report["first_latent_shape"], device=device, dtype=dtype
        )
        native_started = time.perf_counter()
        with (
            torch.no_grad()
            if warmup_no_grad
            else contextlib.nullcontext()
        ):
            native_output, dummy_cache = self.native_stream_decode(
                dummy_first,
                dummy_cache,
                first_chunk=True,
                segment_size=int(segment_size),
                compile_decoder=False,
            )
        torch.cuda.synchronize(device)
        native_seconds = time.perf_counter() - native_started
        if native_output is None or not any(item is not None for item in dummy_cache):
            raise RuntimeError("native VAE warmup did not produce output/cache state")

        compile_started = time.perf_counter()
        compile_kwargs: dict[str, Any] = {
            "dynamic": False,
            "fullgraph": False,
        }
        if compile_mode is not None:
            compile_kwargs["mode"] = compile_mode
        compiled_decoder = torch.compile(native_decoder, **compile_kwargs)
        compiled_decoder._is_compiled = True
        model.decoder = compiled_decoder
        dummy_recurrent = torch.zeros(
            self.report["recurrent_latent_shape"], device=device, dtype=dtype
        )
        with (
            torch.no_grad()
            if warmup_no_grad
            else contextlib.nullcontext()
        ):
            compiled_output, dummy_cache = self.native_stream_decode(
                dummy_recurrent,
                dummy_cache,
                first_chunk=False,
                segment_size=int(segment_size),
                compile_decoder=False,
            )
        torch.cuda.synchronize(device)
        compiled_seconds = time.perf_counter() - compile_started
        if compiled_output is None:
            raise RuntimeError("compiled VAE warmup did not produce output")

        model.decoder = native_decoder
        self.compiled_decoder = compiled_decoder
        vae.stream_decode = self._stream_decode

        del native_output, compiled_output, dummy_first, dummy_recurrent
        dummy_cache.clear()
        gc.collect()
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - setup_started
        state_after = _module_state_signature(model)
        cpu_rng_after = torch.random.get_rng_state()
        cuda_rng_after = torch.cuda.get_rng_state(device)
        self.report.update(
            {
                "status": "ready",
                "setup_wall_seconds": float(setup_seconds),
                "native_dummy_wall_seconds": float(native_seconds),
                "compiled_dummy_wall_seconds": float(compiled_seconds),
                "parameter_buffer_state_unchanged": state_before == state_after,
                "cpu_rng_unchanged": bool(torch.equal(cpu_rng_before, cpu_rng_after)),
                "cuda_rng_unchanged": bool(
                    torch.equal(cuda_rng_before, cuda_rng_after)
                ),
                "allocated_before_bytes": allocated_before,
                "allocated_after_cleanup_bytes": int(
                    torch.cuda.memory_allocated(device)
                ),
                "reserved_before_bytes": reserved_before,
                "reserved_after_cleanup_bytes": int(
                    torch.cuda.memory_reserved(device)
                ),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
            }
        )
        if not (
            self.report["parameter_buffer_state_unchanged"]
            and self.report["cpu_rng_unchanged"]
            and self.report["cuda_rng_unchanged"]
        ):
            raise RuntimeError("VAE AOT warmup mutated model or RNG state")

    def _stream_decode(self, *args: Any, **kwargs: Any):
        first_chunk = bool(kwargs.get("first_chunk", False))
        compile_requested = bool(kwargs.get("compile_decoder", False))
        use_native = self.real_stream_calls == 0
        if use_native and not first_chunk:
            raise RuntimeError("first real VAE AOT call must be the first clip")
        if use_native and compile_requested:
            raise RuntimeError("first real VAE clip must retain native dispatch")
        if self.compiled_decoder is None:
            raise RuntimeError("VAE AOT compiled decoder is unavailable")
        self.vae.model.decoder = (
            self.native_decoder if use_native else self.compiled_decoder
        )
        started = time.perf_counter()
        result = self.native_stream_decode(
            *args, **{**kwargs, "compile_decoder": False}
        )
        if not isinstance(result, tuple) or not result or result[0] is None:
            raise RuntimeError("VAE AOT real decode failed")
        self.real_stream_calls += 1
        self.real_native_calls += int(use_native)
        self.real_compiled_calls += int(not use_native)
        self.real_call_records.append(
            {
                "call": self.real_stream_calls - 1,
                "first_chunk": first_chunk,
                "dispatch": "native" if use_native else "compiled",
                "compile_requested_by_pipeline": compile_requested,
                "wall_seconds": float(time.perf_counter() - started),
            }
        )
        self.report.update(
            {
                "real_stream_calls": self.real_stream_calls,
                "real_native_calls": self.real_native_calls,
                "real_compiled_calls": self.real_compiled_calls,
                "first_real_native": self.real_native_calls == 1,
                "no_real_lazy_compile": True,
            }
        )
        return result

    def summary(self) -> dict[str, Any]:
        return dict(self.report)
