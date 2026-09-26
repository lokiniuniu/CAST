"""Defer serial Matrix VAE decoding until all DiT chunks are complete.

This opt-in source overlay keeps the exact latent tensors, decoder call order,
cache, segment size and streaming video writer of the frozen paper runtime.
Only the placement of those decoder calls changes: the interactive loop first
finishes every DiT chunk, then drains the stored latent references through the
same stateful decoder in chronological order.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

from .matrix_exact_streaming_single_copy_video import (
    _STREAM_FINAL,
    _STREAM_INIT,
)
from .matrix_pinned_async_streaming_single_copy_video import (
    ContiguousPinnedAsyncStreamingSingleCopyWriter,
    FourSlotContiguousPinnedAsyncStreamingSingleCopyWriter,
    FourSlotGpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter,
    FourSlotGpuUint8ViewPinnedAsyncStreamingSingleCopyWriter,
    FourSlotGpuUint8THWCPinnedAsyncStreamingSingleCopyWriter,
    FourSlotDirectGpuTHWCPinnedAsyncStreamingSingleCopyWriter,
    GpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter,
    PinnedAsyncStreamingSingleCopyWriter,
    rewrite_matrix_pipeline_for_pinned_async_streaming_single_copy,
)


_TARGET_MODULE = "pipeline.inference_interactive_pipeline"

_TIMING_INIT = """            total_dit_time = 0.0 
"""
_ASYNC_TIMING_INIT = _TIMING_INIT + """            worldmark_dit_timing_events = []
"""
_TIMING_ITER_START = '''                if self.rank == 0:
                    torch.cuda.synchronize()
                    iter_start_time = time.perf_counter()
                    phase_times = {}
                    t0 = time.perf_counter()
'''
_ASYNC_TIMING_ITER_START = '''                if self.rank == 0:
                    iter_start_time = time.perf_counter()
                    phase_times = {}
                    t0 = time.perf_counter()
'''
_TIMING_ACTION_END = '''                if self.rank == 0:
                    torch.cuda.synchronize()
                    phase_times['1. Action and camera pose setup'] = time.perf_counter() - t0
                    t1 = time.perf_counter()
'''
_ASYNC_TIMING_ACTION_END = '''                if self.rank == 0:
                    phase_times['1. Action and camera pose setup'] = time.perf_counter() - t0
                    t1 = time.perf_counter()
'''
_TIMING_MEMORY_END = '''                if self.rank == 0:
                    torch.cuda.synchronize()
                    phase_times['2. Memory search and feature assembly'] = time.perf_counter() - t1
                    t2 = time.perf_counter()
'''
_ASYNC_TIMING_MEMORY_END = '''                if self.rank == 0:
                    phase_times['2. Memory search and feature assembly'] = time.perf_counter() - t1
                    t2 = time.perf_counter()
                    worldmark_dit_start_event = torch.cuda.Event(enable_timing=True)
                    worldmark_dit_start_event.record()
'''
_TIMING_DIT_END = '''                if self.rank == 0:
                    torch.cuda.synchronize()
                    phase_times['3. DiT denoising loop'] = time.perf_counter() - t2
                    total_dit_time += phase_times['3. DiT denoising loop']
                    t3 = time.perf_counter()
'''
_ASYNC_TIMING_DIT_END = '''                if self.rank == 0:
                    worldmark_dit_end_event = torch.cuda.Event(enable_timing=True)
                    worldmark_dit_end_event.record()
                    worldmark_dit_timing_events.append(
                        (worldmark_dit_start_event, worldmark_dit_end_event)
                    )
                    phase_times['3. DiT denoising loop'] = time.perf_counter() - t2
                    t3 = time.perf_counter()
'''
_TIMING_VAE_END = '''                if self.rank == 0:
                    torch.cuda.synchronize()
                    phase_times['4. VAE decoding and video saving'] = time.perf_counter() - t3
                    
                    total_iter_time = time.perf_counter() - iter_start_time
'''
_ASYNC_TIMING_VAE_END = '''                if self.rank == 0:
                    phase_times['4. VAE decoding and video saving'] = time.perf_counter() - t3
                    
                    total_iter_time = time.perf_counter() - iter_start_time
'''
_TIMING_FINAL = '''            if self.rank == 0:
                print(f"\\n" + "="*50)
                print(f"DiT Core Time: {total_dit_time:.2f} s")
'''
_ASYNC_TIMING_FINAL = '''            if self.rank == 0:
                torch.cuda.synchronize()
                total_dit_time = sum(
                    start.elapsed_time(end) / 1000.0
                    for start, end in worldmark_dit_timing_events
                )
                self._worldmark_async_stage_timing_runtime = {
                    "enabled": True,
                    "complete": True,
                    "event_pairs": len(worldmark_dit_timing_events),
                    "expected_event_pairs": num_iterations,
                    "per_chunk_global_synchronizations_removed": 5,
                    "final_global_synchronizations": 1,
                    "output_mutation": False,
                }
                print(f"\\n" + "="*50)
                print(f"DiT Core Time: {total_dit_time:.2f} s")
'''

_PINNED_STREAM_INIT = _STREAM_INIT.replace(
    "ExactStreamingSingleCopyWriter(",
    "PinnedAsyncStreamingSingleCopyWriter(",
).replace(
    '"exact_streaming_singlecopy_trace.json"',
    '"pinned_async_streaming_singlecopy_trace.json"',
)

_DEFERRED_STREAM_INIT = _PINNED_STREAM_INIT + """            worldmark_deferred_vae_latents = []
"""

_PINNED_DECODE_CHUNK = '''                        vae_segment_size = int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4"))
                        video, vae_cache = self.vae.stream_decode(
                            denoised_pred_for_vae.to(dtype=self.vae.dtype), 
                            vae_cache, 
                            first_chunk=first_clip, 
                            segment_size=vae_segment_size,
                            compile_decoder=getattr(args, "compile_vae", False) and clip_idx >= 1
                        )
                        # Exact streaming SingleCopy: D2H once on the inference
                        # thread; CPU conversion and persistent encoding run on one
                        # bounded background worker in chronological chunk order.
                        worldmark_streaming_video.submit(video, clip_idx=clip_idx)
                        
'''

_DEFERRED_STREAM_CHUNK = '''                        # Runtime-only scheduling: retain the exact latent tensor
                        # reference and execute the unchanged stateful VAE calls after
                        # every DiT chunk has completed.
                        worldmark_deferred_vae_latents.append(denoised_pred_for_vae)
'''

_DEFERRED_STREAM_FINAL = '''            if not self.use_async_vae and self.rank == 0:
                if worldmark_streaming_video is None:
                    raise RuntimeError("deferred VAE writer was not initialized")
                if len(worldmark_deferred_vae_latents) != num_iterations:
                    raise RuntimeError("deferred VAE latent chronology is incomplete")
                for deferred_clip_idx, deferred_latent in enumerate(
                    worldmark_deferred_vae_latents
                ):
                    deferred_first_clip = deferred_clip_idx == 0
                    vae_segment_size = int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4"))
                    video, vae_cache = self.vae.stream_decode(
                        deferred_latent.to(dtype=self.vae.dtype),
                        vae_cache,
                        first_chunk=deferred_first_clip,
                        segment_size=vae_segment_size,
                        compile_decoder=(
                            getattr(args, "compile_vae", False)
                            and deferred_clip_idx >= 1
                        ),
                    )
                    worldmark_streaming_video.submit(
                        video, clip_idx=deferred_clip_idx
                    )
                worldmark_streaming_video.finish()
                self._worldmark_exact_streaming_single_copy_runtime = (
                    worldmark_streaming_video.runtime_summary()
                )
                self._worldmark_deferred_vae_runtime = {
                    "enabled": True,
                    "complete": True,
                    "decoder_calls_deferred": len(worldmark_deferred_vae_latents),
                    "expected_decoder_calls": num_iterations,
                    "chronological": True,
                    "latent_references_not_recomputed": True,
                    "decoder_cache_and_segment_unchanged": True,
                }
                print(f"Saved deferred exact streaming video with {num_iterations} segments")
'''

_COALESCED_STREAM_FINAL = '''            if not self.use_async_vae and self.rank == 0:
                if worldmark_streaming_video is None:
                    raise RuntimeError("coalesced VAE writer was not initialized")
                if len(worldmark_deferred_vae_latents) != num_iterations:
                    raise RuntimeError("coalesced VAE latent chronology is incomplete")
                vae_segment_size = int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4"))
                coalesced_vae_stream_calls = 0
                coalesced_vae_group_sizes = []
                first_latent = worldmark_deferred_vae_latents[0]
                first_video, vae_cache = self.vae.stream_decode(
                    first_latent.to(dtype=self.vae.dtype),
                    vae_cache,
                    first_chunk=True,
                    segment_size=vae_segment_size,
                    compile_decoder=False,
                )
                if first_video is None or int(first_video.shape[2]) != 57:
                    raise RuntimeError("coalesced VAE first chunk shape mismatch")
                worldmark_streaming_video.submit(first_video, clip_idx=0)
                coalesced_vae_stream_calls += 1
                coalesced_vae_group_sizes.append(1)
                deferred_clip_idx = 1
                while deferred_clip_idx < num_iterations:
                    group_end = min(deferred_clip_idx + 2, num_iterations)
                    latent_group = worldmark_deferred_vae_latents[
                        deferred_clip_idx:group_end
                    ]
                    recurrent_latent = (
                        latent_group[0]
                        if len(latent_group) == 1
                        else torch.cat(latent_group, dim=2)
                    )
                    recurrent_video, vae_cache = self.vae.stream_decode(
                        recurrent_latent.to(dtype=self.vae.dtype),
                        vae_cache,
                        first_chunk=False,
                        segment_size=vae_segment_size,
                        compile_decoder=getattr(args, "compile_vae", False),
                    )
                    expected_group_frames = 40 * len(latent_group)
                    if (
                        recurrent_video is None
                        or int(recurrent_video.shape[2]) != expected_group_frames
                    ):
                        raise RuntimeError(
                            "coalesced recurrent VAE output shape mismatch"
                        )
                    for group_offset in range(len(latent_group)):
                        frame_start = 40 * group_offset
                        frame_end = frame_start + 40
                        video_chunk = recurrent_video[
                            :, :, frame_start:frame_end
                        ].contiguous()
                        worldmark_streaming_video.submit(
                            video_chunk,
                            clip_idx=deferred_clip_idx + group_offset,
                        )
                    coalesced_vae_stream_calls += 1
                    coalesced_vae_group_sizes.append(len(latent_group))
                    deferred_clip_idx = group_end
                worldmark_streaming_video.finish()
                self._worldmark_exact_streaming_single_copy_runtime = (
                    worldmark_streaming_video.runtime_summary()
                )
                self._worldmark_deferred_vae_runtime = {
                    "enabled": True,
                    "complete": True,
                    "decoder_calls_deferred": num_iterations,
                    "expected_decoder_calls": num_iterations,
                    "chronological": True,
                    "latent_references_not_recomputed": True,
                    "decoder_cache_and_segment_unchanged": True,
                    "recurrent_pair_coalescing": True,
                    "stream_decode_calls": coalesced_vae_stream_calls,
                    "stream_decode_group_sizes": coalesced_vae_group_sizes,
                    "expected_stream_decode_calls": 1 + (num_iterations - 1 + 1) // 2,
                    "max_recurrent_group_size": 2,
                    "output_chunks_restored": num_iterations,
                }
                print(f"Saved pair-coalesced exact streaming video with {num_iterations} segments")
'''


def rewrite_matrix_pipeline_for_deferred_vae_streaming(
    source: str,
    *,
    coalesce_recurrent_pairs: bool = False,
    async_stage_timing: bool = False,
) -> str:
    rewritten = rewrite_matrix_pipeline_for_pinned_async_streaming_single_copy(
        source
    )
    replacements = (
        (_PINNED_STREAM_INIT, _DEFERRED_STREAM_INIT, "stream initialization"),
        (_PINNED_DECODE_CHUNK, _DEFERRED_STREAM_CHUNK, "per-chunk VAE decode"),
        (
            _STREAM_FINAL,
            (
                _COALESCED_STREAM_FINAL
                if coalesce_recurrent_pairs
                else _DEFERRED_STREAM_FINAL
            ),
            "stream finalization",
        ),
    )
    for native, candidate, label in replacements:
        if rewritten.count(native) != 1:
            raise RuntimeError(
                f"deferred VAE overlay expected one {label} block, found "
                f"{rewritten.count(native)}"
            )
        rewritten = rewritten.replace(native, candidate, 1)
    if async_stage_timing:
        timing_replacements = (
            (_TIMING_INIT, _ASYNC_TIMING_INIT, "timing initialization"),
            (_TIMING_ITER_START, _ASYNC_TIMING_ITER_START, "iteration timing start"),
            (_TIMING_ACTION_END, _ASYNC_TIMING_ACTION_END, "action timing end"),
            (_TIMING_MEMORY_END, _ASYNC_TIMING_MEMORY_END, "memory timing end"),
            (_TIMING_DIT_END, _ASYNC_TIMING_DIT_END, "DiT timing end"),
            (_TIMING_VAE_END, _ASYNC_TIMING_VAE_END, "VAE timing end"),
            (_TIMING_FINAL, _ASYNC_TIMING_FINAL, "final DiT timing"),
        )
        for native, candidate, label in timing_replacements:
            if rewritten.count(native) != 1:
                raise RuntimeError(
                    f"async timing overlay expected one {label} block, found "
                    f"{rewritten.count(native)}"
                )
            rewritten = rewritten.replace(native, candidate, 1)
    return rewritten


def install_matrix_deferred_vae_streaming_video_pipeline(
    matrix_root: Path,
    *,
    contiguous_slots: bool = True,
    coalesce_recurrent_pairs: bool = False,
    async_stage_timing: bool = False,
    pinned_slot_count: int = 2,
    gpu_uint8_delivery: bool = False,
    elide_cpu_thwc_copy: bool = False,
    physical_thwc_layout: bool = False,
    direct_gpu_thwc: bool = False,
) -> None:
    if pinned_slot_count not in {2, 4}:
        raise RuntimeError("deferred VAE pinned slot count must be 2 or 4")
    if pinned_slot_count == 4 and not contiguous_slots:
        raise RuntimeError("four-slot deferred VAE requires contiguous slots")
    if gpu_uint8_delivery and not contiguous_slots:
        raise RuntimeError("GPU uint8 delivery requires contiguous slots")
    if elide_cpu_thwc_copy and not (
        gpu_uint8_delivery and pinned_slot_count == 4
    ):
        raise RuntimeError("CPU THWC copy elision requires four-slot GPU uint8")
    if physical_thwc_layout and not elide_cpu_thwc_copy:
        raise RuntimeError("physical THWC layout requires CPU THWC copy elision")
    if direct_gpu_thwc and not physical_thwc_layout:
        raise RuntimeError("direct GPU THWC requires physical THWC layout")
    if _TARGET_MODULE in sys.modules:
        raise RuntimeError(
            "deferred VAE overlay must be installed before importing "
            f"{_TARGET_MODULE}"
        )
    importlib.import_module("pipeline")
    source_path = matrix_root / "pipeline" / "inference_interactive_pipeline.py"
    rewritten = rewrite_matrix_pipeline_for_deferred_vae_streaming(
        source_path.read_text(encoding="utf-8"),
        coalesce_recurrent_pairs=coalesce_recurrent_pairs,
        async_stage_timing=async_stage_timing,
    )
    module = types.ModuleType(_TARGET_MODULE)
    module.__file__ = str(source_path)
    module.__package__ = "pipeline"
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__.update(
        {
            "PinnedAsyncStreamingSingleCopyWriter": (
                FourSlotDirectGpuTHWCPinnedAsyncStreamingSingleCopyWriter
                if direct_gpu_thwc
                else (
                FourSlotGpuUint8THWCPinnedAsyncStreamingSingleCopyWriter
                if physical_thwc_layout
                else (
                FourSlotGpuUint8ViewPinnedAsyncStreamingSingleCopyWriter
                if elide_cpu_thwc_copy
                else (
                (
                    GpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter
                    if pinned_slot_count == 2
                    else FourSlotGpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter
                )
                if gpu_uint8_delivery
                else (
                    FourSlotContiguousPinnedAsyncStreamingSingleCopyWriter
                    if pinned_slot_count == 4
                    else (
                        ContiguousPinnedAsyncStreamingSingleCopyWriter
                        if contiguous_slots
                        else PinnedAsyncStreamingSingleCopyWriter
                    )
                )
                )
                )
                )
            ),
            "Path": Path,
        }
    )
    sys.modules[_TARGET_MODULE] = module
    try:
        exec(compile(rewritten, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(_TARGET_MODULE, None)
        raise
