"""Pinned asynchronous D2H delivery for exact Matrix streaming video.

The opt-in runtime owns two maximum-chunk pinned buffers and one dedicated
CUDA copy stream.  A slot is returned only after its copy event, exact CPU
pixel conversion, and chronological writer append have all completed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import queue
import sys
import time
import types
from typing import Any

import torch

from .matrix_exact_streaming_single_copy_video import (
    ExactStreamingSingleCopyWriter,
    native_chunk_pixels,
    rewrite_matrix_pipeline_for_exact_streaming_single_copy,
)


_TARGET_MODULE = "pipeline.inference_interactive_pipeline"


def certify_pinned_async_d2h_runtime(
    runtime: dict[str, Any],
    *,
    expected_chunks: int,
    require_contiguous_slots: bool = False,
    expected_slots: int = 2,
    gpu_uint8_delivery: bool = False,
    cpu_thwc_copy_elided: bool = False,
    physical_thwc_layout: bool = False,
    direct_gpu_thwc: bool = False,
) -> dict[str, Any]:
    expected_chunks = int(expected_chunks)
    expected_slots = int(expected_slots)
    if expected_slots not in {2, 4}:
        raise RuntimeError("pinned-async expected slot count must be 2 or 4")
    chunks = runtime.get("chunks")
    expected_mode = f"{('two' if expected_slots == 2 else 'four')}_slot_pinned_async_copy_stream"
    if runtime.get("d2h_mode") != expected_mode:
        raise RuntimeError("pinned-async D2H mode certificate failed")
    if (
        runtime.get("pinned_slot_count") != expected_slots
        or runtime.get("max_slot_frames") != 57
    ):
        raise RuntimeError("pinned-async slot certificate failed")
    if not isinstance(chunks, list) or len(chunks) != expected_chunks:
        raise RuntimeError("pinned-async chunk records are incomplete")
    for index, row in enumerate(chunks):
        if (
            row.get("clip_idx") != index
            or row.get("slot_idx") not in set(range(expected_slots))
        ):
            raise RuntimeError("pinned-async slot chronology failed")
        if row.get("copy_enqueued_non_blocking") is not True:
            raise RuntimeError("pinned-async copy enqueue certificate failed")
        for name in (
            "slot_wait_s",
            "enqueue_s",
            "copy_event_wait_s",
            "copy_cuda_ms",
            "copy_async_lifetime_s",
            "copy_wait_hidden_s",
            "background_copy_wait_overlap_s",
        ):
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"invalid pinned-async timing {name}")
        ready = row.get("copy_event_ready_on_dequeue")
        if not isinstance(ready, bool):
            raise RuntimeError("missing pinned-async event readiness record")
        if (
            require_contiguous_slots
            and not physical_thwc_layout
            and row.get("destination_contiguous") is not True
        ):
            raise RuntimeError("pinned-async destination view is not contiguous")
        if physical_thwc_layout and row.get("destination_thwc_contiguous") is not True:
            raise RuntimeError("pinned-async THWC destination is not contiguous")
    expected_variant = (
        "pinned_async_four_slot_direct_gpu_thwc"
        if direct_gpu_thwc
        else (
        "pinned_async_four_slot_gpu_uint8_thwc"
        if physical_thwc_layout
        else (
        "pinned_async_four_slot_gpu_uint8_numpy_view"
        if cpu_thwc_copy_elided
        else (
        (
            "pinned_async_contiguous_gpu_uint8_streaming_singlecopy"
            if expected_slots == 2
            else "pinned_async_contiguous_four_slot_gpu_uint8_streaming_singlecopy"
        )
        )
        )
        if gpu_uint8_delivery
        else (
        "pinned_async_contiguous_streaming_singlecopy"
        if expected_slots == 2
        else "pinned_async_contiguous_four_slot_streaming_singlecopy"
        )
        )
    )
    if require_contiguous_slots and runtime.get("variant") != expected_variant:
        raise RuntimeError("contiguous pinned-async runtime variant mismatch")
    result = {
        "complete": True,
        "pinned_slots": expected_slots,
        "copy_streams": 1,
        "copy_events": expected_chunks,
        "slot_reuse_after_encoder_consumed": True,
    }
    if require_contiguous_slots:
        result["contiguous_slot_views"] = True
    if gpu_uint8_delivery:
        if (
            runtime.get("gpu_uint8_delivery") is not True
            or runtime.get("pixel_conversion") != "gpu_fp32_scale_clip_uint8"
        ):
            raise RuntimeError("GPU uint8 delivery certificate failed")
        result["gpu_uint8_delivery"] = True
    if cpu_thwc_copy_elided:
        if runtime.get("cpu_thwc_copy_elided") is not True:
            raise RuntimeError("CPU THWC copy-elision certificate failed")
        result["cpu_thwc_copy_elided"] = True
    if physical_thwc_layout:
        if (
            runtime.get("pinned_physical_layout") != "THWC"
            or runtime.get("destination_thwc_contiguous") is not True
        ):
            raise RuntimeError("physical THWC pinned-layout certificate failed")
        result["physical_thwc_layout"] = True
    if direct_gpu_thwc:
        if runtime.get("direct_gpu_thwc") is not True:
            raise RuntimeError("direct GPU THWC certificate failed")
        result["direct_gpu_thwc"] = True
    return result


class PinnedAsyncStreamingSingleCopyWriter(ExactStreamingSingleCopyWriter):
    """Exact writer with two reusable pinned buffers and async D2H events."""

    _pinned_slot_count = 2

    def __init__(self, *args, **kwargs) -> None:
        self._free_slots: queue.Queue[int] = queue.Queue(
            maxsize=self._pinned_slot_count
        )
        self._slots: list[torch.Tensor] | None = None
        self._copy_stream: torch.cuda.Stream | None = None
        self._slot_height: int | None = None
        self._slot_width: int | None = None
        self._slot_dtype: torch.dtype | None = None
        super().__init__(*args, **kwargs)

    def _ensure_copy_resources(self, video: torch.Tensor) -> None:
        if self._slots is not None:
            expected = (self._slot_dtype, self._slot_height, self._slot_width)
            actual = (video.dtype, int(video.shape[3]), int(video.shape[4]))
            if actual != expected:
                raise RuntimeError(
                    f"decoded video shape/dtype changed after pinned allocation: {actual} != {expected}"
                )
            return
        if video.device.type != "cuda":
            raise RuntimeError("pinned-async D2H requires a CUDA decoded video")
        self._slot_dtype = video.dtype
        self._slot_height = int(video.shape[3])
        self._slot_width = int(video.shape[4])
        shape = (1, 3, 57, self._slot_height, self._slot_width)
        self._slots = [
            torch.empty(shape, dtype=video.dtype, device="cpu", pin_memory=True)
            for _ in range(self._pinned_slot_count)
        ]
        for slot_idx in range(self._pinned_slot_count):
            self._free_slots.put(slot_idx)
        self._copy_stream = torch.cuda.Stream(device=video.device)

    def _get_slot_checked(self) -> tuple[int, float]:
        start = time.perf_counter()
        while True:
            failure = self._get_failure()
            if failure is not None:
                raise RuntimeError("pinned-async video worker failed") from failure
            try:
                return self._free_slots.get(timeout=0.05), time.perf_counter() - start
            except queue.Empty:
                continue

    def _slot_view(self, slot_idx: int, frames: int) -> torch.Tensor:
        assert self._slots is not None
        return self._slots[slot_idx][:, :, :frames]

    def _slot_pixels(self, slot_view: torch.Tensor):
        return native_chunk_pixels(slot_view)

    def submit(self, video: torch.Tensor, *, clip_idx: int) -> None:
        if self._finished:
            raise RuntimeError("cannot submit after pinned-async writer finish")
        if clip_idx != len(self._chunks):
            raise RuntimeError(
                f"non-chronological chunk: expected {len(self._chunks)}, got {clip_idx}"
            )
        expected_chunk_frames = 57 if clip_idx == 0 else 40
        if (
            video.ndim != 5
            or video.shape[0] != 1
            or video.shape[1] != 3
            or int(video.shape[2]) != expected_chunk_frames
        ):
            raise RuntimeError(
                f"chunk {clip_idx} expected [1,3,{expected_chunk_frames},H,W], "
                f"got {tuple(video.shape)}"
            )
        self._ensure_copy_resources(video)
        assert self._slots is not None and self._copy_stream is not None
        slot_idx, slot_wait_s = self._get_slot_checked()
        slot_view = self._slot_view(slot_idx, expected_chunk_frames)
        current_stream = torch.cuda.current_stream(device=video.device)
        enqueue_start = time.perf_counter()
        event_create_start = time.perf_counter()
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        event_create_s = time.perf_counter() - event_create_start
        stream_context_start = time.perf_counter()
        with torch.cuda.stream(self._copy_stream):
            wait_stream_start = time.perf_counter()
            self._copy_stream.wait_stream(current_stream)
            wait_stream_s = time.perf_counter() - wait_stream_start
            copy_start_record_start = time.perf_counter()
            copy_start.record(self._copy_stream)
            copy_start_record_s = time.perf_counter() - copy_start_record_start
            copy_call_start = time.perf_counter()
            slot_view.copy_(video, non_blocking=True)
            copy_call_s = time.perf_counter() - copy_call_start
            copy_end_record_start = time.perf_counter()
            copy_end.record(self._copy_stream)
            copy_end_record_s = time.perf_counter() - copy_end_record_start
        stream_context_s = time.perf_counter() - stream_context_start
        enqueue_s = time.perf_counter() - enqueue_start
        # Keep both the tensor object and allocator stream relationship alive
        # until the consumer synchronizes copy_end.
        record_stream_start = time.perf_counter()
        video.record_stream(self._copy_stream)
        record_stream_s = time.perf_counter() - record_stream_start
        row: dict[str, Any] = {
            "clip_idx": int(clip_idx),
            "frames": expected_chunk_frames,
            "slot_idx": slot_idx,
            "d2h_s": enqueue_s,
            "slot_wait_s": slot_wait_s,
            "enqueue_s": enqueue_s,
            "event_create_s": event_create_s,
            "stream_context_s": stream_context_s,
            "wait_stream_s": wait_stream_s,
            "copy_start_record_s": copy_start_record_s,
            "copy_call_s": copy_call_s,
            "copy_end_record_s": copy_end_record_s,
            "record_stream_s": record_stream_s,
            "source_contiguous": bool(video.is_contiguous()),
            "source_stride": [int(value) for value in video.stride()],
            "destination_contiguous": bool(slot_view.is_contiguous()),
            "destination_thwc_contiguous": bool(
                slot_view[0].permute(1, 2, 3, 0).is_contiguous()
            ),
            "destination_stride": [int(value) for value in slot_view.stride()],
            "queue_wait_s": None,
            "copy_event_wait_s": None,
            "copy_event_ready_on_dequeue": None,
            "copy_cuda_ms": None,
            "copy_async_lifetime_s": None,
            "copy_wait_hidden_s": None,
            "background_copy_wait_overlap_s": None,
            "copy_enqueued_non_blocking": True,
            "convert_s": None,
            "append_s": None,
            "_enqueue_host_time": time.perf_counter(),
        }
        self._chunks.append(row)
        row["queue_wait_s"] = self._put_checked(
            (slot_idx, slot_view, copy_start, copy_end, video, row)
        )

    def _run(self) -> None:
        writer = None
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            writer = self._make_writer()
            self._writer_open_s = time.perf_counter() - start
            self._writer_opened = True
            failed = False
            while True:
                item = self._queue.get()
                try:
                    if item is self._STOP:
                        break
                    slot_idx, slot_view, copy_start, copy_end, source_video, row = item
                    ready = bool(copy_end.query())
                    wait_start = time.perf_counter()
                    copy_end.synchronize()
                    wait_s = time.perf_counter() - wait_start
                    lifetime_s = time.perf_counter() - float(row.pop("_enqueue_host_time"))
                    row["copy_event_ready_on_dequeue"] = ready
                    row["copy_event_wait_s"] = wait_s
                    row["copy_cuda_ms"] = float(copy_start.elapsed_time(copy_end))
                    row["copy_async_lifetime_s"] = lifetime_s
                    row["copy_wait_hidden_s"] = max(0.0, lifetime_s - wait_s)
                    # The synchronize is executed only on the background
                    # consumer, so this wait window is available for foreground
                    # inference progress.  End-to-end GPU timing must determine
                    # how much of the opportunity is realized by the next DiT.
                    row["background_copy_wait_overlap_s"] = wait_s
                    # source_video must remain referenced through synchronize().
                    del source_video
                    if failed:
                        continue
                    start = time.perf_counter()
                    pixels = self._slot_pixels(slot_view)
                    row["convert_s"] = time.perf_counter() - start
                    start = time.perf_counter()
                    try:
                        for frame in pixels:
                            writer.append_data(frame)
                    finally:
                        row["append_s"] = time.perf_counter() - start
                    self._frame_count += int(pixels.shape[0])
                except BaseException as error:
                    failed = True
                    self._set_failure(error)
                finally:
                    if item is not self._STOP:
                        # Reuse is forbidden until event wait, conversion, and
                        # encoder append above have all left the critical path.
                        self._free_slots.put(item[0])
                    self._queue.task_done()
        except BaseException as error:
            self._set_failure(error)
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if item is not self._STOP:
                        item[3].synchronize()
                        self._free_slots.put(item[0])
                finally:
                    self._queue.task_done()
                if item is self._STOP:
                    break
        finally:
            if writer is not None:
                start = time.perf_counter()
                try:
                    writer.close()
                except BaseException as error:
                    self._set_failure(error)
                else:
                    self._writer_closed = True
                self._writer_close_s = time.perf_counter() - start

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload.update(
            {
                "variant": "pinned_async_streaming_singlecopy",
                "d2h_mode": (
                    "two_slot_pinned_async_copy_stream"
                    if self._pinned_slot_count == 2
                    else "four_slot_pinned_async_copy_stream"
                ),
                "pinned_slot_count": self._pinned_slot_count,
                "max_slot_frames": 57,
                "dedicated_copy_streams": 1,
                "slot_reuse_after_encoder_consumed": True,
            }
        )
        return payload


class ContiguousPinnedAsyncStreamingSingleCopyWriter(
    PinnedAsyncStreamingSingleCopyWriter
):
    """Pinned writer whose recurrent 40-frame views remain contiguous.

    Slicing the temporal dimension of a [1,3,57,H,W] slot leaves the original
    57-frame channel stride and therefore creates a non-contiguous 40-frame
    destination.  Taking a flat prefix uses the same allocation and byte
    capacity while producing the exact contiguous [1,3,T,H,W] layout expected
    by the decoded source tensor.
    """

    def _slot_view(self, slot_idx: int, frames: int) -> torch.Tensor:
        assert self._slots is not None
        if self._slot_height is None or self._slot_width is None:
            raise RuntimeError("pinned slot geometry is unavailable")
        elements = 3 * int(frames) * self._slot_height * self._slot_width
        return self._slots[slot_idx].view(-1)[:elements].view(
            1, 3, int(frames), self._slot_height, self._slot_width
        )

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload.update(
            {
                "variant": "pinned_async_contiguous_streaming_singlecopy",
                "contiguous_slot_views": True,
            }
        )
        return payload


class FourSlotContiguousPinnedAsyncStreamingSingleCopyWriter(
    ContiguousPinnedAsyncStreamingSingleCopyWriter
):
    """Four-slot ring for deferred decoding; task queue remains bounded at two."""

    _pinned_slot_count = 4

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload["variant"] = "pinned_async_contiguous_four_slot_streaming_singlecopy"
        return payload


class GpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter(
    ContiguousPinnedAsyncStreamingSingleCopyWriter
):
    """Move the exact scale/clip/cast before D2H to shrink the copied tensor."""

    def submit(self, video: torch.Tensor, *, clip_idx: int) -> None:
        if video.device.type != "cuda":
            raise RuntimeError("GPU uint8 delivery requires CUDA decoder output")
        converted = ((video.float() + 1) * 127.5).clip(0, 255).to(torch.uint8)
        super().submit(converted, clip_idx=clip_idx)

    def _slot_pixels(self, slot_view: torch.Tensor):
        if slot_view.dtype != torch.uint8 or slot_view.device.type != "cpu":
            raise RuntimeError("GPU uint8 pinned slot contract failed")
        return slot_view[0].permute(1, 2, 3, 0).numpy().copy(order="C")

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload.update(
            {
                "variant": "pinned_async_contiguous_gpu_uint8_streaming_singlecopy",
                "pixel_conversion": "gpu_fp32_scale_clip_uint8",
                "gpu_uint8_delivery": True,
            }
        )
        payload["certificate"]["cpu_uint8"] = False
        payload["certificate"]["gpu_uint8"] = True
        return payload


class FourSlotGpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter(
    GpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter
):
    """Four-slot variant of exact GPU-side uint8 delivery."""

    _pinned_slot_count = 4

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload["variant"] = (
            "pinned_async_contiguous_four_slot_gpu_uint8_streaming_singlecopy"
        )
        return payload


class FourSlotGpuUint8ViewPinnedAsyncStreamingSingleCopyWriter(
    FourSlotGpuUint8ContiguousPinnedAsyncStreamingSingleCopyWriter
):
    """Expose the pinned uint8 tensor as a zero-copy THWC NumPy view."""

    def _slot_pixels(self, slot_view: torch.Tensor):
        if slot_view.dtype != torch.uint8 or slot_view.device.type != "cpu":
            raise RuntimeError("GPU uint8 pinned slot contract failed")
        return slot_view[0].permute(1, 2, 3, 0).numpy()

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload.update(
            {
                "variant": "pinned_async_four_slot_gpu_uint8_numpy_view",
                "cpu_thwc_copy_elided": True,
            }
        )
        return payload


class FourSlotGpuUint8THWCPinnedAsyncStreamingSingleCopyWriter(
    FourSlotGpuUint8ViewPinnedAsyncStreamingSingleCopyWriter
):
    """Use a physical THWC pinned layout while exposing logical BCTHW copies."""

    def _ensure_copy_resources(self, video: torch.Tensor) -> None:
        if self._slots is not None:
            expected = (self._slot_dtype, self._slot_height, self._slot_width)
            actual = (video.dtype, int(video.shape[3]), int(video.shape[4]))
            if actual != expected:
                raise RuntimeError(
                    f"decoded video shape/dtype changed after pinned allocation: {actual} != {expected}"
                )
            return
        if video.device.type != "cuda" or video.dtype != torch.uint8:
            raise RuntimeError("THWC pinned delivery requires CUDA uint8 video")
        self._slot_dtype = video.dtype
        self._slot_height = int(video.shape[3])
        self._slot_width = int(video.shape[4])
        shape = (57, self._slot_height, self._slot_width, 3)
        self._slots = [
            torch.empty(shape, dtype=video.dtype, device="cpu", pin_memory=True)
            for _ in range(self._pinned_slot_count)
        ]
        for slot_idx in range(self._pinned_slot_count):
            self._free_slots.put(slot_idx)
        self._copy_stream = torch.cuda.Stream(device=video.device)

    def _slot_view(self, slot_idx: int, frames: int) -> torch.Tensor:
        assert self._slots is not None
        return self._slots[slot_idx][:frames].permute(3, 0, 1, 2).unsqueeze(0)

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload.update(
            {
                "variant": "pinned_async_four_slot_gpu_uint8_thwc",
                "pinned_physical_layout": "THWC",
                "destination_thwc_contiguous": True,
            }
        )
        return payload


class FourSlotDirectGpuTHWCPinnedAsyncStreamingSingleCopyWriter(
    PinnedAsyncStreamingSingleCopyWriter
):
    """Convert to contiguous THWC on CUDA, then copy to contiguous THWC pinned slots."""

    _pinned_slot_count = 4

    def _ensure_thwc_resources(self, pixels: torch.Tensor) -> None:
        if self._slots is not None:
            expected = (self._slot_dtype, self._slot_height, self._slot_width)
            actual = (pixels.dtype, int(pixels.shape[1]), int(pixels.shape[2]))
            if actual != expected:
                raise RuntimeError(f"THWC shape/dtype changed: {actual} != {expected}")
            return
        if pixels.device.type != "cuda" or pixels.dtype != torch.uint8:
            raise RuntimeError("direct THWC delivery requires CUDA uint8 pixels")
        self._slot_dtype = pixels.dtype
        self._slot_height = int(pixels.shape[1])
        self._slot_width = int(pixels.shape[2])
        shape = (57, self._slot_height, self._slot_width, 3)
        self._slots = [
            torch.empty(shape, dtype=torch.uint8, device="cpu", pin_memory=True)
            for _ in range(self._pinned_slot_count)
        ]
        for slot_idx in range(self._pinned_slot_count):
            self._free_slots.put(slot_idx)
        self._copy_stream = torch.cuda.Stream(device=pixels.device)

    def submit(self, video: torch.Tensor, *, clip_idx: int) -> None:
        if self._finished:
            raise RuntimeError("cannot submit after pinned-async writer finish")
        if clip_idx != len(self._chunks):
            raise RuntimeError(
                f"non-chronological chunk: expected {len(self._chunks)}, got {clip_idx}"
            )
        expected_frames = 57 if clip_idx == 0 else 40
        if (
            video.ndim != 5
            or video.shape[0] != 1
            or video.shape[1] != 3
            or int(video.shape[2]) != expected_frames
            or video.device.type != "cuda"
        ):
            raise RuntimeError("invalid decoded video for direct THWC delivery")
        pixels = (
            ((video[0].permute(1, 2, 3, 0).float() + 1) * 127.5)
            .clip(0, 255)
            .to(torch.uint8)
            .contiguous()
        )
        self._ensure_thwc_resources(pixels)
        assert self._slots is not None and self._copy_stream is not None
        slot_idx, slot_wait_s = self._get_slot_checked()
        slot_view = self._slots[slot_idx][:expected_frames]
        current_stream = torch.cuda.current_stream(device=pixels.device)
        enqueue_start = time.perf_counter()
        event_create_start = time.perf_counter()
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        event_create_s = time.perf_counter() - event_create_start
        stream_context_start = time.perf_counter()
        with torch.cuda.stream(self._copy_stream):
            wait_stream_start = time.perf_counter()
            self._copy_stream.wait_stream(current_stream)
            wait_stream_s = time.perf_counter() - wait_stream_start
            record_start = time.perf_counter()
            copy_start.record(self._copy_stream)
            copy_start_record_s = time.perf_counter() - record_start
            copy_call_start = time.perf_counter()
            slot_view.copy_(pixels, non_blocking=True)
            copy_call_s = time.perf_counter() - copy_call_start
            record_end = time.perf_counter()
            copy_end.record(self._copy_stream)
            copy_end_record_s = time.perf_counter() - record_end
        stream_context_s = time.perf_counter() - stream_context_start
        enqueue_s = time.perf_counter() - enqueue_start
        record_stream_start = time.perf_counter()
        pixels.record_stream(self._copy_stream)
        record_stream_s = time.perf_counter() - record_stream_start
        row: dict[str, Any] = {
            "clip_idx": int(clip_idx),
            "frames": expected_frames,
            "slot_idx": slot_idx,
            "d2h_s": enqueue_s,
            "slot_wait_s": slot_wait_s,
            "enqueue_s": enqueue_s,
            "event_create_s": event_create_s,
            "stream_context_s": stream_context_s,
            "wait_stream_s": wait_stream_s,
            "copy_start_record_s": copy_start_record_s,
            "copy_call_s": copy_call_s,
            "copy_end_record_s": copy_end_record_s,
            "record_stream_s": record_stream_s,
            "source_contiguous": bool(pixels.is_contiguous()),
            "source_stride": [int(value) for value in pixels.stride()],
            "destination_contiguous": bool(slot_view.is_contiguous()),
            "destination_stride": [int(value) for value in slot_view.stride()],
            "destination_thwc_contiguous": bool(slot_view.is_contiguous()),
            "queue_wait_s": None,
            "copy_event_wait_s": None,
            "copy_event_ready_on_dequeue": None,
            "copy_cuda_ms": None,
            "copy_async_lifetime_s": None,
            "copy_wait_hidden_s": None,
            "background_copy_wait_overlap_s": None,
            "copy_enqueued_non_blocking": True,
            "convert_s": None,
            "append_s": None,
            "_enqueue_host_time": time.perf_counter(),
        }
        self._chunks.append(row)
        row["queue_wait_s"] = self._put_checked(
            (slot_idx, slot_view, copy_start, copy_end, pixels, row)
        )

    def _slot_pixels(self, slot_view: torch.Tensor):
        return slot_view.numpy()

    def _runtime_payload(self, *, status: str, error: str | None) -> dict[str, Any]:
        payload = super()._runtime_payload(status=status, error=error)
        payload.update(
            {
                "variant": "pinned_async_four_slot_direct_gpu_thwc",
                "pixel_conversion": "gpu_fp32_scale_clip_uint8",
                "gpu_uint8_delivery": True,
                "cpu_thwc_copy_elided": True,
                "pinned_physical_layout": "THWC",
                "destination_thwc_contiguous": True,
                "contiguous_slot_views": True,
                "direct_gpu_thwc": True,
            }
        )
        payload["certificate"]["cpu_uint8"] = False
        payload["certificate"]["gpu_uint8"] = True
        return payload


def rewrite_matrix_pipeline_for_pinned_async_streaming_single_copy(source: str) -> str:
    rewritten = rewrite_matrix_pipeline_for_exact_streaming_single_copy(source)
    constructor = "ExactStreamingSingleCopyWriter("
    if rewritten.count(constructor) != 1:
        raise RuntimeError("pinned-async overlay expected one writer constructor")
    rewritten = rewritten.replace(
        constructor, "PinnedAsyncStreamingSingleCopyWriter(", 1
    ).replace(
        '"exact_streaming_singlecopy_trace.json"',
        '"pinned_async_streaming_singlecopy_trace.json"',
        1,
    )
    return rewritten


def install_matrix_pinned_async_streaming_single_copy_video_pipeline(
    matrix_root: Path,
    *,
    contiguous_slots: bool = False,
) -> None:
    if _TARGET_MODULE in sys.modules:
        raise RuntimeError(
            "pinned-async overlay must be installed before importing "
            f"{_TARGET_MODULE}"
        )
    importlib.import_module("pipeline")
    source_path = matrix_root / "pipeline" / "inference_interactive_pipeline.py"
    rewritten = rewrite_matrix_pipeline_for_pinned_async_streaming_single_copy(
        source_path.read_text(encoding="utf-8")
    )
    module = types.ModuleType(_TARGET_MODULE)
    module.__file__ = str(source_path)
    module.__package__ = "pipeline"
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__.update(
        {
            "PinnedAsyncStreamingSingleCopyWriter": (
                ContiguousPinnedAsyncStreamingSingleCopyWriter
                if contiguous_slots
                else PinnedAsyncStreamingSingleCopyWriter
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
