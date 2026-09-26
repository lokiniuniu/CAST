"""Exact streaming final-video delivery for the canonical Matrix method.

This is an opt-in source overlay.  It preserves the native CPU pixel
expression and frame order, but feeds one persistent imageio writer from a
bounded single-worker queue instead of retaining every decoded chunk for a
final ``torch.concat``.  CUDA-to-CPU transfer remains on the inference thread;
no GPU-side uint8 conversion is performed.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import queue
import sys
import threading
import time
import types
from typing import Any, Callable

import numpy as np
import torch


_TARGET_MODULE = "pipeline.inference_interactive_pipeline"
_INIT = """            all_videos_list = []
"""
_STREAM_INIT = """            worldmark_streaming_video = None
            if not self.use_async_vae and self.rank == 0:
                worldmark_streaming_video = ExactStreamingSingleCopyWriter(
                    output_path=Path(self.output_dir) / f"{save_name}.mp4",
                    trace_path=Path(self.output_dir) / "exact_streaming_singlecopy_trace.json",
                    expected_chunks=num_iterations,
                    expected_frames=17 + 40 * num_iterations,
                    fps=17,
                    queue_capacity=2,
                )
"""
_NATIVE_CHUNK = '''                        video_np = np.ascontiguousarray(((rearrange(video[0], "C T H W -> T H W C").float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8))
                        
                        import imageio
                        imageio.mimsave(f"{self.output_dir}/{save_name}_current_iteration_{clip_idx}.mp4", video_np, fps=17)
                        all_videos_list.append(video.cpu())
'''
_STREAM_CHUNK = '''                        # Exact streaming SingleCopy: D2H once on the inference
                        # thread; CPU conversion and persistent encoding run on one
                        # bounded background worker in chronological chunk order.
                        worldmark_streaming_video.submit(video, clip_idx=clip_idx)
'''
_NATIVE_FINAL = '''            if not self.use_async_vae and self.rank == 0:
                if len(all_videos_list) > 0:
                    concatenated_video = np.ascontiguousarray(((rearrange(torch.concat(all_videos_list, dim=2)[0], "C T H W -> T H W C").float() + 1) * 127.5).clip(0, 255).numpy().astype(np.uint8))
                    import imageio
                    imageio.mimsave(f"{self.output_dir}/{save_name}.mp4", concatenated_video, fps=17)
                    print(f"Saved concatenated video with {len(all_videos_list)} segments")
'''
_STREAM_FINAL = '''            if not self.use_async_vae and self.rank == 0:
                if worldmark_streaming_video is None:
                    raise RuntimeError("exact streaming SingleCopy writer was not initialized")
                worldmark_streaming_video.finish()
                self._worldmark_exact_streaming_single_copy_runtime = (
                    worldmark_streaming_video.runtime_summary()
                )
                print(f"Saved exact streaming video with {num_iterations} segments")
'''


def native_chunk_pixels(video_cpu: torch.Tensor) -> np.ndarray:
    """Apply the native final-video expression to one already-CPU chunk."""

    if video_cpu.device.type != "cpu":
        raise RuntimeError("pixel conversion must remain on CPU")
    if video_cpu.ndim != 5 or video_cpu.shape[0] != 1 or video_cpu.shape[1] != 3:
        raise RuntimeError(
            "expected decoded video shape [1,3,T,H,W], got "
            f"{tuple(video_cpu.shape)}"
        )
    # Literal elementwise equivalent of the upstream rearrange expression.
    pixels = (
        (video_cpu[0].permute(1, 2, 3, 0).float() + 1) * 127.5
    ).clip(0, 255).numpy().astype(np.uint8)
    return np.ascontiguousarray(pixels)


def certify_exact_streaming_runtime(
    runtime: dict[str, Any], *, expected_chunks: int, allow_gpu_uint8: bool = False
) -> dict[str, Any]:
    """Fail closed unless a completed runtime has every native segment record."""

    expected_chunks = int(expected_chunks)
    expected_frames = 17 + 40 * expected_chunks
    chunks = runtime.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != expected_chunks:
        raise RuntimeError("exact streaming runtime has incomplete segment records")
    if runtime.get("status") != "passed" or runtime.get("error") is not None:
        raise RuntimeError("exact streaming runtime did not finish successfully")
    exact_scalars = {
        "expected_chunks": expected_chunks,
        "segment_count": expected_chunks,
        "expected_frames": expected_frames,
        "frame_count": expected_frames,
        "queue_capacity": 2,
    }
    for name, expected in exact_scalars.items():
        if int(runtime.get(name, -1)) != expected:
            raise RuntimeError(
                f"exact streaming runtime {name} mismatch: "
                f"{runtime.get(name)!r} != {expected}"
            )
    if runtime.get("writer_opened") is not True or runtime.get("writer_closed") is not True:
        raise RuntimeError("exact streaming writer open/close certificate failed")
    for name in ("writer_open_s", "writer_close_s"):
        value = runtime.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"invalid exact streaming timing {name}")
    peak_queue = runtime.get("peak_queue")
    if not isinstance(peak_queue, int) or not 1 <= peak_queue <= 2:
        raise RuntimeError("exact streaming peak queue is outside [1, 2]")
    for clip_idx, row in enumerate(chunks):
        if not isinstance(row, dict):
            raise RuntimeError("exact streaming chunk record is not a mapping")
        if int(row.get("clip_idx", -1)) != clip_idx:
            raise RuntimeError("exact streaming chunk order certificate failed")
        expected_chunk_frames = 57 if clip_idx == 0 else 40
        if int(row.get("frames", -1)) != expected_chunk_frames:
            raise RuntimeError("exact streaming chunk frame certificate failed")
        for name in ("d2h_s", "convert_s", "queue_wait_s", "append_s"):
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"invalid exact streaming chunk timing {name}")
    certificate = runtime.get("certificate")
    required = {
        "gpu_uint8" if allow_gpu_uint8 else "cpu_uint8",
        "persistent_single_writer",
        "chronological_single_worker",
        "final_join_before_delivery",
    }
    if not isinstance(certificate, dict) or any(
        certificate.get(name) is not True for name in required
    ):
        raise RuntimeError("exact streaming semantic certificate failed")
    return {
        "complete": True,
        "segments": expected_chunks,
        "frames": expected_frames,
        "writer_opened": True,
        "writer_closed": True,
        "queue_capacity": 2,
    }


class ExactStreamingSingleCopyWriter:
    """Bounded one-worker CPU converter and persistent video writer."""

    _STOP = object()

    def __init__(
        self,
        *,
        output_path: Path,
        trace_path: Path,
        expected_chunks: int,
        expected_frames: int,
        fps: int = 17,
        queue_capacity: int = 2,
        writer_factory: Callable[..., Any] | None = None,
    ) -> None:
        native_expected = 17 + 40 * int(expected_chunks)
        if expected_chunks <= 0 or expected_frames != native_expected:
            raise RuntimeError(
                "native frame certificate failed: expected_frames must equal "
                f"17 + 40 * chunks = {native_expected}, got {expected_frames}"
            )
        if queue_capacity != 2:
            raise RuntimeError("canonical exact streaming queue capacity must be 2")
        self.output_path = Path(output_path)
        self.trace_path = Path(trace_path)
        self.expected_chunks = int(expected_chunks)
        self.expected_frames = int(expected_frames)
        self.fps = int(fps)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._writer_factory = writer_factory
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._finished = False
        self._chunks: list[dict[str, Any]] = []
        self._frame_count = 0
        self._peak_queue = 0
        self._writer_open_s = 0.0
        self._writer_close_s = 0.0
        self._writer_opened = False
        self._writer_closed = False
        self._summary: dict[str, Any] | None = None
        self._thread = threading.Thread(
            target=self._run, name="matrix-exact-video-writer", daemon=False
        )
        self._thread.start()

    def _get_failure(self) -> BaseException | None:
        with self._failure_lock:
            return self._failure

    def _set_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error

    def _put_checked(self, item: Any) -> float:
        start = time.perf_counter()
        while True:
            failure = self._get_failure()
            if failure is not None:
                raise RuntimeError("streaming video worker failed") from failure
            try:
                self._queue.put(item, timeout=0.05)
                # A consumer can dequeue between put() and qsize(); retain the
                # observed enqueue as occupancy one so the completion trace is
                # stable without changing queue synchronization.
                self._peak_queue = max(self._peak_queue, 1, self._queue.qsize())
                return time.perf_counter() - start
            except queue.Full:
                continue

    def submit(self, video: torch.Tensor, *, clip_idx: int) -> None:
        if self._finished:
            raise RuntimeError("cannot submit after exact streaming writer finish")
        if clip_idx != len(self._chunks):
            raise RuntimeError(
                f"non-chronological chunk: expected {len(self._chunks)}, got {clip_idx}"
            )
        expected_chunk_frames = 57 if clip_idx == 0 else 40
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
            raise RuntimeError(f"invalid decoded video shape {tuple(video.shape)}")
        if int(video.shape[2]) != expected_chunk_frames:
            raise RuntimeError(
                f"chunk {clip_idx} expected {expected_chunk_frames} frames, "
                f"got {int(video.shape[2])}"
            )
        d2h_start = time.perf_counter()
        video_cpu = video.cpu()
        d2h_s = time.perf_counter() - d2h_start
        row: dict[str, Any] = {
            "clip_idx": int(clip_idx),
            "frames": expected_chunk_frames,
            "d2h_s": d2h_s,
            "convert_s": None,
            "queue_wait_s": None,
            "append_s": None,
        }
        self._chunks.append(row)
        row["queue_wait_s"] = self._put_checked((video_cpu, row))

    def _make_writer(self) -> Any:
        if self._writer_factory is not None:
            return self._writer_factory(self.output_path, fps=self.fps)
        import imageio

        return imageio.get_writer(str(self.output_path), fps=self.fps)

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
                    if failed:
                        continue
                    video_cpu, row = item
                    start = time.perf_counter()
                    pixels = native_chunk_pixels(video_cpu)
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
                    self._queue.task_done()
        except BaseException as error:
            self._set_failure(error)
            # Drain queued work so producers and finish cannot deadlock.
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
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
        return {
            "variant": "exact_streaming_singlecopy",
            "status": status,
            "error": error,
            "output_path": str(self.output_path),
            "fps": self.fps,
            "expected_chunks": self.expected_chunks,
            "segment_count": len(self._chunks),
            "expected_frames": self.expected_frames,
            "frame_count": self._frame_count,
            "peak_queue": self._peak_queue,
            "queue_capacity": self._queue.maxsize,
            "writer_open_s": self._writer_open_s,
            "writer_close_s": self._writer_close_s,
            "writer_opened": self._writer_opened,
            "writer_closed": self._writer_closed,
            "chunks": self._chunks,
            "certificate": {
                "cpu_uint8": True,
                "persistent_single_writer": True,
                "chronological_single_worker": True,
                "final_join_before_delivery": True,
                "native_frame_formula": "17 + 40 * num_iterations",
            },
        }

    def _write_trace(self, *, status: str, error: str | None) -> None:
        payload = self._runtime_payload(status=status, error=error)
        self._summary = payload
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.trace_path.with_suffix(self.trace_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.trace_path)

    def runtime_summary(self) -> dict[str, Any]:
        if not self._finished or self._thread.is_alive() or self._summary is None:
            raise RuntimeError("exact streaming runtime requested before final join")
        # JSON roundtrip makes the returned trace independent of mutable rows.
        return json.loads(json.dumps(self._summary))

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("exact streaming writer finish called twice")
        self._finished = True
        put_error: BaseException | None = None
        # STOP is lifecycle control, not new work.  It must still be delivered
        # after a worker-side failure so the draining worker can leave its
        # queue loop and join cannot deadlock.
        while self._thread.is_alive():
            try:
                self._queue.put(self._STOP, timeout=0.05)
                self._peak_queue = max(self._peak_queue, 1, self._queue.qsize())
                break
            except queue.Full:
                continue
            except BaseException as error:
                put_error = error
                break
        self._thread.join()
        failure = self._get_failure() or put_error
        complete = (
            failure is None
            and len(self._chunks) == self.expected_chunks
            and self._frame_count == self.expected_frames
        )
        error_text = None if failure is None else repr(failure)
        if failure is None and not complete:
            error_text = (
                f"frame certificate failed: chunks={len(self._chunks)}/"
                f"{self.expected_chunks}, frames={self._frame_count}/"
                f"{self.expected_frames}"
            )
        self._write_trace(status="passed" if complete else "failed", error=error_text)
        if failure is not None:
            raise RuntimeError("exact streaming video worker failed") from failure
        if not complete:
            raise RuntimeError(error_text)


def rewrite_matrix_pipeline_for_exact_streaming_single_copy(source: str) -> str:
    """Rewrite exactly the three audited native video-delivery boundaries."""

    replacements = (
        (_INIT, _STREAM_INIT, "video-list initialization"),
        (_NATIVE_CHUNK, _STREAM_CHUNK, "per-chunk delivery"),
        (_NATIVE_FINAL, _STREAM_FINAL, "final delivery"),
    )
    rewritten = source
    for native, proposed, label in replacements:
        count = rewritten.count(native)
        if count != 1:
            raise RuntimeError(f"streaming overlay expected one {label}, found {count}")
        rewritten = rewritten.replace(native, proposed, 1)
    return rewritten


def install_matrix_exact_streaming_single_copy_video_pipeline(matrix_root: Path) -> None:
    """Install the opt-in pipeline overlay without modifying third_party."""

    if _TARGET_MODULE in sys.modules:
        raise RuntimeError(
            "exact streaming overlay must be installed before importing "
            f"{_TARGET_MODULE}"
        )
    importlib.import_module("pipeline")
    source_path = matrix_root / "pipeline" / "inference_interactive_pipeline.py"
    rewritten = rewrite_matrix_pipeline_for_exact_streaming_single_copy(
        source_path.read_text(encoding="utf-8")
    )
    module = types.ModuleType(_TARGET_MODULE)
    module.__file__ = str(source_path)
    module.__package__ = "pipeline"
    module.__loader__ = None
    module.__spec__ = None
    module.__dict__.update(
        {
            "ExactStreamingSingleCopyWriter": ExactStreamingSingleCopyWriter,
            "Path": Path,
        }
    )
    sys.modules[_TARGET_MODULE] = module
    try:
        exec(compile(rewritten, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(_TARGET_MODULE, None)
        raise
