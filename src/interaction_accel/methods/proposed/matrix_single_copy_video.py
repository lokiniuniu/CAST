"""Runtime overlay for single-copy Matrix video delivery.

The upstream interactive pipeline materializes an iteration-level uint8 NumPy
video, writes it, and then transfers the same decoded tensor to CPU a second
time for final concatenation.  WorldMark consumes only the final ``gen.mp4``.
This overlay removes the unused iteration conversion/encode without modifying
the vendored upstream checkout.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types


_TARGET_MODULE = "pipeline.inference_interactive_pipeline"
_NATIVE_BLOCK = '''                        video_np = np.ascontiguousarray(((rearrange(video[0], "C T H W -> T H W C").float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8))
                        
                        import imageio
                        imageio.mimsave(f"{self.output_dir}/{save_name}_current_iteration_{clip_idx}.mp4", video_np, fps=17)
                        all_videos_list.append(video.cpu())
'''
_SINGLE_COPY_BLOCK = '''                        # WorldMark consumes only the final concatenated video.  Keep
                        # one CPU copy per decoded chunk and defer uint8 conversion and
                        # encoding until the final output is assembled.
                        all_videos_list.append(video.cpu())
'''
_PROFILED_SINGLE_COPY_BLOCK = '''                        # Profiling-only overlay: preserve SingleCopy output while exposing
                        # the exact per-clip device-to-host boundary.
                        all_videos_list.append(
                            self._worldmark_full_pipeline_profiler.profile_clip_d2h(
                                video, clip_idx=clip_idx, first_clip=first_clip
                            )
                        )
'''

_LATENT_TO_VAE = "denoised_pred_for_vae.to(dtype=self.vae.dtype),"
_PROFILED_LATENT_TO_VAE = '''self._worldmark_full_pipeline_profiler.profile_latent_to_vae(
                                denoised_pred_for_vae,
                                dtype=self.vae.dtype,
                                clip_idx=clip_idx,
                                first_clip=first_clip,
                            ),'''

_FINAL_POSTPROCESS = '''                    concatenated_video = np.ascontiguousarray(((rearrange(torch.concat(all_videos_list, dim=2)[0], "C T H W -> T H W C").float() + 1) * 127.5).clip(0, 255).numpy().astype(np.uint8))
'''
_PROFILED_FINAL_POSTPROCESS = '''                    concatenated_video = self._worldmark_full_pipeline_profiler.profile_final_postprocess(
                        all_videos_list
                    )
'''


def rewrite_matrix_pipeline_for_single_copy(
    source: str, *, profile_full_pipeline: bool = False
) -> str:
    """Return the upstream source with exactly one audited block rewritten."""

    occurrences = source.count(_NATIVE_BLOCK)
    if occurrences != 1:
        raise RuntimeError(
            "single-copy overlay expected exactly one upstream video block, "
            f"found {occurrences}"
        )
    rewritten = source.replace(
        _NATIVE_BLOCK,
        _PROFILED_SINGLE_COPY_BLOCK if profile_full_pipeline else _SINGLE_COPY_BLOCK,
        1,
    )
    if not profile_full_pipeline:
        return rewritten
    if rewritten.count(_LATENT_TO_VAE) != 1:
        raise RuntimeError("profile overlay expected one latent-to-VAE boundary")
    if rewritten.count(_FINAL_POSTPROCESS) != 1:
        raise RuntimeError("profile overlay expected one final postprocess boundary")
    return rewritten.replace(
        _LATENT_TO_VAE, _PROFILED_LATENT_TO_VAE, 1
    ).replace(_FINAL_POSTPROCESS, _PROFILED_FINAL_POSTPROCESS, 1)


def install_matrix_single_copy_video_pipeline(
    matrix_root: Path, *, profile_full_pipeline: bool = False
) -> None:
    """Load the Matrix pipeline through an in-memory, one-block overlay."""

    if _TARGET_MODULE in sys.modules:
        raise RuntimeError(
            "single-copy pipeline overlay must be installed before importing "
            f"{_TARGET_MODULE}"
        )
    importlib.import_module("pipeline")
    source_path = matrix_root / "pipeline" / "inference_interactive_pipeline.py"
    source = source_path.read_text(encoding="utf-8")
    rewritten = rewrite_matrix_pipeline_for_single_copy(
        source, profile_full_pipeline=profile_full_pipeline
    )

    module = types.ModuleType(_TARGET_MODULE)
    module.__file__ = str(source_path)
    module.__package__ = "pipeline"
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[_TARGET_MODULE] = module
    try:
        exec(compile(rewritten, str(source_path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(_TARGET_MODULE, None)
        raise
