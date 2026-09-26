#!/usr/bin/env python3
"""Offline integrity and source-boundary check for the CAST release."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/interaction_accel/methods/proposed"
FORBIDDEN_SUFFIXES = {".ckpt", ".log", ".mp4", ".pt", ".pth", ".safetensors"}


def main() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["source_files"].items():
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"missing source file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"source hash mismatch: {name}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"generated or heavy artifact included: {path}")
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            if not node.module or path.parent != SOURCE:
                continue
            dependency = SOURCE / f"{node.module}.py"
            if not dependency.is_file():
                raise SystemExit(f"missing relative import {dependency.name} in {path.name}")

    spec = json.loads((ROOT / "configs/cast_matrix.json").read_text())
    expected = {
        "sparse_density": 0.2,
        "enable_fc_pasm": True,
        "fc_reference_numerics": True,
        "routing_mode": "fc_pasm_swap",
        "routing_active_layers": [22, 26],
        "frameweave_shared_int8_qkv_all_paths": True,
        "frameweave_cached_compact_rope_phase": True,
        "frameweave_cached_native_rope_phase": True,
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise SystemExit(f"CAST configuration mismatch: {key}")
    candidate = ROOT / spec["candidate"]
    if not candidate.is_file():
        raise SystemExit("CAST candidate file is missing")
    print(f"CAST Matrix release checked: {len(manifest['source_files'])} source files")


if __name__ == "__main__":
    main()
