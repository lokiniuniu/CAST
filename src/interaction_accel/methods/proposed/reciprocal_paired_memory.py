"""Fixed-cardinality reciprocal-pair memory for modular C3 improvement."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class ReciprocalMemoryPair:
    left_index: int
    right_index: int
    reciprocal: bool
    left_pose_distance: float
    right_pose_distance: float
    pair_pose_distance: float


@dataclass(frozen=True)
class ReciprocalPairedMemorySelection:
    atoms: tuple[Any, ...]
    pairs: tuple[ReciprocalMemoryPair, ...]
    candidate_count: int


class MatrixReciprocalPairedMemoryCompiler:
    """Select one sink and a minimum-cost matching of two history pairs.

    C3-v2's attention and angular-witness solver are held fixed.  This module
    changes only the four remote memory vertices.  A causal pose-nearest
    candidate domain is constructed, then a two-edge vertex-disjoint matching
    is selected lexicographically:

    1. maximize the number of exact reciprocal-action pairs;
    2. minimize the farthest endpoint-to-current-pose chordal distance;
    3. minimize within-pair pose discrepancy;
    4. break ties by immutable archive index.

    The construction has fixed cardinality and no learned score, weighted
    blend, activation gate, or quality metric.  When the causal past contains
    no inverse-action atoms, the same program degenerates uniquely to two
    local secant pairs rather than changing its budget.
    """

    name = "matrix_reciprocal_paired_memory_compiler"

    @staticmethod
    def _index(atom: Any) -> int:
        return int(getattr(atom, "original_time_index"))

    @staticmethod
    def _pose_distance(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.detach().float()
        right = right.detach().float().to(device=left.device)
        if left.shape != (4, 4) or right.shape != (4, 4):
            raise ValueError("memory atom poses must be 4x4 transforms")
        relative = torch.linalg.solve(left, right)
        identity = torch.eye(4, device=relative.device, dtype=relative.dtype)
        return float(torch.linalg.matrix_norm(relative - identity).item())

    def _trajectory_distance(
        self, atom_pose: torch.Tensor, current_c2ws: torch.Tensor
    ) -> float:
        if current_c2ws.ndim != 3 or current_c2ws.shape[1:] != (4, 4):
            raise ValueError("current_c2ws must have shape [T, 4, 4]")
        return min(self._pose_distance(atom_pose, pose) for pose in current_c2ws)

    @staticmethod
    def _is_reciprocal(left: torch.Tensor, right: torch.Tensor) -> bool:
        left = left.detach().float().reshape(-1)
        right = right.detach().float().reshape(-1).to(device=left.device)
        if left.shape != right.shape:
            return False
        scale = max(1.0, float(left.abs().max().item()), float(right.abs().max().item()))
        tolerance = 16.0 * torch.finfo(torch.float32).eps * scale
        return bool(torch.max(torch.abs(left + right)).item() <= tolerance)

    @staticmethod
    def _action_key(action: torch.Tensor) -> tuple[float, ...]:
        return tuple(float(value) for value in action.detach().float().reshape(-1).cpu())

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        top_k: int,
        recent_exclusion: int,
    ) -> ReciprocalPairedMemorySelection:
        if len(atoms) < 5:
            raise RuntimeError("reciprocal-paired memory requires sink plus four atoms")
        sink = next((atom for atom in atoms if bool(getattr(atom, "is_sink", False))), atoms[0])
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink and self._index(atom) <= newest - int(recent_exclusion)
        ]
        if len(eligible) < 4:
            raise RuntimeError("reciprocal-paired memory has fewer than four causal atoms")
        ranked = sorted(
            eligible,
            key=lambda atom: (
                self._trajectory_distance(getattr(atom, "c2w"), current_c2ws),
                self._index(atom),
            ),
        )
        candidates = ranked[: max(4, int(top_k))]
        # A pure pose top-k can contain only one leg of a reciprocal control
        # cycle.  Complete it with the closest four endpoints from the best
        # available inverse-action section.  This is a deterministic domain
        # completion, not a sample-dependent activation gate: if no inverse
        # section exists, the native pose domain is unchanged.
        action_sections: dict[tuple[float, ...], list[Any]] = {}
        for atom in ranked:
            action_sections.setdefault(
                self._action_key(getattr(atom, "action")), []
            ).append(atom)
        reciprocal_section_candidates = []
        visited_sections: set[tuple[float, ...]] = set()
        for key, section in action_sections.items():
            if key in visited_sections:
                continue
            inverse_key = tuple(-value for value in key)
            inverse_section = action_sections.get(inverse_key)
            if inverse_section is None:
                continue
            if inverse_key == key:
                endpoints = section[:4]
            else:
                endpoints = section[:2] + inverse_section[:2]
            if len(endpoints) < 4:
                continue
            visited_sections.update((key, inverse_key))
            endpoint_radius = max(
                self._trajectory_distance(getattr(atom, "c2w"), current_c2ws)
                for atom in endpoints
            )
            reciprocal_section_candidates.append(
                (
                    endpoint_radius,
                    tuple(sorted(self._index(atom) for atom in endpoints)),
                    endpoints,
                )
            )
        if reciprocal_section_candidates:
            _, _, completed_endpoints = min(reciprocal_section_candidates)
            candidate_ids = {id(atom) for atom in candidates}
            for atom in completed_endpoints:
                if id(atom) not in candidate_ids:
                    candidates.append(atom)
                    candidate_ids.add(id(atom))
        endpoint_distance = {
            id(atom): self._trajectory_distance(
                getattr(atom, "c2w"), current_c2ws
            )
            for atom in candidates
        }
        edges = []
        for left, right in combinations(candidates, 2):
            reciprocal = self._is_reciprocal(
                getattr(left, "action"), getattr(right, "action")
            )
            pair_distance = self._pose_distance(
                getattr(left, "c2w"), getattr(right, "c2w")
            )
            indices = tuple(sorted((self._index(left), self._index(right))))
            edges.append(
                (
                    left,
                    right,
                    reciprocal,
                    pair_distance,
                    indices,
                )
            )

        matchings = []
        for first, second in combinations(edges, 2):
            vertices = (first[0], first[1], second[0], second[1])
            if len({id(atom) for atom in vertices}) != 4:
                continue
            reciprocal_penalty = int(not first[2]) + int(not second[2])
            farthest_endpoint = max(endpoint_distance[id(atom)] for atom in vertices)
            pair_discrepancy = first[3] + second[3]
            immutable_indices = tuple(sorted(self._index(atom) for atom in vertices))
            matchings.append(
                (
                    (
                        reciprocal_penalty,
                        farthest_endpoint,
                        pair_discrepancy,
                        immutable_indices,
                    ),
                    (first, second),
                )
            )
        if not matchings:
            raise RuntimeError("reciprocal-paired memory found no two-edge matching")
        _, selected_edges = min(matchings, key=lambda item: item[0])
        ordered_edges = sorted(selected_edges, key=lambda edge: edge[4])
        selected_atoms = [sink]
        pairs = []
        for left, right, reciprocal, pair_distance, _ in ordered_edges:
            endpoints = sorted((left, right), key=self._index)
            selected_atoms.extend(endpoints)
            pairs.append(
                ReciprocalMemoryPair(
                    left_index=self._index(endpoints[0]),
                    right_index=self._index(endpoints[1]),
                    reciprocal=bool(reciprocal),
                    left_pose_distance=endpoint_distance[id(endpoints[0])],
                    right_pose_distance=endpoint_distance[id(endpoints[1])],
                    pair_pose_distance=float(pair_distance),
                )
            )
        if len(selected_atoms) != 5 or len({id(atom) for atom in selected_atoms}) != 5:
            raise RuntimeError("reciprocal-paired memory violated fixed cardinality")
        return ReciprocalPairedMemorySelection(
            atoms=tuple(selected_atoms),
            pairs=tuple(pairs),
            candidate_count=len(candidates),
        )


__all__ = [
    "MatrixReciprocalPairedMemoryCompiler",
    "ReciprocalMemoryPair",
    "ReciprocalPairedMemorySelection",
]
