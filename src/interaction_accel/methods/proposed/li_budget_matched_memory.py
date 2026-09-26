"""Light-Interaction-budget-matched causal pose memory for Matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .reciprocal_paired_memory import MatrixReciprocalPairedMemoryCompiler


@dataclass(frozen=True)
class BudgetMatchedMemoryAtom:
    original_time_index: int
    pose_distance: float


@dataclass(frozen=True)
class BudgetMatchedMemorySelection:
    atoms: tuple[Any, ...]
    remote_atoms: tuple[BudgetMatchedMemoryAtom, ...]
    requested_atoms: int
    candidate_count: int


class MatrixBudgetMatchedPostControlPoseMemoryCompiler(
    MatrixReciprocalPairedMemoryCompiler
):
    """Select a causal pose basis with the released LI cardinality.

    The adapter supplies the number of historical latent slots used by the
    released Light Interaction schedule for the same chunk.  Slot zero is the
    immutable post-control anchor (latent 1), while every remaining slot is
    filled by a distinct causal atom nearest to the current camera trajectory.
    The rule uses neither generated pixels nor benchmark/evaluation signals.
    """

    name = "matrix_budget_matched_post_control_pose_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        if not atoms:
            raise RuntimeError("budget-matched memory requires a non-empty archive")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("budget-matched memory requires latent 1 as its sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError(
                f"only {len(eligible)} causal atoms for {remote_budget} remote slots"
            )
        ranked = sorted(
            eligible,
            key=lambda atom: (
                self._trajectory_distance(getattr(atom, "c2w"), current_c2ws),
                -self._index(atom),
            ),
        )
        selected_remote = ranked[:remote_budget]
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("budget-matched memory violated cardinality or uniqueness")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(
                    getattr(atom, "c2w"), current_c2ws
                ),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedActionSectionPoseMemoryCompiler(
    MatrixBudgetMatchedPostControlPoseMemoryCompiler
):
    """Reserve a same-action causal section, then complete by pose locality.

    The cardinality and latent-1 anchor are identical to released LI.  Among
    the remote slots, ``ceil(K/2)`` are assigned to causal atoms whose control
    vector equals the current control section; all unfilled slots come from
    the global pose-nearest ordering.  The integer split is fixed by the
    memory budget and is not tuned from pixels or evaluation measurements.
    """

    name = "matrix_budget_matched_action_section_pose_memory_compiler"

    @staticmethod
    def _same_action(left: torch.Tensor, right: torch.Tensor) -> bool:
        left = left.detach().float().reshape(-1)
        right = right.detach().float().reshape(-1).to(device=left.device)
        return bool(left.shape == right.shape and torch.equal(left, right))

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("action-section memory requires latent 1 as its sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")
        if remote_budget == 0:
            return BudgetMatchedMemorySelection(
                atoms=(sink,),
                remote_atoms=(),
                requested_atoms=1,
                candidate_count=len(eligible),
            )

        def key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(getattr(atom, "c2w"), current_c2ws),
                -self._index(atom),
            )

        same_action = sorted(
            [
                atom
                for atom in eligible
                if self._same_action(getattr(atom, "action"), current_action)
            ],
            key=key,
        )
        primary_budget = (remote_budget + 1) // 2
        selected_remote = list(same_action[:primary_budget])
        selected_ids = {id(atom) for atom in selected_remote}
        for atom in sorted(eligible, key=key):
            if id(atom) in selected_ids:
                continue
            selected_remote.append(atom)
            selected_ids.add(id(atom))
            if len(selected_remote) == remote_budget:
                break
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("action-section memory violated cardinality or uniqueness")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedLocalPoseSimplexMemoryCompiler(
    MatrixBudgetMatchedPostControlPoseMemoryCompiler
):
    """Build a non-redundant local pose coreset at the released LI budget.

    The nearest causal atom is retained as the local witness.  Remaining slots
    are greedily filled inside the fixed ``2K`` pose-nearest domain by the atom
    that maximizes its minimum SE(3) chordal distance to the selected remote
    basis.  This deterministic max-min simplex avoids spending all four slots
    on adjacent latent slices while preventing arbitrarily distant memories.
    """

    name = "matrix_budget_matched_local_pose_simplex_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("local-pose simplex memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")
        ranked = sorted(
            eligible,
            key=lambda atom: (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            ),
        )
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            selected_remote = [domain[0]]
            remaining = list(domain[1:])
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("local-pose simplex memory violated cardinality or uniqueness")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler(
    MatrixBudgetMatchedLocalPoseSimplexMemoryCompiler
):
    """Use one current-action seed inside the local pose simplex.

    Unlike hard action-section routing, only the first remote vertex is
    action-conditioned.  Its domain is still the same fixed local ``2K`` pose
    neighborhood; every remaining vertex follows the geometry-only max-min
    simplex rule.  Thus action evidence orients the coreset without consuming
    half of its geometric degrees of freedom.
    """

    name = "matrix_budget_matched_action_seeded_pose_simplex_memory_compiler"

    @staticmethod
    def _same_action(left: torch.Tensor, right: torch.Tensor) -> bool:
        left = left.detach().float().reshape(-1)
        right = right.detach().float().reshape(-1).to(device=left.device)
        return bool(left.shape == right.shape and torch.equal(left, right))

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("action-seeded simplex memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            action_local = [
                atom
                for atom in domain
                if self._same_action(atom.action, current_action)
            ]
            seed = min(action_local, key=local_key) if action_local else domain[0]
            selected_remote = [seed]
            remaining = [atom for atom in domain if id(atom) != id(seed)]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("action-seeded simplex memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedActionTangentSecantMemoryCompiler(
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler
):
    """Join one action tangent, one local witness, and pose secants.

    The first remote vertex is the current-action seed from the local domain.
    When the budget permits, a distinct pose-nearest vertex is fixed as the
    local tangent witness.  Remaining vertices are max-min pose secants.  This
    gives the four-slot case a deterministic ``2 local + 2 diverse`` basis.
    """

    name = "matrix_budget_matched_action_tangent_secant_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("action tangent-secant memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            action_local = [
                atom
                for atom in domain
                if self._same_action(atom.action, current_action)
            ]
            seed = min(action_local, key=local_key) if action_local else domain[0]
            selected_remote = [seed]
            if remote_budget >= 2:
                local_witness = next(
                    atom for atom in domain if id(atom) != id(seed)
                )
                selected_remote.append(local_witness)
            selected_ids = {id(atom) for atom in selected_remote}
            remaining = [atom for atom in domain if id(atom) not in selected_ids]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("action tangent-secant memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedActionTerminalPoseSimplexMemoryCompiler(
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler
):
    """Orient R4's local pose simplex at the current trajectory terminal pose.

    R4 chooses its same-action seed by distance to any pose along the current
    trajectory.  That can select a historical atom near the trajectory entry
    even though Memory conditions the next denoising boundary.  This variant
    keeps R4's fixed local ``2K`` domain and max-min completion, but ranks the
    single same-action seed by distance to the causal trajectory endpoint.
    No slot, model-call, or benchmark-dependent budget is added.
    """

    name = "matrix_budget_matched_action_terminal_pose_simplex_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        if current_c2ws.ndim != 3 or tuple(current_c2ws.shape[1:]) != (4, 4):
            raise ValueError("current_c2ws must have shape [T, 4, 4]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("action-terminal simplex memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        def terminal_key(atom: Any) -> tuple[float, float, int]:
            return (
                self._pose_distance(atom.c2w, current_c2ws[-1]),
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            action_local = [
                atom
                for atom in domain
                if self._same_action(atom.action, current_action)
            ]
            seed = min(action_local, key=terminal_key) if action_local else domain[0]
            selected_remote = [seed]
            remaining = [atom for atom in domain if id(atom) != id(seed)]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("action-terminal simplex memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedActionTrajectoryPhaseSimplexMemoryCompiler(
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler
):
    """Spread R4's coreset over the current trajectory phase before pose.

    Each local candidate is assigned to its nearest pose along the current
    causal trajectory.  After R4's one same-action seed, max-min completion is
    lexicographic in normalized trajectory phase and then SE(3) pose.  This
    distinguishes entry-, middle-, and terminal-aligned memories without a
    fitted mixing coefficient or an additional memory slot.
    """

    name = "matrix_budget_matched_action_trajectory_phase_simplex_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        if current_c2ws.ndim != 3 or tuple(current_c2ws.shape[1:]) != (4, 4):
            raise ValueError("current_c2ws must have shape [T, 4, 4]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("trajectory-phase simplex memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        def nearest_phase(atom: Any) -> int:
            return min(
                range(int(current_c2ws.shape[0])),
                key=lambda index: (
                    self._pose_distance(atom.c2w, current_c2ws[index]),
                    index,
                ),
            )

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            phases = {id(atom): nearest_phase(atom) for atom in domain}
            action_local = [
                atom
                for atom in domain
                if self._same_action(atom.action, current_action)
            ]
            seed = min(action_local, key=local_key) if action_local else domain[0]
            selected_remote = [seed]
            remaining = [atom for atom in domain if id(atom) != id(seed)]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            abs(phases[id(remaining[index])] - phases[id(selected)])
                            for selected in selected_remote
                        ),
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("trajectory-phase simplex memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedOrientedReciprocalPoseSimplexMemoryCompiler(
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler
):
    """Give R4's action-primary vertex one inverse-action support vertex.

    The first remote vertex remains the nearest current-action atom in R4's
    local ``2K`` domain.  When another slot and an exact inverse-action atom
    exist, one nearest reciprocal atom is selected as its oriented support.
    Remaining slots retain R4's max-min SE(3) completion.  The rule is a
    deterministic discrete orientation of one coreset, not a second selector.
    """

    name = "matrix_budget_matched_oriented_reciprocal_pose_simplex_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("oriented reciprocal simplex memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            action_local = [
                atom
                for atom in domain
                if self._same_action(atom.action, current_action)
            ]
            seed = min(action_local, key=local_key) if action_local else domain[0]
            selected_remote = [seed]
            if remote_budget >= 2:
                reciprocal_local = [
                    atom
                    for atom in domain
                    if id(atom) != id(seed)
                    and self._is_reciprocal(atom.action, current_action)
                ]
                if reciprocal_local:
                    selected_remote.append(min(reciprocal_local, key=local_key))
            selected_ids = {id(atom) for atom in selected_remote}
            remaining = [atom for atom in domain if id(atom) not in selected_ids]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("oriented reciprocal simplex memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedCompletedReciprocalPoseSimplexMemoryCompiler(
    MatrixBudgetMatchedOrientedReciprocalPoseSimplexMemoryCompiler
):
    """Complete R4's local domain with the nearest oriented action pair.

    A pure pose-nearest ``2K`` domain may contain neither the current-action
    section nor its inverse.  This variant deterministically completes that
    domain with at most the nearest causal current-action primary and nearest
    inverse-action support, then selects the same fixed number of vertices.
    The completion changes candidate availability, never Memory cardinality.
    """

    name = "matrix_budget_matched_completed_reciprocal_pose_simplex_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("completed reciprocal simplex memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = list(ranked[: max(remote_budget, 2 * remote_budget)])
            domain_ids = {id(atom) for atom in domain}
            action_all = [
                atom for atom in ranked if self._same_action(atom.action, current_action)
            ]
            reciprocal_all = [
                atom for atom in ranked if self._is_reciprocal(atom.action, current_action)
            ]
            for witness in (
                action_all[0] if action_all else None,
                reciprocal_all[0] if reciprocal_all else None,
            ):
                if witness is not None and id(witness) not in domain_ids:
                    domain.append(witness)
                    domain_ids.add(id(witness))
            seed = action_all[0] if action_all else domain[0]
            selected_remote = [seed]
            if remote_budget >= 2 and reciprocal_all:
                support = reciprocal_all[0]
                if id(support) != id(seed):
                    selected_remote.append(support)
            selected_ids = {id(atom) for atom in selected_remote}
            remaining = [atom for atom in domain if id(atom) not in selected_ids]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("completed reciprocal simplex memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


class MatrixBudgetMatchedActionChordBridgePoseSimplexMemoryCompiler(
    MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler
):
    """Choose R4's one action seed as a minimax trajectory-chord bridge.

    R4 may attach its seed to any interior pose and R6 over-corrects toward the
    terminal pose.  This final variant instead minimizes the worse of the
    seed's distances to the current trajectory entry and terminal poses.  The
    seed therefore bridges the whole current motion chord; all other vertices
    and the local ``2K`` domain remain R4's continuous SE(3) simplex.
    """

    name = "matrix_budget_matched_action_chord_bridge_pose_simplex_memory_compiler"

    def select(
        self,
        atoms: Sequence[Any],
        *,
        current_c2ws: torch.Tensor,
        current_action: torch.Tensor,
        requested_atoms: int,
        recent_exclusion: int,
    ) -> BudgetMatchedMemorySelection:
        requested_atoms = int(requested_atoms)
        if not 1 <= requested_atoms <= 5:
            raise ValueError("LI-compatible memory budget must be in [1, 5]")
        if current_c2ws.ndim != 3 or tuple(current_c2ws.shape[1:]) != (4, 4):
            raise ValueError("current_c2ws must have shape [T, 4, 4]")
        sink = next(
            (atom for atom in atoms if bool(getattr(atom, "is_sink", False))),
            None,
        )
        if sink is None or self._index(sink) != 1:
            raise RuntimeError("action-chord bridge memory requires latent 1 as sink")
        newest = max(self._index(atom) for atom in atoms)
        eligible = [
            atom
            for atom in atoms
            if atom is not sink
            and self._index(atom) <= newest - int(recent_exclusion)
        ]
        remote_budget = requested_atoms - 1
        if len(eligible) < remote_budget:
            raise RuntimeError("insufficient causal atoms for LI-compatible budget")

        def local_key(atom: Any) -> tuple[float, int]:
            return (
                self._trajectory_distance(atom.c2w, current_c2ws),
                -self._index(atom),
            )

        def bridge_key(atom: Any) -> tuple[float, float, int]:
            entry = self._pose_distance(atom.c2w, current_c2ws[0])
            terminal = self._pose_distance(atom.c2w, current_c2ws[-1])
            return (max(entry, terminal), entry + terminal, -self._index(atom))

        ranked = sorted(eligible, key=local_key)
        if remote_budget == 0:
            selected_remote: list[Any] = []
        else:
            domain = ranked[: max(remote_budget, 2 * remote_budget)]
            action_local = [
                atom
                for atom in domain
                if self._same_action(atom.action, current_action)
            ]
            seed = min(action_local, key=bridge_key) if action_local else domain[0]
            selected_remote = [seed]
            remaining = [atom for atom in domain if id(atom) != id(seed)]
            while len(selected_remote) < remote_budget:
                chosen_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            self._pose_distance(
                                remaining[index].c2w, selected.c2w
                            )
                            for selected in selected_remote
                        ),
                        -self._trajectory_distance(
                            remaining[index].c2w, current_c2ws
                        ),
                        self._index(remaining[index]),
                    ),
                )
                selected_remote.append(remaining.pop(chosen_index))
        selected = (sink, *selected_remote)
        if len(selected) != requested_atoms or len({id(atom) for atom in selected}) != len(selected):
            raise RuntimeError("action-chord bridge memory violated cardinality")
        records = tuple(
            BudgetMatchedMemoryAtom(
                original_time_index=self._index(atom),
                pose_distance=self._trajectory_distance(atom.c2w, current_c2ws),
            )
            for atom in selected_remote
        )
        return BudgetMatchedMemorySelection(
            atoms=tuple(selected),
            remote_atoms=records,
            requested_atoms=requested_atoms,
            candidate_count=len(eligible),
        )


__all__ = [
    "BudgetMatchedMemoryAtom",
    "BudgetMatchedMemorySelection",
    "MatrixBudgetMatchedPostControlPoseMemoryCompiler",
    "MatrixBudgetMatchedActionSectionPoseMemoryCompiler",
    "MatrixBudgetMatchedLocalPoseSimplexMemoryCompiler",
    "MatrixBudgetMatchedActionSeededPoseSimplexMemoryCompiler",
    "MatrixBudgetMatchedActionTangentSecantMemoryCompiler",
    "MatrixBudgetMatchedActionTerminalPoseSimplexMemoryCompiler",
    "MatrixBudgetMatchedActionTrajectoryPhaseSimplexMemoryCompiler",
    "MatrixBudgetMatchedOrientedReciprocalPoseSimplexMemoryCompiler",
    "MatrixBudgetMatchedCompletedReciprocalPoseSimplexMemoryCompiler",
    "MatrixBudgetMatchedActionChordBridgePoseSimplexMemoryCompiler",
]
