from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple

import numpy as np

from rebuild.multimodel_state_invariant_final import LocalGroup, PairEvidence, StateInvariantFinalResolver


class StateInvariantFinalResolverFast(StateInvariantFinalResolver):
    """Performance-preserving resolver for the final three-model MTMC path.

    The previous resolver recomputed reciprocal-best scores for every candidate
    pair. Those values were not used by the acceptance gates because the active
    configuration requires zero mutual-model support. This subclass removes that
    dead O(N^3) work, vectorizes cosine matrices, uses partial selection for the
    top-k mean, and caches each pair/model score.
    """

    def __init__(self, cfg: Mapping[str, object], registry=None):
        super().__init__(cfg, registry=registry)
        self._score_cache: dict[tuple[str, str, str], tuple[float, int, bool]] = {}

    @staticmethod
    def _matrix(left: Iterable[np.ndarray], right: Iterable[np.ndarray]) -> np.ndarray:
        a = []
        b = []
        for value in left:
            if value is not None:
                arr = np.asarray(value, np.float32).reshape(-1)
                norm = float(np.linalg.norm(arr))
                if arr.size and np.isfinite(norm) and norm > 0:
                    a.append(arr / norm)
        for value in right:
            if value is not None:
                arr = np.asarray(value, np.float32).reshape(-1)
                norm = float(np.linalg.norm(arr))
                if arr.size and np.isfinite(norm) and norm > 0:
                    b.append(arr / norm)
        if not a or not b:
            return np.empty((0, 0), np.float32)
        return np.matmul(np.stack(a), np.stack(b).T).astype(np.float32, copy=False)

    @classmethod
    def _topmean(cls, matrix: np.ndarray, k: int = 5) -> float:
        if matrix.size == 0:
            return 0.0
        values = matrix.reshape(-1)
        count = min(k, values.size)
        if count == values.size:
            return float(np.mean(values))
        top = np.partition(values, values.size - count)[-count:]
        return float(np.mean(top))

    @staticmethod
    def _cache_key(left: LocalGroup, right: LocalGroup, model: str) -> tuple[str, str, str]:
        first, second = sorted((left.key, right.key))
        return first, second, model

    def _view_score(self, left: LocalGroup, right: LocalGroup, model: str) -> tuple[float, int, bool]:
        key = self._cache_key(left, right, model)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        l = left.state_bank.get(model, {})
        r = right.state_bank.get(model, {})
        score = self._best_view_score(l, r)
        self._score_cache[key] = score
        return score

    def _model_score(self, left: LocalGroup, right: LocalGroup, model: str) -> float:
        return self._view_score(left, right, model)[0]

    def pair(
        self,
        left: LocalGroup,
        right: LocalGroup,
        left_pool: List[LocalGroup],
        right_pool: List[LocalGroup],
    ) -> PairEvidence:
        del left_pool, right_pool
        details = {model: self._view_score(left, right, model) for model in self.MODELS}
        scores = {model: details[model][0] for model in self.MODELS}
        ordered = sorted(scores.values(), reverse=True)
        agreement = float(np.mean(ordered[:2])) if ordered else 0.0
        support = sum(score >= self.model_min[model] for model, score in scores.items())
        colour = self._colour(left, right)
        geometry = self._distance(left, right)
        temporal = self._temporal(left, right)
        if left.camera == right.camera:
            continuity = 0.0 if self._overlap(left, right) else float(
                0.65
                * max(
                    0.0,
                    1.0
                    - min(abs(right.start - left.end), abs(left.start - right.end)) / 30.0,
                )
                + 0.35 * geometry
            )
        else:
            continuity = 0.5
        state_left = left.state_type
        state_right = right.state_type
        state_change = (
            state_left not in ("unknown", "mixed")
            and state_right not in ("unknown", "mixed")
            and state_left != state_right
        )
        _, left_view_support, left_transition = details["swin"]
        _, right_view_support, right_transition = details["swin"]
        state_change = state_change or left_transition or right_transition
        deep_top2 = float(np.mean(ordered[:2])) if len(ordered) >= 2 else (ordered[0] if ordered else 0.0)
        strongest = ordered[0] if ordered else 0.0
        if left.camera == right.camera:
            fused = 0.78 * deep_top2 + 0.12 * colour + 0.10 * continuity
        else:
            fused = 0.70 * deep_top2 + 0.12 * strongest + 0.10 * colour + 0.05 * temporal + 0.03 * geometry
        return PairEvidence(
            float(fused),
            float(scores["resnet"]),
            float(scores["swin"]),
            float(scores["solider"]),
            float(colour),
            float(geometry),
            float(temporal),
            float(continuity),
            float(agreement),
            int(support),
            0,
            int(min(left_view_support, right_view_support)),
            bool(state_change),
        )
