from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(
    sets: Sequence[Mapping[int, Mapping[str, float]]],
    accept: Callable[[Mapping[str, float], float], bool],
    floor: float = 0.35,
):
    gids = sorted({int(gid) for rows in sets for gid in rows})
    count = len(sets)
    if count == 0 or not gids:
        return {}

    banned = set()
    while True:
        cols = len(gids) + count
        matrix = np.full((count, cols), float(floor), np.float32)
        for i, rows in enumerate(sets):
            for j, gid in enumerate(gids):
                if (int(i), int(gid)) in banned:
                    continue
                item = rows.get(int(gid))
                if item is not None:
                    matrix[i, j] = float(item["score"])

        rr, cc = linear_sum_assignment(-matrix)
        chosen = {
            int(i): int(gids[j])
            for i, j in zip(rr.tolist(), cc.tolist())
            if j < len(gids) and matrix[i, j] > float(floor)
        }
        rejected = []
        for i, gid in chosen.items():
            item = sets[i].get(gid)
            if item is None:
                rejected.append((i, gid))
                continue

            used_elsewhere = {
                other for row, other in chosen.items() if row != i
            }
            alternatives = [
                float(value["score"])
                for other_gid, value in sets[i].items()
                if int(other_gid) != int(gid)
                and int(other_gid) not in used_elsewhere
                and (int(i), int(other_gid)) not in banned
            ]
            second = max(alternatives, default=0.0)
            if not accept(item, second):
                rejected.append((int(i), int(gid)))

        if not rejected:
            return chosen

        before = len(banned)
        banned.update(rejected)
        if len(banned) == before:
            return chosen
