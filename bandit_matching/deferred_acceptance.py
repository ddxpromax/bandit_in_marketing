"""Gale–Shapley deferred acceptance; agents propose, arms dispose (agent-optimal stable matching)."""

from __future__ import annotations

import numpy as np


def prefs_from_utility_rows(U: np.ndarray) -> list[list[int]]:
    """Strict total order: row i lists arm indices from most to least preferred by utility."""
    n = U.shape[1]
    order = np.argsort(-U, axis=1)
    return [list(order[i]) for i in range(U.shape[0])]


def gale_shapley_agent_optimal(
    agent_prefs: list[list[int]],
    arm_prefs: list[list[int]],
) -> tuple[list[int], list[int]]:
    """
    Returns:
        a2b: agent i -> matched arm index
        b2a: arm j -> matched agent index
    """
    n = len(agent_prefs)
    if len(arm_prefs) != n:
        raise ValueError("Expect square market (|agents| == |arms|).")

    arm_rank: list[list[int]] = []
    for j in range(n):
        r = [0] * n
        for rank, a in enumerate(arm_prefs[j]):
            r[a] = rank
        arm_rank.append(r)

    nxt = [0] * n
    b2a: list[int | None] = [None] * n
    free = list(range(n))

    while free:
        i = free.pop()
        if nxt[i] >= n:
            continue
        j = agent_prefs[i][nxt[i]]
        nxt[i] += 1

        cur = b2a[j]
        if cur is None:
            b2a[j] = i
        else:
            if arm_rank[j][i] < arm_rank[j][cur]:
                b2a[j] = i
                free.append(cur)
            else:
                free.append(i)

    if any(x is None for x in b2a):
        raise RuntimeError("Incomplete matching; check preference lists.")

    b2a_i = [x for x in b2a]
    a2b = [0] * n
    for j in range(n):
        a2b[b2a_i[j]] = j
    return a2b, b2a_i
