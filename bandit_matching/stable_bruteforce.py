"""Enumerate all one-to-one matchings (small N only): stable check + welfare objectives."""

from __future__ import annotations

import itertools

import numpy as np


def _is_stable(a2b: tuple[int, ...], Ua: np.ndarray, Ub: np.ndarray) -> bool:
    n = len(a2b)
    b2a = [0] * n
    for i, j in enumerate(a2b):
        b2a[j] = i
    for i in range(n):
        for j in range(n):
            if a2b[i] == j:
                continue
            if Ua[i, j] > Ua[i, a2b[i]] and Ub[j, i] > Ub[j, b2a[j]]:
                return False
    return True


def utilitarian_welfare(a2b: tuple[int, ...], Ua: np.ndarray, Ub: np.ndarray) -> float:
    n = len(a2b)
    return float(sum(Ua[i, a2b[i]] + Ub[a2b[i], i] for i in range(n)))


def rawlsian_welfare(a2b: tuple[int, ...], Ua: np.ndarray, Ub: np.ndarray) -> float:
    """Maximin over 2N individual utilities (agent + arm payoffs)."""
    n = len(a2b)
    payoffs: list[float] = []
    for i in range(n):
        j = a2b[i]
        payoffs.append(float(Ua[i, j]))
        payoffs.append(float(Ub[j, i]))
    return min(payoffs)


def best_stable_by_objective(
    Ua: np.ndarray,
    Ub: np.ndarray,
    objective: str,
) -> tuple[tuple[int, ...], float]:
    """
    objective: 'utilitarian' | 'rawlsian'
    """
    n = Ua.shape[0]
    best_m: tuple[int, ...] | None = None
    best_v = float("-inf")
    for perm in itertools.permutations(range(n)):
        if not _is_stable(perm, Ua, Ub):
            continue
        if objective == "utilitarian":
            v = utilitarian_welfare(perm, Ua, Ub)
        elif objective == "rawlsian":
            v = rawlsian_welfare(perm, Ua, Ub)
        else:
            raise ValueError(objective)
        if v > best_v:
            best_v = v
            best_m = perm
    if best_m is None:
        raise RuntimeError("No stable matching found (unexpected).")
    return best_m, best_v
