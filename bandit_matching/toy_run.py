"""
Minimal simulation: random utilities, oracle utilitarian stable matching (small N),
and a naive exploration phase + DA on estimated preferences (not the full paper ETC).
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许在 bandit_matching/ 目录下直接 `python toy_run.py`（把项目根加入 path）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from bandit_matching.deferred_acceptance import gale_shapley_agent_optimal, prefs_from_utility_rows
from bandit_matching.stable_bruteforce import best_stable_by_objective


def main() -> None:
    rng = np.random.default_rng(0)
    n = 4
    # Ground-truth means (subgaussian noise can be layered here later)
    Ua = rng.uniform(0.5, 3.0, size=(n, n))
    Ub = rng.uniform(0.5, 3.0, size=(n, n))

    opt_match, opt_w = best_stable_by_objective(Ua, Ub, "utilitarian")
    print("Oracle utilitarian-optimal stable matching (agent -> arm):", opt_match)
    print("Oracle utilitarian welfare:", round(opt_w, 4))

    # Naive exploration: each (agent, arm) pair pulled `pulls_per_pair` times
    pulls_per_pair = 40
    noise_scale = 0.3
    est_a = np.zeros((n, n))
    est_b = np.zeros((n, n))
    counts = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            for _ in range(pulls_per_pair):
                est_a[i, j] += rng.normal(Ua[i, j], noise_scale)
                est_b[j, i] += rng.normal(Ub[j, i], noise_scale)
                counts[i, j] += 1
    est_a /= counts
    est_b /= counts

    ap = prefs_from_utility_rows(est_a)
    # Ub[j, i]: arm j's utility for agent i -> row j is arm j's vector over agents.
    arm_prefs = prefs_from_utility_rows(est_b)

    a2b, _ = gale_shapley_agent_optimal(ap, arm_prefs)
    learned_perm = tuple(a2b)
    from bandit_matching.stable_bruteforce import utilitarian_welfare

    learned_w = utilitarian_welfare(learned_perm, Ua, Ub)
    print("After estimation + agent-proposing DA, true utilitarian welfare:", round(learned_w, 4))
    print("Regret (oracle - learned):", round(opt_w - learned_w, 4))


if __name__ == "__main__":
    main()
