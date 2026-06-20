"""
Derive all binary prediction market probabilities from a joint score matrix.

The core model outputs P(score_a=i, score_b=j) as an NxN matrix.
Every downstream market is a linear function of that matrix — no extra models needed.

Usage:
    from markets import score_matrix_markets
    markets = score_matrix_markets(joint)   # joint[i,j] = P(team_a=i goals, team_b=j goals)
"""
from __future__ import annotations

import numpy as np


def score_matrix_markets(joint: np.ndarray) -> dict[str, float]:
    """
    Derive all binary/ternary market probabilities from the joint score matrix.

    Args:
        joint: NxN array where joint[i, j] = P(team_a scores i, team_b scores j).
               Does not need to be normalized — function normalizes internally.

    Returns:
        Dict of market names → probabilities in [0, 1].
    """
    j = joint / joint.sum()
    N = j.shape[0]
    total_goals = np.add.outer(np.arange(N), np.arange(N))  # total_goals[i,j] = i+j

    return {
        # 1X2
        "home_win":    float(np.tril(j, -1).sum()),     # i > j (team_a wins)
        "draw":        float(np.diag(j).sum()),          # i == j
        "away_win":    float(np.triu(j, 1).sum()),       # j > i (team_b wins)

        # Over / Under
        "over_0_5":    float((j * (total_goals >= 1)).sum()),
        "over_1_5":    float((j * (total_goals >= 2)).sum()),
        "over_2_5":    float((j * (total_goals >= 3)).sum()),
        "over_3_5":    float((j * (total_goals >= 4)).sum()),
        "over_4_5":    float((j * (total_goals >= 5)).sum()),

        # Goal scorer markets
        "btts":         float(j[1:, 1:].sum()),                         # both teams score
        "home_scores":  float(1.0 - j[0, :].sum()),                     # team_a scores >= 1
        "away_scores":  float(1.0 - j[:, 0].sum()),                     # team_b scores >= 1
        "home_cs":      float(j[:, 0].sum()),                           # team_b scoreless
        "away_cs":      float(j[0, :].sum()),                           # team_a scoreless
        "home_2plus":   float(j[2:, :].sum()),                          # team_a scores >= 2
        "away_2plus":   float(j[:, 2:].sum()),                          # team_b scores >= 2
        "home_3plus":   float(j[3:, :].sum()),
        "away_3plus":   float(j[:, 3:].sum()),
    }


def expected_goals_from_matrix(joint: np.ndarray) -> tuple[float, float]:
    """Return (xG_a, xG_b) from the joint matrix."""
    j = joint / joint.sum()
    N = j.shape[0]
    idx = np.arange(N, dtype=float)
    xg_a = float((j.sum(axis=1) * idx).sum())
    xg_b = float((j.sum(axis=0) * idx).sum())
    return xg_a, xg_b
