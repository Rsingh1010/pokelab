"""Plain-language explanations for an optimized team.

Per member:
  - covers:          defending types it hits super-effectively (own-type attacks)
  - only_answer:     types NOBODY else on the team hits super-effectively
  - resists:         attacking types it resists or is immune to
  - patches:         attacking types it resists that a teammate is WEAK to
  - weak_to:         its own weaknesses
  - roles:           fast / bulky / physical / special tags
  - contribution:    team score minus the score of the other five alone
                     ("what would we lose without it"). Can be small or negative:
                     a member may be there mainly because 6 slots must be filled
                     within the weakness / type limits.
Team level: gaps (types not hit / not resisted), thin spots (covered once), shared
weaknesses, and the stat that limits the balance score.
"""
import numpy as np
import pandas as pd

from src.features.preprocessing import STAT_COLS
from src.optimization.team_optimizer import DEFAULT_WEIGHTS, ROLES, Matrices, score_team
from src.optimization.type_chart import TYPES


def _types(mask) -> list:
    return [t for t, on in zip(TYPES, mask) if on]


def explain_members(M: Matrices, idx, weights=DEFAULT_WEIGHTS) -> pd.DataFrame:
    idx = list(idx)
    total = score_team(M, idx, weights)["total"]
    rows = []
    for i in idx:
        others = [j for j in idx if j != i]
        others_off = M.offense[others].sum(axis=0)
        others_weak = M.weak[others].sum(axis=0)
        rows.append({
            "name": M.names[i],
            "covers": _types(M.offense[i]),
            "only_answer": _types(M.offense[i] & (others_off == 0)),
            "resists": _types(M.resist[i]),
            "patches": _types(M.resist[i] & (others_weak > 0)),
            "weak_to": _types(M.weak[i]),
            "roles": [r for r, on in zip(ROLES, M.roles[i]) if on],
            "contribution": total - score_team(M, others, weights)["total"],
        })
    return pd.DataFrame(rows)


def explain_team(M: Matrices, idx, weights=DEFAULT_WEIGHTS) -> dict:
    idx = list(idx)
    off, res, weak = M.offense[idx].sum(0), M.resist[idx].sum(0), M.weak[idx].sum(0)
    team_pct = M.pct[idx].mean(axis=0)
    return {
        "scores": score_team(M, idx, weights),
        "no_super_effective_answer": _types(off == 0),
        "only_one_answer": _types(off == 1),
        "nothing_resists": _types(res == 0),
        "only_one_resist": _types(res == 1),
        "shared_weaknesses": {t: int(n) for t, n in zip(TYPES, weak) if n >= 2},
        "roles_missing": [r for r, n in zip(ROLES, M.roles[idx].sum(0)) if n == 0],
        "weakest_stat": STAT_COLS[int(np.argmin(team_pct))],
        "weakest_stat_percentile": float(team_pct.min()),
    }


if __name__ == "__main__":
    import json

    from src.features.preprocessing import load_pokemon
    from src.optimization.team_optimizer import optimize
    from src.optimization.type_chart import load_type_chart

    team, _, pool, M = optimize(load_pokemon(), load_type_chart())
    idx = [M.names.index(n) for n in team["name"]]
    pd.set_option("display.max_colwidth", 60, "display.width", 200)
    print(explain_members(M, idx).round(3).to_string(index=False))
    print("\nTeam:", json.dumps(explain_team(M, idx), indent=2))
