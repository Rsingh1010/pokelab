"""Six-Pokémon team optimizer: an integer program solved with PuLP + CBC.

Decision variables
    x_i in {0,1}   1 if Pokémon i is on the team
    c_t in [0,2]   offensive coverage DEPTH of defending type t  (c_t <= sum_i OFF[i,t] x_i)
    d_t in [0,2]   defensive coverage DEPTH vs attacking type t  (d_t <= sum_i RES[i,t] x_i)
    g_r in [0,1]   role r covered (fast/bulky/physical/special) (g_r <= sum_i ROLE[i,r] x_i)
    m   in [0,1]   balance: lowest team-average stat percentile (m <= avg_i PCT[i,s] for every stat s)

Maximize   w_cov * sum(c)/36 + w_def * sum(d)/36 + w_role * sum(g)/4 + w_bal * m
           (+ a tiny tie-breaker favoring higher total BST among equally good teams)

Subject to
    sum_i x_i = 6
    sum_i WEAK[i,t] x_i <= max_weak         for every attacking type t
                                             (WEAK = 1 for a 2x weakness, 2 for a 4x weakness)
    sum_i HAS_TYPE[i,T] x_i <= max_per_type  for every type T
    x_i = 1 for locked Pokémon
Pool filters (applied before solving): fully evolved, BST >= min_bst, legendaries optional.

Why the c/d/g/m variables work: the objective pushes them up, the constraints cap
them at what the chosen team actually achieves. With x integer, c_t ends up equal to
min(number of members covering type t, 2). This keeps everything linear.

Why depth 2: with "at least one member" (depth 1), many teams hit a perfect score on
coverage and defense, so the weights stop mattering. Depth 2 also matches play: if your
only answer to a type is knocked out, you have none left.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp
from sklearn.decomposition import PCA

from src.config import RANDOM_SEED
from src.features.preprocessing import STAT_COLS, make_features
from src.optimization.type_chart import TYPES, defensive_profile, offensive_profile

TEAM_SIZE = 6
TERMS = ("coverage", "defense", "roles", "balance")
DEFAULT_WEIGHTS = {"coverage": 0.4, "defense": 0.3, "roles": 0.2, "balance": 0.1}
ROLES = ("fast", "bulky", "physical", "special")
DEFAULT_MIN_BST = 500   # median BST of fully evolved Pokémon
DEFAULT_MAX_WEAK = 2    # rule of thumb: weakness load per attacking type <= 2 (two 2x members, or one 4x)
DEFAULT_MAX_PER_TYPE = 2
TIEBREAK = 1e-5
COVER_DEPTH = 2         # a type counts as fully covered when 2 members cover it


class InfeasibleTeamError(ValueError):
    pass


# ------------------------------------------------------------------ inputs
def role_axes(fully_evolved: pd.DataFrame) -> pd.DataFrame:
    """Each Pokémon's position on the two validated role axes (PCA of relative stats).

    PCA axis signs are arbitrary, so we flip them to fixed meanings:
      speed_axis    high = fast & frail,  low = slow & bulky
      physical_axis high = physical,      low = special
    """
    X, _ = make_features(fully_evolved, "relative")
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(X)
    comps = pca.components_.copy()
    if comps[0, STAT_COLS.index("speed")] < 0:
        comps[0] *= -1
    if comps[1, STAT_COLS.index("attack")] < 0:
        comps[1] *= -1
    scores = (X.to_numpy() - pca.mean_) @ comps.T
    return pd.DataFrame(scores, index=fully_evolved.index, columns=["speed_axis", "physical_axis"])


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fully evolved Pokémon with role tags (top/bottom third of each axis) and stat percentiles."""
    fe = df[df["is_fully_evolved"]].copy()
    fe = fe.join(role_axes(fe))
    lo_s, hi_s = fe["speed_axis"].quantile([1 / 3, 2 / 3])
    lo_p, hi_p = fe["physical_axis"].quantile([1 / 3, 2 / 3])
    fe["fast"], fe["bulky"] = fe["speed_axis"] >= hi_s, fe["speed_axis"] <= lo_s
    fe["physical"], fe["special"] = fe["physical_axis"] >= hi_p, fe["physical_axis"] <= lo_p
    for s in STAT_COLS:
        fe[f"{s}_pct"] = fe[s].rank(pct=True)   # percentile among ALL fully evolved Pokémon
    return fe


def select_pool(fe: pd.DataFrame, min_bst: int, allow_legendary: bool, locked=()) -> pd.DataFrame:
    """Filter candidates. Locked Pokémon are always kept: the user asked for them explicitly."""
    locked = list(locked)
    missing = set(locked) - set(fe["name"])
    if missing:
        raise InfeasibleTeamError(f"Not fully evolved or unknown: {sorted(missing)}")
    keep = fe["bst"] >= min_bst
    if not allow_legendary:
        keep &= ~(fe["is_legendary"] | fe["is_mythical"])
    keep |= fe["name"].isin(locked)
    return fe[keep].reset_index(drop=True)


@dataclass
class Matrices:
    """0/1 matrices, one row per pool Pokémon. This is everything the solver sees."""
    names: list
    bst: np.ndarray
    offense: np.ndarray    # n x 18: hits defending type t super-effectively with an own-type attack
    resist: np.ndarray     # n x 18: takes <= 0.5x from attacking type t (resist or immune)
    weak: np.ndarray       # n x 18: weakness load. 0 = not weak, 1 = takes 2x, 2 = takes 4x
    has_type: np.ndarray   # n x 18: is of type T
    roles: np.ndarray      # n x 4
    pct: np.ndarray        # n x 6 stat percentiles


def build_matrices(pool: pd.DataFrame, chart: pd.DataFrame) -> Matrices:
    off, res, weak, has = [], [], [], []
    for t1, t2 in zip(pool["type_1"], pool["type_2"]):
        d = defensive_profile(chart, t1, t2)
        off.append((offensive_profile(chart, t1, t2)[TYPES] >= 2).to_numpy())
        res.append((d[TYPES] <= 0.5).to_numpy())
        # A 4x weakness counts double: two Pokémon with 4x Ground weakness are far
        # riskier than two with 2x, and should hit the shared-weakness limit.
        weak.append(np.select([d[TYPES] >= 4, d[TYPES] >= 2], [2, 1], 0))
        has.append(np.array([t in (t1, t2) for t in TYPES]))
    return Matrices(
        names=pool["name"].tolist(),
        bst=pool["bst"].to_numpy(),
        offense=np.array(off, dtype=int), resist=np.array(res, dtype=int),
        weak=np.array(weak, dtype=int), has_type=np.array(has, dtype=int),
        roles=pool[list(ROLES)].to_numpy(dtype=int),
        pct=pool[[f"{s}_pct" for s in STAT_COLS]].to_numpy(),
    )


# ------------------------------------------------------------------ scoring (used by solver checks and benchmarks)
def normalize(weights: dict) -> dict:
    total = sum(weights[k] for k in TERMS)
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return {k: weights[k] / total for k in TERMS}


def score_team(M: Matrices, idx, weights=DEFAULT_WEIGHTS) -> dict:
    """Objective value of ANY team, computed directly (no solver). Rows idx of M."""
    idx = list(idx)
    w = normalize(weights)
    terms = {
        "coverage": np.minimum(M.offense[idx].sum(axis=0), COVER_DEPTH).sum() / (len(TYPES) * COVER_DEPTH),
        "defense": np.minimum(M.resist[idx].sum(axis=0), COVER_DEPTH).sum() / (len(TYPES) * COVER_DEPTH),
        "roles": M.roles[idx].any(axis=0).sum() / len(ROLES),
        "balance": M.pct[idx].mean(axis=0).min(),
    }
    terms = {k: float(v) for k, v in terms.items()}   # plain floats, not numpy types
    return {**terms, "total": sum(w[k] * terms[k] for k in TERMS)}


def team_constraints_ok(M: Matrices, idx, max_weak: int, max_per_type: int) -> bool:
    idx = list(idx)
    return (len(set(idx)) == TEAM_SIZE
            and M.weak[idx].sum(axis=0).max() <= max_weak
            and M.has_type[idx].sum(axis=0).max() <= max_per_type)


# ------------------------------------------------------------------ solver
def solve(M: Matrices, weights=DEFAULT_WEIGHTS, max_weak=DEFAULT_MAX_WEAK,
          max_per_type=DEFAULT_MAX_PER_TYPE, locked_idx=()) -> list:
    """Return the row indices of the optimal team."""
    n, T, R, S = len(M.names), len(TYPES), len(ROLES), len(STAT_COLS)
    if n < TEAM_SIZE:
        raise InfeasibleTeamError(f"Only {n} Pokémon in the pool; need {TEAM_SIZE}.")
    w = normalize(weights)
    prob = pulp.LpProblem("pokelab_team", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    c = [pulp.LpVariable(f"cov_{t}", 0, COVER_DEPTH) for t in range(T)]
    d = [pulp.LpVariable(f"def_{t}", 0, COVER_DEPTH) for t in range(T)]
    g = [pulp.LpVariable(f"role_{r}", 0, 1) for r in range(R)]
    m = pulp.LpVariable("balance", 0, 1)

    max_bst = max(M.bst.max(), 1)
    depth_scale = T * COVER_DEPTH
    prob += (w["coverage"] * pulp.lpSum(c) / depth_scale + w["defense"] * pulp.lpSum(d) / depth_scale
             + w["roles"] * pulp.lpSum(g) / R + w["balance"] * m
             + TIEBREAK * pulp.lpSum(M.bst[i] * x[i] for i in range(n)) / (TEAM_SIZE * max_bst))

    prob += pulp.lpSum(x) == TEAM_SIZE
    for t in range(T):
        prob += c[t] <= pulp.lpSum(M.offense[i, t] * x[i] for i in range(n))
        prob += d[t] <= pulp.lpSum(M.resist[i, t] * x[i] for i in range(n))
        prob += pulp.lpSum(M.weak[i, t] * x[i] for i in range(n)) <= max_weak
        prob += pulp.lpSum(M.has_type[i, t] * x[i] for i in range(n)) <= max_per_type
    for r in range(R):
        prob += g[r] <= pulp.lpSum(M.roles[i, r] * x[i] for i in range(n))
    for s in range(S):
        prob += m <= pulp.lpSum(M.pct[i, s] * x[i] for i in range(n)) / TEAM_SIZE
    for i in locked_idx:
        prob += x[i] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise InfeasibleTeamError(
            f"No valid team ({pulp.LpStatus[prob.status]}). Try lowering min BST, raising the "
            "weakness-load or shared-type limits, allowing legendaries, or unlocking a Pokémon.")
    return [i for i in range(n) if x[i].value() > 0.5]


def optimize(df: pd.DataFrame, chart: pd.DataFrame, weights=DEFAULT_WEIGHTS,
             min_bst=DEFAULT_MIN_BST, allow_legendary=False, max_weak=DEFAULT_MAX_WEAK,
             max_per_type=DEFAULT_MAX_PER_TYPE, locked=()):
    """Full pipeline: features -> pool -> matrices -> solve. Returns (team rows, scores, pool, matrices)."""
    pool = select_pool(add_team_features(df), min_bst, allow_legendary, locked)
    M = build_matrices(pool, chart)
    locked_idx = [M.names.index(n) for n in locked]
    idx = solve(M, weights, max_weak, max_per_type, locked_idx)
    return pool.loc[idx], score_team(M, idx, weights), pool, M


if __name__ == "__main__":
    from src.features.preprocessing import load_pokemon
    from src.optimization.type_chart import load_type_chart

    team, scores, pool, _ = optimize(load_pokemon(), load_type_chart())
    print(f"Pool: {len(pool)} Pokémon (fully evolved, BST >= {DEFAULT_MIN_BST}, no legendaries)\n")
    print(team[["name", "type_1", "type_2", "bst", *ROLES]].to_string(index=False))
    print("\nScores (0-1):", {k: round(v, 3) for k, v in scores.items()})
