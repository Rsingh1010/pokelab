"""Tests for the team optimizer.

The key test: on a small pool we can try EVERY possible team by brute force.
The solver must find a team exactly as good as the brute-force best.
"""
from itertools import combinations

import numpy as np
import pytest

from src.optimization import team_optimizer as to
from src.optimization.type_chart import TYPES


def random_matrices(n=10, seed=0):
    rng = np.random.default_rng(seed)
    T = len(TYPES)
    has_type = np.zeros((n, T), dtype=int)
    for i in range(n):
        has_type[i, rng.choice(T, size=rng.integers(1, 3), replace=False)] = 1
    return to.Matrices(
        names=[f"p{i}" for i in range(n)],
        bst=rng.integers(450, 600, n),
        offense=rng.integers(0, 2, (n, T)), resist=rng.integers(0, 2, (n, T)),
        weak=(rng.random((n, T)) < 0.15).astype(int), has_type=has_type,
        roles=rng.integers(0, 2, (n, 4)), pct=rng.random((n, 6)),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solver_matches_brute_force(seed):
    M = random_matrices(n=10, seed=seed)
    feasible = [t for t in combinations(range(10), 6) if to.team_constraints_ok(M, t, 2, 2)]
    if not feasible:
        pytest.skip("no feasible team for this random pool")
    best = max(to.score_team(M, t)["total"] for t in feasible)
    idx = to.solve(M, max_weak=2, max_per_type=2)
    assert to.team_constraints_ok(M, idx, 2, 2)
    assert to.score_team(M, idx)["total"] == pytest.approx(best, abs=1e-4)


def test_locked_members_are_kept():
    M = random_matrices(n=12, seed=3)
    idx = to.solve(M, max_weak=6, max_per_type=6, locked_idx=[0, 5])
    assert {0, 5} <= set(idx) and len(idx) == 6


def test_infeasible_is_reported_clearly():
    M = random_matrices(n=8)
    M.weak[:] = 1                     # everyone weak to everything
    with pytest.raises(to.InfeasibleTeamError):
        to.solve(M, max_weak=2)


def test_weights_change_the_choice():
    M = random_matrices(n=12, seed=4)
    a = to.solve(M, weights={"coverage": 1, "defense": 0, "roles": 0, "balance": 0}, max_weak=6, max_per_type=6)
    b = to.solve(M, weights={"coverage": 0, "defense": 0, "roles": 0, "balance": 1}, max_weak=6, max_per_type=6)
    assert to.score_team(M, a, {"coverage": 1, "defense": 0, "roles": 0, "balance": 0})["coverage"] >= \
           to.score_team(M, b, {"coverage": 1, "defense": 0, "roles": 0, "balance": 0})["coverage"]
    assert to.score_team(M, b)["balance"] >= to.score_team(M, a)["balance"] - 1e-4


def test_weights_must_be_positive():
    with pytest.raises(ValueError):
        to.normalize({k: 0 for k in to.TERMS})


# ------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def real():
    from src.features.preprocessing import load_pokemon
    from src.optimization.type_chart import load_type_chart
    return load_pokemon(), load_type_chart()


def test_role_axes_have_fixed_meaning(real):
    fe = to.add_team_features(real[0]).set_index("name")
    assert fe.loc["Regieleki", "fast"] and fe.loc["Shuckle", "bulky"]
    assert fe.loc["Alakazam", "special"]
    assert fe.loc["Kartana", "physical"]


def test_real_team_respects_every_rule(real):
    df, chart = real
    team, scores, pool, M = to.optimize(df, chart, min_bst=480, locked=["Garchomp"])
    assert len(team) == 6 and team["name"].is_unique
    assert "Garchomp" in set(team["name"])
    assert (team["bst"] >= 480).all()
    assert not (team["is_legendary"] | team["is_mythical"]).any()
    assert team["is_fully_evolved"].all()
    idx = [M.names.index(n) for n in team["name"]]
    assert to.team_constraints_ok(M, idx, to.DEFAULT_MAX_WEAK, to.DEFAULT_MAX_PER_TYPE)
    assert 0 <= scores["total"] <= 1


def test_legendary_toggle(real):
    df, chart = real
    fe = to.add_team_features(df)
    no = to.select_pool(fe, 500, allow_legendary=False)
    yes = to.select_pool(fe, 500, allow_legendary=True)
    assert len(yes) > len(no)
    assert not (no["is_legendary"] | no["is_mythical"]).any()


def test_unknown_or_unevolved_lock_rejected(real):
    df, chart = real
    with pytest.raises(to.InfeasibleTeamError):
        to.optimize(df, chart, locked=["Pikachu"])   # not fully evolved


def test_coverage_counts_depth_up_to_two():
    T = len(TYPES)
    M = random_matrices(n=6)
    M.offense[:] = 0
    M.offense[0, 0] = 1                       # type 0 covered once
    M.offense[[1, 2, 3], 1] = 1               # type 1 covered three times (capped at 2)
    s = to.score_team(M, range(6))
    assert s["coverage"] == pytest.approx((1 + 2) / (T * to.COVER_DEPTH))
