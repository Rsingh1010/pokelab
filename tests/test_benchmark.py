"""Tests for src/optimization/benchmark.py."""
import numpy as np
import pytest

from src.optimization import benchmark as bm
from src.optimization import team_optimizer as to
from tests.test_team_optimizer import random_matrices


def test_random_teams_are_valid_draws():
    M = random_matrices(n=15)
    teams = bm.random_teams(M, n=50, seed=1)
    assert len(teams) == 50
    assert all(len(set(t)) == 6 and max(t) < 15 for t in teams)


def test_top_bst_picks_the_six_strongest():
    M = random_matrices(n=15)
    top = bm.top_bst_team(M)
    assert sorted(M.bst[top]) == sorted(np.sort(M.bst)[-6:])


@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_greedy_follows_rules_and_never_beats_optimum(seed):
    M = random_matrices(n=14, seed=seed)
    g = bm.greedy_team(M, max_weak=2, max_per_type=2)
    if g is None:
        pytest.skip("greedy got stuck on this random pool")
    assert bm.evaluate(M, g)["follows_rules"]
    opt = to.solve(M, max_weak=2, max_per_type=2)
    assert bm.evaluate(M, opt)["total"] >= bm.evaluate(M, g)["total"] - 1e-9


def test_evaluate_flags_rule_breaks():
    M = random_matrices(n=8)
    M.weak[:] = 0
    M.weak[:, 0] = 1                       # everyone weak to type 0 -> load 6
    assert not bm.evaluate(M, range(6))["follows_rules"]
    assert bm.evaluate(M, range(6))["max_weakness_load"] == 6
