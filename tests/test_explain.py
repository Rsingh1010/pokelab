"""Tests for src/optimization/explain.py on a hand-built team where every answer is known."""
import numpy as np
import pytest

from src.optimization import explain as ex
from src.optimization import team_optimizer as to
from src.optimization.type_chart import TYPES

T = len(TYPES)
FIRE, WATER, GRASS = TYPES.index("fire"), TYPES.index("water"), TYPES.index("grass")


@pytest.fixture
def team():
    n = 6
    M = to.Matrices(
        names=[f"p{i}" for i in range(n)], bst=np.full(n, 500),
        offense=np.zeros((n, T), int), resist=np.zeros((n, T), int), weak=np.zeros((n, T), int),
        has_type=np.zeros((n, T), int), roles=np.zeros((n, 4), int), pct=np.full((n, 6), 0.5),
    )
    M.offense[0, GRASS] = 1               # only p0 hits grass
    M.offense[[1, 2], WATER] = 1          # two members hit water
    M.weak[1, FIRE] = M.weak[2, FIRE] = 1 # p1, p2 weak to fire
    M.resist[3, FIRE] = 1                 # p3 resists fire -> patches p1/p2
    M.roles[0, 0] = 1                     # p0 is fast
    M.pct[:, 2] = 0.2                     # defense is the weakest stat
    return M


def test_member_details(team):
    m = ex.explain_members(team, range(6)).set_index("name")
    assert m.loc["p0", "only_answer"] == ["grass"]
    assert m.loc["p1", "only_answer"] == []            # water is also covered by p2
    assert m.loc["p3", "patches"] == ["fire"]
    assert m.loc["p1", "weak_to"] == ["fire"]
    assert m.loc["p0", "roles"] == ["fast"]


def test_contribution_is_leave_one_out(team):
    m = ex.explain_members(team, range(6)).set_index("name")
    full = to.score_team(team, range(6))["total"]
    without_p0 = to.score_team(team, [1, 2, 3, 4, 5])["total"]
    assert m.loc["p0", "contribution"] == pytest.approx(full - without_p0)
    assert m.loc["p0", "contribution"] > m.loc["p4", "contribution"]   # p0 brings unique coverage + a role


def test_team_summary(team):
    s = ex.explain_team(team, range(6))
    assert s["only_one_answer"] == ["grass"]
    assert "water" not in s["no_super_effective_answer"] and "fire" in s["no_super_effective_answer"]
    assert s["weakness_load_at_limit"] == {"fire": 2}
    assert s["only_one_resist"] == ["fire"]
    assert s["roles_missing"] == ["bulky", "physical", "special"]
    assert s["weakest_stat"] == "defense" and s["weakest_stat_percentile"] == pytest.approx(0.2)
