"""Tests for the type chart. Expected values are well-known facts about the current games.

The chart is where silent bugs would quietly ruin the optimizer, so these tests
check specific matchups, all 8 immunities, and dual-type math.
"""
import pytest

from src.optimization import type_chart as tc


@pytest.fixture(scope="module")
def chart():
    try:
        return tc.load_type_chart()
    except FileNotFoundError:
        pytest.skip("Run python -m src.optimization.type_chart first")


def test_parser_on_fake_record():
    fake = {t: {"damage_relations": {"double_damage_to": [], "half_damage_to": [], "no_damage_to": []}}
            for t in tc.TYPES}
    fake["fire"]["damage_relations"]["double_damage_to"] = [{"name": "grass"}]
    fake["fire"]["damage_relations"]["half_damage_to"] = [{"name": "water"}]
    fake["normal"]["damage_relations"]["no_damage_to"] = [{"name": "ghost"}, {"name": "stellar"}]
    c = tc.chart_from_type_records(fake)
    assert c.loc["fire", "grass"] == 2 and c.loc["fire", "water"] == 0.5
    assert c.loc["normal", "ghost"] == 0
    assert c.loc["water", "fire"] == 1                 # unlisted = neutral
    assert "stellar" not in c.columns                  # non-battle types ignored


def test_shape_and_values(chart):
    assert chart.shape == (18, 18)
    assert set(chart.to_numpy().ravel()) <= {0.0, 0.5, 1.0, 2.0}
    assert list(chart.index) == list(chart.columns)


@pytest.mark.parametrize("attacking, defending, expected", [
    ("fire", "grass", 2), ("water", "fire", 2), ("grass", "water", 2),
    ("fire", "water", 0.5), ("dragon", "steel", 0.5), ("fairy", "dragon", 2),
    ("ice", "dragon", 2), ("steel", "fairy", 2), ("normal", "normal", 1),
])
def test_known_single_matchups(chart, attacking, defending, expected):
    assert chart.loc[attacking, defending] == expected


def test_exactly_the_eight_immunities(chart):
    immunities = {(a, d) for a in chart.index for d in chart.columns if chart.loc[a, d] == 0}
    assert immunities == {
        ("normal", "ghost"), ("fighting", "ghost"), ("poison", "steel"), ("ground", "flying"),
        ("ghost", "normal"), ("electric", "ground"), ("psychic", "dark"), ("dragon", "fairy"),
    }


def test_fighting_super_effective_list(chart):
    hits = set(chart.columns[chart.loc["fighting"] == 2])
    assert hits == {"normal", "ice", "rock", "dark", "steel"}


@pytest.mark.parametrize("attacking, t1, t2, expected", [
    ("rock", "fire", "flying", 4),       # Charizard: double weakness
    ("ground", "fire", "flying", 0),     # immunity beats weakness
    ("fire", "grass", "steel", 4),       # Ferrothorn
    ("grass", "water", "ground", 4),     # Swampert
    ("electric", "water", "ground", 0),  # Swampert immune to Electric
    ("electric", "water", "flying", 4),  # Gyarados
    ("water", "water", "dragon", 0.25),  # Kingdra: double resistance
    ("fire", "bug", "steel", 4),         # Scizor
    ("dragon", "steel", "fairy", 0),     # Mawile
])
def test_dual_type_math(chart, attacking, t1, t2, expected):
    assert tc.defensive_multiplier(chart, attacking, t1, t2) == expected


def test_type_order_does_not_matter(chart):
    for a in tc.TYPES:
        assert tc.defensive_multiplier(chart, a, "fire", "flying") == tc.defensive_multiplier(chart, a, "flying", "fire")


def test_single_type_handles_missing_second_type(chart):
    nan = float("nan")
    assert tc.defensive_multiplier(chart, "water", "fire", nan) == 2
    assert tc.defensive_multiplier(chart, "water", "fire", None) == 2


def test_offensive_profile_takes_best_own_type(chart):
    off = tc.offensive_profile(chart, "fire", "flying")
    assert off["grass"] == 2                  # both hit grass 2x
    assert off["fighting"] == 2               # flying covers it
    assert off["rock"] == 0.5                 # both resisted by rock
    assert tc.offensive_profile(chart, "normal")["ghost"] == 0
