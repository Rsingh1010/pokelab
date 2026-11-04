"""Tests for src/features/build_app_data.py: the app must show the same numbers as the analysis."""
import pytest

from src.features import build_app_data as bad
from src.optimization.team_optimizer import add_team_features


@pytest.fixture(scope="module")
def built():
    try:
        return bad.build()
    except FileNotFoundError:
        pytest.skip("Run the anomaly script first")


def test_one_row_per_species(built):
    table, _ = built
    assert len(table) == 1025 and table["dex_id"].is_unique


def test_role_fields_only_for_fully_evolved(built):
    table, _ = built
    assert table.loc[table["is_fully_evolved"], "speed_axis"].notna().all()
    assert table.loc[~table["is_fully_evolved"], "speed_axis"].isna().all()


def test_role_axes_match_optimizer(built):
    """Same axes the Team Builder uses, so a 'fast' Pokémon is fast everywhere in the app."""
    from src.features.preprocessing import load_pokemon
    table, _ = built
    fe = add_team_features(load_pokemon()).set_index("dex_id")
    t = table.set_index("dex_id").loc[fe.index]
    assert (t["speed_axis"] - fe["speed_axis"]).abs().max() < 1e-9
    assert (t["fast"] == fe["fast"]).all()


def test_strength_axis_points_toward_strong(built):
    table, _ = built
    assert table["strength_pc1"].corr(table["bst"]) > 0.9


def test_similar_excludes_self_and_has_five(built):
    table, _ = built
    row = table.set_index("name").loc["Garchomp"]
    sims = row["similar"].split("|")
    assert len(sims) == 5 and "Garchomp" not in sims
