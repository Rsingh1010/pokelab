"""Tests for src/data/fetch_competitive.py (no network: parsers run on small hand-written samples)."""
import pandas as pd
import pytest

from src.data import fetch_competitive as fc

FORMATS_TS = '''export const FormatsData = {
\tbulbasaur: {
\t\ttier: "LC",
\t},
\tvenusaur: {
\t\ttier: "PU",
\t\tdoublesTier: "(DUU)",
\t\tnatDexTier: "RU",
\t},
\tmewtwo: {
\t\ttier: "Uber",
\t\tnatDexTier: "Uber",
\t},
\tlandorustherian: {
\t\ttier: "OU",
\t},
};
'''

ITEMS_TS = '''export const Items = {
\tcharizarditex: {
\t\tname: "Charizardite X",
\t\tmegaStone: "Charizard-Mega-X",
\t},
\tleftovers: {
\t\tname: "Leftovers",
\t},
\teviolite: {
\t\tname: "Eviolite",
\t},
};
'''


def test_to_id():
    assert fc.to_id("Mr. Mime") == "mrmime"
    assert fc.to_id("Landorus-Therian") == "landorustherian"
    assert fc.to_id("ho-oh") == "hooh"
    assert fc.to_id("Farfetch’d") == "farfetchd"


def test_parse_tiers_prefers_natdex_and_ignores_other_tier_fields():
    t = fc.parse_natdex_tiers(FORMATS_TS)
    assert t["venusaur"] == "RU"            # natDexTier wins; doublesTier ignored
    assert t["bulbasaur"] == "LC"           # falls back to tier
    assert t["mewtwo"] == "Uber"


def test_parse_mega_stones():
    assert fc.parse_mega_stones(ITEMS_TS) == {"charizarditex"}   # Eviolite is not a Mega Stone


def test_parse_usage_mega_share():
    chaos = {"data": {
        "Charizard": {"usage": 0.10, "Raw count": 50, "Items": {"charizarditex": 30.0, "leftovers": 10.0}},
        "Venusaur": {"usage": 0.02, "Raw count": 9, "Items": {"leftovers": 5.0}},
    }}
    u = fc.parse_usage(chaos, {"charizarditex"}).set_index("smogon_name")
    assert u.loc["Charizard", "mega_share"] == pytest.approx(0.75)
    assert u.loc["Venusaur", "mega_share"] == 0


def test_build_usage_join_legality_and_unmatched():
    pokemon = pd.DataFrame({
        "dex_id": [3, 150, 645, 2], "name": ["Venusaur", "Mewtwo", "Landorus", "Ivysaur"],
        "species_slug": ["venusaur", "mewtwo", "landorus", "ivysaur"],
        "is_fully_evolved": [True, True, True, False],
    })
    tiers = {"venusaur": "RU", "mewtwo": "Uber", "landorus": "UU"}
    month = pd.DataFrame({"smogon_name": ["Venusaur", "Landorus-Therian"],
                          "showdown_id": ["venusaur", "landorustherian"],
                          "usage": [0.04, 0.20], "raw_count": [1, 1], "mega_share": [0.0, 0.0]})
    usage, unmatched = fc.build_usage(pokemon, {"2026-01": month, "2026-02": month.assign(usage=[0.02, 0.2])}, tiers)
    u = usage.set_index("name")
    assert "Ivysaur" not in u.index                          # fully evolved only
    assert u.loc["Venusaur", "usage_mean"] == pytest.approx(0.03)
    assert u.loc["Landorus", "usage_mean"] == 0              # Therian usage NOT given to base form
    assert not u.loc["Mewtwo", "legal"] and u.loc["Venusaur", "legal"]
    assert unmatched["smogon_name"].tolist() == ["Landorus-Therian"]   # reported, not dropped
