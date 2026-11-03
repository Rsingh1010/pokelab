"""Tests for the data pipeline (fetch_pokemon.py and build_dataset.py).

Three groups:
1. Fetch logic: caching and retries, tested WITHOUT the internet (we fake requests.get).
2. Parsing and validation: tested on small hand-made inputs where we know the right answer.
3. The real processed file: sanity checks on data/processed/pokemon.csv.
"""
import json

import pandas as pd
import pytest
import requests

from src.config import PROCESSED_DIR
from src.data import build_dataset as bd
from src.data import fetch_pokemon as fp


# ---------------------------------------------------------------- helpers
class FakeResponse:
    """Stands in for a requests.Response so tests never touch the network."""

    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    """Point the cache at a temp folder, make sleeps instant, and record every request.

    Returns a function: give it a list of responses, it serves them in order.
    """
    monkeypatch.setattr(fp, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fp.time, "sleep", lambda s: None)
    calls = []

    def serve(responses):
        def fake_get(url, timeout):
            calls.append(url)
            return responses[len(calls) - 1]
        monkeypatch.setattr(fp.requests, "get", fake_get)
        return calls

    return serve


def make_species(name="testmon", en_name="Testmon", generation="generation-iv", defaults=1):
    varieties = [{"is_default": i < defaults, "pokemon": {"name": f"{name}-form{i}"}} for i in range(2)]
    return {
        "id": 999,
        "name": name,
        "names": [{"language": {"name": "ja"}, "name": "テスト"}, {"language": {"name": "en"}, "name": en_name}],
        "generation": {"name": generation},
        "is_legendary": False,
        "is_mythical": True,
        "evolves_from_species": None,
        "varieties": varieties,
    }


def make_pokemon(stats=(50, 60, 70, 80, 90, 100), types=(("fire", 1),)):
    api_names = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]
    return {
        "name": "testmon-form0",
        "stats": [{"stat": {"name": n}, "base_stat": v} for n, v in zip(api_names, stats)],
        "types": [{"slot": slot, "type": {"name": t}} for t, slot in types],
        "abilities": [
            {"ability": {"name": "blaze"}, "is_hidden": False},
            {"ability": {"name": "solar-power"}, "is_hidden": True},
        ],
        "sprites": {"front_default": "s.png", "other": {"official-artwork": {"front_default": "a.png"}}},
    }


def valid_df():
    """A tiny table that should pass validation. Tests break it on purpose."""
    rows = [bd.parse_one(make_species(), make_pokemon())]
    rows.append({**rows[0], "dex_id": 1000, "name": "Othermon", "species_slug": "othermon"})
    return bd.add_evolution_features(pd.DataFrame(rows))


# ---------------------------------------------------------------- 1. fetch logic
def test_cache_path_is_windows_safe():
    path = fp.cache_path("pokemon-species/?limit=100000")
    assert "?" not in str(path.name)


def test_cache_hit_makes_no_request(fake_api):
    calls = fake_api([])  # any request would crash: no responses to serve
    path = fp.cache_path("pokemon/x")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"name": "x"}))
    assert fp.get_json("pokemon/x") == {"name": "x"}
    assert calls == []


def test_success_is_cached_then_reused(fake_api):
    calls = fake_api([FakeResponse(200, {"name": "x"})])
    fp.get_json("pokemon/x")
    fp.get_json("pokemon/x")
    assert len(calls) == 1
    assert fp.cache_path("pokemon/x").exists()


def test_retries_temporary_errors(fake_api):
    calls = fake_api([FakeResponse(503), FakeResponse(429), FakeResponse(200, {"ok": True})])
    assert fp.get_json("pokemon/x") == {"ok": True}
    assert len(calls) == 3


def test_404_fails_immediately_without_retry(fake_api):
    calls = fake_api([FakeResponse(404)] * 5)
    with pytest.raises(requests.HTTPError):
        fp.get_json("pokemon/missing")
    assert len(calls) == 1
    assert not fp.cache_path("pokemon/missing").exists()  # errors are never cached


def test_gives_up_after_max_retries(fake_api):
    calls = fake_api([FakeResponse(500)] * fp.MAX_RETRIES)
    with pytest.raises(RuntimeError):
        fp.get_json("pokemon/x")
    assert len(calls) == fp.MAX_RETRIES


@pytest.mark.parametrize("defaults", [0, 2])
def test_default_form_must_be_unique(defaults):
    with pytest.raises(ValueError):
        fp.default_pokemon_name(make_species(defaults=defaults))


# ---------------------------------------------------------------- 2. parsing + validation
def test_parse_one_fields():
    row = bd.parse_one(make_species(), make_pokemon())
    assert row["name"] == "Testmon"          # English name, not Japanese, not slug
    assert row["generation"] == 4
    assert row["bst"] == 50 + 60 + 70 + 80 + 90 + 100
    assert row["type_2"] is None             # single-type
    assert row["abilities"] == "blaze"       # hidden ability kept separate
    assert row["hidden_ability"] == "solar-power"


def test_parse_one_orders_types_by_slot():
    # API lists slot 2 first: primary type must still come from slot 1
    row = bd.parse_one(make_species(), make_pokemon(types=(("flying", 2), ("fire", 1))))
    assert (row["type_1"], row["type_2"]) == ("fire", "flying")


def test_validate_accepts_clean_data():
    bd.validate(valid_df())  # should not raise


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda df: df.assign(dex_id=1),              # duplicate ID
        lambda df: df.assign(name="Same"),           # duplicate name
        lambda df: df.assign(speed=0),               # stat out of range
        lambda df: df.assign(attack=None),           # missing stat
        lambda df: df.assign(type_1="sound"),        # not a real type
        lambda df: df.assign(type_2="fire"),         # type_2 == type_1
        lambda df: df.assign(evolves_from="ghostmon"),  # parent species doesn't exist
    ],
)
def test_validate_rejects_bad_data(corrupt):
    with pytest.raises(ValueError):
        bd.validate(corrupt(valid_df()))


# ---------------------------------------------------------------- 3. real processed file
@pytest.fixture(scope="module")
def processed():
    path = PROCESSED_DIR / "pokemon.csv"
    if not path.exists():
        pytest.skip("Run python -m src.data.build_dataset first")
    return pd.read_csv(path)


def test_processed_shape_and_ids(processed):
    assert len(processed) == 1025
    assert processed["dex_id"].tolist() == list(range(1, 1026))  # unique, no gaps, sorted


def test_processed_stats_numeric_and_bst_consistent(processed):
    assert all(pd.api.types.is_integer_dtype(processed[c]) for c in bd.STAT_COLS)
    assert (processed[bd.STAT_COLS].sum(axis=1) == processed["bst"]).all()


def test_processed_no_missing_required_fields(processed):
    required = ["dex_id", "name", "generation", "type_1", *bd.STAT_COLS, "bst"]
    assert processed[required].notna().all().all()


# ---------------------------------------------------------------- 4. form overrides
from src.data.form_overrides import FORM_OVERRIDES  # noqa: E402


@pytest.mark.parametrize("species_slug", sorted(FORM_OVERRIDES))
def test_override_is_a_real_form_of_its_species(species_slug):
    path = fp.cache_path(f"pokemon-species/{species_slug}")
    species_files = list(path.parent.glob("[0-9]*.json"))
    if not species_files:
        pytest.skip("Raw cache missing")
    species = next(
        s for s in (json.loads(f.read_text(encoding="utf-8")) for f in species_files)
        if s["name"] == species_slug
    )
    forms = {v["pokemon"]["name"] for v in species["varieties"]}
    assert FORM_OVERRIDES[species_slug] in forms


def test_chosen_form_respects_switch(monkeypatch):
    species = make_species(name="testmon")
    monkeypatch.setitem(fp.FORM_OVERRIDES, "testmon", "testmon-form1")
    assert fp.chosen_pokemon_name(species, use_overrides=True) == "testmon-form1"
    assert fp.chosen_pokemon_name(species, use_overrides=False) == "testmon-form0"


def test_bad_override_is_rejected(monkeypatch):
    monkeypatch.setitem(fp.FORM_OVERRIDES, "testmon", "testmon-typo")
    with pytest.raises(ValueError):
        fp.chosen_pokemon_name(make_species(name="testmon"))


def test_processed_uses_overrides(processed):
    used = processed.set_index("species_slug")["pokemon_slug"]
    for species_slug, form in FORM_OVERRIDES.items():
        assert used[species_slug] == form
    assert processed["form_override"].sum() == len(FORM_OVERRIDES)


# ---------------------------------------------------------------- 5. evolution features
def test_evolution_features_on_known_tree():
    # a -> b -> c is a 3-stage line; a also branches to e; d never evolves
    df = pd.DataFrame({
        "species_slug": ["a", "b", "c", "d", "e"],
        "evolves_from": [None, "a", "b", None, "a"],
    })
    out = bd.add_evolution_features(df).set_index("species_slug")
    assert out["evolution_stage"].to_dict() == {"a": 1, "b": 2, "c": 3, "d": 1, "e": 2}
    assert out["is_fully_evolved"].to_dict() == {"a": False, "b": False, "c": True, "d": True, "e": True}


def test_processed_evolution_examples(processed):
    p = processed.set_index("species_slug")
    assert p.loc[["charmander", "charmeleon", "charizard"], "evolution_stage"].tolist() == [1, 2, 3]
    assert p.loc[["pichu", "pikachu", "raichu"], "is_fully_evolved"].tolist() == [False, False, True]
    assert p.loc["tauros", "is_fully_evolved"]            # never evolves -> counts as final
    assert not p.loc["eevee", "is_fully_evolved"]         # branches into 8 evolutions
    assert p.loc[["vaporeon", "sylveon"], "is_fully_evolved"].all()
