"""Stages 2 and 3 of the pipeline: parse cached raw JSON into one clean table,
validate it, and save it to data/processed/pokemon.csv.

This file never calls the API. It only reads data/raw/, so you can rerun it
as often as you like while fixing parsing logic.
"""
import json

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR
from src.data.fetch_pokemon import cache_path, chosen_pokemon_name, default_pokemon_name, list_species_ids

STAT_NAMES = {  # API name -> our column name
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "special-attack": "sp_attack",
    "special-defense": "sp_defense",
    "speed": "speed",
}
STAT_COLS = list(STAT_NAMES.values())

VALID_TYPES = {
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "steel", "fairy",
}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9}
MAX_GENERATION = 9  # project scope: Generations I-IX


def read_cached(endpoint: str) -> dict:
    """Read a cached response. Fails if it's missing: run fetch_pokemon first."""
    path = cache_path(endpoint)
    if not path.exists():
        raise FileNotFoundError(f"{path} not cached. Run: python -m src.data.fetch_pokemon")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_one(species: dict, pokemon: dict) -> dict:
    """Turn one species record + its default Pokémon record into one table row."""
    stats = {STAT_NAMES[s["stat"]["name"]]: s["base_stat"] for s in pokemon["stats"]}
    types = sorted(pokemon["types"], key=lambda t: t["slot"])  # slot 1 = primary type
    english = [n["name"] for n in species["names"] if n["language"]["name"] == "en"]

    return {
        "dex_id": species["id"],
        "name": english[0] if english else species["name"],  # "Mr. Mime", not "mr-mime"
        "species_slug": species["name"],                      # API id for the species
        "pokemon_slug": pokemon["name"],                      # default form, e.g. "deoxys-normal"
        "generation": ROMAN[species["generation"]["name"].split("-")[1]],
        "type_1": types[0]["type"]["name"],
        "type_2": types[1]["type"]["name"] if len(types) > 1 else None,
        **stats,
        "bst": sum(stats.values()),
        "abilities": "|".join(a["ability"]["name"] for a in pokemon["abilities"] if not a["is_hidden"]),
        "hidden_ability": next((a["ability"]["name"] for a in pokemon["abilities"] if a["is_hidden"]), None),
        "evolves_from": (species["evolves_from_species"] or {}).get("name"),  # species slug or None
        "is_legendary": species["is_legendary"],
        "is_mythical": species["is_mythical"],
        "sprite_url": pokemon["sprites"]["front_default"],
        "artwork_url": pokemon["sprites"]["other"]["official-artwork"]["front_default"],
    }


def add_evolution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add evolution_stage (1 = first in its line) and is_fully_evolved.

    Uses only each species' 'evolves_from' link, so no extra API calls.
    - Stage: count how many steps back the chain goes (Pichu 1, Pikachu 2, Raichu 3).
    - Fully evolved: no other species evolves FROM this one.
      Single-stage Pokémon (e.g. Tauros) count as fully evolved.
    """
    parent = dict(zip(df["species_slug"], df["evolves_from"]))

    def stage(slug):
        steps = 1
        while isinstance(parent.get(slug), str):
            slug = parent[slug]
            steps += 1
        return steps

    has_child = set(df["evolves_from"].dropna())
    return df.assign(
        evolution_stage=df["species_slug"].map(stage),
        is_fully_evolved=~df["species_slug"].isin(has_child),
    )


def build(use_overrides: bool = True) -> pd.DataFrame:
    """use_overrides=False gives the plain default-form dataset (for comparison runs)."""
    rows = []
    for species_id in list_species_ids():
        species = read_cached(f"pokemon-species/{species_id}")
        pokemon = read_cached(f"pokemon/{chosen_pokemon_name(species, use_overrides)}")
        row = parse_one(species, pokemon)
        row["form_override"] = pokemon["name"] != default_pokemon_name(species)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("dex_id").reset_index(drop=True)
    before = len(df)
    df = df[df["generation"] <= MAX_GENERATION].reset_index(drop=True)
    df = add_evolution_features(df)
    print(f"Parsed {before} species; kept {len(df)} in Gen I-{MAX_GENERATION}")
    print(f"Form overrides applied: {df.loc[df['form_override'], 'pokemon_slug'].tolist()}")
    return df


def validate(df: pd.DataFrame) -> None:
    """Hard checks. Any failure stops the pipeline instead of saving bad data."""
    problems = []
    if not df["dex_id"].is_unique:
        problems.append("duplicate dex_id")
    if not df["name"].is_unique:
        problems.append("duplicate name")
    if df[STAT_COLS].isna().any().any():
        problems.append("missing base stats")
    if not df[STAT_COLS].apply(lambda c: c.between(1, 255)).all().all():
        problems.append("base stat outside 1-255")
    if not set(df["type_1"]).issubset(VALID_TYPES):
        problems.append(f"unknown type_1: {set(df['type_1']) - VALID_TYPES}")
    if not set(df["type_2"].dropna()).issubset(VALID_TYPES):
        problems.append(f"unknown type_2: {set(df['type_2'].dropna()) - VALID_TYPES}")
    if (df["type_1"] == df["type_2"]).any():
        problems.append("type_1 equals type_2")
    if df[["sprite_url", "artwork_url"]].isna().any().any():
        problems.append("missing image URL")
    if not set(df["evolves_from"].dropna()).issubset(set(df["species_slug"])):
        problems.append("evolves_from points to a species not in the dataset")
    if not df["evolution_stage"].between(1, 3).all():
        problems.append("evolution_stage outside 1-3")
    if problems:
        raise ValueError("Validation failed: " + "; ".join(problems))
    print("Validation passed")


if __name__ == "__main__":
    df = build()
    validate(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "pokemon.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {out}")
