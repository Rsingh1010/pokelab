"""The real type-effectiveness chart, built from PokéAPI's /type endpoint.

Build once (uses the same cache as the Pokémon data):
    python -m src.optimization.type_chart
Saves data/processed/type_chart.csv: rows = ATTACKING type, columns = DEFENDING type,
values = damage multiplier (0, 0.5, 1 or 2).

Dual types multiply: Fire/Flying vs Rock = 2 x 2 = 4. Ground vs Fire/Flying = 2 x 0 = 0.
"""
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.build_dataset import VALID_TYPES
from src.data.fetch_pokemon import get_json

TYPES = sorted(VALID_TYPES)  # the 18 battle types (PokéAPI also lists "unknown", "shadow", "stellar")
CHART_PATH = PROCESSED_DIR / "type_chart.csv"
RELATION_MULTIPLIER = {"double_damage_to": 2.0, "half_damage_to": 0.5, "no_damage_to": 0.0}


def chart_from_type_records(records: dict) -> pd.DataFrame:
    """records: {attacking type name: PokéAPI /type JSON}. Uses the CURRENT damage relations.

    We only read the '..._to' lists (what this type does when attacking).
    Anything not listed is neutral (1x).
    """
    chart = pd.DataFrame(1.0, index=TYPES, columns=TYPES)
    chart.index.name, chart.columns.name = "attacking", "defending"
    for attacker, record in records.items():
        for relation, mult in RELATION_MULTIPLIER.items():
            for target in record["damage_relations"][relation]:
                if target["name"] in TYPES:
                    chart.loc[attacker, target["name"]] = mult
    return chart


def build_type_chart() -> pd.DataFrame:
    return chart_from_type_records({t: get_json(f"type/{t}") for t in TYPES})


def load_type_chart() -> pd.DataFrame:
    if not CHART_PATH.exists():
        raise FileNotFoundError("Run: python -m src.optimization.type_chart")
    return pd.read_csv(CHART_PATH, index_col=0)


def defensive_multiplier(chart: pd.DataFrame, attacking: str, type_1: str, type_2=None) -> float:
    """Damage multiplier when an attack of type `attacking` hits a Pokémon of type_1(/type_2)."""
    mult = chart.loc[attacking, type_1]
    if isinstance(type_2, str):          # type_2 is None or NaN for single-type Pokémon
        mult *= chart.loc[attacking, type_2]
    return float(mult)


def defensive_profile(chart: pd.DataFrame, type_1: str, type_2=None) -> pd.Series:
    """Multiplier this Pokémon takes from each of the 18 attacking types."""
    return pd.Series({a: defensive_multiplier(chart, a, type_1, type_2) for a in TYPES})


def offensive_profile(chart: pd.DataFrame, type_1: str, type_2=None) -> pd.Series:
    """Best multiplier this Pokémon's OWN types (STAB) deal to each single defending type.

    Simplification: real Pokémon learn moves of other types. We only model
    attacks matching the Pokémon's own types.
    """
    own = [type_1] + ([type_2] if isinstance(type_2, str) else [])
    return chart.loc[own].max(axis=0)


if __name__ == "__main__":
    chart = build_type_chart()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chart.to_csv(CHART_PATH)
    print(f"Saved {CHART_PATH}")
    print(chart.replace({1.0: "."}).to_string())
