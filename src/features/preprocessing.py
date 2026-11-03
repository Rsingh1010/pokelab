"""Turn the processed Pokémon table into numeric feature matrices for modeling.

Two choices define every experiment:
- subset:          which Pokémon go in ("all" or "fully_evolved")
- representation:  how stats become numbers ("standardized" or "relative")
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR

STAT_COLS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]
SUBSETS = ("all", "fully_evolved")
REPRESENTATIONS = ("standardized", "relative")


def load_pokemon() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "pokemon.csv")


def select_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return df
    if subset == "fully_evolved":
        return df[df["is_fully_evolved"]]
    raise ValueError(f"Unknown subset {subset!r}; choose from {SUBSETS}")


def relative_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Each stat as a share of the Pokémon's total. Every row sums to 1."""
    return df[STAT_COLS].div(df[STAT_COLS].sum(axis=1), axis=0)


def make_features(df: pd.DataFrame, representation: str):
    """Return (features, scaler).

    standardized: each stat rescaled to mean 0, std 1 across the given rows.
                  Keeps overall strength in the data. Scaler is returned so the
                  same transform can be reused (e.g. by the app) without refitting.
    relative:     stat / total. Removes overall strength, keeps the 'shape'.
                  Not rescaled: the six shares already have similar spreads
                  (checked in EDA), and leaving them as shares keeps centroids
                  readable ("28% of its stats are Speed"). Scaler is None.
    """
    if representation == "standardized":
        scaler = StandardScaler()
        values = scaler.fit_transform(df[STAT_COLS])
        return pd.DataFrame(values, index=df.index, columns=STAT_COLS), scaler
    if representation == "relative":
        return relative_stats(df), None
    raise ValueError(f"Unknown representation {representation!r}; choose from {REPRESENTATIONS}")
