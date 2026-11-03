"""Milestone 8 data: Smogon usage statistics for Gen 9 National Dex.

Run: python -m src.data.fetch_competitive

Sources (cached in data/raw/smogon/ after the first download):
  - Usage: https://www.smogon.com/stats/<month>/chaos/gen9nationaldex-1630.json
           'usage' = share of teams using the Pokémon, weighted toward stronger players.
  - Legality: Pokémon Showdown data/formats-data.ts (natDexTier). Used ONLY to remove
           banned Pokémon. Tiers are computed from usage, so they are never a model input.
  - Mega Stones: Pokémon Showdown data/items.ts, to measure how often a Pokémon is
           used for its Mega form (stats we do not model).

Output: data/processed/usage_gen9natdex.csv and results/usage_unmatched.csv
Usage is NOT win rate or strength.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import MAX_RETRIES, PROCESSED_DIR, RAW_DIR, REQUEST_TIMEOUT_S, RESULTS_DIR
from src.data.fetch_pokemon import RETRYABLE_STATUS

FORMAT = "gen9nationaldex"
CUTOFF = 1630                                   # Smogon's "standard" skill level
MONTHS = ["2026-07", "2026-08", "2026-09"]      # pinned snapshot: latest 3 months confirmed available
SHOWDOWN_RAW = "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data"
BANNED_TIERS = {"Uber", "AG", "Illegal"}        # not usable in National Dex OU
SMOGON_DIR = RAW_DIR / "smogon"


# ------------------------------------------------------------------ download (cached)
def fetch_cached(url: str, path: Path) -> str:
    """Download once, keep the file forever. Same retry rules as the PokéAPI fetcher."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT_S * 4)   # usage files are several MB
        except (requests.ConnectionError, requests.Timeout) as err:
            error = err
        else:
            if r.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(r.text, encoding="utf-8")
                return r.text
            if r.status_code not in RETRYABLE_STATUS:
                r.raise_for_status()
            error = f"HTTP {r.status_code}"
        print(f"  {url}: {error}, retrying in {2 ** attempt}s")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Gave up on {url}")


def usage_url(month: str) -> str:
    return f"https://www.smogon.com/stats/{month}/chaos/{FORMAT}-{CUTOFF}.json"


# ------------------------------------------------------------------ parsing
def to_id(name: str) -> str:
    """Showdown's ID rule: lowercase, letters and digits only. 'Mr. Mime' -> 'mrmime'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _blocks(ts_source: str):
    """Yield (key, body) for each top-level entry like '\\tbulbasaur: {...\\n\\t},'."""
    for m in re.finditer(r"^\t(\w+): \{\n(.*?)^\t\},?", ts_source, flags=re.M | re.S):
        yield m.group(1), m.group(2)


def parse_natdex_tiers(formats_data_ts: str) -> dict:
    """{showdown id: National Dex tier}. If natDexTier is missing, Showdown falls back to 'tier'."""
    tiers = {}
    for key, body in _blocks(formats_data_ts):
        nat = re.search(r'natDexTier: "([^"]+)"', body)
        reg = re.search(r'(?<![A-Za-z])tier: "([^"]+)"', body)
        if nat or reg:
            tiers[key] = (nat or reg).group(1)
    return tiers


def parse_mega_stones(items_ts: str) -> set:
    """Item ids that are Mega Stones (their entry has a megaStone field)."""
    return {key for key, body in _blocks(items_ts) if "megaStone" in body}


def parse_usage(chaos: dict, mega_stones: set) -> pd.DataFrame:
    rows = []
    for name, d in chaos["data"].items():
        items = d.get("Items", {})
        total_items = sum(items.values()) or 1
        rows.append({
            "smogon_name": name,
            "showdown_id": to_id(name),
            "usage": d["usage"],
            "raw_count": d.get("Raw count", 0),
            "mega_share": sum(v for k, v in items.items() if k in mega_stones) / total_items,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ build
def build_usage(pokemon: pd.DataFrame, monthly: dict, tiers: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join usage to our species. Match on the species' BASE id ('landorus', not 'landorustherian').

    Forms Smogon lists separately (e.g. Landorus-Therian) don't match: we only have the
    default form's stats. They are reported in the unmatched table, never silently dropped.
    A legal Pokémon absent from a month's file had 0% usage that month.
    """
    base = pokemon[pokemon["is_fully_evolved"]][["dex_id", "name", "species_slug"]].copy()
    base["showdown_id"] = base["species_slug"].map(to_id)
    base["natdex_tier"] = base["showdown_id"].map(tiers)
    base["legal"] = base["natdex_tier"].notna() & ~base["natdex_tier"].isin(BANNED_TIERS)

    unmatched = []
    for month, table in monthly.items():
        t = table.set_index("showdown_id")
        base[f"usage_{month}"] = base["showdown_id"].map(t["usage"]).fillna(0.0)
        base[f"mega_share_{month}"] = base["showdown_id"].map(t["mega_share"])
        miss = table[~table["showdown_id"].isin(base["showdown_id"])]
        unmatched.append(miss.assign(month=month))

    months = list(monthly)
    base["usage_mean"] = base[[f"usage_{m}" for m in months]].mean(axis=1)
    base["mega_share"] = base[[f"mega_share_{m}" for m in months]].mean(axis=1)
    base = base.drop(columns=[f"mega_share_{m}" for m in months])
    unmatched = pd.concat(unmatched).groupby(["smogon_name", "showdown_id"], as_index=False)["usage"].mean()
    return base, unmatched.sort_values("usage", ascending=False)


if __name__ == "__main__":
    from src.features.preprocessing import load_pokemon

    print("Downloading (cached after first run)...")
    formats_ts = fetch_cached(f"{SHOWDOWN_RAW}/formats-data.ts", SMOGON_DIR / "formats-data.ts")
    items_ts = fetch_cached(f"{SHOWDOWN_RAW}/items.ts", SMOGON_DIR / "items.ts")
    tiers, megas = parse_natdex_tiers(formats_ts), parse_mega_stones(items_ts)
    monthly = {}
    for month in MONTHS:
        raw = fetch_cached(usage_url(month), SMOGON_DIR / month / f"{FORMAT}-{CUTOFF}.json")
        chaos = json.loads(raw)
        print(f"  {month}: {chaos['info'].get('number of battles', '?')} battles, {len(chaos['data'])} Pokémon listed")
        monthly[month] = parse_usage(chaos, megas)
    print(f"Parsed {len(tiers)} tier entries, {len(megas)} Mega Stones")

    usage, unmatched = build_usage(load_pokemon(), monthly, tiers)
    usage.to_csv(PROCESSED_DIR / "usage_gen9natdex.csv", index=False)
    unmatched.to_csv(RESULTS_DIR / "usage_unmatched.csv", index=False)

    legal = usage[usage["legal"]]
    print(f"\nFully evolved: {len(usage)} | no tier found: {usage['natdex_tier'].isna().sum()} "
          f"| banned: {usage['natdex_tier'].isin(BANNED_TIERS).sum()} | legal: {len(legal)}")
    print("Tier counts:", usage["natdex_tier"].value_counts(dropna=False).to_dict())
    print(f"Legal with usage > 0: {(legal['usage_mean'] > 0).sum()}")
    print("\nTop 15 legal by mean usage:")
    print(legal.nlargest(15, "usage_mean")[["name", "natdex_tier", "usage_mean", "mega_share"]].round(3).to_string(index=False))
    print(f"\nUnmatched Smogon names: {len(unmatched)}. Top 15 by usage:")
    print(unmatched.head(15).round(4).to_string(index=False))
