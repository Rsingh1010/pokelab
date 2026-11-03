"""Stage 1 of the pipeline: download raw JSON from PokéAPI, with an on-disk cache.

Rule: every API response is saved to data/raw/ exactly as received.
If the file already exists, we read it from disk and never call the API again.
"""
import json
import time

import requests

from src.data.form_overrides import FORM_OVERRIDES
from src.config import (
    API_BASE,
    MAX_RETRIES,
    POLITE_DELAY_S,
    RAW_DIR,
    REQUEST_TIMEOUT_S,
)

# HTTP codes worth retrying: rate-limited (429) or the server had a problem (5xx)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def cache_path(endpoint: str):
    """Map an API endpoint to a file in data/raw/.

    'pokemon-species/25'             -> data/raw/pokemon-species/25.json
    'pokemon-species/?limit=100000'  -> data/raw/pokemon-species/_limit-100000.json

    Windows does not allow '?' in file names, so query characters are replaced.
    """
    safe = endpoint.strip("/").replace("/?", "/_").replace("?", "_").replace("=", "-").replace("&", "_")
    return RAW_DIR / f"{safe}.json"


def get_json(endpoint: str) -> dict:
    """Return the JSON for an endpoint, using the cache when possible."""
    path = cache_path(endpoint)

    # 1. Cache hit: no network call at all.
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # 2. Cache miss: request with retries and growing waits (1s, 2s, 4s, ...).
    url = f"{API_BASE}/{endpoint.strip('/')}"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as err:
            error = err  # network blip: retry
        else:
            if response.status_code == 200:
                data = response.json()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data), encoding="utf-8")
                time.sleep(POLITE_DELAY_S)  # be polite to a free API
                return data
            if response.status_code not in RETRYABLE_STATUS:
                # e.g. 404: retrying won't help, and silently skipping hides bugs
                response.raise_for_status()
            error = f"HTTP {response.status_code}"

        wait = 2 ** attempt
        print(f"  {endpoint}: {error} (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait}s")
        time.sleep(wait)

    raise RuntimeError(f"Gave up on {url} after {MAX_RETRIES} attempts")


def list_species_ids() -> list[int]:
    """Every species ID the API knows about, read from one list request.

    The list gives URLs like '.../pokemon-species/25/', so we take the last part.
    We do NOT hardcode 1025: if the API grows, we filter by generation later.
    """
    data = get_json("pokemon-species/?limit=100000")
    ids = [int(item["url"].rstrip("/").split("/")[-1]) for item in data["results"]]
    return sorted(ids)


def default_pokemon_name(species: dict) -> str:
    """A species has one or more forms ('varieties'); exactly one is the default.

    Examples: pikachu -> 'pikachu', deoxys -> 'deoxys-normal'.
    """
    defaults = [v["pokemon"]["name"] for v in species["varieties"] if v["is_default"]]
    if len(defaults) != 1:
        raise ValueError(f"{species['name']}: expected 1 default variety, found {len(defaults)}")
    return defaults[0]


def chosen_pokemon_name(species: dict, use_overrides: bool = True) -> str:
    """The ONE place that decides which form represents a species.

    Both the fetcher and the parser call this, so they can never disagree.
    """
    if use_overrides and species["name"] in FORM_OVERRIDES:
        override = FORM_OVERRIDES[species["name"]]
        varieties = {v["pokemon"]["name"] for v in species["varieties"]}
        if override not in varieties:
            raise ValueError(f"Override {override!r} is not a form of {species['name']}")
        return override
    return default_pokemon_name(species)


def download_all() -> None:
    """Make sure every species and its default Pokémon is in the cache.

    If this crashes halfway, just run it again: finished downloads are cached,
    so it continues where it stopped.
    """
    ids = list_species_ids()
    print(f"API lists {len(ids)} species")

    for n, species_id in enumerate(ids, start=1):
        species = get_json(f"pokemon-species/{species_id}")
        get_json(f"pokemon/{chosen_pokemon_name(species)}")
        if n % 100 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} done")

    print("All raw data cached in", RAW_DIR)


if __name__ == "__main__":
    download_all()
