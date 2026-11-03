"""Project-wide paths and constants. Change things here, not scattered across files."""
from pathlib import Path

# Repo root = two levels up from this file (src/config.py -> pokelab/)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # untouched API responses (cache)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned tables the app reads
RESULTS_DIR = ROOT / "results"

# PokéAPI
API_BASE = "https://pokeapi.co/api/v2"
REQUEST_TIMEOUT_S = 15     # give up on a single request after this long
MAX_RETRIES = 5            # attempts before failing loudly
POLITE_DELAY_S = 0.1       # pause between *new* (uncached) requests

# One seed for every random process, so results are reproducible
RANDOM_SEED = 42
