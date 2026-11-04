"""Rebuild every result from scratch, in order.  python -m src.pipeline [--skip-download]

--skip-download: use the committed data/processed/ files instead of calling PokéAPI and Smogon
(the raw cache in data/raw/ is not committed). Everything after the downloads is fully reproducible
from the committed processed files with fixed seeds.
"""
import subprocess
import sys
import time

DOWNLOAD = [
    "src.data.fetch_pokemon",          # PokéAPI -> data/raw (cached)
    "src.data.build_dataset",          # parse + validate -> data/processed/pokemon.csv
    "src.optimization.type_chart",     # PokéAPI type chart -> data/processed/type_chart.csv
    "src.data.fetch_competitive",      # Smogon + Showdown -> data/processed/usage_gen9natdex.csv
]
ANALYSIS = [
    "src.models.clustering",           # K-Means k-sweep
    "src.models.k_selection",          # baseline + bootstrap stability + pre-registered rule
    "src.models.gmm",                  # EXP-03
    "src.models.anomaly_detection",    # Isolation Forest (needs the raw cache for the override check)
    "src.models.competitive_analysis", # EXP-06 association
    "src.models.competitive_prediction",  # EXP-06 held-out prediction
    "src.optimization.benchmark",      # OPT-01
    "src.evaluation.record_experiments",  # results/experiments/*.json
    "src.features.build_app_data",     # results/app/ for the Streamlit app
]

if __name__ == "__main__":
    steps = ANALYSIS if "--skip-download" in sys.argv else DOWNLOAD + ANALYSIS
    for i, module in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] python -m {module}", flush=True)
        start = time.time()
        if subprocess.run([sys.executable, "-m", module]).returncode != 0:
            sys.exit(f"Stopped: {module} failed.")
        print(f"  done in {time.time() - start:.0f}s")
    print("\nAll results rebuilt. Start the app: streamlit run app/dashboard.py")
