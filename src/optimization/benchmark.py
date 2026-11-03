"""Milestone 7: how much better is the optimizer than simpler ways of picking a team?

Run: python -m src.optimization.benchmark

Baselines (all drawn from the SAME pool the optimizer uses):
  random   - 10,000 teams of 6 picked uniformly at random
  top_bst  - the 6 highest-BST Pokémon ("just pick the strongest")
  greedy   - add one Pokémon at a time, each time the one that raises the score
             most without breaking a rule. A classic fast heuristic: shows what
             EXACT optimization buys over a reasonable shortcut.

Honest caveat: the optimizer is scored on the same objective it maximizes, so it
must win on 'total'. The useful questions are: by how much, how rare such teams
are by chance, whether baselines even satisfy the rules, and how far greedy falls short.
"""
import json

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RESULTS_DIR
from src.optimization.team_optimizer import (
    DEFAULT_MAX_PER_TYPE, DEFAULT_MAX_WEAK, DEFAULT_MIN_BST, DEFAULT_WEIGHTS, TEAM_SIZE,
    TERMS, Matrices, optimize, score_team,
)

N_RANDOM = 10_000


def evaluate(M: Matrices, idx, weights=DEFAULT_WEIGHTS,
             max_weak=DEFAULT_MAX_WEAK, max_per_type=DEFAULT_MAX_PER_TYPE) -> dict:
    idx = list(idx)
    weak_load = M.weak[idx].sum(axis=0).max()
    type_count = M.has_type[idx].sum(axis=0).max()
    return {
        **score_team(M, idx, weights),
        "max_weakness_load": int(weak_load),
        "max_same_type": int(type_count),
        "follows_rules": bool(weak_load <= max_weak and type_count <= max_per_type),
        "mean_bst": float(M.bst[idx].mean()),
    }


def random_teams(M: Matrices, n=N_RANDOM, seed=RANDOM_SEED) -> list:
    rng = np.random.default_rng(seed)
    return [rng.choice(len(M.names), TEAM_SIZE, replace=False).tolist() for _ in range(n)]


def top_bst_team(M: Matrices) -> list:
    return np.argsort(-M.bst, kind="stable")[:TEAM_SIZE].tolist()


def greedy_team(M: Matrices, weights=DEFAULT_WEIGHTS,
                max_weak=DEFAULT_MAX_WEAK, max_per_type=DEFAULT_MAX_PER_TYPE):
    """Build a team one pick at a time. Returns None if it paints itself into a corner."""
    team = []
    for _ in range(TEAM_SIZE):
        best, best_score = None, -1.0
        for i in range(len(M.names)):
            if i in team:
                continue
            trial = team + [i]
            if (M.weak[trial].sum(axis=0).max() > max_weak
                    or M.has_type[trial].sum(axis=0).max() > max_per_type):
                continue
            s = score_team(M, trial, weights)["total"]
            if s > best_score:
                best, best_score = i, s
        if best is None:
            return None
        team.append(best)
    return team


def run(min_bst=DEFAULT_MIN_BST, allow_legendary=False) -> tuple[pd.DataFrame, dict]:
    from src.features.preprocessing import load_pokemon
    from src.optimization.type_chart import load_type_chart

    team, _, pool, M = optimize(load_pokemon(), load_type_chart(), min_bst=min_bst, allow_legendary=allow_legendary)
    opt_idx = [M.names.index(n) for n in team["name"]]

    rand = pd.DataFrame([evaluate(M, t) for t in random_teams(M)])
    rows = {"optimized": evaluate(M, opt_idx), "top_bst": evaluate(M, top_bst_team(M))}
    g = greedy_team(M)
    rows["greedy"] = evaluate(M, g) if g is not None else {"total": None, "follows_rules": False}
    valid_rand = rand[rand["follows_rules"]]
    rows["random_mean"] = rand.mean(numeric_only=True).to_dict() | {"follows_rules": float(rand["follows_rules"].mean())}
    rows["random_valid_mean"] = valid_rand.mean(numeric_only=True).to_dict() | {"follows_rules": 1.0}
    rows["random_valid_best"] = valid_rand.loc[valid_rand["total"].idxmax()].to_dict()

    table = pd.DataFrame(rows).T
    opt_total = rows["optimized"]["total"]
    summary = {
        "pool_size": len(M.names), "min_bst": min_bst, "allow_legendary": allow_legendary,
        "n_random": N_RANDOM,
        "random_share_following_rules": round(float(rand["follows_rules"].mean()), 4),
        "random_share_scoring_at_or_above_optimum": float((rand["total"] >= opt_total - 1e-9).mean()),
        "optimized_vs_best_valid_random": round(opt_total - rows["random_valid_best"]["total"], 4),
        "optimized_vs_greedy": None if g is None else round(opt_total - rows["greedy"]["total"], 4),
        "teams": {
            "optimized": [M.names[i] for i in opt_idx],
            "top_bst": [M.names[i] for i in top_bst_team(M)],
            "greedy": None if g is None else [M.names[i] for i in g],
        },
    }
    return table, summary


if __name__ == "__main__":
    out = {}
    for label, legendary in (("default", False), ("with_legendaries", True)):
        print(f"\n===== {label} =====")
        table, summary = run(allow_legendary=legendary)
        cols = [*TERMS, "total", "follows_rules", "max_weakness_load", "max_same_type", "mean_bst"]
        print(table[cols].astype(float).round(3).to_string())
        print(json.dumps({k: v for k, v in summary.items() if k != "teams"}, indent=2))
        print("Teams:", json.dumps(summary["teams"], indent=2))
        table.to_csv(RESULTS_DIR / f"optimizer_benchmark_{label}.csv")
        out[label] = summary
    (RESULTS_DIR / "optimizer_benchmark_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
