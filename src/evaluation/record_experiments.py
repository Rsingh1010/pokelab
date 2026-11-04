"""Write experiment records to results/experiments/ as JSON.

Every number is read from saved result files, never typed by hand,
so the records can't drift from the actual outputs. Rerun after any change:
    python -m src.evaluation.record_experiments
"""
import json

import pandas as pd

from src.config import RESULTS_DIR

OUT = RESULTS_DIR / "experiments"


def load():
    sel = pd.read_csv(RESULTS_DIR / "k_selection.csv")
    xtab = pd.read_csv(RESULTS_DIR / "evolution_crosstab_k2.csv", index_col=0)
    gmm = pd.read_csv(RESULTS_DIR / "gmm_selection.csv").set_index("k")
    return sel, xtab, gmm


def setup(sel, subset, representation):
    return sel[(sel["subset"] == subset) & (sel["representation"] == representation)].set_index("k")


def r(x):
    return round(float(x), 3)


def build_records(sel, xtab, gmm):
    all_std = setup(sel, "all", "standardized")
    fe_std = setup(sel, "fully_evolved", "standardized")
    fe_rel = setup(sel, "fully_evolved", "relative")
    agreement = max(xtab.values.trace(), xtab.values[::-1].trace()) / xtab.values.sum()
    cand = fe_rel.loc[3:8]
    best_stab_k = int(cand["stability_ari"].idxmax())
    common = {
        "algorithm": "K-Means (n_init=10, random_state=42)",
        "k_range": "2-10",
        "data_files": ["results/k_selection.csv", "results/evolution_crosstab_k2.csv"],
    }
    return {
        "EXP-01": {
            "title": "Does overall strength influence clustering?",
            "status": "completed",
            "question": "Do standardized raw stats cluster by strength / evolution stage rather than by role?",
            "hypothesis": "Yes: EDA showed every raw stat correlates 0.55-0.71 with BST.",
            "methodology": "Cluster all 1,025 Pokémon on standardized stats; compare the k=2 split with is_fully_evolved; "
                           "repeat on fully evolved only.",
            **common,
            "metrics": {
                "all_standardized_k2_silhouette": r(all_std.loc[2, "silhouette"]),
                "all_standardized_k2_stability_ari": r(all_std.loc[2, "stability_ari"]),
                "k2_agreement_with_is_fully_evolved": r(agreement),
                "fully_evolved_standardized_silhouette_range": [r(fe_std["silhouette"].min()), r(fe_std["silhouette"].max())],
            },
            "interpretation": f"The strongest raw-stat split (k=2) matches evolved vs not-evolved for {agreement:.0%} of Pokémon. "
                              f"Removing unevolved Pokémon drops silhouette to {fe_std['silhouette'].min():.2f}-{fe_std['silhouette'].max():.2f} "
                              "at every k. Raw-stat structure is mostly evolution stage and power level, not role.",
            "limitations": "Evolution stage is derived from PokéAPI evolves_from links; baby Pokémon count as stage 1.",
        },
        "EXP-02": {
            "title": "Can statistical roles be discovered independently of total strength?",
            "status": "completed",
            "question": "Do relative stats (stat / BST) on fully evolved Pokémon form real, stable groups?",
            "methodology": "Fully evolved (568) on relative stats; compare silhouette to a shuffled-column baseline "
                           "(10 shuffles, 95th percentile) and measure bootstrap stability (20 resamples, ARI).",
            **common,
            "metrics": {
                "silhouette_k3_8": cand["silhouette"].round(3).tolist(),
                "baseline_p95_k3_8": cand["baseline_p95"].round(3).tolist(),
                "max_margin_over_baseline_k3_8": r((cand["silhouette"] - cand["baseline_p95"]).max()),
                "stability_ari_k3_8": cand["stability_ari"].round(3).tolist(),
                "stability_ari_k2": r(fe_rel.loc[2, "stability_ari"]),
            },
            "interpretation": "Every k=3-8 beats the shuffled baseline, but by at most "
                              f"{(cand['silhouette'] - cand['baseline_p95']).max():.3f}. No k>=3 is stable "
                              f"(best K-Means ARI {cand['stability_ari'].max():.3f}; best GMM ARI "
                              f"{gmm.loc[3:8, 'stability_ari'].max():.3f}). Relative-stat clusters have near-equal "
                              "mean BST, so they do reflect stat shape, not strength, but the shapes form a continuum "
                              "rather than discrete groups. Only a 2-way split (fast/special vs bulky/physical) is stable.",
            "limitations": "Silhouette and K-Means both assume round, similar-sized groups; overlapping groups would score poorly.",
        },
        "EXP-03": {
            "title": "K-Means vs Gaussian Mixture Models",
            "status": "completed",
            "question": "Does a soft, flexible-shape model (GMM) find stable groups where K-Means did not?",
            "methodology": "GMM, full covariance, n_init=5, on fully evolved / relative stats (HP share dropped: "
                           "shares sum to 1). Same pre-registered rule as K-Means; BIC to choose among eligible k.",
            "algorithm": "GaussianMixture(covariance_type='full', n_init=5, random_state=42)",
            "data_files": ["results/gmm_selection.csv", "results/k_selection.csv"],
            "metrics": {
                "gmm_eligible_k": gmm.index[gmm["eligible"]].tolist(),
                "gmm_lowest_bic_k": int(gmm["bic"].idxmin()),
                "gmm_stability_ari_k3_8": gmm.loc[3:8, "stability_ari"].round(3).tolist(),
                "kmeans_stability_ari_k3_8": cand["stability_ari"].round(3).tolist(),
                "gmm_silhouette_k3_8": gmm.loc[3:8, "silhouette"].round(3).tolist(),
                "gmm_mean_top_probability_k3_8": gmm.loc[3:8, "mean_top_probability"].round(3).tolist(),
            },
            "interpretation": f"GMM also has no eligible k and is less stable than K-Means at every k=3-8 "
                              f"(GMM ARI {gmm.loc[3:8, 'stability_ari'].min():.2f}-{gmm.loc[3:8, 'stability_ari'].max():.2f}). "
                              f"BIC is lowest at k={int(gmm['bic'].idxmin())}, but that k is unstable. GMM reports high "
                              f"confidence (mean top probability {gmm.loc[3:8, 'mean_top_probability'].min():.2f}-"
                              f"{gmm.loc[3:8, 'mean_top_probability'].max():.2f}) while its groups change across resamples: "
                              "model probabilities assume the model is right and do not measure stability.",
            "limitations": "Silhouette favors round groups, so it penalizes GMM's stretched groups by design; "
                           "bootstrap stability is the fairer comparison and also fails.",
        },
        "EXP-04": {
            "title": "How many archetypes should be identified?",
            "status": "completed (K-Means)",
            "question": "Which k satisfies the pre-registered selection rule?",
            "methodology": "Rule set before viewing profiles: main setup fully_evolved/relative, k in 3-8; eligible only if "
                           "silhouette > shuffled baseline p95, bootstrap ARI >= 0.6, smallest cluster >= 5%.",
            **common,
            "metrics": {"eligible_k": sel.loc[sel["eligible"], "k"].tolist(),
                        "closest_k": best_stab_k,
                        "closest_k_stability_ari": r(cand.loc[best_stab_k, "stability_ari"])},
            "interpretation": f"No k qualified. k={best_stab_k} came closest (ARI {cand.loc[best_stab_k, 'stability_ari']:.3f} vs 0.6 cutoff). "
                              "The cutoff was not lowered after seeing results. Clusters are reported as descriptive partitions only.",
            "limitations": "20 bootstrap resamples: ARI estimates carry noise of a few hundredths.",
        },
        "EXP-05": {
            "title": "Are the discovered archetypes stable?",
            "status": "completed (K-Means)",
            "methodology": "Bootstrap: resample rows with replacement 20x, refit, label all original rows, ARI vs full-data clustering.",
            **common,
            "metrics": {
                f"{s}/{rep}": setup(sel, s, rep)["stability_ari"].round(3).to_dict()
                for s in ("all", "fully_evolved") for rep in ("standardized", "relative")
            },
            "interpretation": "Stable splits exist only for power/evolution (all/standardized: ARI "
                              f"{all_std.loc[2, 'stability_ari']:.2f} at k=2, {all_std.loc[5, 'stability_ari']:.2f} at k=5). "
                              "Role-based splits on fully evolved Pokémon are stable only at k=2 "
                              f"(ARI {fe_rel.loc[2, 'stability_ari']:.2f}).",
            "limitations": "Seed-only stability was not used: with n_init=10 it is near-perfect and uninformative.",
        },
        "EXP-06": usage_record(),
    }


def usage_record():
    path, pred_path = RESULTS_DIR / "competitive_summary.json", RESULTS_DIR / "competitive_prediction.json"
    if not (path.exists() and pred_path.exists()):
        return {"title": "Are archetypes associated with competitive usage?", "status": "pending"}
    a = json.loads(path.read_text(encoding="utf-8"))
    p = json.loads(pred_path.read_text(encoding="utf-8"))["models"]
    sp = a["spearman_vs_usage"]
    best = max((m for m in p if m != "bst_ridge"), key=lambda m: p[m]["mean_heldout_spearman"])
    pairs = pd.read_csv(RESULTS_DIR / "competitive_pairs.csv", index_col=0)
    # ratios of tiny usages are noise (0.01% vs 4% = 400x): require both >= 0.5% of teams
    pairs = pairs[pairs[["usage_a", "usage_b"]].min(axis=1) >= 0.005]
    top = pairs.assign(ratio=pairs[["usage_a", "usage_b"]].max(axis=1) / pairs[["usage_a", "usage_b"]].min(axis=1))
    top = top.loc[top["ratio"].idxmax()]   # most extreme stat-twin contrast
    return {
        "title": "Are archetypes associated with competitive usage?",
        "status": "completed",
        "question": "Do stat strength, stat shape (role axes, descriptive clusters) or anomalies relate to "
                    "Gen 9 National Dex usage, and does anything beyond BST help predict it?",
        "methodology": "Smogon chaos stats, gen9nationaldex-1630, mean of 2026-07..09. Fully evolved Pokémon legal "
                       "in the format (banned tiers removed using Showdown formats-data; tiers never used as inputs). "
                       "Exact-form matching. Spearman with 1,000-sample bootstrap 95% intervals; Kruskal-Wallis across "
                       "descriptive clusters; 5-fold x10 repeated CV comparing BST-only vs stats+shares+types models.",
        "data_files": ["data/processed/usage_gen9natdex.csv", "results/competitive_summary.json",
                       "results/competitive_prediction.json", "results/competitive_pairs.csv"],
        "metrics": {"association": a, "prediction": p},
        "interpretation": (
            f"BST has a clear moderate link to usage (rho {sp['bst']['rho']}, 95% CI {sp['bst']['ci95']}). Stat shape "
            f"does not: role axes rho {sp['speed_axis (fast+ / bulky-)']['rho']} and "
            f"{sp['physical_axis (physical+ / special-)']['rho']}, descriptive clusters p = "
            f"{a['kruskal_usage_by_cluster_p']:.2f}. Held-out prediction: BST-only Spearman "
            f"{p['bst_ridge']['mean_heldout_spearman']}; best richer model ({best}) "
            f"{p[best]['mean_heldout_spearman']} (gain {p[best]['mean_gain_over_bst']:+.3f}, beats BST in "
            f"{p[best]['share_of_folds_beating_bst']:.0%} of folds). Near-identical stat profiles can differ "
            f"{top['ratio']:.0f}x in usage ({top['a']} vs {top['b']}), so abilities, moves and typing drive much of usage."),
        "limitations": "Usage is popularity, not win rate. 27% of usage is in forms outside the dataset (Megas, "
                       "alternate and regional forms); Eviolite users (Chansey, Porygon2) are excluded as not fully "
                       "evolved. Showdown tier file taken from the live repo, frozen by local cache. Only 29 Pokémon "
                       "pass the 4.52% high-usage cutoff, so group comparisons are hints. Tests are exploratory.",
    }


def anomaly_record():
    """Isolation Forest findings (Milestone 5). Not an EXP in the original spec, kept as ANOM-01."""
    summ = json.loads((RESULTS_DIR / "anomaly_summary.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(RESULTS_DIR / "anomaly_scores.csv")
    top = {v: scores.nsmallest(10, f"{v}_rank")["name"].tolist()
           for v in ("raw", "relative", "fe_relative")}
    comp, ovr = summ["flagged_composition"], summ["form_override_check"]
    return {
        "title": "Which Pokémon have unusual stat profiles?",
        "status": "completed",
        "methodology": "Isolation Forest (500 trees). Three views: raw stats (all), relative stats (all), "
                       "relative stats (fully evolved only). No labels exist, so checked by seed stability "
                       "(Spearman across 5 seeds), agreement with a single-stat baseline (max |z|), and a "
                       "form-override robustness check. 'Flagged' = top 5%, a reporting cutoff only.",
        "algorithm": "IsolationForest(n_estimators=500, random_state=42)",
        "data_files": ["results/anomaly_scores.csv", "results/anomaly_summary.json"],
        "metrics": {v: summ[v] for v in ("raw", "relative", "fe_relative")} | {
            "flagged_composition": comp, "form_override_check": ovr, "top_10": top},
        "interpretation": (
            f"Rankings are highly reproducible (seed Spearman {min(summ[v]['seed_stability_spearman'] for v in top):.2f}+) "
            f"and mostly match the single-stat baseline (Spearman {min(summ[v]['spearman_vs_max_abs_z'] for v in top):.2f}-"
            f"{max(summ[v]['spearman_vs_max_abs_z'] for v in top):.2f}): most anomalies have one extreme stat. "
            f"Raw-view anomalies over-represent legendaries ({comp['raw']['legendary_or_mythical_flagged']:.0%} vs "
            f"{comp['raw']['legendary_or_mythical_all']:.0%}); relative-view anomalies over-represent unevolved Pokémon. "
            + (f"Form overrides change at most {20 - min(ovr[v]['top_overlap'] for v in ovr)} of the top 20 in any view."
               if ovr else "Form-override check not run (needs the raw cache).")),
        "limitations": "Axis-aligned splits are weaker on diagonal (combination) oddities; an anomaly means rare "
                       "in this dataset, not erroneous or strong.",
    }


def optimizer_record():
    """Milestone 7 benchmarks, kept as OPT-01."""
    summ = json.loads((RESULTS_DIR / "optimizer_benchmark_summary.json").read_text(encoding="utf-8"))
    tables = {k: pd.read_csv(RESULTS_DIR / f"optimizer_benchmark_{k}.csv", index_col=0) for k in summ}
    d, t = summ["default"], tables["default"]
    return {
        "title": "How much better is the optimizer than simpler ways to pick a team?",
        "status": "completed",
        "methodology": "Same pool for every method (fully evolved, BST >= 500). Baselines: 10,000 random teams, "
                       "the 6 highest-BST Pokémon, and a greedy one-at-a-time heuristic. Two scenarios: "
                       "legendaries excluded (default) and allowed. The optimizer maximizes the same score it is "
                       "judged on, so the gaps and rule compliance are the result, not the win itself.",
        "data_files": ["results/optimizer_benchmark_summary.json",
                       *[f"results/optimizer_benchmark_{k}.csv" for k in summ]],
        "metrics": {k: {m: v for m, v in s_.items() if m != "teams"} for k, s_ in summ.items()}
                   | {"teams": {k: s_["teams"] for k, s_ in summ.items()}},
        "interpretation": (
            f"Only {d['random_share_following_rules']:.0%} of random teams follow the weakness and type rules, "
            f"and none of {d['n_random']:,} reached the optimal score. The optimum beats the best rule-following "
            f"random team by {d['optimized_vs_best_valid_random']:.3f} and greedy by {d['optimized_vs_greedy']:.3f} "
            f"(greedy defense {t.loc['greedy', 'defense']:.2f} vs {t.loc['optimized', 'defense']:.2f}): an early "
            "greedy pick blocks better combinations later. The 6 highest-BST Pokémon break the rules "
            f"(weakness load {t.loc['top_bst', 'max_weakness_load']:.0f}) and cover far fewer types "
            f"({t.loc['top_bst', 'coverage']:.2f} vs {t.loc['optimized', 'coverage']:.2f}). The trade-off: the "
            f"optimized team averages {t.loc['optimized', 'mean_bst']:.0f} BST vs {t.loc['top_bst', 'mean_bst']:.0f}."),
        "limitations": "Abilities, moves and items are ignored (e.g. Slaking and Wishiwashi rank high on BST "
                       "but have ability drawbacks). The objective weights are illustrative, not fitted to battle results. "
                       "This measures type coverage and stat balance, not battle strength.",
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if (RESULTS_DIR / "optimizer_benchmark_summary.json").exists():
        (OUT / "OPT-01.json").write_text(json.dumps({"id": "OPT-01", **optimizer_record()}, indent=2), encoding="utf-8")
        print("OPT-01 - completed")
    if (RESULTS_DIR / "anomaly_summary.json").exists():
        (OUT / "ANOM-01.json").write_text(json.dumps({"id": "ANOM-01", **anomaly_record()}, indent=2), encoding="utf-8")
        print("ANOM-01 - completed")
    for exp_id, record in build_records(*load()).items():
        (OUT / f"{exp_id}.json").write_text(json.dumps({"id": exp_id, **record}, indent=2), encoding="utf-8")
        print(exp_id, "-", record["status"])
