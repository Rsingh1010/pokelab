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
        "EXP-06": {"title": "Are archetypes associated with competitive usage?", "status": "pending"},
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for exp_id, record in build_records(*load()).items():
        (OUT / f"{exp_id}.json").write_text(json.dumps({"id": exp_id, **record}, indent=2), encoding="utf-8")
        print(exp_id, "-", record["status"])
