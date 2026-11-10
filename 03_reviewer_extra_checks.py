#!/usr/bin/env python3
"""
03_reviewer_extra_checks.py -- extra analyses the reviewers asked for that the main pipeline does not cover
===========================================================================================================
Reads outputs/model_results.pkl (written by 01_training_and_validation.py v2.0.0) and writes to
outputs/extra_checks/ :

  heldout_independence.csv      how correlated the "held-out" quantities are with the clustering bands
                                (R1 #6: held-out is not independent if they move together)
  warping_window_sensitivity.csv  Hard-DTW refitted for several Sakoe-Chiba radii incl. none (R2 #5)
  band_subset_ablation.csv      does a subset of bands reproduce the full partition? (R2 #3: why hyperspectral,
                                which few bands carry the signal)
  medium_confounding.csv/_crosstab_*.csv   ARI / NMI / permutation p of every partition vs culture medium
                                (R1 #1) -- needs strain_metadata.csv with columns Species, Medium

Run:  python 03_reviewer_extra_checks.py --out_dir outputs [--metadata strain_metadata.csv]
"""
import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

SEED = 42
_trapz = getattr(np, "trapezoid", None) or np.trapz


def load(out_dir):
    with open(Path(out_dir) / "model_results.pkl", "rb") as fh:
        return pickle.load(fh)


def heldout_independence(B):
    traj, bands = B["traj_full"], B["band_names_all"]
    days = np.asarray(B["day_order"], float)
    held = bands.index(B["heldout_band"])
    auc = {b: _trapz(traj[:, :, i], days, axis=1) for i, b in enumerate(bands)}
    rows = []
    for b in B["clustering_bands"]:
        i = bands.index(b)
        rho, p = spearmanr(auc[B["heldout_band"]], auc[b])
        within = [np.corrcoef(traj[s, :, held], traj[s, :, i])[0, 1] for s in range(traj.shape[0])]
        rows.append({"quantity": f"{B['heldout_band']} AUC", "vs": f"{b} AUC", "spearman_rho_across_strains": rho,
                     "p": p, "median_within_strain_pearson_of_trajectories": float(np.nanmedian(within))})
    st = B["species_table"].set_index("Species").loc[B["species_order"]]
    for feat in ("NDVI", "GNDVI", "Simple_Ratio"):
        for b in B["clustering_bands"]:
            rho, p = spearmanr(st[feat].values, auc[b])
            rows.append({"quantity": f"{feat} (species mean)", "vs": f"{b} AUC", "spearman_rho_across_strains": rho,
                         "p": p, "median_within_strain_pearson_of_trajectories": np.nan})
    return pd.DataFrame(rows)


def warping_window_sensitivity(B, radii=(0, 1, 2, 3, 5, 8, 10, 30, None)):
    sc = TimeSeriesScalerMeanVariance().fit_transform(B["traj_cluster"])
    k = int(B["res"]["Hard-DTW"]["params"]["k"])
    selected = np.asarray(B["res"]["Hard-DTW"]["labels"])

    def fit(r):
        mp = None if r is None else dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=r)
        return TimeSeriesKMeans(n_clusters=k, metric="dtw", metric_params=mp, max_iter=50, n_init=5,
                                random_state=SEED).fit_predict(sc)

    labs = {r: fit(r) for r in radii}
    unc = labs[None]
    return pd.DataFrame([{"sakoe_chiba_radius_days": "none" if r is None else r, "k": k,
                          "ARI_vs_selected_partition": adjusted_rand_score(selected, l),
                          "ARI_vs_unconstrained_refit": adjusted_rand_score(unc, l)} for r, l in labs.items()])


def band_subset_ablation(B):
    """Hierarchical/DTW with the model's own linkage and k on every subset of the clustering bands."""
    P = B["res"]["Hierarchical-DTW"]["params"]
    k, lk = int(P["k"]), str(P["linkage"])
    full = np.asarray(B["res"]["Hierarchical-DTW"]["labels"])
    X, names = B["traj_cluster"], list(B["clustering_bands"])
    rows = []
    for size in range(1, len(names)):
        for sub in itertools.combinations(range(len(names)), size):
            sc = TimeSeriesScalerMeanVariance().fit_transform(X[:, :, list(sub)])
            D = cdist_dtw(sc)
            D = (D + D.T) / 2
            np.fill_diagonal(D, 0)
            lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=lk).fit_predict(D)
            rows.append({"n_bands": size, "bands": "+".join(names[i] for i in sub),
                         "ARI_vs_full_5band_partition": adjusted_rand_score(full, lab)})
    return pd.DataFrame(rows).sort_values(["n_bands", "ARI_vs_full_5band_partition"], ascending=[True, False])


def medium_confounding(B, meta_path, out, n_perm=4999):
    meta = pd.read_csv(meta_path)
    if not {"Species", "Medium"} <= set(meta.columns):
        raise ValueError("strain_metadata.csv needs columns Species and Medium")
    med = meta.drop_duplicates("Species").set_index("Species").reindex(B["species_order"])["Medium"]
    if med.isna().any():
        raise ValueError(f"No medium given for: {med[med.isna()].index.tolist()}")
    codes = pd.factorize(med.values)[0]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, lab in B["partitions"].items():
        lab = np.asarray(lab)
        a0 = adjusted_rand_score(codes, lab)
        hits = sum(adjusted_rand_score(rng.permutation(codes), lab) >= a0 - 1e-12 for _ in range(n_perm))
        rows.append({"partition": name, "ARI_vs_medium": a0, "NMI_vs_medium": normalized_mutual_info_score(codes, lab),
                     "p_permutation_ARI": (hits + 1) / (n_perm + 1), "n_media": int(len(set(codes)))})
        pd.crosstab(med.values, lab).to_csv(out / f"medium_confounding_crosstab_{name}.csv")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--metadata", default=None, help="CSV with columns Species, Medium (optional: Salinity, Accession)")
    args, _ = ap.parse_known_args(argv)
    B = load(args.out_dir)
    sp = B["species_order"]
    cons = dict(zip(B["consensus_df"]["Species"], B["consensus_df"]["Consensus_Cluster"]))
    B["partitions"] = {**{n: np.asarray(r["labels"]) for n, r in B["res"].items()},
                       "Consensus": np.array([cons[s] for s in sp])}
    out = Path(args.out_dir) / "extra_checks"
    out.mkdir(exist_ok=True)
    df = heldout_independence(B)
    df.to_csv(out / "heldout_independence.csv", index=False)
    print("\nHeld-out vs clustering bands (Spearman across strains):\n", df.round(3).to_string(index=False))
    df = warping_window_sensitivity(B)
    df.to_csv(out / "warping_window_sensitivity.csv", index=False)
    print("\nWarping-window sensitivity (Hard-DTW refit):\n", df.round(3).to_string(index=False))
    df = band_subset_ablation(B)
    df.to_csv(out / "band_subset_ablation.csv", index=False)
    print("\nBest subset per size (ARI vs full 5-band partition):\n",
          df.groupby("n_bands").head(1).round(3).to_string(index=False))
    if args.metadata:
        df = medium_confounding(B, args.metadata, out)
        df.to_csv(out / "medium_confounding.csv", index=False)
        print("\nPartitions vs culture medium:\n", df.round(3).to_string(index=False))
    else:
        print("\n(no --metadata given: medium-confounding test skipped; supply strain_metadata.csv)")
    print("\nWritten to", out)


if __name__ == "__main__":
    main()
