#!/usr/bin/env python3
"""
02_figures_corrected.py   (single figure entry point, v2.0.0)
=============================================================
Generates every manuscript / supplementary figure from outputs/model_results.pkl, the single saved
result object written by 01_training_and_validation.py. Because text, tables and figures all read
that one object, cluster counts, assignments and labels cannot disagree between them
(reviewer comment 5: Fig. 4 showed k = 6 while text/table said k = 5).

Run:
    python 02_figures_corrected.py                      # uses ./outputs
    python 02_figures_corrected.py --out_dir outputs --dpi 600

Conventions
  * No "Figure N" numbers are printed inside the images; numbering belongs in the manuscript captions.
    figure_audit_table.csv lists, for every file, the function, feature set and source object, so
    each manuscript figure can be traced to a script + input + saved output (reviewer comment 2).
  * Feature-set wording used throughout:
        5-band clustering tensor   = Chlorophyll_680, Carotenoid_480, Phycobiliprotein_620,
                                     Lipid_CH_930, Protein_Amide_910   (what the models saw)
        held-out band              = Water_NIR_970 (never seen by any clustering model)
        6-band panels              = DESCRIPTIVE plots only
  * Titles say "spectral trajectory" / "candidate harvest-window hypothesis": the measured signals are
    hyperspectral band responses, not biomass, lipid, pigment yield or a verified harvest optimum.
  * Times-like font, six-colour palette (extra categories use hatching).
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import textwrap
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from tslearn.metrics import cdist_dtw, dtw_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

warnings.filterwarnings("ignore", message="findfont.*")
_trapz = getattr(np, "trapezoid", None) or np.trapz
SEED = 42
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # running inside Jupyter / IPython: no __file__
    BASE_DIR = Path.cwd()

FEAT_5 = "5-band clustering tensor (Water_NIR_970 excluded)"
FEAT_6 = "6-band DESCRIPTIVE plot (Water_NIR_970 held out, not used for clustering)"
FEAT_H = "held-out summaries: Water_NIR_970 descriptors + NDVI/GNDVI/Simple_Ratio"
FEAT_0 = "schematic (no data)"

# ----------------------------------------------------------------------------
# House style
# ----------------------------------------------------------------------------
C = {"olive": "#494623", "ochre": "#c3892b", "sage": "#8e883d", "brown": "#8f7c2f", "cream": "#e8e7d8", "white": "#ffffff"}
CYCLE = [C["olive"], C["ochre"], C["sage"], C["brown"]]
HATCHES = ["", "//", "xx", "..", "\\\\", "++", "oo", "**"]
_TIMES = ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "Tinos", "DejaVu Serif"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
TIMES_FONT = next((n for n in _TIMES if n in _avail), "DejaVu Serif")

DPI = 600
FIG_DIR = Path("outputs/figures")
REGISTRY = []


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [TIMES_FONT] + _TIMES, "mathtext.fontset": "stix",
        "figure.facecolor": C["white"], "axes.facecolor": C["white"], "savefig.facecolor": C["white"],
        "axes.edgecolor": C["olive"], "axes.labelcolor": C["olive"], "text.color": C["olive"],
        "xtick.color": C["olive"], "ytick.color": C["olive"], "axes.linewidth": 1.1,
        "grid.color": C["cream"], "grid.linestyle": "--", "grid.alpha": 0.9,
        "axes.titlesize": 11.5, "axes.labelsize": 10.5, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def ccolor(i):
    i = int(i)
    return CYCLE[i % len(CYCLE)], HATCHES[(i // len(CYCLE)) % len(HATCHES)]


def wrap(t, w=78):
    return "\n".join(textwrap.wrap(t, width=w))


def pretty(b):
    return b.replace("_", " ")


def save(fig, stub, description, features, source):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stub}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stub}.pdf", bbox_inches="tight")
    plt.close(fig)
    REGISTRY.append(dict(output=f"{stub}.png/.pdf", description=description, input_feature_set=features,
                         source_object=source, script="02_figures_corrected.py"))
    print(f"  saved {stub}")


# ----------------------------------------------------------------------------
# Data access: ONE saved result object
# ----------------------------------------------------------------------------
def load_bundle(out_dir):
    with open(Path(out_dir) / "model_results.pkl", "rb") as fh:
        B = pickle.load(fh)
    sp = B["species_order"]
    cons = dict(zip(B["consensus_df"]["Species"], B["consensus_df"]["Consensus_Cluster"]))
    B["partitions"] = {**{n: np.asarray(r["labels"]) for n, r in B["res"].items()},
                       "Consensus": np.array([cons[s] for s in sp])}
    if B["manifest"]["data_source"] != "real":
        raise RuntimeError("Bundle was not produced from real data.")
    return B


def band_idx(B, name):
    return B["band_names_all"].index(name)


def scaled_cluster(B):
    s = TimeSeriesScalerMeanVariance().fit_transform(B["traj_cluster"])
    return s, s.reshape(s.shape[0], -1)


def k_of(lab):
    return len(np.unique(lab))


# ----------------------------------------------------------------------------
# F01 / F07  schematics
# ----------------------------------------------------------------------------
def fig_algorithm_mechanics(B):
    style()
    g = B["res"]["Soft-DTW"]["params"]["gamma"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    x = np.linspace(0, 10, 60)
    a, b = np.sin(x / 2) + 0.15 * x, np.sin((x - 1.4) / 2) + 0.15 * (x - 1.4) + 0.15
    for ax in axes:
        ax.plot(x, a, color=C["olive"], lw=2.2, label="Strain A")
        ax.plot(x, b, color=C["ochre"], lw=2.2, label="Strain B")
        ax.set_xlabel("Cultivation day (schematic)")
        ax.grid(True)
    for i in range(0, len(x), 6):
        axes[0].plot([x[i]] * 2, [a[i], b[i]], color=C["sage"], lw=0.9)
    for i in range(0, len(x), 5):
        for j in range(max(0, i - 3), min(len(x), i + 4)):
            axes[1].plot([x[i], x[j]], [a[i], b[j]], color=C["brown"], lw=0.4, alpha=0.18)
    w = int(np.argmax(np.abs(a - b)))
    axes[2].plot([x[w]] * 2, [a[w], b[w]], color=C["sage"], lw=2.6)
    axes[2].annotate("complete linkage:\nmaximum pairwise distance", xy=(x[w], (a[w] + b[w]) / 2), xytext=(0.4, 0.1),
                     textcoords="axes fraction", fontsize=8, arrowprops=dict(arrowstyle="->", color=C["olive"]))
    axes[0].set_title(wrap("Hard-DTW k-means: single minimum-cost alignment path", 40))
    axes[1].set_title(wrap(f"Soft-DTW k-means ($\\gamma$ = {g:.3g}): weighted expectation over all paths", 40))
    axes[2].set_title(wrap("Hierarchical / DTW: complete-linkage distance", 40))
    axes[0].set_ylabel("Band signal (a.u., schematic)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "F01_algorithm_mechanics", "Schematic of the three clustering strategies", FEAT_0, "res['Soft-DTW'].params.gamma")


def fig_dtw_vs_euclidean(B):
    style()
    x = np.linspace(0, 10, 80)
    a, b = np.sin(x / 2 - 0.3) + 0.1 * x, np.sin((x - 1.6) / 2 - 0.3) + 0.1 * (x - 1.6) + 0.2
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True, constrained_layout=True)
    for ax in axes:
        ax.plot(x, a, color=C["olive"], lw=2.2, label="Strain A")
        ax.plot(x, b, color=C["ochre"], lw=2.2, label="Strain B")
        ax.set_xlabel("Cultivation day (schematic)")
        ax.grid(True)
    for i in range(0, len(x), 4):
        axes[0].plot([x[i]] * 2, [a[i], b[i]], color=C["sage"], lw=0.8)
    for i in range(0, len(x) - 8, 4):
        axes[1].plot([x[i], x[i + 8]], [a[i], b[i + 8]], color=C["brown"], lw=0.8)
    axes[0].set_title("Rigid Euclidean matching (fixed $t_i \\leftrightarrow t_i$)")
    axes[1].set_title("Elastic DTW alignment (warped path)")
    axes[0].set_ylabel("Band signal (a.u., schematic)")
    axes[0].legend(frameon=False)
    save(fig, "F07_dtw_vs_euclidean", "Schematic: Euclidean vs DTW matching", FEAT_0, "none")


# ----------------------------------------------------------------------------
# F02 Sankey -- built from the same saved partitions as text and tables
# ----------------------------------------------------------------------------
def _node_layout(labels, pad=0.8):
    counts = pd.Series(labels).value_counts().sort_index()
    lay, cur = {}, 0.0
    for c, n in counts.items():
        lay[int(c)] = (cur, cur + n)
        cur += n + pad
    return lay, dict(counts), cur - pad


def _ribbon(ax, x0, x1, a0, a1, b0, b1, color, n=60):
    xs = np.linspace(x0, x1, n)
    t = (xs - x0) / (x1 - x0)
    s = t * t * (3 - 2 * t)
    ax.fill_between(xs, a0 + (b0 - a0) * s, a1 + (b1 - a1) * s, color=color, alpha=0.55, lw=0, zorder=1)


def fig_cluster_flow(B):
    style()
    P = B["partitions"]
    names = ["Hard-DTW", "Soft-DTW", "Hierarchical-DTW", "Consensus"]
    pref = ["M1", "M2", "M3", "CN"]
    stages = [[int(v) for v in P[n]] for n in names]
    titles = [f"{n} (k = {k_of(s)})" for n, s in zip(names, stages)]
    lays = [_node_layout(s) for s in stages]
    hmax = max(l[2] for l in lays)
    gap, nw, ns = 2.6, 0.22, len(stages[0])
    xs = [i * gap for i in range(4)]
    fig, ax = plt.subplots(figsize=(15.5, max(6.8, hmax * 0.42)), constrained_layout=True)
    for s in range(3):
        trans = {}
        for i in range(ns):
            trans[(stages[s][i], stages[s + 1][i])] = trans.get((stages[s][i], stages[s + 1][i]), 0) + 1
        lc, rc = {c: lays[s][0][c][0] for c in lays[s][0]}, {c: lays[s + 1][0][c][0] for c in lays[s + 1][0]}
        for (cl, cr) in sorted(trans):
            n = trans[(cl, cr)]
            a0, b0 = lc[cl], rc[cr]
            lc[cl] += n
            rc[cr] += n
            _ribbon(ax, xs[s] + nw, xs[s + 1] - nw, a0, a0 + n, b0, b0 + n, ccolor(cl)[0])
    for si, (lay, cnt, _) in enumerate(lays):
        for cid, (y0, y1) in lay.items():
            col, h = ccolor(cid)
            ax.add_patch(mpatches.Rectangle((xs[si] - nw, y0), 2 * nw, y1 - y0, lw=1.3, ec=C["olive"], fc=col, hatch=h, alpha=0.95, zorder=3))
            ax.text(xs[si], (y0 + y1) / 2, f"{pref[si]}-C{cid}\n(n={cnt[cid]})", ha="center", va="center", fontsize=7.4,
                    fontweight="bold", color=C["white"], zorder=4)
    for si, t in enumerate(titles):
        ax.text(xs[si], -1.6, t, ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xlim(xs[0] - 1, xs[-1] + 1)
    ax.set_ylim(hmax + 0.5, -2.6)
    ax.axis("off")
    ident = [n for n in names[:3] if adjusted_rand_score(P["Consensus"], P[n]) >= 0.999]
    sub = f" Consensus partition is identical to {', '.join(ident)}." if ident else ""
    fig.suptitle(wrap("Species flow across the three models and the consensus (ribbon width = number of strains)." + sub, 105), fontsize=11.5)
    save(fig, "F02_cluster_flow_sankey", "Sankey of partitions; k labels computed from saved labels", FEAT_5, "partitions (model_results.pkl)")


# ----------------------------------------------------------------------------
# F03  per-band spectral trajectories per cluster (any partition)
# ----------------------------------------------------------------------------
def fig_all_bands(B, name):
    style()
    lab = B["partitions"][name]
    cl = sorted(np.unique(lab))
    bands, days = B["band_names_all"], B["day_order"]
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.6), constrained_layout=True)
    for bi, (ax, band) in enumerate(zip(axes.ravel(), bands)):
        for ci, c in enumerate(cl):
            m = B["traj_full"][lab == c, :, bi]
            mu, sd = m.mean(0), m.std(0)
            col, _ = ccolor(ci)
            ax.plot(days, mu, color=col, lw=2.2, marker="o", ms=3, label=f"Cluster {c} (n={int((lab == c).sum())})")
            ax.fill_between(days, mu - sd, mu + sd, color=col, alpha=0.15)
        held = band == B["heldout_band"]
        ax.set_title(pretty(band) + ("  [held out: not used for clustering]" if held else "  [clustering band]"), fontsize=10)
        if held:
            ax.set_facecolor(C["cream"])
        ax.set_xlabel("Cultivation day")
        ax.set_ylabel("Mean band signal")
        ax.grid(True)
    axes[0, 0].legend(fontsize=7.4, frameon=True)
    fig.suptitle(wrap(f"Spectral trajectories per cluster, {name} (k = {len(cl)}): mean $\\pm$ SD across strains", 100), fontsize=12)
    save(fig, f"F03_allbands_{name.replace('-', '_')}", f"Per-band spectral trajectories, {name}", FEAT_6, "partitions + traj_full")


# ----------------------------------------------------------------------------
# F04 held-out summaries (points, medians, q-values, effect sizes)
# ----------------------------------------------------------------------------
def fig_heldout_summaries(B, name="Consensus"):
    style()
    V = B["validation"][name]
    merged, tests = V["merged"], V["tests_df"].set_index("feature")
    clusters = sorted(merged["Cluster"].unique())
    feats = [("NDVI", "NDVI"), ("GNDVI", "GNDVI"), ("Water_AUC", "Water/NIR AUC (held-out band)"), ("Water_Day_of_Max", "Water/NIR day of maximum")]
    fig, axes = plt.subplots(1, len(feats), figsize=(16, 4.8), constrained_layout=True)
    rng = np.random.default_rng(SEED)
    for ax, (f, title) in zip(axes, feats):
        for i, c in enumerate(clusters):
            v = merged.loc[merged["Cluster"] == c, f].dropna().values
            col, _ = ccolor(i)
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(v)), v, s=46, color=col, edgecolor=C["olive"], zorder=3)
            if len(v):
                ax.hlines(np.median(v), i + 0.7, i + 1.3, color=C["olive"], lw=2.2, zorder=4)
        t = tests.loc[f]
        ax.set_title(f"{title}\n$H$={t['H']:.2f}, $q_{{BH}}$={t['q_fdr_bh']:.3f}, $\\varepsilon^2$={t['epsilon_squared']:.2f}", fontsize=10)
        ax.set_xticks(range(1, len(clusters) + 1))
        ax.set_xticklabels([f"C{c}\n(n={int((merged['Cluster'] == c).sum())})" for c in clusters])
        ax.grid(True, axis="y")
    fig.suptitle(wrap(f"Held-out spectral summaries across {name} clusters (Kruskal-Wallis, exploratory; bars = medians; "
                      f"BH family = 8 tests). Not independent biological assays.", 120), fontsize=11)
    save(fig, f"F04_heldout_summaries_{name}", "Held-out spectral summaries per cluster", FEAT_H, "validation[...].merged / tests_df")


# ----------------------------------------------------------------------------
# F05 channel audit, F06 taxonomy
# ----------------------------------------------------------------------------
def fig_channel_audit(B):
    style()
    df = B["channel_report"]
    if df is None or df.empty:
        print("  [channel audit] no report in bundle; skipped")
        return
    bands, chans = sorted(df["band"].unique()), sorted(df["channel"].unique())
    fig, ax = plt.subplots(figsize=(9.8, 5.4), constrained_layout=True)
    x, w = np.arange(len(bands)), 0.8 / len(chans)
    for ci, ch in enumerate(chans):
        col, h = ccolor(ci)
        v = [df[(df["band"] == b) & (df["channel"] == ch)]["fraction_files_with_real_signal"].values[0] for b in bands]
        ax.bar(x + ci * w, v, w, color=col, edgecolor=C["olive"], hatch=h, label=ch)
    ax.set_xticks(x + w * (len(chans) - 1) / 2)
    ax.set_xticklabels([pretty(b) for b in bands], rotation=30, ha="right")
    ax.set_ylabel("Fraction of sampled files with non-zero, non-constant signal")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, title="Channel")
    ax.grid(True, axis="y")
    fig.suptitle(f"Channel signal-coverage audit (channel used for all results: {B['manifest']['channel_used']})", fontsize=11.5)
    save(fig, "F05_channel_signal_audit", "Channel audit", "all channels x 6 bands", "channel_report")


def fig_taxonomy(B):
    style()
    V = B["validation"]["Consensus"]["taxonomy"]
    ct = V["crosstab"]
    if ct is None:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    bottom, x = np.zeros(ct.shape[1]), np.arange(ct.shape[1])
    for ti, tx in enumerate(ct.index):
        col, h = ccolor(ti)
        v = ct.loc[tx].values.astype(float)
        ax.bar(x, v, bottom=bottom, color=col, edgecolor=C["olive"], hatch=h, label=wrap(tx, 26), width=0.6)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in ct.columns])
    ax.set_xlabel("Consensus cluster")
    ax.set_ylabel("Number of strains")
    ax.legend(frameon=False, fontsize=7.8, title="Taxon", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, axis="y")
    fig.suptitle(wrap(f"Taxon composition per consensus cluster (ARI = {V['ari']:.3f}, NMI = {V['nmi']:.3f}, "
                      f"permutation p(ARI) = {V['p_perm_ari']:.3f})", 90), fontsize=11.5)
    save(fig, "F06_taxonomy_vs_consensus", "Taxonomy (external metadata) vs consensus", "external taxonomy", "validation['Consensus'].taxonomy")


# ----------------------------------------------------------------------------
# F08 silhouette profiles in DTW geometry
# ----------------------------------------------------------------------------
def fig_silhouette_profiles(B):
    style()
    D = B["shared_dtw_dist"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5), constrained_layout=True)
    for ax, n in zip(axes, ["Hard-DTW", "Soft-DTW", "Hierarchical-DTW"]):
        lab = B["partitions"][n]
        s = silhouette_samples(D, lab, metric="precomputed")
        y = 10
        for ci, c in enumerate(sorted(np.unique(lab))):
            v = np.sort(s[lab == c])[::-1]
            col, h = ccolor(ci)
            ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, facecolor=col, hatch=h, edgecolor=C["olive"], label=f"C{c}")
            y += len(v) + 5
        ax.axvline(s.mean(), color=C["brown"], ls="--", lw=2, label=f"mean = {s.mean():.3f}")
        ax.set_title(f"{n} (k = {k_of(lab)})", fontsize=10.5)
        ax.set_xlabel("Silhouette coefficient (precomputed DTW distances)")
        ax.yaxis.set_visible(False)
        ax.set_xlim(min(-0.4, s.min() - 0.05), 1)
        ax.grid(True, axis="x")
        ax.legend(loc="upper right", fontsize=7.4)
    fig.suptitle("Silhouette profiles in DTW geometry (modest values = limited-to-moderate separation)", fontsize=11.5)
    save(fig, "F08_silhouette_profiles_dtw", "Silhouette profiles using DTW distances", FEAT_5, "shared_dtw_dist + partitions")


# ----------------------------------------------------------------------------
# F09 DTW alignment example
# ----------------------------------------------------------------------------
def fig_dtw_alignment(B, band="Chlorophyll_680"):
    style()
    D, sc = B["shared_dtw_dist"], scaled_cluster(B)[0]
    sp, days = B["species_order"], B["day_order"]
    iu = np.triu_indices(len(sp), 1)
    near, far = np.argmin(D[iu]), np.argmax(D[iu])
    bi = B["clustering_bands"].index(band)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    for ax, k, t in zip(axes, (near, far), ("Most similar pair (minimum 5-band DTW distance)", "Most dissimilar pair (maximum 5-band DTW distance)")):
        i, j = iu[0][k], iu[1][k]
        s1, s2 = sc[i, :, bi].reshape(-1, 1), sc[j, :, bi].reshape(-1, 1)
        path, d = dtw_path(s1, s2)
        ax.plot(days, s1[:, 0], color=C["olive"], lw=2.2, marker="o", ms=3, label=sp[i])
        ax.plot(days, s2[:, 0] + 3.5, color=C["ochre"], lw=2.2, marker="o", ms=3, label=sp[j])
        for a, b in path[::2]:
            ax.plot([days[a], days[b]], [s1[a, 0], s2[b, 0] + 3.5], color=C["sage"], lw=0.6, alpha=0.6)
        ax.set_title(f"{wrap(t, 38)}\n{band} shown; univariate DTW = {d:.2f}", fontsize=10)
        ax.set_xlabel("Cultivation day")
        ax.legend(frameon=False)
        ax.grid(True)
    save(fig, "F09_dtw_alignment_example", "Worked DTW alignment (pair chosen from 5-band distance matrix)", FEAT_5, "shared_dtw_dist + traj_cluster")


# ----------------------------------------------------------------------------
# F10 co-association heatmap (saved matrix; sorted by consensus cluster)
# ----------------------------------------------------------------------------
def fig_coassociation(B):
    style()
    co, sp, cons = B["coassoc"], B["species_order"], B["partitions"]["Consensus"]
    order = sorted(range(len(sp)), key=lambda i: (cons[i], sp[i]))
    M = co[np.ix_(order, order)]
    lab = [sp[i] for i in order]
    n = len(sp)
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 3), constrained_layout=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("p", [C["white"], C["cream"], C["ochre"], C["olive"]])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(lab, rotation=90, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(lab, fontsize=9)
    cs = np.array([cons[i] for i in order])
    for b in np.where(np.diff(cs) != 0)[0]:
        ax.axhline(b + 0.5, color=C["olive"], lw=1)
        ax.axvline(b + 0.5, color=C["olive"], lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fraction of the 3 models placing the pair together")
    fig.suptitle(wrap("Cross-model co-association (robustness summary, not external validation); strains sorted by consensus cluster", 70), fontsize=11)
    save(fig, "F10_coassociation_heatmap", "Co-association matrix from saved object", FEAT_5, "coassoc")


# ----------------------------------------------------------------------------
# F11 Soft-DTW gamma sensitivity from the saved sweep (both geometries)
# ----------------------------------------------------------------------------
def fig_gamma_sensitivity(B):
    style()
    sw = B["res"]["Soft-DTW"]["sweep_df"]
    P = B["res"]["Soft-DTW"]["params"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, constrained_layout=True)
    for ax, col, t in zip(axes, ("silhouette_dtw", "silhouette_euclid"), ("Silhouette on DTW distances", "Silhouette on flattened Euclidean")):
        for ki, k in enumerate(sorted(sw["k"].unique())):
            d = sw[sw["k"] == k].sort_values("gamma")
            col_, _ = ccolor(ki)
            ax.plot(d["gamma"], d[col], marker="o", ms=5, lw=1.8, color=col_, ls=["-", "--", "-.", ":"][(ki // 4) % 4], label=f"k = {k}")
        ax.axvline(P["gamma"], color=C["brown"], lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Soft-DTW $\\gamma$")
        ax.set_title(t, fontsize=10.5)
        ax.grid(True)
    axes[0].set_ylabel("Mean silhouette")
    axes[1].legend(fontsize=7.5, ncol=2, frameon=True)
    fig.suptitle(f"Soft-DTW $\\gamma$ sensitivity from the full candidate sweep (dotted line: selected $\\gamma$ = {P['gamma']:.3g}, k = {P['k']})", fontsize=11)
    save(fig, "F11_softdtw_gamma_sensitivity", "gamma x k sweep in both geometries", FEAT_5, "res['Soft-DTW'].sweep_df")


# ----------------------------------------------------------------------------
# F12 warping-window robustness
# ----------------------------------------------------------------------------
def fig_warping(B):
    style()
    sc = scaled_cluster(B)[0]
    D0 = B["shared_dtw_dist"]
    iu = np.triu_indices(sc.shape[0], 1)
    radii = [1, 2, 3, 5, 8]
    cor = [pearsonr(D0[iu], cdist_dtw(sc, global_constraint="sakoe_chiba", sakoe_chiba_radius=r)[iu])[0] for r in radii]
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    col, h = ccolor(0)
    ax.bar([str(r) for r in radii], cor, color=col, edgecolor=C["olive"], hatch=h, width=0.6)
    ax.set_xlabel("Sakoe-Chiba radius (days)")
    ax.set_ylabel("Pearson r vs unconstrained DTW distances")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")
    ax.set_title(wrap(f"Pairwise-distance robustness to the warping window (Hard-DTW fitted with radius {B['sakoe_chiba_radius']})", 60), fontsize=10.5)
    save(fig, "F12_warping_window_robustness", "Distance robustness vs Sakoe-Chiba radius", FEAT_5, "traj_cluster")


# ----------------------------------------------------------------------------
# F13 elastic-metric benchmark WITH Euclidean baseline (reviewer 9)
# ----------------------------------------------------------------------------
def _lcss(a, b, eps):
    n, m = len(a), len(b)
    L = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            L[i, j] = L[i - 1, j - 1] + 1 if abs(a[i - 1] - b[j - 1]) <= eps else max(L[i - 1, j], L[i, j - 1])
    return 1.0 - L[n, m] / min(n, m)


def _edr(a, b, eps):
    n, m = len(a), len(b)
    E = np.zeros((n + 1, m + 1))
    E[:, 0], E[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i, j] = min(E[i - 1, j - 1] + (0 if abs(a[i - 1] - b[j - 1]) <= eps else 1), E[i - 1, j] + 1, E[i, j - 1] + 1)
    return E[n, m] / max(n, m)


def _twed(a, b, nu=0.001, lam=1.0):
    n, m = len(a), len(b)
    pa, pb = np.r_[a[0], a[:-1]], np.r_[b[0], b[:-1]]
    T = np.full((n + 1, m + 1), np.inf)
    T[0, 0] = 0
    da, db = np.abs(a - pa) + nu + lam, np.abs(b - pb) + nu + lam
    for i in range(1, n + 1):
        T[i, 0] = T[i - 1, 0] + da[i - 1]
    for j in range(1, m + 1):
        T[0, j] = T[0, j - 1] + db[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            T[i, j] = min(T[i - 1, j - 1] + abs(a[i - 1] - b[j - 1]) + abs(pa[i - 1] - pb[j - 1]), T[i - 1, j] + da[i - 1], T[i, j - 1] + db[j - 1])
    return T[n, m]


def fig_elastic_benchmark(B, out_dir):
    style()
    sc, flat = scaled_cluster(B)
    n, nb = sc.shape[0], sc.shape[2]
    cons = B["partitions"]["Consensus"]
    kc = k_of(cons)
    mats = {"Euclidean (baseline)": squareform(pdist(flat)), "DTW": B["shared_dtw_dist"]}
    for nm, fn in (("LCSS", lambda a, b: _lcss(a, b, 0.5)), ("EDR", lambda a, b: _edr(a, b, 0.5)), ("TWED", _twed)):
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = np.mean([fn(sc[i, :, b], sc[j, :, b]) for b in range(nb)])  # mean over the 5 bands
        mats[nm] = M
    rows = []
    for nm, M in mats.items():
        Z = scipy_linkage(squareform(M, checks=False), method="average")
        rows.append(dict(metric=nm, ARI_vs_consensus=adjusted_rand_score(cons, fcluster(Z, t=kc, criterion="maxclust"))))
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "elastic_metric_benchmark.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    for i, r in df.iterrows():
        col, h = ccolor(i)
        ax.bar(r["metric"], r["ARI_vs_consensus"], color=col, edgecolor=C["olive"], hatch=h, width=0.6)
    ax.axhline(0, color=C["olive"], lw=0.8)
    ax.set_ylabel(f"ARI vs consensus partition (average linkage, k = {kc})")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([wrap(m, 12) for m in df["metric"]])
    ax.grid(True, axis="y")
    ax.set_title(wrap("Agreement of alternative distances with the consensus (elastic distances are means over the 5 clustering bands; Euclidean baseline included)", 70), fontsize=10)
    save(fig, "F13_elastic_metric_benchmark", "Euclidean + DTW/LCSS/EDR/TWED agreement with consensus", FEAT_5, "traj_cluster")


# ----------------------------------------------------------------------------
# F14-F16 descriptive spectral dynamics
# ----------------------------------------------------------------------------
def fig_phase_portraits(B):
    style()
    ci, ki = band_idx(B, "Chlorophyll_680"), band_idx(B, "Carotenoid_480")
    days, cons = np.asarray(B["day_order"], float), B["partitions"]["Consensus"]
    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    for k, c in enumerate(sorted(np.unique(cons))):
        col, _ = ccolor(k)
        for i in np.where(cons == c)[0]:
            chl, car = B["traj_full"][i, :, ci], B["traj_full"][i, :, ki]
            ax.plot(chl / np.maximum(car, 1e-6), np.gradient(chl, days), color=col, alpha=0.7, lw=1.4)
            ax.scatter((chl / np.maximum(car, 1e-6))[-1], np.gradient(chl, days)[-1], color=col, s=30, edgecolor=C["olive"], zorder=3)
        ax.plot([], [], color=col, lw=2.2, label=f"Cluster {c}")
    ax.set_xlabel("Band ratio Chlorophyll_680 / Carotenoid_480")
    ax.set_ylabel("Rate of change of Chlorophyll_680 band signal (per day)")
    ax.legend(frameon=True)
    ax.grid(True)
    fig.suptitle(wrap("Spectral phase portraits per consensus cluster (marker = final day); band signals, not pigment concentrations", 85), fontsize=11)
    save(fig, "F14_spectral_phase_portraits", "Phase portraits", FEAT_6, "traj_full + Consensus")


def fig_leadlag(B):
    style()
    mt, nd, bands = B["traj_full"].mean(0), len(B["day_order"]), B["band_names_all"]
    nb, ml = len(bands), min(4, len(B["day_order"]) - 2)
    M = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            s1, s2 = mt[:, i] - mt[:, i].mean(), mt[:, j] - mt[:, j].mean()
            best, bl = -2, 0
            for lag in range(-ml, ml + 1):
                a, b = (s1[lag:], s2[:nd - lag]) if lag >= 0 else (s1[:nd + lag], s2[-lag:])
                if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                cc = np.corrcoef(a, b)[0, 1]
                if cc > best:
                    best, bl = cc, lag
            M[i, j] = bl
    fig, ax = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("d", [C["ochre"], C["white"], C["olive"]])
    im = ax.imshow(M, cmap=cmap, vmin=-ml, vmax=ml)
    lbl = [pretty(b) + (" (held out)" if b == B["heldout_band"] else "") for b in bands]
    ax.set_xticks(range(nb)); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(nb)); ax.set_yticklabels(lbl, fontsize=8.5)
    for i in range(nb):
        for j in range(nb):
            ax.text(j, i, f"{int(M[i, j]):+d}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Best-correlation lag (days); row leads column when negative")
    fig.suptitle(wrap("Lead/lag structure between band signals (strain-mean trajectories)", 80), fontsize=11)
    save(fig, "F15_band_leadlag", "Cross-correlation lead/lag", FEAT_6, "traj_full")


def fig_spectral_timing(B):
    style()
    band = B["heldout_band"]
    b, days, sp, cons = band_idx(B, band), np.asarray(B["day_order"], float), B["species_order"], B["partitions"]["Consensus"]
    rows = []
    for i, s in enumerate(sp):
        y = B["traj_full"][i, :, b]
        above = np.where(y >= y.min() + 0.1 * (y.max() - y.min()))[0]
        rows.append(dict(Species=s, Cluster=int(cons[i]), T_onset=days[above[0]], T_max_rise=days[int(np.argmax(np.gradient(y, days)))], T_peak=days[int(np.argmax(y))]))
    df = pd.DataFrame(rows).sort_values(["Cluster", "Species"]).reset_index(drop=True)
    df.to_csv(FIG_DIR.parent / "spectral_timing_waterNIR.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(df) + 2.5), constrained_layout=True)
    y = np.arange(len(df))
    ucl = sorted(df["Cluster"].unique())
    colr = {c: ccolor(i)[0] for i, c in enumerate(ucl)}
    ax.barh(y, df["T_peak"] - df["T_onset"], left=df["T_onset"], color=[colr[c] for c in df["Cluster"]], alpha=0.35, height=0.6, edgecolor=C["olive"])
    ax.scatter(df["T_onset"], y, marker="|", s=120, color=C["olive"], label="onset (10% of range)")
    ax.scatter(df["T_max_rise"], y, marker="o", s=45, color=C["ochre"], edgecolor=C["olive"], label="steepest rise", zorder=3)
    ax.scatter(df["T_peak"], y, marker="s", s=45, color=C["brown"], edgecolor=C["olive"], label="peak signal", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} (C{c})" for s, c in zip(df["Species"], df["Cluster"])], fontsize=8)
    ax.set_xlabel(f"Cultivation day (spectral timing from held-out band {band})")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, axis="x")
    fig.suptitle(wrap("Spectral timing events of the Water/NIR band signal. The peak is a spectral endpoint, not a verified biomass or product optimum.", 90), fontsize=10.5)
    save(fig, "F16_spectral_timing_waterNIR", "Onset / steepest rise / peak of the Water/NIR band", "held-out band Water_NIR_970", "traj_full")


# ----------------------------------------------------------------------------
# F17-F18 stability proxies (Euclidean k-means on the 5-band clustering tensor)
# ----------------------------------------------------------------------------
def fig_montecarlo(B, n_rep=25):
    style()
    cons = B["partitions"]["Consensus"]
    kc = k_of(cons)
    flat = scaled_cluster(B)[1]
    rng = np.random.default_rng(SEED)
    sigmas = [0.05, 0.10, 0.20]
    data = [[adjusted_rand_score(cons, KMeans(kc, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(flat + rng.normal(0, s, flat.shape)))
             for _ in range(n_rep)] for s in sigmas]
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    for i, bx in enumerate(bp["boxes"]):
        col, h = ccolor(i)
        bx.set(facecolor=col, edgecolor=C["olive"], hatch=h, alpha=0.9)
    for m in bp["medians"]:
        m.set(color=C["white"], lw=1.8)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([f"$\\sigma$ = {s}" for s in sigmas])
    ax.set_ylabel("ARI vs consensus (Euclidean k-means proxy)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y")
    fig.suptitle(wrap("Noise-injection stability (proxy clustering on z-scored 5-band trajectories, not the DTW pipeline itself)", 80), fontsize=11)
    save(fig, "F17_montecarlo_stability_proxy", "Noise-injection stability proxy", FEAT_5, "traj_cluster")


def fig_leave_one_out(B):
    style()
    cons = B["partitions"]["Consensus"]
    kc, sp = k_of(cons), B["species_order"]
    flat = scaled_cluster(B)[1]
    n = len(sp)
    ar = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        ar.append(adjusted_rand_score(cons[m], KMeans(min(kc, m.sum()), n_init=5, random_state=SEED).fit_predict(flat[m])))
    order = np.argsort(ar)
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    col, h = ccolor(0)
    ax.bar(range(n), np.array(ar)[order], color=col, edgecolor=C["olive"], hatch=h, width=0.7)
    ax.axhline(np.mean(ar), color=C["brown"], ls="--", lw=1.6, label=f"mean = {np.mean(ar):.3f}")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"-{sp[i]}" for i in order], rotation=90, fontsize=8.5)
    ax.set_xlabel("Strain left out")
    ax.set_ylabel("ARI of remaining strains vs their consensus labels\n(Euclidean k-means proxy)")
    ax.legend(frameon=True)
    ax.grid(True, axis="y")
    fig.suptitle("Leave-one-strain-out reassignment stability (proxy clustering)", fontsize=11)
    save(fig, "F18_leave_one_out_proxy", "Leave-one-strain-out proxy", FEAT_5, "traj_cluster")


# ----------------------------------------------------------------------------
# F19 candidate harvest-window HYPOTHESES (spectral; not validated)
# ----------------------------------------------------------------------------
def fig_harvest_hypotheses(B):
    style()
    X = B["traj_full"]
    days, sp, cons = np.asarray(B["day_order"], float), B["species_order"], B["partitions"]["Consensus"]
    N = np.zeros_like(X)
    for b in range(X.shape[2]):
        rg = X[:, :, b].max() - X[:, :, b].min()
        N[:, :, b] = (X[:, :, b] - X[:, :, b].min()) / rg if rg > 1e-9 else 0.5
    S = N.mean(2)
    order = np.lexsort((np.array(sp), cons))
    M = S[order]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(days)), max(6, 0.34 * len(sp))), constrained_layout=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("h", [C["white"], C["cream"], C["ochre"], C["olive"]])
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for r in range(len(sp)):
        ax.scatter(np.where(M[r] >= 0.9 * M[r].max())[0], [r] * int((M[r] >= 0.9 * M[r].max()).sum()), s=10, color=C["white"], edgecolor=C["olive"], lw=0.4, zorder=3)
        ax.scatter(int(np.argmax(M[r])), r, marker="*", s=90, color=C["white"], edgecolor=C["olive"], lw=0.8, zorder=4)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"{d:g}" for d in days], fontsize=8)
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels([f"{sp[i]} (C{cons[i]})" for i in order], fontsize=8)
    ax.set_xlabel("Cultivation day")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Composite spectral signal (mean of 6 min-max-normalised bands)")
    fig.suptitle(wrap("Candidate harvest-window hypotheses from spectral signal (star = day of maximum composite signal; dots = within 90% of it). "
                      "Not validated against biomass, lipid, pigment or yield.", 100), fontsize=10.5)
    save(fig, "F19_candidate_harvest_window_hypotheses", "Spectral candidate windows (hypothesis only)", FEAT_6, "traj_full + Consensus")


# ----------------------------------------------------------------------------
# F20 band importance, F21 radar
# ----------------------------------------------------------------------------
def fig_band_importance(B):
    style()
    days, cons, bands = np.asarray(B["day_order"], float), B["partitions"]["Consensus"], B["band_names_all"]
    H = []
    for bi, b in enumerate(bands):
        auc = np.array([_trapz(B["traj_full"][i, :, bi], days) for i in range(len(cons))])
        g = [auc[cons == c] for c in np.unique(cons)]
        try:
            H.append(kruskal(*g)[0])
        except ValueError:
            H.append(0.0)
    df = pd.DataFrame(dict(band=bands, H=H)).sort_values("H", ascending=False).reset_index(drop=True)
    df["cum_pct"] = 100 * df["H"].cumsum() / max(df["H"].sum(), 1e-9)
    fig, ax1 = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    for i, r in df.iterrows():
        held = r["band"] == B["heldout_band"]
        ax1.bar(i, r["H"], color=C["ochre"] if held else C["olive"], edgecolor=C["olive"], hatch="//" if held else "", width=0.6)
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([pretty(b) + (" (held out)" if b == B["heldout_band"] else "") for b in df["band"]], rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Kruskal-Wallis $H$ (band AUC vs consensus cluster)")
    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df["cum_pct"], color=C["brown"], marker="o", lw=2)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Cumulative % of summed H")
    ax2.grid(False)
    fig.suptitle(wrap("Between-cluster separation per band. Clustering bands are not independent of the clusters they defined; the hatched held-out band is.", 85), fontsize=10.5)
    save(fig, "F20_band_importance", "Band separation Pareto (held-out flagged)", FEAT_6, "traj_full + Consensus")


def fig_radar(B):
    style()
    days, cons, bands = np.asarray(B["day_order"], float), B["partitions"]["Consensus"], B["band_names_all"]
    cl = sorted(np.unique(cons))
    raw = np.array([[np.mean([_trapz(B["traj_full"][i, :, b], days) for i in np.where(cons == c)[0]]) for b in range(len(bands))] for c in cl])
    rg = raw.max(0) - raw.min(0)
    sc = np.where(rg > 1e-9, (raw - raw.min(0)) / np.where(rg > 1e-9, rg, 1), 0.5)
    pd.DataFrame(raw, index=[f"Cluster {c}" for c in cl], columns=bands).to_csv(FIG_DIR.parent / "radar_data_raw_AUC.csv")
    ang = np.linspace(0, 2 * np.pi, len(bands), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), constrained_layout=True)
    for i, c in enumerate(cl):
        v = sc[i].tolist() + sc[i][:1].tolist()
        col, h = ccolor(i)
        ax.plot(ang, v, color=col, lw=2.2, label=f"Cluster {c}")
        ax.fill(ang, v, color=col, alpha=0.15, hatch=h)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([wrap(pretty(b) + (" (held out)" if b == B["heldout_band"] else ""), 14) for b in bands], fontsize=9)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.suptitle(wrap("Normalised band-AUC fingerprint per consensus cluster (0-1 scaled across clusters per band)", 70), fontsize=11)
    save(fig, "F21_radar_band_auc", "Radar of band AUC per cluster", FEAT_6, "traj_full + Consensus")


# ----------------------------------------------------------------------------
# F22 PCA of the clustering matrix; F23-F24 strain trajectories
# ----------------------------------------------------------------------------
def fig_pca(B, name):
    style()
    flat = scaled_cluster(B)[1]
    p = PCA(n_components=2, random_state=SEED)
    xy = p.fit_transform(flat)
    ve = p.explained_variance_ratio_ * 100
    lab, sp = B["partitions"][name], B["species_order"]
    fig, ax = plt.subplots(figsize=(10.5, 8.4), constrained_layout=True)
    for ci, c in enumerate(sorted(np.unique(lab))):
        idx = np.where(lab == c)[0]
        col, _ = ccolor(ci)
        ax.scatter(xy[idx, 0], xy[idx, 1], s=170, color=col, edgecolor=C["olive"], lw=1.1, label=f"Cluster {c}", zorder=3,
                   marker=["o", "s", "^", "D", "v", "P", "X", "*"][(ci // len(CYCLE)) % 8])
        for i in idx:
            ax.annotate(sp[i], xy[i], fontsize=11, xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ve[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({ve[1]:.1f}% variance)", fontsize=12)
    ax.legend(frameon=True, fontsize=11, loc="best")
    ax.grid(True, alpha=0.5)
    ax.set_title(wrap(f"{name}: PCA of the z-scored 5-band clustering trajectories (20 strains x {flat.shape[1]} features = 31 days x 5 bands, fitted on all strains)", 75), fontsize=11)
    save(fig, f"F22_pca_{name.replace('-', '_')}", f"PCA projection coloured by {name}", FEAT_5, "traj_cluster (z-scored, flattened)")


def fig_species_trajectories(B, mode):
    style()
    sp, days, cons = B["species_order"], B["day_order"], B["partitions"]["Consensus"]
    fig, axes = plt.subplots(4, 5, figsize=(17, 11.5), sharex=True, constrained_layout=True)
    wb = band_idx(B, B["heldout_band"])
    for i, (ax, s) in enumerate(zip(axes.ravel(), sp)):
        col, _ = ccolor(sorted(np.unique(cons)).index(cons[i]))
        if mode == "water_raw":
            y, sd = B["traj_full"][i, :, wb], np.nan_to_num(B["traj_sd_full"][i, :, wb])
            ax.plot(days, y, color=col, lw=2, marker="o", ms=3)
            ax.fill_between(days, y - sd, y + sd, color=col, alpha=0.2)
        else:
            for bi, b in enumerate(B["band_names_all"]):
                y = B["traj_full"][i, :, bi]
                y = (y - y.min()) / (y.max() - y.min() + 1e-12)
                c2, _ = ccolor(bi)
                ax.plot(days, y, color=c2, lw=1.4, ls=["-", "--", "-.", ":"][(bi // 4) % 4], label=pretty(b) if i == 0 else None)
        ax.set_title(f"{s} (C{cons[i]})", fontsize=10)
        ax.grid(True)
    for ax in axes[-1]:
        ax.set_xlabel("Cultivation day")
    if mode == "allbands_norm":
        axes[0, 0].legend(fontsize=6.5, frameon=True)
    t = ("Raw Water/NIR (held-out band) spectral trajectory per strain, mean $\\pm$ SD across images"
         if mode == "water_raw" else "All six band trajectories per strain, each min-max normalised within strain")
    fig.suptitle(wrap(t + ". Spectral signals, not biomass or growth curves.", 110), fontsize=12)
    save(fig, f"F23_strain_trajectories_{mode}", t, FEAT_6, "traj_full, traj_sd_full")


# ----------------------------------------------------------------------------
def write_checksums(out_dir):
    out_dir = Path(out_dir)
    L = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out_dir).as_posix()}"
         for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != "checksums.sha256"]
    (out_dir / "checksums.sha256").write_text("\n".join(L) + "\n")


def generate_all(out_dir="outputs", dpi=600):
    global DPI, FIG_DIR
    DPI, FIG_DIR = dpi, Path(out_dir) / "figures"
    B = load_bundle(out_dir)
    print(f"Loaded bundle: consensus k = {B['consensus_k']}, clustering bands = {B['clustering_bands']}, held-out = {B['heldout_band']}")
    jobs = [("algorithm", fig_algorithm_mechanics, ()), ("dtw_vs_euclid", fig_dtw_vs_euclidean, ()), ("sankey", fig_cluster_flow, ())]
    jobs += [(f"bands_{n}", fig_all_bands, (n,)) for n in B["partitions"]]
    jobs += [("heldout", fig_heldout_summaries, ()), ("channel", fig_channel_audit, ()), ("taxonomy", fig_taxonomy, ()),
             ("silhouette", fig_silhouette_profiles, ()), ("alignment", fig_dtw_alignment, ()), ("coassoc", fig_coassociation, ()),
             ("gamma", fig_gamma_sensitivity, ()), ("warping", fig_warping, ()), ("elastic", fig_elastic_benchmark, (out_dir,)),
             ("phase", fig_phase_portraits, ()), ("leadlag", fig_leadlag, ()), ("timing", fig_spectral_timing, ()),
             ("mc", fig_montecarlo, ()), ("loo", fig_leave_one_out, ()), ("harvest", fig_harvest_hypotheses, ()),
             ("bandimp", fig_band_importance, ()), ("radar", fig_radar, ())]
    jobs += [(f"pca_{n}", fig_pca, (n,)) for n in B["partitions"]]
    jobs += [("traj_water", fig_species_trajectories, ("water_raw",)), ("traj_all", fig_species_trajectories, ("allbands_norm",))]
    failed = []
    for name, fn, extra in jobs:
        try:
            fn(B, *extra)
        except Exception:
            failed.append(name)
            print(f"  !! {name} FAILED:\n{traceback.format_exc()}")
    pd.DataFrame(REGISTRY).to_csv(Path(out_dir) / "audit_table_figures.csv", index=False)
    write_checksums(out_dir)
    print(f"Done: {len(REGISTRY)} figure files registered, {len(failed)} failed {failed if failed else ''}")
    return failed



def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate all manuscript and supplementary figures from outputs/model_results.pkl")
    ap.add_argument("--out_dir", default="outputs", help="folder containing model_results.pkl (written by 01_training_and_validation.py)")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args(argv)
    failed = generate_all(args.out_dir, dpi=args.dpi)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
