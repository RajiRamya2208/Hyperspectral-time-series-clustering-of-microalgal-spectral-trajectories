#!/usr/bin/env python3
"""
01_training_and_validation.py   (single analysis entry point, v2.0.0)
======================================================================
Strain-aware spectral-trajectory clustering of 20 microalgal / cyanobacterial
strains (hyperspectral ROI spectra, 31-day cultivation).

Run:
    python 01_training_and_validation.py --data_dir path/to/hyperspectral_xlsx
    python 01_training_and_validation.py --selection_geometry euclidean   # reproduces the
                                                                        # original (pre-review) selection rule
Outputs (all in --out_dir, default ./outputs):
    model_results.pkl            everything 02_figures_corrected.py needs (plain dicts/arrays)
    *.csv                        every table reported in the manuscript (see audit_table_results.csv)
    run_manifest.json            arguments, seed, package versions, bands, channel, data source
    checksums.sha256             SHA-256 of every output file

---------------------------------------------------------------------------
WHAT IS (AND IS NOT) IN THE ANALYSIS  -- reviewer comments 2, 3
---------------------------------------------------------------------------
Feature sets
  * CLUSTERING tensor : 5 bands (Chlorophyll_680, Carotenoid_480, Phycobiliprotein_620,
                        Lipid_CH_930, Protein_Amide_910)  -> shape (20 species, 31 days, 5)
  * HELD-OUT band     : Water_NIR_970. It is never given to any clustering model. It is used only
                        to derive held-out spectral summaries (AUC, max, day of max, slopes).
  * NDVI / GNDVI / Simple_Ratio come from the ROI_Metadata sheet of the same acquisitions. They are
    held-out *spectral summaries*, NOT independent biological assays.
  * Taxonomy is genuinely external metadata.

Ordered preprocessing flow actually executed in THIS script (nothing else is applied):
  Upstream (acquisition software / export; cannot be verified from this code and must be
  described in the manuscript exactly as it was done): white/dark-reference calibration,
  Savitzky-Golay smoothing, resampling to 1 nm, global 0-1 normalisation  -> the spreadsheet column
  family "<ROI>_<channel>" (channel = Normalized by default; see --use_channel).
  In Python:
   1. read column family "<ROI>_<channel>" from sheet Spectral_Data (601 wavelengths)
   2. mean over each band's wavelength window (DTW_BAND_INDICES)         -> ROI x band
   3. mean of ROI values within each image (one .xlsx file = one image)   -> image x band
   4. mean of image values within each species-day                        -> species x day x band
      (also SD across images and image counts; saved to trajectory_tensor_long.csv)
   5. linear interpolation of any missing species-day cell (count logged)
   6. per-series z-scoring over time (tslearn TimeSeriesScalerMeanVariance) -> clustering therefore
      compares trajectory SHAPE, not absolute level
   7. DTW distances / DTW k-means / Soft-DTW k-means / hierarchical clustering
  SNV, 5-point moving average, first derivative and PCA(30) that older versions computed on
  all_features were NEVER used for clustering; they have been removed. PCA is used only for the
  2-D visualisation in 02_figures_corrected.py (on the z-scored 5-band clustering trajectories).
  There is no synthetic-data fallback in this version; data_source is always "real".

Geometry of every tuning / reporting step  -- reviewer comment 4
  * DTW distance matrix used for scoring: unconstrained DTW on the z-scored 5-band tensor
    (computed once, shared by all models).
  * Hard-DTW k-means *fitting* uses a Sakoe-Chiba band (--sakoe_chiba_radius, default 3 days).
    An unconstrained refit is run as a sensitivity check (ARI reported).
  * Model selection (k / gamma / linkage) maximises silhouette in the geometry given by
    --selection_geometry (default "dtw" = silhouette on the precomputed DTW matrix).
    Both silhouettes are stored for every candidate (sweep_*.csv) and geometry_sensitivity.csv
    shows whether the other geometry would have chosen a different partition.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
import re
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

SCRIPT_VERSION = "2.0.0"
_trapz = getattr(np, "trapezoid", None) or np.trapz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DTW_Pipeline")

# ----------------------------------------------------------------------------
# Paths: relative to this file, overridable by CLI or environment variable
# ----------------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # running inside Jupyter / IPython: no __file__
    BASE_DIR = Path.cwd()
DEFAULT_DATA_DIR = str(BASE_DIR / "data" / "hyperspectral_xlsx")
DEFAULT_OUT_DIR = os.environ.get("MICROALGAE_OUTPUT_DIR", str(BASE_DIR / "outputs"))

SEED = 42
MAX_DOMINANT_FRAC = 0.60

# ----------------------------------------------------------------------------
# Dataset constants
# ----------------------------------------------------------------------------
FILENAME_PATTERN = re.compile(r"Day[_-]?(\d+)_([A-Za-z]+)_(\d+)_(\d+)", re.IGNORECASE)

CANONICAL_SPECIES = ["ANBA", "APHA", "CHLO", "CLAM", "COCO", "COEL", "DESM", "DUNA", "GLEO", "MESH",
                     "MONO", "NANO", "NOST", "PAVL", "PICH", "PLAT", "SCNE", "SPIR", "TEDE", "TETR"]

SPECIES_FULL_NAMES = {
    "ANBA": "Anabaena sp.", "APHA": "Aphanocapsa montana", "CHLO": "Chlorella vulgaris",
    "CLAM": "Chlamydomonas sp.", "COCO": "Chroococcus minor", "COEL": "Coelastrella sp.",
    "DESM": "Desmodesmus sp.", "DUNA": "Dunaliella sp.", "GLEO": "Gloeocapsa sp.",
    "MESH": "Messastrum sp.", "MONO": "Monoraphidium sp.", "NANO": "Nannochloropsis oculata",
    "NOST": "Nostoc sp.", "PAVL": "Pavlova lutheri", "PICH": "Picochlorum sp.",
    "PLAT": "Platymonas sp.", "SCNE": "Scenedesmus sp.", "SPIR": "Spirulina sp.",
    "TEDE": "Tetradesmus sp.", "TETR": "Tetraselmis chuii",
}

# Coarse taxonomy: external metadata, used ONLY post hoc.
SPECIES_TAXONOMY = {
    "ANBA": "Cyanobacteria", "APHA": "Cyanobacteria", "CHLO": "Chlorophyta", "CLAM": "Chlorophyta",
    "COCO": "Cyanobacteria", "COEL": "Chlorophyta", "DESM": "Chlorophyta", "DUNA": "Chlorophyta",
    "GLEO": "Cyanobacteria", "MESH": "Chlorophyta", "MONO": "Chlorophyta",
    "NANO": "Ochrophyta (Eustigmatophyceae)", "NOST": "Cyanobacteria", "PAVL": "Haptophyta",
    "PICH": "Chlorophyta", "PLAT": "Chlorophyta", "SCNE": "Chlorophyta", "SPIR": "Cyanobacteria",
    "TEDE": "Chlorophyta", "TETR": "Chlorophyta",
}

DTW_BAND_INDICES = {
    "Chlorophyll_680": (670, 690),
    "Carotenoid_480": (470, 500),
    "Phycobiliprotein_620": (610, 640),
    "Lipid_CH_930": (910, 950),
    "Protein_Amide_910": (890, 920),
    "Water_NIR_970": (960, 980),
}
HELD_OUT_BAND = "Water_NIR_970"
CLUSTERING_BANDS = [b for b in DTW_BAND_INDICES if b != HELD_OUT_BAND]

SPECTRAL_INDEX_FEATURES = ["NDVI", "GNDVI", "Simple_Ratio"]
WATER_FEATURES = ["Water_AUC", "Water_Max", "Water_Day_of_Max", "Water_Early_Slope", "Water_Late_Slope"]
HELDOUT_FEATURES = SPECTRAL_INDEX_FEATURES + WATER_FEATURES


# ============================================================================
# 1. Species-code canonicalisation (unchanged behaviour)
# ============================================================================
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def canonicalize_species(raw, parent=None, source_file=None, max_edit=1):
    raw_u = raw.upper().strip()
    if raw_u in CANONICAL_SPECIES:
        return raw_u, False, False
    if parent:
        p = parent.upper().strip()
        if p in CANONICAL_SPECIES and _levenshtein(raw_u, p) <= 2:
            logger.info("Filename typo %r -> %r via parent folder (%s)", raw, p, source_file or "")
            return p, True, False
    cands = [c for c in CANONICAL_SPECIES if _levenshtein(raw_u, c) <= max_edit]
    if len(cands) == 1:
        logger.info("Species-name correction %r -> %r (%s)", raw, cands[0], source_file or "")
        return cands[0], True, False
    return raw_u, False, True


# ============================================================================
# 2. Channel audit + data loading (real data only)
# ============================================================================
def discover_files(data_dir):
    files = []
    for root, _, names in os.walk(data_dir):
        for fn in names:
            if fn.lower().endswith(".xlsx") and not fn.startswith("~$"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def channel_band_coverage_report(files, band_defs=DTW_BAND_INDICES, sample_size=30):
    """Per channel x band: fraction of sampled files with non-constant, non-zero signal.
    Each sampled workbook is parsed once."""
    rng = np.random.RandomState(SEED)
    sample = list(files) if len(files) <= sample_size else \
        [files[i] for i in sorted(rng.choice(len(files), sample_size, replace=False))]
    counts = {}
    for path in sample:
        try:
            sdf = pd.read_excel(path, sheet_name="Spectral_Data")
        except Exception:
            continue
        if "Wavelength_nm" not in sdf.columns:
            continue
        wl = sdf["Wavelength_nm"].values.astype(float)
        channels = sorted({c.rsplit("_", 1)[-1] for c in sdf.columns if c != "Wavelength_nm" and "_" in c})
        for ch in channels:
            cols = [c for c in sdf.columns if c.endswith(f"_{ch}")]
            X = sdf[cols].values.astype(float).T
            for band, (lo, hi) in band_defs.items():
                m = (wl >= lo) & (wl <= hi)
                if not m.any():
                    continue
                v = X[:, m]
                rec = counts.setdefault((ch, band), [0, 0])
                rec[1] += 1
                if np.isfinite(v).any() and np.nanstd(v) > 1e-9 and not np.allclose(np.nan_to_num(v), 0.0):
                    rec[0] += 1
    rows = [{"channel": ch, "band": b, "fraction_files_with_real_signal": round(r / n, 3) if n else 0.0,
             "n_files_checked": n} for (ch, b), (r, n) in sorted(counts.items())]
    report = pd.DataFrame(rows)
    if report.empty:
        raise ValueError("Channel audit found no '<ROI>_<channel>' columns in 'Spectral_Data'.")
    per_channel = report.groupby("channel")["fraction_files_with_real_signal"].mean().sort_values(ascending=False)
    return report, per_channel


class Dataset:
    """Plain container: ROI-level spectra + metadata. No preprocessing beyond the interpolation of
    a file's wavelength axis onto the first file's axis (only if axes differ)."""


def load_dataset(files, channel):
    ds = Dataset()
    spectra, species, days, image_id, pig = [], [], [], [], []
    wl_ref, n_ok, n_skip, n_noday = None, 0, 0, 0
    corrections, unresolved = {}, set()
    logger.info("Reading %d workbooks (channel=%s)...", len(files), channel)
    for f_idx, path in enumerate(files):
        parent, fn = os.path.basename(os.path.dirname(path)), os.path.basename(path)
        m = FILENAME_PATTERN.search(fn)
        if not m:
            n_noday += 1
            logger.warning("Filename %s does not match Day_<d>_<SPECIES>_<i>_<j>; file excluded.", fn)
            continue
        day, sp_raw = int(m.group(1)), m.group(2).upper()
        sp, corrected, unres = canonicalize_species(sp_raw, parent, fn)
        if corrected:
            corrections[(sp_raw, sp)] = corrections.get((sp_raw, sp), 0) + 1
        if unres:
            unresolved.add(sp)
        try:
            xl = pd.ExcelFile(path)
            sdf = xl.parse("Spectral_Data")
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)
            n_skip += 1
            continue
        if "Wavelength_nm" not in sdf.columns:
            n_skip += 1
            continue
        wl = sdf["Wavelength_nm"].values.astype(np.float32)
        wl_ref = wl if wl_ref is None else wl_ref
        cols = [c for c in sdf.columns if c.endswith(f"_{channel}")]
        if not cols:
            logger.warning("Skipping %s: no '_%s' columns", fn, channel)
            n_skip += 1
            continue
        meta = {}
        try:
            mdf = xl.parse("ROI_Metadata")
            if "ROI_Name" in mdf.columns:
                meta = {r["ROI_Name"]: r for _, r in mdf.iterrows()}
        except Exception:
            pass
        for col in cols:
            roi = col[: -(len(channel) + 1)]
            s = sdf[col].values.astype(np.float32)
            if len(wl) != len(wl_ref) or not np.allclose(wl, wl_ref):
                s = np.interp(wl_ref, wl, s)
            spectra.append(s)
            species.append(sp)
            days.append(day)
            image_id.append(f_idx)
            r = meta.get(roi)
            pig.append((float(r.get("NDVI", np.nan)), float(r.get("GNDVI", np.nan)),
                        float(r.get("Simple_Ratio", np.nan))) if r is not None else (np.nan,) * 3)
        n_ok += 1
    if not spectra:
        raise ValueError("No usable spectra were read.")
    if corrections:
        logger.info("Species-name corrections: %s", corrections)
    if unresolved:
        logger.warning("Unresolved species labels kept as-is: %s", sorted(unresolved))
    ds.wavelengths, ds.spectra = wl_ref, np.stack(spectra)
    ds.species, ds.days = np.array(species), np.array(days)
    ds.image_id, ds.pigment = np.array(image_id), np.array(pig, dtype=np.float32)
    ds.channel, ds.data_source = channel, "real"
    ds.n_files_ok, ds.n_files_skipped, ds.n_files_excluded_no_day = n_ok, n_skip, n_noday
    ds.unresolved_species = sorted(unresolved)
    logger.info("Parsed %d workbooks (%d skipped, %d excluded) -> %d ROI spectra x %d wavelengths",
                n_ok, n_skip, n_noday, *ds.spectra.shape)
    return ds


# ============================================================================
# 3. Trajectories (ROI -> image -> species-day)
# ============================================================================
def band_average(X, wl, band_defs):
    out = {}
    for name, (lo, hi) in band_defs.items():
        m = (wl >= lo) & (wl <= hi)
        if not m.any():
            raise ValueError(f"Band {name} ({lo}-{hi} nm) not covered by the wavelength axis.")
        out[name] = X[:, m].mean(axis=1)
    return pd.DataFrame(out)


def build_trajectories(ds, band_defs=DTW_BAND_INDICES):
    bands = list(band_defs)
    bdf = band_average(ds.spectra, ds.wavelengths, band_defs)
    bdf["Species"], bdf["Day"], bdf["Image"] = ds.species, ds.days, ds.image_id
    n_roi = bdf.groupby(["Species", "Day"]).size()
    img = bdf.groupby(["Species", "Day", "Image"], as_index=False)[bands].mean()  # ROI -> image
    g = img.groupby(["Species", "Day"])
    mean_df, sd_df, n_img = g[bands].mean(), g[bands].std(ddof=1), g.size()
    species_order, day_order = sorted(img["Species"].unique()), sorted(img["Day"].unique())
    idx = pd.MultiIndex.from_product([species_order, day_order], names=["Species", "Day"])
    S, T, B = len(species_order), len(day_order), len(bands)
    traj = mean_df.reindex(idx).values.reshape(S, T, B).astype(float)
    sd = sd_df.reindex(idx).values.reshape(S, T, B).astype(float)
    n_images = n_img.reindex(idx).fillna(0).values.reshape(S, T).astype(int)
    n_rois = n_roi.reindex(idx).fillna(0).values.reshape(S, T).astype(int)
    n_missing = int(np.isnan(traj[:, :, 0]).sum())
    if n_missing:
        logger.warning("%d species-day cell(s) missing -> linear interpolation along time.", n_missing)
        for i in range(S):
            for k in range(B):
                traj[i, :, k] = pd.Series(traj[i, :, k]).interpolate(limit_direction="both").values
    design = pd.DataFrame([{"Species": s, "Day": d, "n_images": n_images[i, j], "n_roi_spectra": n_rois[i, j]}
                           for i, s in enumerate(species_order) for j, d in enumerate(day_order)])
    long_rows = []
    for i, s in enumerate(species_order):
        for j, d in enumerate(day_order):
            for k, b in enumerate(bands):
                long_rows.append({"Species": s, "Day": d, "Band": b, "mean": traj[i, j, k], "sd_across_images": sd[i, j, k],
                                  "n_images": n_images[i, j], "n_roi_spectra": n_rois[i, j],
                                  "used_in_clustering": b != HELD_OUT_BAND})
    return dict(traj=traj, sd=sd, species_order=species_order, day_order=day_order, band_names=bands,
                design=design, long=pd.DataFrame(long_rows), n_missing_cells=n_missing)


def water_descriptors(traj_full, species_order, day_order, band_names, band=HELD_OUT_BAND):
    b = band_names.index(band)
    days, mid = np.asarray(day_order, float), len(day_order) // 2
    rows = []
    for i, sp in enumerate(species_order):
        y = traj_full[i, :, b]
        rows.append({"Species": sp, "Water_AUC": float(_trapz(y, days)), "Water_Max": float(y.max()),
                     "Water_Day_of_Max": float(days[int(np.argmax(y))]),
                     "Water_Early_Slope": float((y[mid] - y[0]) / max(days[mid] - days[0], 1e-6)),
                     "Water_Late_Slope": float((y[-1] - y[mid]) / max(days[-1] - days[mid], 1e-6))})
    return pd.DataFrame(rows)


# ============================================================================
# 4. Geometry-explicit metrics
# ============================================================================
def dtw_matrix(scaled, radius=None):
    kw = dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=radius) if radius else {}
    D = cdist_dtw(scaled, **kw)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def partition_metrics(flat, D, labels):
    """Both geometries, always: silhouette on precomputed DTW matrix and on flattened Euclidean."""
    labels = np.asarray(labels)
    u = len(np.unique(labels))
    out = dict(silhouette_dtw=np.nan, silhouette_euclid=np.nan, davies_bouldin_euclid=np.nan)
    if 1 < u < len(labels):
        out["silhouette_dtw"] = float(silhouette_score(D, labels, metric="precomputed"))
        out["silhouette_euclid"] = float(silhouette_score(flat, labels))
        out["davies_bouldin_euclid"] = float(davies_bouldin_score(flat, labels))
    counts = pd.Series(labels).value_counts()
    out.update(dominant_fraction=float(counts.max() / len(labels)), active_clusters=int(len(counts)),
               min_cluster_size=int(counts.min()))
    return out


def balance_pool(sweep):
    ok = sweep[sweep["dominant_fraction"] <= MAX_DOMINANT_FRAC]
    if ok.empty:
        logger.warning("No candidate satisfies dominant fraction <= %.2f; using all.", MAX_DOMINANT_FRAC)
        return sweep
    return ok


def best_row(sweep, col):
    pool = balance_pool(sweep).reset_index(drop=True)
    return pool.loc[pool[col].idxmax()]


def stability_ari(fit_fn, k, n_seeds):
    labs = [fit_fn(k, s) for s in range(SEED, SEED + n_seeds)]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris)) if aris else 1.0


def _k_range(args, n):
    return range(args.k_min, min(args.k_max, n - 1) + 1)


# ============================================================================
# 5. Models
# ============================================================================
def fit_hard(scaled, k, seed, radius, n_init=5, max_iter=50):
    mp = dict(global_constraint="sakoe_chiba", sakoe_chiba_radius=radius) if radius else None
    return TimeSeriesKMeans(n_clusters=k, metric="dtw", metric_params=mp, max_iter=max_iter,
                            n_init=n_init, random_state=seed).fit_predict(scaled)


def fit_soft(scaled, k, seed, gamma, n_init=5, max_iter=100):
    return TimeSeriesKMeans(n_clusters=k, metric="softdtw", metric_params={"gamma": gamma},
                            n_init=n_init, max_iter=max_iter, random_state=seed).fit_predict(scaled)


def fit_hier(D, k, linkage):
    return AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=linkage).fit_predict(D)


def _result(name, col, species, labels, flat, D, params, stab, sweep, extra=None):
    rep = dict(method=name, **params, **partition_metrics(flat, D, labels), stability_mean_pairwise_ARI=stab)
    rep.update(extra or {})
    return dict(name=name, cluster_col=col, species_order=list(species), labels=np.asarray(labels),
                result_df=pd.DataFrame({"Species": species, col: labels}), report=rep, sweep_df=sweep, params=params)


def run_hard_dtw(scaled, flat, D, species, args):
    radius = args.sakoe_chiba_radius or None
    rows, cache = [], {}
    for k in _k_range(args, len(species)):
        lab = fit_hard(scaled, k, SEED, radius)
        if len(np.unique(lab)) < 2:
            continue
        rows.append(dict(k=k, **partition_metrics(flat, D, lab)))
        cache[k] = lab
    sweep = pd.DataFrame(rows)
    sel, alt = best_row(sweep, args.sel_col), best_row(sweep, args.alt_col)
    k = int(sel["k"])
    labels = cache[k]
    stab = stability_ari(lambda kk, s: fit_hard(scaled, kk, s, radius), k, args.n_stability_seeds)
    unc = fit_hard(scaled, k, SEED, None)  # unconstrained refit (reviewer 2, major 5)
    extra = dict(ARI_vs_unconstrained_refit=float(adjusted_rand_score(labels, unc)),
                 sakoe_chiba_radius=radius if radius else "unconstrained")
    res = _result("Hard-DTW", "DTW_Cluster", species, labels, flat, D, dict(k=k), stab, sweep, extra)
    sens = dict(model="Hard-DTW", selected_by=args.geometry, selected_params=f"k={k}",
                alternative_geometry=args.alt_geometry, alternative_params=f"k={int(alt['k'])}",
                ARI_selected_vs_alternative=float(adjusted_rand_score(labels, cache[int(alt['k'])])))
    return res, sens


def default_gamma_grid(flat):
    n = flat.shape[0]
    d2 = [np.sum((flat[i] - flat[j]) ** 2) for i in range(n) for j in range(i + 1, n)]
    med = max(float(np.median(d2)) if d2 else 1.0, 1e-3)
    return sorted({round(med * f, 5) for f in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)})


def run_soft_dtw(scaled, flat, D, species, args, top_n=4):
    grid = default_gamma_grid(flat)
    logger.info("Soft-DTW gamma grid: %s", grid)
    rows, cache = [], {}
    for g in grid:
        for k in _k_range(args, len(species)):
            lab = fit_soft(scaled, k, SEED, g)
            if len(np.unique(lab)) < 2:
                continue
            rows.append(dict(gamma=g, k=k, **partition_metrics(flat, D, lab)))
            cache[(g, k)] = lab
    sweep = pd.DataFrame(rows)
    cand = balance_pool(sweep).sort_values(args.sel_col, ascending=False).head(top_n).reset_index(drop=True)
    stabs = []
    for _, r in cand.iterrows():
        g = float(r["gamma"])
        stabs.append(stability_ari(lambda kk, s, _g=g: fit_soft(scaled, kk, s, _g), int(r["k"]), args.n_stability_seeds))
    cand["stability_mean_pairwise_ARI"] = stabs
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    cand["combined_score"] = 0.5 * mm(cand[args.sel_col]) + 0.5 * mm(cand["stability_mean_pairwise_ARI"])
    top = cand.sort_values("combined_score", ascending=False).iloc[0]
    g, k = float(top["gamma"]), int(top["k"])
    labels = cache[(g, k)]
    alt = best_row(sweep, args.alt_col)
    alt_lab = cache[(float(alt["gamma"]), int(alt["k"]))]
    res = _result("Soft-DTW", "SoftDTW_Cluster", species, labels, flat, D, dict(k=k, gamma=g),
                  float(top["stability_mean_pairwise_ARI"]), sweep, dict(candidate_ranking=cand))
    sens = dict(model="Soft-DTW", selected_by=args.geometry, selected_params=f"gamma={g:.5g}, k={k}",
                alternative_geometry=args.alt_geometry,
                alternative_params=f"gamma={float(alt['gamma']):.5g}, k={int(alt['k'])} (silhouette only, no stability step)",
                ARI_selected_vs_alternative=float(adjusted_rand_score(labels, alt_lab)))
    return res, sens


def run_hierarchical(scaled, flat, D, species, args, linkages=("average", "complete", "single")):
    rows, cache = [], {}
    for lk in linkages:
        for k in _k_range(args, len(species)):
            lab = fit_hier(D, k, lk)
            if len(np.unique(lab)) < 2:
                continue
            rows.append(dict(linkage=lk, k=k, **partition_metrics(flat, D, lab)))
            cache[(lk, k)] = lab
    sweep = pd.DataFrame(rows)
    sel, alt = best_row(sweep, args.sel_col), best_row(sweep, args.alt_col)
    lk, k = str(sel["linkage"]), int(sel["k"])
    labels = cache[(lk, k)]

    def fit_fn(kk, s):
        noise = np.random.RandomState(s).normal(0, 0.02, scaled.shape)
        return fit_hier(dtw_matrix(scaled + noise), kk, lk)

    stab = stability_ari(fit_fn, k, args.n_stability_seeds)
    res = _result("Hierarchical-DTW", "Hierarchical_DTW_Cluster", species, labels, flat, D,
                  dict(k=k, linkage=lk), stab, sweep,
                  dict(stability_note="deterministic algorithm; stability = input-noise perturbation (sd 0.02)"))
    sens = dict(model="Hierarchical-DTW", selected_by=args.geometry, selected_params=f"linkage={lk}, k={k}",
                alternative_geometry=args.alt_geometry,
                alternative_params=f"linkage={alt['linkage']}, k={int(alt['k'])}",
                ARI_selected_vs_alternative=float(adjusted_rand_score(labels, cache[(str(alt['linkage']), int(alt['k']))])))
    return res, sens


# ============================================================================
# 6. Consensus (co-association) -- a robustness summary, not external validation
# ============================================================================
def build_coassociation(label_arrays):
    co = np.mean([(l[:, None] == l[None, :]).astype(float) for l in label_arrays], axis=0)
    np.fill_diagonal(co, 1.0)
    return co


def select_consensus_k(co, D, args, linkage="average"):
    n = co.shape[0]
    dist = 1.0 - co
    np.fill_diagonal(dist, 0.0)
    Z = scipy_linkage(squareform(dist, checks=False), method=linkage)
    rows = []
    for k in _k_range(args, n):
        lab = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        ag, singles = [], 0
        for i in range(n):
            mates = np.where(lab == lab[i])[0]
            mates = mates[mates != i]
            if len(mates):
                ag.append(float(co[i, mates].mean()))
            else:
                ag.append(0.0)
                singles += 1
        bal = pd.Series(lab).value_counts()
        rows.append(dict(k=k, silhouette_dtw=float(silhouette_score(D, lab, metric="precomputed")),
                         mean_species_agreement=float(np.mean(ag)), n_singletons=singles,
                         dominant_fraction=float(bal.max() / n), active_clusters=int(len(bal))))
    sweep = pd.DataFrame(rows)
    pool = balance_pool(sweep).copy()
    mm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    pool["combined_score"] = 0.5 * mm(pool["silhouette_dtw"]) + 0.5 * mm(pool["mean_species_agreement"])
    sweep = sweep.merge(pool[["k", "combined_score"]], on="k", how="left")
    return int(pool.loc[pool["combined_score"].idxmax(), "k"]), sweep, Z


def agreement_outputs(species, label_map, cons_labels, co):
    names = list(label_map) + ["Consensus"]
    arrs = list(label_map.values()) + [cons_labels]
    ari = pd.DataFrame([[adjusted_rand_score(a, b) for b in arrs] for a in arrs], index=names, columns=names)
    n, tol = len(species), 1e-9
    pairs, per_sp = [], []
    for i in range(n):
        for j in range(i + 1, n):
            v = co[i, j]
            cat = "unanimous_together" if v > 1 - tol else "unanimous_apart" if v < tol else "contested"
            pairs.append(dict(Species_A=species[i], Species_B=species[j], coassociation=v, category=cat,
                              same_consensus_cluster=bool(cons_labels[i] == cons_labels[j])))
    for i, s in enumerate(species):
        mates = np.where(cons_labels == cons_labels[i])[0]
        mates = mates[mates != i]
        contested = (co[i] > tol) & (co[i] < 1 - tol)
        per_sp.append(dict(Species=s, Consensus_Cluster=int(cons_labels[i]),
                           mean_coassoc_with_consensus_mates=float(co[i, mates].mean()) if len(mates) else np.nan,
                           n_contested_pairs=int(contested.sum()), disputed=bool(contested.any()),
                           **{f"{m}_cluster": int(l[i]) for m, l in label_map.items()}))
    return ari, pd.DataFrame(pairs), pd.DataFrame(per_sp)


# ============================================================================
# 7. Held-out spectral summaries + taxonomy (reviewer comment 6)
# ============================================================================
def benjamini_hochberg(p):
    p = np.asarray([np.nan if v is None else v for v in p], float)
    q = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    if not len(valid):
        return q
    order = valid[np.argsort(p[valid])]
    raw = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    q[order] = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0, 1)
    return q


def permutation_kw_p(values, labels, n_perm, rng):
    """Species-level permutation p for the Kruskal-Wallis H (cluster sizes preserved)."""
    ok = np.isfinite(values)
    v, codes = values[ok], pd.factorize(np.asarray(labels)[ok])[0]
    n, k = len(v), codes.max() + 1 if ok.any() else 0
    if k < 2 or n < 3:
        return np.nan
    ranks = rankdata(v)
    _, tc = np.unique(v, return_counts=True)
    tie = 1 - np.sum(tc ** 3 - tc) / (n ** 3 - n)
    if tie <= 0:
        return np.nan

    def H(c):
        cnt = np.bincount(c, minlength=k)
        return (12.0 / (n * (n + 1)) * np.sum(np.bincount(c, weights=ranks, minlength=k) ** 2 / cnt) - 3 * (n + 1)) / tie

    h0 = H(codes)
    hits = sum(H(rng.permutation(codes)) >= h0 - 1e-9 for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def heldout_validation(cluster_df, col, species_table, taxonomy, n_perm, rng, label):
    merged = species_table.merge(cluster_df[["Species", col]].rename(columns={col: "Cluster"}), on="Species")
    clusters = sorted(merged["Cluster"].unique())
    tests, desc = [], []
    for feat in HELDOUT_FEATURES:
        vals = merged[feat].values.astype(float)
        lab = merged["Cluster"].values
        groups = [vals[(lab == c) & np.isfinite(vals)] for c in clusters]
        for c, g in zip(clusters, groups):
            desc.append(dict(method=label, feature=feat, cluster=int(c), n=len(g),
                             median=float(np.median(g)) if len(g) else np.nan,
                             q1=float(np.percentile(g, 25)) if len(g) else np.nan,
                             q3=float(np.percentile(g, 75)) if len(g) else np.nan))
        groups = [g for g in groups if len(g)]
        n, k = sum(len(g) for g in groups), len(groups)
        H = p = eps = eta = pperm = np.nan
        if k > 1 and np.nanstd(vals) > 1e-9:
            try:
                H, p = kruskal(*groups)
                eps = H / (n - 1)                              # epsilon-squared
                eta = (H - k + 1) / (n - k) if n > k else np.nan  # eta-squared_H
                pperm = permutation_kw_p(vals, lab, n_perm, rng)
            except ValueError:
                pass
        tests.append(dict(method=label, feature=feat, kind="spectral index (same acquisitions)" if feat in SPECTRAL_INDEX_FEATURES
                          else "held-out band Water_NIR_970 descriptor", n_species=n, n_clusters=k,
                          min_cluster_n=min(len(g) for g in groups) if groups else 0,
                          H=H, p_value=p, p_permutation=pperm, epsilon_squared=eps, eta_squared_H=eta))
    tdf = pd.DataFrame(tests)
    tdf["q_fdr_bh"] = benjamini_hochberg(tdf["p_value"].values)
    tdf["fdr_family"] = f"{label}: {len(tdf)} tests = {len(SPECTRAL_INDEX_FEATURES)} spectral indices + {len(WATER_FEATURES)} Water/NIR descriptors"
    tdf["low_n_flag"] = tdf["min_cluster_n"] < 3

    d = merged.dropna(subset=["Species"]).copy()
    d["Taxon"] = d["Species"].map(taxonomy)
    d = d.dropna(subset=["Taxon"])
    tax = dict(ari=np.nan, nmi=np.nan, p_perm_ari=np.nan, crosstab=None, n_species=len(d))
    if d["Taxon"].nunique() > 1 and d["Cluster"].nunique() > 1:
        t = pd.factorize(d["Taxon"])[0]
        c = d["Cluster"].values
        a0 = adjusted_rand_score(t, c)
        hits = sum(adjusted_rand_score(rng.permutation(t), c) >= a0 - 1e-12 for _ in range(n_perm))
        tax.update(ari=float(a0), nmi=float(normalized_mutual_info_score(t, c)),
                   p_perm_ari=float((hits + 1) / (n_perm + 1)), crosstab=pd.crosstab(d["Taxon"], d["Cluster"]))
    return dict(tests_df=tdf, descriptives_df=pd.DataFrame(desc), taxonomy=tax, merged=merged)


# ============================================================================
# 8. Bookkeeping: manifest, checksums, audit table
# ============================================================================
def package_versions():
    out = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "tslearn", "matplotlib", "openpyxl"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_checksums(out_dir):
    out_dir = Path(out_dir)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out_dir).as_posix()}"
             for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != "checksums.sha256"]
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")


RESULT_AUDIT = [
    ("Species x day x band trajectories (mean, SD across images, n)", "build_trajectories", "6 bands stored; flag used_in_clustering", "trajectory_tensor_long.csv"),
    ("Experimental design / nesting (images and ROI spectra per species-day)", "build_trajectories", "n/a", "design_summary.csv"),
    ("Channel audit", "channel_band_coverage_report", "all channels x 6 bands", "channel_audit.csv"),
    ("k / gamma / linkage candidate sweeps (both silhouettes)", "run_hard_dtw / run_soft_dtw / run_hierarchical", "5-band clustering tensor", "sweep_hard_dtw.csv; sweep_soft_dtw.csv; sweep_hierarchical.csv"),
    ("Selection-geometry sensitivity", "run_* (alt_col)", "5-band clustering tensor", "geometry_sensitivity.csv"),
    ("Internal validity table (DTW + Euclidean silhouette, DB, stability, ARI vs unconstrained)", "run_pipeline", "5-band clustering tensor", "internal_validity_comparison.csv"),
    ("Consensus k sweep", "select_consensus_k", "co-association of 3 models", "sweep_consensus.csv"),
    ("Pairwise model agreement (ARI)", "agreement_outputs", "3 model partitions + consensus", "model_agreement_ARI.csv"),
    ("Pairwise co-association, unanimous / contested pairs", "agreement_outputs", "3 model partitions", "coassociation_pairs.csv"),
    ("Species-level agreement / disputed strains", "agreement_outputs", "3 model partitions", "species_agreement.csv"),
    ("Held-out spectral summaries: H, p, permutation p, effect sizes, BH q", "heldout_validation", "NDVI/GNDVI/SR + Water_NIR_970 descriptors (not used for clustering)", "heldout_validation_tests.csv"),
    ("Cluster-wise n, median, IQR bounds", "heldout_validation", "same as above", "heldout_cluster_descriptives.csv"),
    ("Taxonomy agreement (ARI, NMI, permutation p)", "heldout_validation", "external taxonomy", "taxonomy_validation.csv"),
]


# ============================================================================
# 9. Main
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description="DTW spectral-trajectory clustering (reviewed version)")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--use_channel", default="Normalized",
                   help="Column suffix to read ('Normalized' was auto-selected in the original run). 'auto' re-runs the audit and picks the best channel.")
    p.add_argument("--selection_geometry", choices=["dtw", "euclidean"], default="dtw",
                   help="Geometry of the silhouette used to choose k / gamma / linkage. 'euclidean' reproduces the pre-review selection rule.")
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--n_stability_seeds", type=int, default=5)
    p.add_argument("--sakoe_chiba_radius", type=int, default=3, help="0 = unconstrained Hard-DTW fitting")
    p.add_argument("--consensus_k", type=int, default=None)
    p.add_argument("--n_perm", type=int, default=4999, help="permutations for species-level checks")
    p.add_argument("--skip_channel_audit", action="store_true")
    return p


def run_pipeline(args):
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(out / "run_log.txt", mode="w"))
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    args.geometry = args.selection_geometry
    args.alt_geometry = "euclidean" if args.geometry == "dtw" else "dtw"
    args.sel_col = "silhouette_dtw" if args.geometry == "dtw" else "silhouette_euclid"
    args.alt_col = "silhouette_euclid" if args.geometry == "dtw" else "silhouette_dtw"
    logger.info("v%s | data_dir=%s | selection geometry=%s", SCRIPT_VERSION, args.data_dir, args.geometry)

    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir} (use --data_dir or MICROALGAE_DATA_DIR)")
    files = discover_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No .xlsx files under {args.data_dir}")

    # --- channel ---------------------------------------------------------
    channel_report = pd.DataFrame()
    channel = args.use_channel
    if channel == "auto" or not args.skip_channel_audit:
        channel_report, per_channel = channel_band_coverage_report(files)
        channel_report.to_csv(out / "channel_audit.csv", index=False)
        logger.info("Channel audit (mean real-signal coverage): %s", per_channel.to_dict())
        if channel == "auto":
            channel = per_channel.index[0]
            logger.info("Auto-selected channel: %s", channel)
        elif channel not in per_channel.index:
            raise ValueError(f"Channel '{channel}' not found; available: {list(per_channel.index)}")

    # --- data & trajectories --------------------------------------------
    ds = load_dataset(files, channel)
    assert ds.data_source == "real"
    T = build_trajectories(ds)
    traj_full, species, days, bands = T["traj"], T["species_order"], T["day_order"], T["band_names"]
    if sorted(species) != sorted(CANONICAL_SPECIES):
        logger.warning("Species found differ from CANONICAL_SPECIES: %s", sorted(set(species) ^ set(CANONICAL_SPECIES)))
    T["design"].to_csv(out / "design_summary.csv", index=False)
    T["long"].to_csv(out / "trajectory_tensor_long.csv", index=False)
    logger.info("Design: %d images, %d ROI spectra; images per species-day min/max = %d/%d; ROI spectra per image median = %.1f",
                len(np.unique(ds.image_id)), len(ds.spectra), T["design"]["n_images"].min(), T["design"]["n_images"].max(),
                len(ds.spectra) / len(np.unique(ds.image_id)))

    assert HELD_OUT_BAND in bands and HELD_OUT_BAND not in CLUSTERING_BANDS
    cidx = [bands.index(b) for b in CLUSTERING_BANDS]
    traj_cluster = traj_full[:, :, cidx]
    logger.info("Clustering tensor %s | bands=%s | held-out=%s", traj_cluster.shape, CLUSTERING_BANDS, HELD_OUT_BAND)

    scaled = TimeSeriesScalerMeanVariance().fit_transform(traj_cluster)
    flat = scaled.reshape(len(species), -1)
    D = dtw_matrix(scaled)  # unconstrained, shared by all scoring steps

    # --- models ------------------------------------------------------------
    logger.info("Model 1: Hard-DTW k-means");      m1, s1 = run_hard_dtw(scaled, flat, D, species, args)
    logger.info("Model 2: Soft-DTW k-means");      m2, s2 = run_soft_dtw(scaled, flat, D, species, args)
    logger.info("Model 3: Hierarchical / DTW");    m3, s3 = run_hierarchical(scaled, flat, D, species, args)
    models = {"Hard-DTW": m1, "Soft-DTW": m2, "Hierarchical-DTW": m3}
    m1["sweep_df"].to_csv(out / "sweep_hard_dtw.csv", index=False)
    m2["sweep_df"].to_csv(out / "sweep_soft_dtw.csv", index=False)
    m3["sweep_df"].to_csv(out / "sweep_hierarchical.csv", index=False)
    m2["report"]["candidate_ranking"].to_csv(out / "sweep_soft_dtw_stability_ranking.csv", index=False)
    sens_df = pd.DataFrame([s1, s2, s3])
    sens_df.to_csv(out / "geometry_sensitivity.csv", index=False)

    rows = []
    for m in models.values():
        r = {k: v for k, v in m["report"].items() if k != "candidate_ranking"}
        rows.append(r)
    internal_df = pd.DataFrame(rows)
    internal_df.to_csv(out / "internal_validity_comparison.csv", index=False)
    print(internal_df.to_string(index=False))
    print(sens_df.to_string(index=False))

    # --- consensus ---------------------------------------------------------
    label_map = {n: m["labels"] for n, m in models.items()}
    co = build_coassociation(list(label_map.values()))
    if args.consensus_k:
        dist = 1 - co
        np.fill_diagonal(dist, 0)
        Zc, kc, cons_sweep = scipy_linkage(squareform(dist, checks=False), method="average"), args.consensus_k, pd.DataFrame()
    else:
        kc, cons_sweep, Zc = select_consensus_k(co, D, args)
    cons_labels = fcluster(Zc, t=kc, criterion="maxclust")
    consensus_df = pd.DataFrame({"Species": species, "Consensus_Cluster": cons_labels})
    cons_sweep.to_csv(out / "sweep_consensus.csv", index=False)
    ari_df, pairs_df, sp_agree_df = agreement_outputs(species, label_map, cons_labels, co)
    ari_df.to_csv(out / "model_agreement_ARI.csv")
    pairs_df.to_csv(out / "coassociation_pairs.csv", index=False)
    sp_agree_df.to_csv(out / "species_agreement.csv", index=False)
    pd.DataFrame(co, index=species, columns=species).to_csv(out / "coassociation_matrix.csv")
    identical = [n for n in label_map if adjusted_rand_score(cons_labels, label_map[n]) >= 0.999]
    logger.info("Consensus k=%d. Pair categories: %s", kc, pairs_df["category"].value_counts().to_dict())
    if identical:
        logger.warning("Consensus partition is IDENTICAL to %s. It is a relabelling of that model, NOT independent "
                       "three-way confirmation; all three models use the same DTW geometry and trajectories.", identical)
    print(ari_df.round(3).to_string())

    # --- held-out spectral summaries + taxonomy ---------------------------------
    pig = pd.DataFrame(ds.pigment, columns=SPECTRAL_INDEX_FEATURES)
    pig["Species"] = ds.species
    species_table = water_descriptors(traj_full, species, days, bands).merge(pig.groupby("Species").mean().reset_index(), on="Species")
    species_table.to_csv(out / "species_heldout_features.csv", index=False)
    partitions = {**{n: m["result_df"] for n, m in models.items()}, "Consensus": consensus_df}
    cols = {**{n: m["cluster_col"] for n, m in models.items()}, "Consensus": "Consensus_Cluster"}
    validation = {n: heldout_validation(df, cols[n], species_table, SPECIES_TAXONOMY, args.n_perm, rng, n)
                  for n, df in partitions.items()}
    pd.concat([v["tests_df"] for v in validation.values()]).to_csv(out / "heldout_validation_tests.csv", index=False)
    pd.concat([v["descriptives_df"] for v in validation.values()]).to_csv(out / "heldout_cluster_descriptives.csv", index=False)
    pd.DataFrame([dict(method=n, ARI=v["taxonomy"]["ari"], NMI=v["taxonomy"]["nmi"], p_permutation_ARI=v["taxonomy"]["p_perm_ari"],
                       n_species=v["taxonomy"]["n_species"]) for n, v in validation.items()]).to_csv(out / "taxonomy_validation.csv", index=False)
    for n, v in validation.items():
        t = v["tests_df"]
        print(f"\n[{n}] held-out spectral summaries (BH family = {t['fdr_family'].iloc[0]})")
        print(t[["feature", "min_cluster_n", "H", "p_value", "p_permutation", "epsilon_squared", "q_fdr_bh"]].round(4).to_string(index=False))

    # --- save --------------------------------------------------------------------
    manifest = dict(script_version=SCRIPT_VERSION, seed=SEED, args={k: v for k, v in vars(args).items()},
                    python=sys.version, platform=platform.platform(), packages=package_versions(),
                    channel_used=channel, data_source=ds.data_source, clustering_bands=CLUSTERING_BANDS,
                    held_out_band=HELD_OUT_BAND, tensor_shape_clustering=list(traj_cluster.shape),
                    tensor_shape_full=list(traj_full.shape), n_images=int(len(np.unique(ds.image_id))),
                    n_roi_spectra=int(len(ds.spectra)), n_files_ok=ds.n_files_ok, n_files_skipped=ds.n_files_skipped,
                    n_files_excluded_no_day=ds.n_files_excluded_no_day, n_interpolated_species_day_cells=T["n_missing_cells"],
                    dtw_scoring_matrix="unconstrained DTW on z-scored 5-band tensor",
                    selection_geometry=args.geometry, consensus_k=int(kc), consensus_identical_to=identical,
                    runtime_s=round(time.time() - t0, 1))
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    pd.DataFrame(RESULT_AUDIT, columns=["result", "function", "input_feature_set", "saved_output"]).to_csv(out / "audit_table_results.csv", index=False)
    bundle = dict(manifest=manifest, species_order=species, day_order=days, band_names_all=bands,
                  clustering_bands=CLUSTERING_BANDS, heldout_band=HELD_OUT_BAND, traj_full=traj_full, traj_sd_full=T["sd"],
                  traj_cluster=traj_cluster, shared_dtw_dist=D, res=models, consensus_df=consensus_df, coassoc=co,
                  consensus_k=int(kc), channel_report=channel_report, taxonomy=SPECIES_TAXONOMY,
                  species_full_names=SPECIES_FULL_NAMES, species_table=species_table, validation=validation,
                  agreement_ari=ari_df, species_agreement=sp_agree_df, internal_df=internal_df,
                  geometry_sensitivity=sens_df, selection_geometry=args.geometry,
                  sakoe_chiba_radius=args.sakoe_chiba_radius)
    with open(out / "model_results.pkl", "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    write_checksums(out)
    logger.info("Done in %.1f s. Outputs in %s", time.time() - t0, out)
    return bundle


def main(argv=None):
    args, _ = build_parser().parse_known_args(argv)
    return run_pipeline(args)



if __name__ == "__main__":
    main()
