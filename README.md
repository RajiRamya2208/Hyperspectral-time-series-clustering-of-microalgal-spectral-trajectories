# Hyperspectral time-series clustering of microalgal spectral trajectories

Code and data accompanying the manuscript *"Hyperspectral time-series clustering of microalgal spectral
trajectories for strain-aware harvest window identification"* (SRM University-AP).

**Version:** 2.0.0 | **Python:** 3.12.7 | **Random seed:** 42

## Access conditions

The code and data in this repository are provided for peer review and for reproducing the results reported in the
manuscript. They are released without an open-source licence: all rights are reserved by the authors and
SRM University-AP. Any reuse beyond verification requires written permission from the corresponding author
(rajendran.k@srmap.edu.in).

## What this repository does

Clusters 20 microalgal/cyanobacterial strains by the temporal behaviour of five spectral bands
(Chlorophyll_680, Carotenoid_480, Phycobiliprotein_620, Lipid_CH_930, Protein_Amide_910) over 31 days,
using Hard-DTW k-means, Soft-DTW k-means and hierarchical DTW. The three partitions are combined by a
co-association consensus and described with held-out summaries (Water_NIR_970 descriptors, NDVI, GNDVI,
Simple Ratio) and taxonomy. The Water/NIR band is never given to any clustering model.
There is no synthetic-data fallback; `data_source` is always `real` and is logged in `run_manifest.json`.

## Files

| File | Purpose |
|---|---|
| `01_training_and_validation.py` | single analysis entry point |
| `02_figures_corrected.py` | single figure-generation entry point |
| `03_reviewer_extra_checks.py` | sensitivity analyses (warping window, band subsets, medium confounding) |
| `strain_metadata.csv` | strain, phylum and culture medium for all 20 strains |
| `hyperspectral_data_combined.xlsx` | all band-level data in one workbook (see Data) |
| `data/` | the 1,860 original Specim IQ workbooks, one zip per strain (see Data) |
| `requirements.txt` | Python packages |

## Data

* `hyperspectral_data_combined.xlsx` has two data sheets: `ROI_band_means` (37,200 ROI spectra: six band means,
  NDVI, GNDVI, Simple Ratio) and `Strain_day_means` (620 strain-days; mean of the six bands before z-scoring).
* `data/` contains the 1,860 original workbooks (20 strains x 31 days x 3 images; 20 ROI spectra each; one culture
  flask per strain, so images and ROIs are technical replicates). Script 01 reads these workbooks.
  Unzip every archive so that the strain folders sit in `data/hyperspectral_xlsx/`, for example
  `data/hyperspectral_xlsx/ANBA/Day_0_ANBA_1_2830.xlsx`.

## Install

```
pip install -r requirements.txt
```

## Run (in this order)

```
python 01_training_and_validation.py --data_dir data/hyperspectral_xlsx
python 02_figures_corrected.py --out_dir outputs --dpi 600
python 03_reviewer_extra_checks.py --out_dir outputs --metadata strain_metadata.csv
```

## Expected outputs (created in `outputs/`)

* `model_results.pkl` - the single saved result object read by scripts 02 and 03
* `*.csv` - every table reported in the manuscript; `audit_table_results.csv` and `audit_table_figures.csv`
  link each result and figure to its function, feature set and saved output
* `run_manifest.json` - arguments, seed, package versions, bands, channel, `data_source`
* `checksums.sha256` - SHA-256 of every output file (verify with `sha256sum -c checksums.sha256` inside `outputs/`)
* `figures/` - PNG and PDF figures
* `extra_checks/` - outputs of script 03

## Contact

Corresponding author: rajendran.k@srmap.edu.in
