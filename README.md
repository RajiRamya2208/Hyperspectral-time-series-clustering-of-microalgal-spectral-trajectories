**Hyperspectral time-series clustering of microalgal spectral trajectories**

Code accompanying the manuscript "Hyperspectral time-series clustering of microalgal spectral trajectories for strain-aware harvest window identification" (SRM University-AP).

Version: 2.0.0 | Python 3.12.7 | Random seed: 42 | Licence: MIT (code)

**What this repository does**

Clusters 20 microalgal/cyanobacterial strains by the temporal behaviour of five spectral bands (Chlorophyll_680, Carotenoid_480, Phycobiliprotein_620, Lipid_CH_930, Protein_Amide_910) over 31 days using Hard-DTW k-means, Soft-DTW k-means and hierarchical DTW, combines them by a co-association consensus, and describes the clusters with held-out summaries (Water_NIR_970 descriptors, NDVI, GNDVI, Simple Ratio) and taxonomy. The Water/NIR band is never given to any clustering model. There is no synthetic-data fallback; data_source is always real and is logged in run_manifest.json.

**Folder structure**
01_training_and_validation.py   single analysis entry point

02_figures_corrected.py         single figure-generation entry point

03_reviewer_extra_checks.py     sensitivity analyses (warping window, band subsets, medium confounding)

strain_metadata.csv             Species, Medium (+ optional Salinity, Accession) for all 20 strains

requirements.txt

data/hyperspectral_xlsx/        1,860 workbooks (20 strains x 31 days x 3 images); see Data below

outputs/                        created by the scripts

**Install**
pip install -r requirements.txt

**Run (in this order)**
python 01_training_and_validation.py --data_dir data/hyperspectral_xlsx
python 02_figures_corrected.py --out_dir outputs --dpi 600
python 03_reviewer_extra_checks.py --out_dir outputs --metadata strain_metadata.csv


**Expected outputs (in outputs/)**
model_results.pkl : the single saved result object read by scripts 02 and 03
*.csv : every table reported in the manuscript; audit_table_results.csv and audit_table_figures.csv link each result/figure to its function, feature set and saved output
run_manifest.json : arguments, seed, package versions, bands, channel, data_source
checksums.sha256 : SHA-256 of every output file
figures/ : PNG + PDF figures
extra_checks/ : outputs of script 03

Verify outputs: cd outputs && sha256sum -c checksums.sha256

**Data**

ROI spectra (37,200 spectra from 1,860 images) and daily strain-level trajectory tables: [Zenodo DOI to be added], licence CC BY 4.0.
