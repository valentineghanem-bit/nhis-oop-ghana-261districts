[![CI](https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE · RECORD-Spatial · TRIPOD+AI
**Date:** 2026
**Status:** Manuscript in preparation

---

## 1. Overview

This repository contains the complete reproducible analytical pipeline for a cross-sectional ecological study of National Health Insurance Scheme (NHIS) non-enrolment across all 261 administrative districts of Ghana. The analysis integrates Ghana Population and Housing Census 2021 data with Demographic and Health Survey data (2019, 2022) to characterise the spatial distribution and structural socioeconomic determinants of district-level NHIS uninsurance.

**Key finding:** NHIS non-enrolment is strongly spatially concentrated (Global Moran's I = 0.461, z = 11.94, p < 0.001) and predictable from routinely available census data (RF AUC-ROC = 0.918). Poverty incidence is the dominant determinant across all analytical frameworks; women's NHIS coverage is a robust protective factor (LASSO OR = 0.697).

## 2. Repository Structure

```
nhis-oop-ghana-261districts/
├── data/
│   ├── raw/                          # Source data (GeoJSON, PHC CSVs)
│   └── processed/                    # Analytical outputs (spatial_results.csv, ml_results.csv)
├── scripts/
│   ├── 00_data_cleaning.py           # Raw data ingestion and cleaning
│   ├── 01_data_wrangling.py          # Feature engineering and zone classification
│   ├── 02_spatial_analysis.py        # Queen weights, Moran's I, LISA, Gi*, GWR
│   ├── 03_ml_pipeline.py             # RF, HGB, LASSO CV + permutation importance
│   ├── spatial_utils.py              # Reusable spatial analysis utilities
│   ├── analysis_pipeline.py          # Full pipeline orchestrator
│   └── spatial_diagnostics.R         # R spatial regression diagnostics (spdep)
├── figures/                          # Publication figures (PNG, 300 DPI)
├── tables/                           # Summary tables (CSV)
├── dashboard/
│   └── NHIS_OOP_Ghana_Dashboard.html # Interactive vanilla-JS dashboard
├── poster/
│   └── NHIS_OOP_Ghana_Poster.html    # A0 conference poster (HTML/CSS)
├── tests/                            # pytest test suite
├── app.py                            # Dashboard launcher (stdlib HTTP server)
├── analysis.R                        # Spatial regression diagnostics (R)
├── run_all.sh                        # One-shot pipeline driver
├── Dockerfile                        # Full computational environment
├── requirements.txt                  # Pinned Python dependencies
├── CITATION.cff                      # Machine-readable citation
├── LICENSE                           # MIT License
└── .github/workflows/ci.yml         # GitHub Actions CI
```

## 3. Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Queen contiguity | Pure Python (vertex-set) | Spatial weights matrix (n=260, 708 links) |
| Global Moran's I | NumPy (vectorised, 999 perms) | Global spatial autocorrelation test |
| Univariate LISA | NumPy permutation | Local HH/LL/HL/LH cluster detection |
| Bivariate LISA | NumPy permutation | Poverty & illiteracy co-clustering |
| Getis-Ord Gi* | NumPy (self-inclusive W) | Hotspot/coldspot delineation |
| GWR | NumPy WLS (adaptive Gaussian) | Spatially varying coefficients |
| Random Forest | scikit-learn 1.7.2 | Binary risk classification |
| Gradient Boosting | HistGradientBoostingClassifier | Ensemble comparator |
| LASSO Logistic Regression | scikit-learn (saga solver) | Variable selection + odds ratios |
| Permutation importance | scikit-learn (30 reps) | Feature importance (SHAP substitute) |
| Spatial regression diagnostics | spdep / spatialreg (R) | SLM/SEM model comparison |

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| Ghana Population and Housing Census (PHC) | Poverty, illiteracy, employment, youth dependency, NHIS status | 2021 | Public (GSS) |
| Ghana Demographic and Health Survey | Women's NHIS coverage (regional) | 2019 | Open (DHS Program) |
| Ghana Demographic and Health Survey | ANC coverage, female education, home delivery (regional) | 2022 | Open (DHS Program) |
| Ghana New 260 District GeoJSON | Administrative boundary polygons | 2021 | Public |

DHS variables are regional-level proxies assigned to all districts within each survey region. One district (Guan, Oti Region) lacks standardised boundary geometry and is included in descriptive and ML analyses but excluded from spatial analyses (n=260 spatial; n=261 total).

## 5. Key Findings

| Metric | Value |
|--------|-------|
| Districts analysed (total / spatial) | 261 / 260 |
| National mean uninsurance rate | 30.9% (SD 13.5%) |
| High-uninsurance districts (>30.2%) | 130 / 261 (49.8%) |
| Global Moran's I (uninsurance) | 0.461 (z=11.94, p<0.001) |
| LISA HH hotspot districts | 21 (8.1%) |
| LISA LL coldspot districts | 32 (12.3%) |
| Gi* confirmed hotspots (p<0.01) | 18 districts |
| Gi* confirmed coldspots (p<0.01) | 27 districts |
| GWR mean local R² | 0.624 (vs OLS 0.371) |
| GWR optimal bandwidth | 0.561° (≈62 km) |
| GWR β — poverty incidence | +0.625 (dominant) |
| GWR β — women's NHIS coverage | −0.110 (protective) |
| RF AUC-ROC (5-fold CV) | 0.918 ± 0.037 |
| RF AUC-PR (5-fold CV) | 0.926 ± 0.036 |
| LASSO OR — poverty incidence | 2.609 |
| LASSO OR — women's NHIS coverage | 0.697 (protective) |
| LASSO OR — Southern Belt | 1.626 |

## 6. Installation and Dependencies

```bash
# Clone repository
git clone https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts.git
cd nhis-oop-ghana-261districts

# Python environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# R packages (optional — spatial regression diagnostics only)
Rscript -e "install.packages(c('spdep','spatialreg','sf','dplyr','ggplot2'))"
```

**Minimum requirements:** Python 3.12 · scikit-learn 1.7.2 · pandas 2.x · numpy 2.x · geopandas · matplotlib

## 7. Reproducing the Analysis

The full pipeline runs in four sequential stages. Execute the one-shot driver or run scripts individually.

### 7.1 One-shot driver

```bash
bash run_all.sh
```

### 7.2 Stage 1 — Data preparation

```bash
python scripts/00_data_cleaning.py    # ingest + clean PHC/DHS source data
python scripts/01_data_wrangling.py   # feature engineering; output: district_master_261_analytical.csv
```

### 7.3 Stage 2 — Spatial analysis

```bash
python scripts/02_spatial_analysis.py  # Queen weights, Moran's I, LISA, Gi*, GWR
# Optional R diagnostics:
Rscript scripts/spatial_diagnostics.R  # spdep LM tests, SLM/SEM comparison
```

### 7.4 Stage 3 — Machine learning pipeline

```bash
python scripts/03_ml_pipeline.py  # RF + HGB + LASSO; 5-fold CV; permutation importance
```

### 7.5 Stage 4 — Figures and outputs

```bash
python scripts/analysis_pipeline.py  # regenerate all figures + summary tables
```

### 7.6 Launching the interactive dashboard

```bash
# macOS
open dashboard/NHIS_OOP_Ghana_Dashboard.html
# Windows
start dashboard/NHIS_OOP_Ghana_Dashboard.html
# Linux
xdg-open dashboard/NHIS_OOP_Ghana_Dashboard.html
```

Or via the Python launcher:

```bash
python app.py              # serves on http://localhost:8050
python app.py --port 8080  # custom port
```

## 8. Outputs

All outputs are generated reproducibly from the scripts above. Do not manually edit generated files; re-run the relevant script stage instead.

Primary data output: `data/processed/spatial_results.csv` (261 × 54) — the Master CSV for all cross-output reconciliation.

## 8a. Downloadable Artefacts

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|---------------|--------------|---------------------------|
| Interactive dashboard | [View](https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts/blob/main/dashboard/NHIS_OOP_Ghana_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts/blob/main/dashboard/NHIS_OOP_Ghana_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/nhis-oop-ghana-261districts/main/dashboard/NHIS_OOP_Ghana_Dashboard.html) |
| Conference poster | [View](https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts/blob/main/poster/NHIS_OOP_Ghana_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts/blob/main/poster/NHIS_OOP_Ghana_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/nhis-oop-ghana-261districts/main/poster/NHIS_OOP_Ghana_Poster.html) |

> **Tip:** The dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies. Machine learning components follow **TRIPOD+AI**; spatial statistical components follow **RECORD-Spatial**.

## 10. Ethics

Retrospective analysis of publicly available, de-identified aggregate administrative and survey data. Ethical oversight: Ghana Health Service Ethics Review Board. No individual-level data were accessed.

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Spatial distribution and socioeconomic determinants of NHIS non-enrolment across 261 districts in Ghana: a geographically weighted regression and machine learning analysis.* GitHub. https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts

**BibTeX:**
```bibtex
@misc{ghanem2026nhis,
  author = {Ghanem, Valentine Golden},
  title  = {Spatial distribution and socioeconomic determinants of NHIS non-enrolment across 261 districts in Ghana},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/nhis-oop-ghana-261districts}
}
```

A machine-readable citation is provided in `CITATION.cff`.

## 12. License

Code is released under the **MIT License** — see [LICENSE](LICENSE) for details.
Outputs and figures: **CC BY 4.0**.

## 13. Author & Contact

**Valentine Golden Ghanem**
Ghana COCOBOD Cocoa Clinic, Accra, Ghana
Email: valentineghanem@gmail.com
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

## 14. Acknowledgements

The author thanks the Ghana Statistical Service for publicly releasing PHC 2021 district-level data and the DHS Program for open-access survey microdata. Analysis system: AIPOCH v6.5 — 9-Connector Non-Destructive Engine.
