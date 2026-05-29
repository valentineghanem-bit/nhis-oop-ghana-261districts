#!/usr/bin/env bash
# run_all.sh — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
# One-shot pipeline driver. Runs all four analytical stages in sequence.
# Usage: bash run_all.sh
# Requires: Python 3.12 with requirements.txt installed; R 4.3+ for Stage 2b (optional).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "NHIS OOP Ghana 261 Districts — Full Pipeline"
echo "AIPOCH v6.5 | $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

echo ""
echo "--- Stage 1: Data Preparation ---"
python3 scripts/00_data_cleaning.py
python3 scripts/01_data_wrangling.py

echo ""
echo "--- Stage 2a: Spatial Analysis (Python) ---"
python3 scripts/02_spatial_analysis.py

echo ""
echo "--- Stage 2b: Spatial Diagnostics (R, optional) ---"
if command -v Rscript &>/dev/null; then
    Rscript scripts/spatial_diagnostics.R
else
    echo "  [SKIP] Rscript not found — skipping R spatial diagnostics"
fi

echo ""
echo "--- Stage 3: Machine Learning Pipeline ---"
python3 scripts/03_ml_pipeline.py

echo ""
echo "--- Stage 4: Figure and Output Generation ---"
python3 scripts/analysis_pipeline.py

echo ""
echo "=========================================="
echo "Pipeline complete. Outputs:"
echo "  data/processed/spatial_results.csv"
echo "  data/processed/ml_results.csv"
echo "  figures/  (fig_01 – fig_13)"
echo "  tables/   (table_ml_performance.csv, etc.)"
echo "=========================================="
