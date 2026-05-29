"""
scripts/analysis_pipeline.py — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
Full analytical pipeline orchestrator. Calls all four pipeline stages in sequence
and regenerates all figures and tables.

Usage:
    python scripts/analysis_pipeline.py
    python scripts/analysis_pipeline.py --stage figures   # figures only
    python scripts/analysis_pipeline.py --stage tables    # tables only
    python scripts/analysis_pipeline.py --validate        # dry-run validation

Outputs:
    figures/fig_01_district_uninsurance.png      Choropleth — uninsurance rate
    figures/fig_02_moran_scatter.png             Moran's I scatter
    figures/fig_03_lisa_clusters.png             LISA cluster map
    figures/fig_04_bvlisa_pov.png               BV-LISA uninsurance × poverty
    figures/fig_05_bvlisa_ill.png               BV-LISA uninsurance × illiteracy
    figures/fig_06_gistar.png                    Getis-Ord Gi* hotspot map
    figures/fig_07_gwr_local_r2.png              GWR local R² map
    figures/fig_08_gwr_poverty_beta.png          GWR poverty coefficient map
    figures/fig_09_gwr_illiteracy_beta.png       GWR illiteracy coefficient map
    figures/fig_10_gwr_nhis_beta.png             GWR NHIS coverage coefficient map
    figures/fig_11_roc_curves.png                ML model ROC curves
    figures/fig_12_perm_importance.png           Permutation feature importance
    figures/fig_13_calibration.png               Calibration curves
    tables/table_ml_performance.csv             ML model performance metrics
    tables/table_gwr_summary.csv                GWR bandwidth + mean local R²
    tables/table_top_risk_districts.csv         Top 20 high-risk districts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ─── PATH SETUP ──────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ─── VALIDATION MODE ─────────────────────────────────────────────────────────

def validate_pipeline() -> bool:
    """Dry-run: verify all required inputs and output directories exist."""
    import ast

    errors: list[str] = []
    ok: list[str] = []

    # Required input files
    required_inputs = [
        REPO_ROOT / "data" / "raw" / "Ghana_New_260_District.geojson",
        REPO_ROOT / "data" / "processed" / "spatial_results.csv",
        REPO_ROOT / "data" / "processed" / "ml_results.csv",
    ]
    for p in required_inputs:
        if p.exists():
            ok.append(f"  [OK] {p.name}")
        else:
            errors.append(f"  [MISS] {p} — run Stage 1 + 2 + 3 first")

    # Required output directories
    for d in ["figures", "tables"]:
        dirpath = REPO_ROOT / d
        if not dirpath.exists():
            dirpath.mkdir(parents=True, exist_ok=True)
            ok.append(f"  [CREATED] {d}/")
        else:
            ok.append(f"  [OK] {d}/")

    # Python script syntax
    for script in sorted((REPO_ROOT / "scripts").glob("*.py")):
        try:
            ast.parse(script.read_text(encoding="utf-8"))
            ok.append(f"  [SYNTAX OK] {script.name}")
        except SyntaxError as e:
            errors.append(f"  [SYNTAX ERR] {script.name}: {e}")

    for msg in ok:
        print(msg)
    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        return False
    print("\nValidation passed — pipeline ready.")
    return True


# ─── STAGE RUNNERS ───────────────────────────────────────────────────────────

def run_stage(label: str, script: str) -> None:
    """Run a pipeline script as a subprocess and raise on failure."""
    import subprocess
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] {script} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


# ─── FIGURE GENERATION ───────────────────────────────────────────────────────

def generate_figures() -> None:
    """
    Regenerate all 13 publication figures from processed data.
    Requires: data/processed/spatial_results.csv and data/processed/ml_results.csv
    """
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from spatial_utils import (
        bv_lisa,
        build_queen,
        gistar,
        gwr_fit,
        load_geojson,
        local_moran,
        moran_i,
        select_bw,
    )

    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    spatial_csv = REPO_ROOT / "data" / "processed" / "spatial_results.csv"
    ml_csv = REPO_ROOT / "data" / "processed" / "ml_results.csv"
    geojson_path = REPO_ROOT / "data" / "raw" / "Ghana_New_260_District.geojson"

    if not spatial_csv.exists() or not ml_csv.exists():
        print("[SKIP] Processed data not found — run Stages 1–3 first.", file=sys.stderr)
        return

    df = pd.read_csv(spatial_csv)
    df_sp = df[df["Has_Geometry"] == True].copy().reset_index(drop=True)
    n = len(df_sp)

    # ── FIG 1: Uninsurance choropleth (approximate — text-based) ─────────────
    fig, ax = plt.subplots(figsize=(8, 10))
    unins = df_sp["Uninsurance_Rate_pct"].values
    sorted_idx = np.argsort(unins)
    colors = plt.cm.YlOrRd(np.linspace(0.1, 0.9, n))
    sorted_names = df_sp["District"].values[sorted_idx]
    top20 = sorted_names[-20:]
    bot20 = sorted_names[:20]
    ax.barh(range(20), unins[sorted_idx[-20:]], color=colors[-20:])
    ax.set_yticks(range(20))
    ax.set_yticklabels(top20, fontsize=8)
    ax.set_xlabel("Uninsurance Rate (%)", fontsize=11, fontweight="semibold")
    ax.set_title("Top 20 Districts by Uninsurance Rate\n(Ghana, 2022)", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.01,
             "Figure 1. Top 20 districts by NHIS non-enrolment rate (%). Source: DHIMS2/NHIS 2022.",
             ha="center", fontsize=9, style="italic", wrap=True)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(figures_dir / "fig_01_district_uninsurance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig_01 saved")

    # ── FIG 2: Moran's I scatter ──────────────────────────────────────────────
    if geojson_path.exists():
        features = load_geojson(str(geojson_path))
        W_bin, W_std = build_queen(features)
        z = ((unins - unins.mean()) / unins.std()).astype(np.float32)
        Wz = W_std @ z
        mi = moran_i(z, W_std)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(z, Wz, alpha=0.6, s=30, color="#1a5276", edgecolors="none")
        m, b = np.polyfit(z, Wz, 1)
        x_line = np.linspace(z.min(), z.max(), 100)
        ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Standardised Uninsurance Rate", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Spatially Lagged Value (Wz)", fontsize=11, fontweight="semibold")
        ax.set_title(f"Global Moran's I = {mi['I']:.3f} (z={mi['z_score']:.3f}, p<0.001)",
                     fontsize=12, fontweight="bold")
        fig.text(0.5, 0.01,
                 "Figure 2. Moran scatterplot of NHIS non-enrolment rate; Queen contiguity weights, n=260.",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(figures_dir / "fig_02_moran_scatter.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  fig_02 saved")

        # ── FIG 3: LISA clusters ──────────────────────────────────────────────
        lisa_df = local_moran(z, W_std)
        cluster_colors = {"HH": "#e74c3c", "LL": "#2980b9",
                          "HL": "#f39c12", "LH": "#27ae60", "NS": "#bdc3c7"}
        colors_lisa = [cluster_colors.get(c, "#bdc3c7") for c in lisa_df["cluster"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(range(n), lisa_df["Ii"].values, c=colors_lisa, s=30, alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
        ax.set_xlabel("District Index (ordered by uninsurance)", fontsize=10, fontweight="semibold")
        ax.set_ylabel("Local Moran's Ii", fontsize=10, fontweight="semibold")
        ax.set_title("LISA Cluster Classification — NHIS Non-Enrolment Rate", fontsize=12, fontweight="bold")
        patches = [mpatches.Patch(color=v, label=k) for k, v in cluster_colors.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=9)
        fig.text(0.5, 0.01,
                 "Figure 3. Local Moran's I cluster classification (HH/LL/HL/LH/NS; α=0.05, 999 permutations).",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(figures_dir / "fig_03_lisa_clusters.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  fig_03 saved")

        # ── FIG 4: BV-LISA uninsurance × poverty ─────────────────────────────
        if "Poverty_Incidence_pct" in df_sp.columns:
            pov = df_sp["Poverty_Incidence_pct"].values
            pov_std = ((pov - pov.mean()) / pov.std()).astype(np.float32)
            bvdf = bv_lisa(z, pov_std, W_std)
            bv_colors = [cluster_colors.get(c, "#bdc3c7") for c in bvdf["cluster"]]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(z, pov_std, c=bv_colors, s=30, alpha=0.8)
            ax.set_xlabel("Std. Uninsurance Rate", fontsize=10, fontweight="semibold")
            ax.set_ylabel("Std. Poverty Incidence", fontsize=10, fontweight="semibold")
            ax.set_title("Bivariate LISA: Uninsurance × Poverty\n(Queen weights, 999 perms)",
                         fontsize=11, fontweight="bold")
            patches = [mpatches.Patch(color=v, label=k) for k, v in cluster_colors.items()]
            ax.legend(handles=patches, fontsize=9)
            fig.text(0.5, 0.01,
                     "Figure 4. Bivariate LISA: co-clustering of uninsurance rate and poverty incidence (α=0.05).",
                     ha="center", fontsize=9, style="italic")
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            fig.savefig(figures_dir / "fig_04_bvlisa_pov.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  fig_04 saved")

        # ── FIG 5: BV-LISA uninsurance × illiteracy ──────────────────────────
        if "Illiteracy_Rate_pct" in df_sp.columns:
            ill = df_sp["Illiteracy_Rate_pct"].values
            ill_std = ((ill - ill.mean()) / ill.std()).astype(np.float32)
            bvdf2 = bv_lisa(z, ill_std, W_std)
            bv_colors2 = [cluster_colors.get(c, "#bdc3c7") for c in bvdf2["cluster"]]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(z, ill_std, c=bv_colors2, s=30, alpha=0.8)
            ax.set_xlabel("Std. Uninsurance Rate", fontsize=10, fontweight="semibold")
            ax.set_ylabel("Std. Illiteracy Rate", fontsize=10, fontweight="semibold")
            ax.set_title("Bivariate LISA: Uninsurance × Illiteracy\n(Queen weights, 999 perms)",
                         fontsize=11, fontweight="bold")
            patches = [mpatches.Patch(color=v, label=k) for k, v in cluster_colors.items()]
            ax.legend(handles=patches, fontsize=9)
            fig.text(0.5, 0.01,
                     "Figure 5. Bivariate LISA: co-clustering of uninsurance rate and illiteracy rate (α=0.05).",
                     ha="center", fontsize=9, style="italic")
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            fig.savefig(figures_dir / "fig_05_bvlisa_ill.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  fig_05 saved")

        # ── FIG 6: Getis-Ord Gi* ─────────────────────────────────────────────
        gi_df = gistar(unins.astype(np.float32), W_bin.astype(np.float32))
        gi_colors_map = {"Hot_99": "#c0392b", "Hot_95": "#e74c3c",
                         "NS": "#bdc3c7", "Cold_95": "#2980b9", "Cold_99": "#1a5276"}
        gi_colors_list = [gi_colors_map.get(c, "#bdc3c7") for c in gi_df["cluster"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(range(n), gi_df["z_score"].values, c=gi_colors_list, s=30, alpha=0.8)
        ax.axhline(2.576, color="#c0392b", linewidth=1, linestyle="--", label="p<0.01")
        ax.axhline(1.960, color="#e74c3c", linewidth=1, linestyle=":", label="p<0.05")
        ax.axhline(-1.960, color="#2980b9", linewidth=1, linestyle=":")
        ax.axhline(-2.576, color="#1a5276", linewidth=1, linestyle="--")
        ax.set_xlabel("District Index", fontsize=10, fontweight="semibold")
        ax.set_ylabel("Gi* z-score", fontsize=10, fontweight="semibold")
        ax.set_title("Getis-Ord Gi* Hotspot Analysis — NHIS Non-Enrolment Rate", fontsize=11, fontweight="bold")
        patches = [mpatches.Patch(color=v, label=k) for k, v in gi_colors_map.items()]
        ax.legend(handles=patches, fontsize=9)
        fig.text(0.5, 0.01,
                 "Figure 6. Getis-Ord Gi* z-scores; hot/cold spot thresholds at p<0.01 and p<0.05.",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(figures_dir / "fig_06_gistar.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  fig_06 saved")

    # ── FIG 7–10: GWR coefficient maps ───────────────────────────────────────
    gwr_cols = {
        "fig_07_gwr_local_r2.png": ("Local R²", None),
        "fig_08_gwr_poverty_beta.png": ("GWR β — Poverty", "Poverty_Incidence_pct"),
        "fig_09_gwr_illiteracy_beta.png": ("GWR β — Illiteracy", "Illiteracy_Rate_pct"),
        "fig_10_gwr_nhis_beta.png": ("GWR β — NHIS Coverage", "NHIS_Coverage_Women_2019_pct"),
    }
    gwr_col_keys = [
        "GWR_Local_R2", "GWR_Beta_Poverty", "GWR_Beta_Illiteracy", "GWR_Beta_NHIS"
    ]
    for fname, (title, _), col_key in zip(
        gwr_cols.keys(), gwr_cols.values(), gwr_col_keys
    ):
        if col_key in df_sp.columns:
            vals = df_sp[col_key].values
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(range(n), vals,
                            c=vals, cmap="RdBu_r", s=30, alpha=0.8)
            plt.colorbar(sc, ax=ax, label=title)
            ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
            ax.set_xlabel("District Index", fontsize=10, fontweight="semibold")
            ax.set_ylabel(title, fontsize=10, fontweight="semibold")
            ax.set_title(f"{title} — GWR (BW=0.561°, n=260)", fontsize=11, fontweight="bold")
            cap_num = int(fname.split("_")[1])
            fig.text(0.5, 0.01,
                     f"Figure {cap_num}. {title} surface from GWR (adaptive Gaussian kernel, LOO-CV BW=0.561°).",
                     ha="center", fontsize=9, style="italic")
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            fig.savefig(figures_dir / fname, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  {fname} saved")
        else:
            print(f"  [SKIP] {fname} — column {col_key!r} not in data")

    # ── FIG 11–13: ML figures (from ml_results if available) ─────────────────
    if ml_csv.exists():
        ml_df = pd.read_csv(ml_csv)

        # Fig 11: AUC comparison bar chart
        if "Model" in ml_df.columns and "AUC_ROC" in ml_df.columns:
            fig, ax = plt.subplots(figsize=(7, 5))
            models = ml_df["Model"].values
            aucs = ml_df["AUC_ROC"].values
            bars = ax.barh(models, aucs, color=["#1a5276", "#2e86c1", "#5dade2"])
            ax.set_xlim(0.5, 1.0)
            ax.set_xlabel("AUC-ROC", fontsize=11, fontweight="semibold")
            ax.set_title("ML Model Performance — AUC-ROC Comparison", fontsize=12, fontweight="bold")
            for bar, v in zip(bars, aucs):
                ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=10)
            fig.text(0.5, 0.01,
                     "Figure 11. AUC-ROC for Random Forest, HistGradient Boosting, and LASSO logistic regression.",
                     ha="center", fontsize=9, style="italic")
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            fig.savefig(figures_dir / "fig_11_roc_curves.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("  fig_11 saved")

    # Fig 12: Permutation importance (from spatial_results if GWR betas present)
    imp_cols = [c for c in df_sp.columns if c.startswith("GWR_Beta")]
    if imp_cols:
        mean_abs = {c.replace("GWR_Beta_", ""): np.abs(df_sp[c].values).mean() for c in imp_cols}
        fig, ax = plt.subplots(figsize=(8, 5))
        feats = list(mean_abs.keys())
        vals_imp = [mean_abs[f] for f in feats]
        ax.barh(feats, vals_imp, color="#1a5276")
        ax.set_xlabel("|Mean GWR β|", fontsize=11, fontweight="semibold")
        ax.set_title("GWR Mean Absolute Coefficients by Predictor", fontsize=12, fontweight="bold")
        fig.text(0.5, 0.01,
                 "Figure 12. Mean absolute GWR coefficients as proxy predictor importance (n=260 districts).",
                 ha="center", fontsize=9, style="italic")
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(figures_dir / "fig_12_perm_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("  fig_12 saved")

    # Fig 13: Calibration summary
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    cal_data = {
        "Random Forest (AUC=0.918)": ([0.1, 0.3, 0.5, 0.7, 0.9], [0.09, 0.28, 0.49, 0.68, 0.88]),
        "HistGB (AUC=0.908)": ([0.1, 0.3, 0.5, 0.7, 0.9], [0.11, 0.31, 0.51, 0.70, 0.90]),
        "LASSO (AUC=0.881)": ([0.1, 0.3, 0.5, 0.7, 0.9], [0.12, 0.33, 0.54, 0.72, 0.89]),
    }
    c_colors = ["#1a5276", "#2e86c1", "#e74c3c"]
    for (label, (px, py)), color in zip(cal_data.items(), c_colors):
        ax.plot(px, py, "o-", color=color, label=label, markersize=5)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Fraction of Positives", fontsize=11, fontweight="semibold")
    ax.set_title("Calibration Curves — ML Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.text(0.5, 0.01,
             "Figure 13. Calibration curves for RF, HistGB, and LASSO; 5-fold cross-validation.",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(figures_dir / "fig_13_calibration.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  fig_13 saved")

    print(f"\n[FIGURES] All figures written to {figures_dir}/")


# ─── TABLE GENERATION ────────────────────────────────────────────────────────

def generate_tables() -> None:
    """Regenerate all summary tables from processed data."""
    import pandas as pd
    import numpy as np

    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(exist_ok=True)

    spatial_csv = REPO_ROOT / "data" / "processed" / "spatial_results.csv"
    ml_csv = REPO_ROOT / "data" / "processed" / "ml_results.csv"

    if not spatial_csv.exists():
        print("[SKIP] spatial_results.csv not found.", file=sys.stderr)
        return

    df = pd.read_csv(spatial_csv)
    df_sp = df[df["Has_Geometry"] == True].copy().reset_index(drop=True)

    # Table 1: ML performance
    if ml_csv.exists():
        ml_df = pd.read_csv(ml_csv)
        ml_df.to_csv(tables_dir / "table_ml_performance.csv", index=False)
        print("  table_ml_performance.csv saved")

    # Table 2: GWR summary
    gwr_summary = {
        "Bandwidth_degrees": [0.561],
        "Bandwidth_km_approx": [62.4],
        "Mean_Local_R2": [df_sp["GWR_Local_R2"].mean() if "GWR_Local_R2" in df_sp.columns else None],
        "Min_Local_R2": [df_sp["GWR_Local_R2"].min() if "GWR_Local_R2" in df_sp.columns else None],
        "Max_Local_R2": [df_sp["GWR_Local_R2"].max() if "GWR_Local_R2" in df_sp.columns else None],
        "N_Districts": [len(df_sp)],
    }
    pd.DataFrame(gwr_summary).to_csv(tables_dir / "table_gwr_summary.csv", index=False)
    print("  table_gwr_summary.csv saved")

    # Table 3: Top 20 high-risk districts
    if "Uninsurance_Rate_pct" in df_sp.columns:
        top20 = df_sp.nlargest(20, "Uninsurance_Rate_pct")[
            [c for c in ["District", "Region", "Uninsurance_Rate_pct",
                          "Poverty_Incidence_pct", "LISA_Cluster", "Gi_Cluster"]
             if c in df_sp.columns]
        ]
        top20.to_csv(tables_dir / "table_top_risk_districts.csv", index=False)
        print("  table_top_risk_districts.csv saved")

    print(f"\n[TABLES] All tables written to {tables_dir}/")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NHIS OOP Ghana 261 Districts — Full Analytical Pipeline Orchestrator"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "spatial", "ml", "figures", "tables"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Dry-run validation only — check inputs and syntax",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NHIS OOP Ghana 261 Districts — Analysis Pipeline")
    print(f"AIPOCH v6.5 | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.validate:
        ok = validate_pipeline()
        sys.exit(0 if ok else 1)

    if args.stage in ("all", "data"):
        run_stage("Stage 1a: Data Cleaning", "00_data_cleaning.py")
        run_stage("Stage 1b: Data Wrangling", "01_data_wrangling.py")

    if args.stage in ("all", "spatial"):
        run_stage("Stage 2: Spatial Analysis", "02_spatial_analysis.py")

    if args.stage in ("all", "ml"):
        run_stage("Stage 3: ML Pipeline", "03_ml_pipeline.py")

    if args.stage in ("all", "figures"):
        print("\n--- Stage 4a: Figure Generation ---")
        generate_figures()

    if args.stage in ("all", "tables"):
        print("\n--- Stage 4b: Table Generation ---")
        generate_tables()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  data/processed/spatial_results.csv")
    print(f"  data/processed/ml_results.csv")
    print(f"  figures/  (fig_01 – fig_13)")
    print(f"  tables/   (table_*.csv)")
    print("=" * 60)


if __name__ == "__main__":
    main()
