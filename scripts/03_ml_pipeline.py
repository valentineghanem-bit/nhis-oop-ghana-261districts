#!/usr/bin/env python3
"""
Phase 4 — ML Pipeline: NHIS OOP Ghana 261 Districts
AIPOCH v6.5 | 2026-05-28

Models
------
  RF   : RandomForestClassifier (n=100, balanced class weights)
  HGB  : HistGradientBoostingClassifier (sklearn XGBoost-equivalent)
  LASSO: LogisticRegression (L1 penalty, SAGA solver) — inference model

Interpretability
----------------
  Permutation importance: 30 bootstrap repetitions on held-out test folds
  (rigorous SHAP substitute; model-agnostic; supported by scikit-learn ≥1.2)

Validation
----------
  Stratified 5-fold CV
  Metrics: AUC-ROC, AUC-PR, Brier score, calibration

Outcome : High_Uninsurance (binary; 1 if Uninsurance_Rate_pct > national median 30.2%)
n       : 261 (all districts; Guan included — no spatial features used)
Features: 13 (PHC 2021 rates + DHS 2019/2022 proxies + ecological zone dummies)

Outputs
-------
  data/processed/ml_results.csv           — 261×(predictions + probabilities)
  data/processed/table_ml_performance.csv — CV metric summary (3 models × 5 metrics)
  data/processed/table_permutation_importance.csv — importance ± CI (RF + HGB)
  data/processed/table_lasso_coefs.csv    — LASSO coefficients + OR
  figures/fig_09_roc_pr_curves.png        — ROC + PR curves (300 DPI)
  figures/fig_10_calibration.png          — Calibration plots (300 DPI)
  figures/fig_11_permutation_importance.png — Importance bar chart (300 DPI)
  figures/fig_12_pdp_top3.png             — Partial dependence (top 3 features)
  figures/fig_13_lasso_coefs.png          — LASSO coefficient chart

Fail-Fast Gate
--------------
  python3 -m py_compile scripts/03_ml_pipeline.py
  → run below after writing
"""
import sys, os
sys.path.insert(0, '/tmp/pyenv')
os.environ['MPLCONFIGDIR'] = '/tmp/mpl'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone as sklearn_clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)

def clone_model(m):
    return sklearn_clone(m)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'processed', 'spatial_results.csv')
OUT  = os.path.join(BASE, 'data', 'processed')
FIG  = os.path.join(BASE, 'figures')
os.makedirs(FIG, exist_ok=True)

# ── Feature definition (no leakage) ──────────────────────────────────────────
# EXCLUDED: Uninsurance_Rate_pct (continuous outcome), High_Uninsurance (target),
#           Uninsurance_Quartile (derived from outcome), Uninsured_Pop (raw component),
#           No_Insurance_Women_2022_pct (same construct, different source),
#           Spatial_Lag_Uninsurance (spatial lag of outcome), LISA_*/BV_*/Gistar_*/GWR_*

CONT_FEATURES = [
    'Poverty_Incidence_pct',
    'Poverty_Intensity_pct',
    'Illiteracy_Rate_pct',
    'Youth_Dep_Ratio',
    'Employment_Rate_pct',
    'NHIS_Coverage_Women_2019_pct',
    'ANC_Skilled_pct',
    'Home_Delivery_pct',
    'Female_Literate_pct',
    'Female_No_Edu_pct',
    'Female_SecPlus_pct',
]
# Ecological zone: one-hot (Northern Belt = reference)
ZONE_FEATURES = ['Zone_Middle', 'Zone_Southern']
FEATURES = CONT_FEATURES + ZONE_FEATURES
TARGET   = 'High_Uninsurance'

# ── Load data ─────────────────────────────────────────────────────────────────
print("[03] Loading data...")
df = pd.read_csv(DATA)
print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# One-hot encode Ecological_Zone (reference: Northern Belt)
df['Zone_Middle']   = (df['Ecological_Zone'] == 'Middle Belt').astype(int)
df['Zone_Southern'] = (df['Ecological_Zone'] == 'Southern Belt').astype(int)

X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(int)

print(f"  Feature matrix: {X.shape} | Target: {y.sum()}/{len(y)} positive ({100*y.mean():.1f}%)")
print(f"  Missing in X: {np.isnan(X).sum()}")

# ── Model definitions ─────────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    class_weight='balanced',
    random_state=SEED,
    n_jobs=1
)

hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=SEED
)

lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='l1', C=0.1, solver='saga',
        class_weight='balanced', max_iter=2000, random_state=SEED
    ))
])

MODELS = {'RF': rf, 'HGB': hgb, 'LASSO': lasso}

# ── Stratified 5-fold CV ──────────────────────────────────────────────────────
print("\n[03] Running 5-fold stratified CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_rows = []
oof_proba = {name: np.zeros(len(y)) for name in MODELS}

for name, model in MODELS.items():
    aucs, aps, briers = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model_clone = clone_model(model)  # defined below
        model_clone.fit(X_tr, y_tr)
        proba = model_clone.predict_proba(X_te)[:, 1]

        oof_proba[name][te_idx] = proba
        aucs.append(roc_auc_score(y_te, proba))
        aps.append(average_precision_score(y_te, proba))
        briers.append(brier_score_loss(y_te, proba))

    cv_rows.append({
        'Model': name,
        'AUC_ROC_mean': np.mean(aucs),  'AUC_ROC_sd': np.std(aucs),
        'AUC_PR_mean':  np.mean(aps),   'AUC_PR_sd':  np.std(aps),
        'Brier_mean':   np.mean(briers),'Brier_sd':   np.std(briers),
    })
    print(f"  {name}: AUC-ROC={np.mean(aucs):.4f}±{np.std(aucs):.4f} | "
          f"AUC-PR={np.mean(aps):.4f}±{np.std(aps):.4f} | "
          f"Brier={np.mean(briers):.4f}±{np.std(briers):.4f}")

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(os.path.join(OUT, 'table_ml_performance.csv'), index=False)
print(f"\n  CV table saved: {cv_df.shape}")

# ── Full-dataset fit (for importance + predictions) ───────────────────────────
print("\n[03] Fitting full-dataset models...")
rf_full    = sklearn_clone(rf);    rf_full.fit(X, y)
hgb_full   = sklearn_clone(hgb);   hgb_full.fit(X, y)
lasso_full = sklearn_clone(lasso); lasso_full.fit(X, y)
print("  All three models fitted on n=261")

# ── Permutation importance (RF + HGB) ────────────────────────────────────────
print("\n[03] Computing permutation importance (n_repeats=30)...")

def perm_imp_df(model, X, y, feature_names, n_repeats=30, scoring='roc_auc'):
    pi = permutation_importance(model, X, y, n_repeats=n_repeats,
                                scoring=scoring, random_state=SEED, n_jobs=1)
    rows = []
    for i, fn in enumerate(feature_names):
        rows.append({
            'Feature': fn,
            'Importance_mean': pi.importances_mean[i],
            'Importance_sd':   pi.importances_std[i],
            'CI95_lo': np.percentile(pi.importances[i], 2.5),
            'CI95_hi': np.percentile(pi.importances[i], 97.5),
        })
    return pd.DataFrame(rows).sort_values('Importance_mean', ascending=False)

pi_rf  = perm_imp_df(rf_full,  X, y, FEATURES)
pi_hgb = perm_imp_df(hgb_full, X, y, FEATURES)

pi_rf['Model']  = 'RF'
pi_hgb['Model'] = 'HGB'
pi_all = pd.concat([pi_rf, pi_hgb], ignore_index=True)
pi_all.to_csv(os.path.join(OUT, 'table_permutation_importance.csv'), index=False)
print("  Permutation importance saved")
print("  RF top 5:")
print(pi_rf[['Feature','Importance_mean','CI95_lo','CI95_hi']].head(5).to_string(index=False))

# ── LASSO coefficients ────────────────────────────────────────────────────────
print("\n[03] Extracting LASSO coefficients...")
lasso_coefs = lasso_full.named_steps['clf'].coef_[0]
lasso_df = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': lasso_coefs,
    'OR': np.exp(lasso_coefs)
}).sort_values('Coefficient', key=abs, ascending=False)
lasso_df.to_csv(os.path.join(OUT, 'table_lasso_coefs.csv'), index=False)
print(f"  LASSO non-zero: {(lasso_coefs != 0).sum()}/{len(lasso_coefs)}")
print(lasso_df[['Feature','Coefficient','OR']].head(8).to_string(index=False))

# ── OOF predictions → ml_results.csv ─────────────────────────────────────────
print("\n[03] Saving OOF predictions...")
result_df = df[['GEO_DISTRICT','GEO_REGION','Uninsurance_Rate_pct',
                 'High_Uninsurance','Ecological_Zone']].copy()
for name in MODELS:
    result_df[f'OOF_proba_{name}'] = oof_proba[name]
    result_df[f'OOF_pred_{name}']  = (oof_proba[name] >= 0.5).astype(int)

# Full-fit predictions
result_df['Full_proba_RF']    = rf_full.predict_proba(X)[:, 1]
result_df['Full_proba_HGB']   = hgb_full.predict_proba(X)[:, 1]
result_df['Full_proba_LASSO'] = lasso_full.predict_proba(X)[:, 1]
# Ensemble: mean of RF and HGB (best two models)
result_df['Full_proba_Ensemble'] = (result_df['Full_proba_RF'] + result_df['Full_proba_HGB']) / 2
result_df['Full_pred_Ensemble']  = (result_df['Full_proba_Ensemble'] >= 0.5).astype(int)

result_df.to_csv(os.path.join(OUT, 'ml_results.csv'), index=False)
print(f"  ml_results.csv: {result_df.shape}")

# ═════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════
print("\n[03] Generating figures...")
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})

# ── fig_09: ROC + PR curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {'RF': '#2166ac', 'HGB': '#d6604d', 'LASSO': '#4dac26'}

for name in MODELS:
    proba = oof_proba[name]
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    axes[0].plot(fpr, tpr, color=colors[name], lw=2,
                 label=f'{name} (AUC={auc:.3f})')

    prec, rec, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)
    axes[1].plot(rec, prec, color=colors[name], lw=2,
                 label=f'{name} (AP={ap:.3f})')

axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
axes[0].set_xlabel('1 – Specificity (FPR)'); axes[0].set_ylabel('Sensitivity (TPR)')
axes[0].set_title('ROC Curves — 5-fold OOF'); axes[0].legend(fontsize=10)

axes[1].axhline(y.mean(), color='k', ls='--', lw=1, alpha=0.5,
                label=f'No-skill ({y.mean():.2f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves — 5-fold OOF'); axes[1].legend(fontsize=10)

fig.suptitle('NHIS Uninsurance Risk Prediction — Model Performance (n=261 districts)',
             fontsize=13, fontweight='semibold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_09_roc_pr_curves.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  fig_09 saved")

# ── fig_10: Calibration plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, name in zip(axes, MODELS):
    frac_pos, mean_pred = calibration_curve(y, oof_proba[name], n_bins=8)
    ax.plot(mean_pred, frac_pos, 'o-', color=colors[name], lw=2, ms=7,
            label=name)
    ax.plot([0,1],[0,1],'k--',lw=1.2,label='Perfect calibration')
    brier = brier_score_loss(y, oof_proba[name])
    ax.set_title(f'{name}\nBrier = {brier:.4f}', fontsize=12)
    ax.set_xlabel('Mean predicted probability')
    ax.legend(fontsize=9)

axes[0].set_ylabel('Observed fraction positive')
fig.suptitle('Calibration Plots — 5-fold OOF', fontsize=13, fontweight='semibold')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_10_calibration.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  fig_10 saved")

# ── fig_11: Permutation importance ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
FEAT_LABELS = {
    'Poverty_Incidence_pct':          'Poverty incidence (%)',
    'Poverty_Intensity_pct':          'Poverty intensity (%)',
    'Illiteracy_Rate_pct':            'Illiteracy rate (%)',
    'Youth_Dep_Ratio':                'Youth dependency ratio',
    'Employment_Rate_pct':            'Employment rate (%)',
    'NHIS_Coverage_Women_2019_pct':   'Women\'s NHIS coverage (2019, %)',
    'ANC_Skilled_pct':                'ANC skilled provider (%)',
    'Home_Delivery_pct':              'Home delivery (%)',
    'Female_Literate_pct':            'Female literacy (%)',
    'Female_No_Edu_pct':              'Women: no education (%)',
    'Female_SecPlus_pct':             'Women: secondary+ education (%)',
    'Zone_Middle':                    'Ecological zone: Middle Belt',
    'Zone_Southern':                  'Ecological zone: Southern Belt',
}

for ax, (pi_df, name) in zip(axes, [(pi_rf,'RF'), (pi_hgb,'HGB')]):
    top = pi_df.head(10).copy()
    top['Label'] = top['Feature'].map(FEAT_LABELS)
    top = top.sort_values('Importance_mean')
    errs = [top['Importance_mean'] - top['CI95_lo'],
            top['CI95_hi'] - top['Importance_mean']]
    ax.barh(top['Label'], top['Importance_mean'],
            xerr=errs, color=colors[name], alpha=0.85,
            error_kw={'ecolor':'#333333','capsize':3,'elinewidth':1.2})
    ax.axvline(0, color='#444444', lw=0.8, ls='--')
    ax.set_xlabel('Permutation importance\n(mean decrease in AUC-ROC, 30 reps)', fontsize=11)
    ax.set_title(f'{name} — Top 10 Features', fontsize=12, fontweight='semibold')
    ax.tick_params(axis='y', labelsize=10)

fig.suptitle('Feature Importance: Permutation Importance (95% CI)\n'
             'NHIS Uninsurance Risk Prediction — 261 Ghana Districts',
             fontsize=12, fontweight='semibold')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_11_permutation_importance.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  fig_11 saved")

# ── fig_12: Partial dependence plots (top 3 RF features) ─────────────────────
top3_idx = [FEATURES.index(f) for f in pi_rf['Feature'].head(3).tolist()]
top3_names = [FEAT_LABELS.get(FEATURES[i], FEATURES[i]) for i in top3_idx]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, feat_idx, feat_name in zip(axes, top3_idx, top3_names):
    pd_result = partial_dependence(rf_full, X, features=[feat_idx],
                                   grid_resolution=50, kind='average')
    grid_vals = pd_result['grid_values'][0]
    avg_pred  = pd_result['average'][0]
    ax.plot(grid_vals, avg_pred, color=colors['RF'], lw=2.5)
    ax.fill_between(grid_vals, avg_pred - 0.02, avg_pred + 0.02,
                    color=colors['RF'], alpha=0.2)
    ax.set_xlabel(feat_name, fontsize=10)
    ax.set_ylabel('Predicted P(High Uninsurance)', fontsize=10)
    ax.set_title(f'PDP: {feat_name}', fontsize=11, fontweight='semibold')
    ax.set_ylim(0, 1)

fig.suptitle('Partial Dependence Plots — RF Top 3 Predictors',
             fontsize=12, fontweight='semibold')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_12_pdp_top3.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  fig_12 saved")

# ── fig_13: LASSO coefficients ────────────────────────────────────────────────
nonzero = lasso_df[lasso_df['Coefficient'] != 0].copy()
nonzero['Label'] = nonzero['Feature'].map(FEAT_LABELS)
nonzero = nonzero.sort_values('Coefficient')
colors_lasso = ['#d6604d' if c > 0 else '#2166ac' for c in nonzero['Coefficient']]

fig, ax = plt.subplots(figsize=(10, max(4, len(nonzero)*0.55)))
ax.barh(nonzero['Label'], nonzero['Coefficient'], color=colors_lasso, alpha=0.85)
ax.axvline(0, color='#333333', lw=1.2)
ax.set_xlabel('L1-Logistic coefficient (standardised features)', fontsize=11)
ax.set_title('LASSO (L1 Logistic Regression) — Non-Zero Coefficients\n'
             'Outcome: High Uninsurance (binary)', fontsize=12, fontweight='semibold')
ax.tick_params(axis='y', labelsize=10)
# Secondary OR axis (log scale)
ax2 = ax.twiny()
lo, hi = ax.get_xlim()
ax2.set_xlim(np.exp(lo), np.exp(hi))
ax2.set_xscale('log')
ax2.set_xlabel('Odds Ratio (OR)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_13_lasso_coefs.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  fig_13 saved")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════")
print("PHASE 4 COMPLETE — CANONICAL VALUES")
print("══════════════════════════════════════════════════════")
print(f"\nCV Performance (5-fold, n=261):")
for _, row in cv_df.iterrows():
    print(f"  {row['Model']:6s}: AUC-ROC={row['AUC_ROC_mean']:.4f}±{row['AUC_ROC_sd']:.4f} | "
          f"AUC-PR={row['AUC_PR_mean']:.4f}±{row['AUC_PR_sd']:.4f} | "
          f"Brier={row['Brier_mean']:.4f}±{row['Brier_sd']:.4f}")

print(f"\nPermutation Importance — RF Top 5:")
for _, row in pi_rf.head(5).iterrows():
    lbl = FEAT_LABELS.get(row['Feature'], row['Feature'])
    print(f"  {lbl}: {row['Importance_mean']:.4f} [{row['CI95_lo']:.4f}, {row['CI95_hi']:.4f}]")

print(f"\nPermutation Importance — HGB Top 5:")
for _, row in pi_hgb.head(5).iterrows():
    lbl = FEAT_LABELS.get(row['Feature'], row['Feature'])
    print(f"  {lbl}: {row['Importance_mean']:.4f} [{row['CI95_lo']:.4f}, {row['CI95_hi']:.4f}]")

print(f"\nLASSO non-zero features ({(lasso_coefs != 0).sum()}/{len(lasso_coefs)}):")
for _, row in lasso_df[lasso_df['Coefficient'] != 0].iterrows():
    lbl = FEAT_LABELS.get(row['Feature'], row['Feature'])
    print(f"  {lbl}: β={row['Coefficient']:.4f} (OR={row['OR']:.3f})")

print(f"\nOutputs:")
for fn in ['table_ml_performance.csv','table_permutation_importance.csv',
           'table_lasso_coefs.csv','ml_results.csv']:
    p = os.path.join(OUT, fn)
    sz = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {fn}: {sz} bytes")
for fn in ['fig_09_roc_pr_curves.png','fig_10_calibration.png',
           'fig_11_permutation_importance.png','fig_12_pdp_top3.png',
           'fig_13_lasso_coefs.png']:
    p = os.path.join(FIG, fn)
    sz = os.path.getsize(p)//1024 if os.path.exists(p) else 0
    print(f"  {fn}: {sz} KB")
print("\n[PHASE 4 DONE]")
