#!/usr/bin/env python3
"""
02_spatial_analysis.py - Phase 3: Spatial Analysis
NHIS OOP Ghana 261 Districts | AIPOCH v6.5 | 2026-05-28

Pure NumPy -- no libpysal/geopandas/R required.
Permutation tests fully vectorised: Z_perms @ W_std.T in a single BLAS call.

Outputs:
  data/processed/spatial_weights_W_binary.csv
  data/processed/spatial_results.csv
  data/processed/spatial_analysis_audit.txt
  figures/fig_01 ... fig_08
"""

import json, time, warnings, pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE      = Path(__file__).parent.parent
DATA_RAW  = BASE / 'data' / 'raw'
DATA_PROC = BASE / 'data' / 'processed'
FIGURES   = BASE / 'figures'
FIGURES.mkdir(exist_ok=True)

GEOJSON_PATH   = DATA_RAW  / 'Ghana_New_260_District.geojson'
ANALYTICAL_CSV = DATA_PROC / 'district_master_261_analytical.csv'
AUDIT_PATH     = DATA_PROC / 'spatial_analysis_audit.txt'

N_PERMS    = 999
ALPHA      = 0.05
GWR_X_COLS = ['Poverty_Incidence_pct', 'Illiteracy_Rate_pct',
               'NHIS_Coverage_Women_2019_pct']
_LOG = []

def log(msg):
    ts  = datetime.now().strftime('%H:%M:%S')
    out = f'[{ts}] {msg}'
    print(out, flush=True)
    _LOG.append(out)

# ==================================================================
# GEOMETRY HELPERS
# ==================================================================

def _flat(feature):
    geom = feature['geometry']
    pts  = []
    if geom['type'] == 'Polygon':
        for ring in geom['coordinates']:
            pts.extend(ring)
    else:
        for poly in geom['coordinates']:
            for ring in poly:
                pts.extend(ring)
    return pts

def vertex_set(feature, prec=4):
    return frozenset((round(c[0], prec), round(c[1], prec))
                     for c in _flat(feature))

def bbox(feature):
    a = np.array(_flat(feature))
    return [a[:,0].min(), a[:,1].min(), a[:,0].max(), a[:,1].max()]

def bboxes_overlap(b1, b2, buf=0.001):
    return (b1[0]-buf <= b2[2] and b2[0]-buf <= b1[2] and
            b1[1]-buf <= b2[3] and b2[1]-buf <= b1[3])

# ==================================================================
# QUEEN CONTIGUITY
# ==================================================================

def build_queen(features):
    n  = len(features)
    vs = [vertex_set(f) for f in features]
    bb = [bbox(f) for f in features]
    W  = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i+1, n):
            if bboxes_overlap(bb[i], bb[j]):
                if vs[i] & vs[j]:
                    W[i,j] = W[j,i] = 1.0
    rs      = W.sum(1)
    islands = [i for i,s in enumerate(rs) if s == 0]
    d       = rs.copy(); d[d==0] = 1.0
    W_std   = W / d[:,None]
    return W.astype(np.float32), W_std.astype(np.float32), int(W.sum())//2, islands

# ==================================================================
# MORAN'S I  (vectorised permutation)
# ==================================================================

def moran_i(z, W_std, n_perms=N_PERMS):
    """
    Global Moran's I.
    Vectorised: generates (n_perms, n) permutation matrix,
    then W_std.T matmul in one BLAS call.
    """
    z   = z.astype(np.float64)
    Wz  = W_std.astype(np.float64) @ z
    I0  = float((z @ Wz) / (z @ z))

    Zp  = np.array([np.random.permutation(z) for _ in range(n_perms)])  # (P,n)
    WZp = Zp @ W_std.T.astype(np.float64)                               # (P,n)
    Ip  = (Zp * WZp).sum(1) / (Zp**2).sum(1)                            # (P,)

    mu  = float(Ip.mean()); sig = float(Ip.std())
    zsc = (I0 - mu) / sig if sig > 0 else 0.0
    pv  = float(np.mean(Ip >= I0))
    return dict(I=I0, EI=mu, z_score=zsc, p_value=pv,
                p_two=float(2*min(np.mean(Ip>=I0), np.mean(Ip<=I0))),
                perms_std=sig)

# ==================================================================
# UNIVARIATE LISA  (vectorised)
# ==================================================================

def local_moran(z, W_std, n_perms=N_PERMS):
    z    = z.astype(np.float64)
    W64  = W_std.astype(np.float64)
    lagz = W64 @ z
    Ii0  = z * lagz                                      # (n,)

    Zp   = np.array([np.random.permutation(z)
                     for _ in range(n_perms)])           # (P,n)
    WZp  = Zp @ W64.T                                   # (P,n)
    Iip  = Zp * WZp                                     # (P,n)

    psim    = np.mean(np.abs(Iip) >= np.abs(Ii0), axis=0)
    cluster = np.full(len(z), 'NS', dtype=object)
    sig = psim < ALPHA
    cluster[sig & (Ii0>0) & (z>0)] = 'HH'
    cluster[sig & (Ii0>0) & (z<0)] = 'LL'
    cluster[sig & (Ii0<0) & (z>0)] = 'HL'
    cluster[sig & (Ii0<0) & (z<0)] = 'LH'
    return dict(Ii=Ii0, lag_z=lagz, p_sim=psim, cluster=cluster)

# ==================================================================
# BIVARIATE LISA  (vectorised)
# ==================================================================

def bv_lisa(x_std, y_std, W_std, n_perms=N_PERMS):
    x, y = x_std.astype(np.float64), y_std.astype(np.float64)
    W64  = W_std.astype(np.float64)
    lagy = W64 @ y
    Ii0  = x * lagy

    Yp   = np.array([np.random.permutation(y)
                     for _ in range(n_perms)])           # (P,n)
    WYp  = Yp @ W64.T                                   # (P,n)
    Iip  = x * WYp                                      # (P,n)  broadcast x

    psim    = np.mean(np.abs(Iip) >= np.abs(Ii0), axis=0)
    cluster = np.full(len(x), 'NS', dtype=object)
    sig = psim < ALPHA
    cluster[sig & (Ii0>0) & (x>0)] = 'HH'
    cluster[sig & (Ii0>0) & (x<0)] = 'LL'
    cluster[sig & (Ii0<0) & (x>0)] = 'HL'
    cluster[sig & (Ii0<0) & (x<0)] = 'LH'
    return dict(Ii=Ii0, lag_y=lagy, p_sim=psim, cluster=cluster)

# ==================================================================
# GETIS-ORD Gi*
# ==================================================================

def gistar(x, W_bin):
    n    = len(x)
    xb   = x.mean(); s = x.std()
    Ws   = W_bin.copy(); np.fill_diagonal(Ws, 1.0)
    Ws   = Ws.astype(np.float64)
    wi   = Ws.sum(1); wi2 = (Ws**2).sum(1)
    num  = Ws @ x - xb * wi
    den  = s * np.sqrt((n*wi2 - wi**2) / (n-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        gz = np.where(den > 0, num/den, 0.0)
    cl  = np.full(n, 'NS', dtype=object)
    cl[gz >=  2.576] = 'Hot_99'
    cl[(gz >= 1.960) & (gz < 2.576)] = 'Hot_95'
    cl[gz <= -2.576] = 'Cold_99'
    cl[(gz <= -1.960) & (gz > -2.576)] = 'Cold_95'
    return gz, cl

# ==================================================================
# GWR  (Gaussian kernel, CV bandwidth selection)
# ==================================================================

def _kern(d, bw): return np.exp(-(d/bw)**2)

def gwr_fit(y, X, coords, bw):
    n, k = X.shape
    betas = np.full((n,k), np.nan); lr2 = np.full(n, np.nan)
    preds = np.full(n, np.nan)
    for i in range(n):
        d  = np.sqrt(((coords - coords[i])**2).sum(1))
        w  = _kern(d, bw)
        Xw = X * w[:,None]
        try:
            b  = np.linalg.solve(Xw.T @ X, Xw.T @ y)
        except np.linalg.LinAlgError:
            continue
        yh = X @ b; yw = np.dot(w,y)/w.sum()
        ssr= np.dot(w,(y-yh)**2); sst = np.dot(w,(y-yw)**2)
        betas[i]=b; lr2[i]=(1-ssr/sst) if sst>0 else np.nan
        preds[i]=float(X[i]@b)
    return betas, lr2, preds

def gwr_cv(y, X, coords, bw):
    n = len(y); sq = np.full(n, np.nan)
    for i in range(n):
        m  = np.ones(n, bool); m[i]=False
        d  = np.sqrt(((coords[m]-coords[i])**2).sum(1))
        w  = _kern(d, bw)
        Xm = X[m]; Xw = Xm * w[:,None]
        try:
            b = np.linalg.solve(Xw.T@Xm, Xw.T@y[m])
            sq[i]=(y[i]-float(X[i]@b))**2
        except:
            sq[i]=1e10
    return float(np.nanmean(sq))

def select_bw(y, X, coords, bmin=0.3, bmax=5.0, nc=10):
    cands  = np.geomspace(bmin, bmax, nc)
    scores = [gwr_cv(y, X, coords, bw) for bw in cands]
    best   = cands[int(np.argmin(scores))]
    return best, cands, scores

# ==================================================================
# MAP DRAWING
# ==================================================================

LISA_C = {'HH':'#d7191c','LL':'#2c7bb6','HL':'#fdae61',
           'LH':'#abd9e9','NS':'#eeeeee'}
GI_C   = {'Hot_99':'#a50026','Hot_95':'#f46d43','NS':'#eeeeee',
           'Cold_95':'#74add1','Cold_99':'#313695'}

def _rings(feat):
    geom = feat['geometry']
    if geom['type']=='Polygon':
        return [np.array(geom['coordinates'][0])[:,:2]]
    return [np.array(p[0])[:,:2] for p in geom['coordinates']]

def pc_cont(feats, vals, cmap_name, vmin, vmax):
    cmap = plt.cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ps, fc = [], []
    for feat, v in zip(feats, vals):
        for r in _rings(feat):
            ps.append(MplPolygon(r, closed=True))
            fc.append((0.85,0.85,0.85,1.) if np.isnan(v) else cmap(norm(v)))
    pc = PatchCollection(ps, facecolors=fc, edgecolors='white', linewidths=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    return pc, sm

def pc_cat(feats, cats, cmap):
    ps, fc = [], []
    for feat, cat in zip(feats, cats):
        h = cmap.get(str(cat),'#eeeeee')
        r,g,b = int(h[1:3],16)/255,int(h[3:5],16)/255,int(h[5:7],16)/255
        for ring in _rings(feat):
            ps.append(MplPolygon(ring, closed=True))
            fc.append((r,g,b,1.))
    return PatchCollection(ps, facecolors=fc, edgecolors='white', linewidths=0.25)

def setup_ax(ax):
    ax.set_xlim(-3.4,1.4); ax.set_ylim(4.4,11.3)
    ax.set_aspect('equal'); ax.axis('off')

def lisa_leg(ax):
    ax.legend(handles=[mpatches.Patch(facecolor=LISA_C[c],label=c)
                        for c in ['HH','LL','HL','LH','NS']],
              loc='lower left', fontsize=8, title='Cluster', framealpha=0.9)

def gi_leg(ax):
    ax.legend(handles=[mpatches.Patch(facecolor=GI_C[c],label=c)
                        for c in ['Hot_99','Hot_95','NS','Cold_95','Cold_99']],
              loc='lower left', fontsize=8, title='Gi* cluster', framealpha=0.9)

# ==================================================================
# MAIN
# ==================================================================

def main():
    log('='*70)
    log('PHASE 3: SPATIAL ANALYSIS -- NHIS OOP Ghana 261 Districts')
    log(f'AIPOCH v6.5 | {datetime.now().strftime("%Y-%m-%d")}')
    log('='*70)

    # --- 1. Load GeoJSON ---
    log('\n[STEP 1] Load GeoJSON')
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)
    features = gj['features']
    geo_dist = [f['properties']['DISTRICT'] for f in features]
    log(f'  Features: {len(features)}')

    # --- 2. Match analytical data ---
    log('\n[STEP 2] Match analytical data to GeoJSON order')
    df_full = pd.read_csv(ANALYTICAL_CSV)
    df_sp   = df_full[df_full['Has_Geometry']].set_index('GEO_DISTRICT')
    miss    = [d for d in geo_dist if d not in df_sp.index]
    if miss: log(f'  WARNING: unmatched {miss}')
    else:    log(f'  All {len(geo_dist)} matched')
    df_geo  = df_sp.loc[geo_dist].reset_index()
    log(f'  Spatial frame: {df_geo.shape}')

    # --- 3. Queen weights ---
    log('\n[STEP 3] Queen contiguity weights')
    t0 = time.time()
    W_bin, W_std, n_links, islands = build_queen(features)
    log(f'  Built in {time.time()-t0:.1f}s | links={n_links} | islands={len(islands)}')
    rs = W_std.sum(1)
    non_isl = np.array([i not in set(islands) for i in range(len(features))])
    assert np.allclose(rs[non_isl], 1.0, atol=1e-4), 'W_std row-sum fail'
    log(f'  Validation PASSED | mean neighbors={W_bin.sum(1).mean():.2f}')
    pd.DataFrame(W_bin.astype(np.int8), index=geo_dist, columns=geo_dist
                 ).to_csv(DATA_PROC/'spatial_weights_W_binary.csv')
    log('  Saved spatial_weights_W_binary.csv')

    # --- 4. Global Moran's I ---
    log('\n[STEP 4] Global Morans I (vectorised, 999 perms each)')
    moran_out = {}
    for col in ['Uninsurance_Rate_pct','Poverty_Incidence_pct','Illiteracy_Rate_pct']:
        x = df_geo[col].values.astype(np.float64)
        z = (x - x.mean()) / x.std()
        res = moran_i(z, W_std)
        moran_out[col] = res
        log(f'  {col}: I={res["I"]:.4f}  z={res["z_score"]:.3f}  p={res["p_value"]:.4f}')

    x_u = df_geo['Uninsurance_Rate_pct'].values.astype(np.float64)
    z_u = (x_u - x_u.mean()) / x_u.std()

    # --- 5. Univariate LISA ---
    log('\n[STEP 5] Univariate LISA (999 perms)')
    lisa = local_moran(z_u, W_std)
    for cat in ['HH','LL','HL','LH','NS']:
        log(f'  {cat}: {int(np.sum(lisa["cluster"]==cat))}')
    hh_d = [geo_dist[i] for i,c in enumerate(lisa['cluster']) if c=='HH']
    ll_d = [geo_dist[i] for i,c in enumerate(lisa['cluster']) if c=='LL']
    log(f'  HH: {hh_d}')
    log(f'  LL: {ll_d}')

    # --- 6. Bivariate LISA ---
    log('\n[STEP 6] Bivariate LISA (999 perms x2)')
    x_p = df_geo['Poverty_Incidence_pct'].values.astype(np.float64)
    z_p = (x_p - x_p.mean()) / x_p.std()
    x_i = df_geo['Illiteracy_Rate_pct'].values.astype(np.float64)
    z_i = (x_i - x_i.mean()) / x_i.std()
    bv_pov   = bv_lisa(z_u, z_p, W_std)
    bv_illit = bv_lisa(z_u, z_i, W_std)
    for lbl, res in [('UxPoverty',bv_pov),('UxIlliteracy',bv_illit)]:
        parts = ' | '.join(f'{c}:{int(np.sum(res["cluster"]==c))}'
                           for c in ['HH','LL','HL','LH','NS'])
        log(f'  [{lbl}] {parts}')

    # --- 7. Gi* ---
    log('\n[STEP 7] Getis-Ord Gi*')
    gi_z, gi_cl = gistar(x_u, W_bin)
    for cat in ['Hot_99','Hot_95','NS','Cold_95','Cold_99']:
        log(f'  {cat}: {int(np.sum(gi_cl==cat))}')
    hot_d = [geo_dist[i] for i,c in enumerate(gi_cl) if c in ('Hot_99','Hot_95')]
    log(f'  Hotspot districts p<0.05: {len(hot_d)} -> {hot_d}')

    # --- 8. GWR ---
    log('\n[STEP 8] GWR (Gaussian kernel, CV bandwidth selection)')
    coords = df_geo[['Longitude','Latitude']].values
    y_std  = (x_u - x_u.mean()) / x_u.std()
    X_raw  = df_geo[GWR_X_COLS].values
    X_std  = (X_raw - X_raw.mean(0)) / X_raw.std(0)
    X_g    = np.column_stack([np.ones(len(X_std)), X_std])
    col_nm = ['intercept'] + GWR_X_COLS

    Xt = X_g.T; b_ols = np.linalg.solve(Xt@X_g, Xt@y_std)
    r2_ols = float(1 - ((y_std - X_g@b_ols)**2).sum() /
                       ((y_std - y_std.mean())**2).sum())
    log(f'  Global OLS R2: {r2_ols:.4f}')

    log('  CV bandwidth selection (10 candidates)...')
    best_bw, cands, cv_sc = select_bw(y_std, X_g, coords, 0.3, 5.0, 10)
    for bw, sc in zip(cands, cv_sc):
        log(f'    bw={bw:.3f}  CV={sc:.6f}')
    log(f'  Optimal bw: {best_bw:.3f} deg ({best_bw*111:.0f} km)')

    log('  Fitting GWR at 260 locations...')
    betas_g, lr2, preds_g = gwr_fit(y_std, X_g, coords, best_bw)
    mlr2 = float(np.nanmean(lr2))
    log(f'  Mean local R2: {mlr2:.4f}  range [{np.nanmin(lr2):.3f},{np.nanmax(lr2):.3f}]')
    for ci, cn in enumerate(col_nm):
        b = betas_g[:,ci]
        log(f'  beta_{cn}: {np.nanmean(b):.4f} [{np.nanmin(b):.3f},{np.nanmax(b):.3f}]')

    # --- 9. Compile spatial_results.csv ---
    log('\n[STEP 9] Compile spatial_results.csv')
    sr = df_geo[['GEO_DISTRICT']].copy()
    sr['Uninsurance_z']           = z_u
    sr['Spatial_Lag_Uninsurance'] = lisa['lag_z']
    sr['LISA_Ii']                 = lisa['Ii']
    sr['LISA_p']                  = lisa['p_sim']
    sr['LISA_cluster']            = lisa['cluster']
    sr['BV_Poverty_Ii']           = bv_pov['Ii']
    sr['BV_Poverty_p']            = bv_pov['p_sim']
    sr['BV_Poverty_cluster']      = bv_pov['cluster']
    sr['BV_Illiteracy_Ii']        = bv_illit['Ii']
    sr['BV_Illiteracy_p']         = bv_illit['p_sim']
    sr['BV_Illiteracy_cluster']   = bv_illit['cluster']
    sr['Gistar_z']                = gi_z
    sr['Gistar_cluster']          = gi_cl
    sr['GWR_local_R2']            = lr2
    for ci, cn in enumerate(col_nm):
        sr[f'GWR_beta_{cn.replace("_pct","")}'] = betas_g[:,ci]
    df_out = df_full.merge(sr, on='GEO_DISTRICT', how='left')
    df_out.to_csv(DATA_PROC/'spatial_results.csv', index=False)
    log(f'  spatial_results.csv: {df_out.shape}')
    log(f'  Guan NaN check: {df_out.loc[df_out.GEO_DISTRICT=="GUAN","LISA_cluster"].isna().all()}')

    # --- 10. Figures ---
    log('\n[STEP 10] Figures')

    def savefig(name):
        plt.tight_layout()
        plt.savefig(FIGURES/name, dpi=300, bbox_inches='tight')
        plt.close()
        log(f'  Saved {name}')

    # fig_01
    fig, ax = plt.subplots(figsize=(7,8))
    pc, sm = pc_cont(features, x_u, 'YlOrRd', 5, 75)
    ax.add_collection(pc); setup_ax(ax)
    plt.colorbar(sm, ax=ax, label='Uninsurance Rate (%)', fraction=0.033, pad=0.04)
    ax.set_title('NHIS Uninsurance Rate (PHC 2021)\nAcross 261 Ghanaian Districts',
                 fontsize=11, fontweight='semibold')
    savefig('fig_01_uninsurance_choropleth.png')

    # fig_02
    fig, ax = plt.subplots(figsize=(6,6))
    cc = [LISA_C.get(str(c),'#cccccc') for c in lisa['cluster']]
    ax.scatter(z_u, lisa['lag_z'], c=cc, s=22, alpha=0.85, edgecolors='w', lw=0.3)
    ax.axhline(0, color='grey', lw=0.8, ls='--'); ax.axvline(0, color='grey', lw=0.8, ls='--')
    xl = np.linspace(z_u.min(), z_u.max(), 100)
    ax.plot(xl, np.polyval(np.polyfit(z_u, lisa['lag_z'], 1), xl), 'k-', lw=1.5)
    mr = moran_out['Uninsurance_Rate_pct']
    ax.set_title(f"Moran's I: I={mr['I']:.4f}  z={mr['z_score']:.3f}  p={mr['p_value']:.4f}",
                 fontsize=10, fontweight='semibold')
    ax.set_xlabel('Uninsurance (standardized)', fontweight='semibold', fontsize=10)
    ax.set_ylabel('Spatial Lag (standardized)', fontweight='semibold', fontsize=10)
    for cat, col in LISA_C.items():
        if cat != 'NS': ax.scatter([],[],c=col,label=cat,s=40)
    ax.legend(title='Cluster', fontsize=8)
    savefig('fig_02_moran_scatter.png')

    # fig_03
    fig, ax = plt.subplots(figsize=(7,8))
    ax.add_collection(pc_cat(features, lisa['cluster'], LISA_C))
    setup_ax(ax); lisa_leg(ax)
    ax.set_title('Univariate LISA: Uninsurance\n(p < 0.05, 999 perms)',
                 fontsize=11, fontweight='semibold')
    savefig('fig_03_lisa_clusters.png')

    # fig_04
    fig, ax = plt.subplots(figsize=(7,8))
    ax.add_collection(pc_cat(features, bv_pov['cluster'], LISA_C))
    setup_ax(ax); lisa_leg(ax)
    ax.set_title('Bivariate LISA: Uninsurance × Poverty\n(p < 0.05, 999 perms)',
                 fontsize=11, fontweight='semibold')
    savefig('fig_04_bv_lisa_poverty.png')

    # fig_05
    fig, ax = plt.subplots(figsize=(7,8))
    ax.add_collection(pc_cat(features, bv_illit['cluster'], LISA_C))
    setup_ax(ax); lisa_leg(ax)
    ax.set_title('Bivariate LISA: Uninsurance × Illiteracy\n(p < 0.05, 999 perms)',
                 fontsize=11, fontweight='semibold')
    savefig('fig_05_bv_lisa_illiteracy.png')

    # fig_06
    fig, ax = plt.subplots(figsize=(7,8))
    ax.add_collection(pc_cat(features, gi_cl, GI_C))
    setup_ax(ax); gi_leg(ax)
    ax.set_title('Getis-Ord Gi* Hotspot Map\nNHIS Uninsurance Rate',
                 fontsize=11, fontweight='semibold')
    savefig('fig_06_gistar_hotspots.png')

    # fig_07
    fig, ax = plt.subplots(figsize=(7,8))
    pc, sm = pc_cont(features, lr2, 'RdYlGn', 0.0, 1.0)
    ax.add_collection(pc); setup_ax(ax)
    plt.colorbar(sm, ax=ax, label='Local R²', fraction=0.033, pad=0.04)
    ax.set_title(f'GWR Local R² (bw={best_bw:.2f}°)\n'
                 f'Uninsurance ~ Poverty + Illiteracy + NHIS Women',
                 fontsize=11, fontweight='semibold')
    savefig('fig_07_gwr_local_r2.png')

    # fig_08
    fig, ax = plt.subplots(figsize=(7,8))
    pb   = betas_g[:,1]
    vabs = float(np.nanpercentile(np.abs(pb), 95))
    pc, sm = pc_cont(features, pb, 'RdBu_r', -vabs, vabs)
    ax.add_collection(pc); setup_ax(ax)
    plt.colorbar(sm, ax=ax, label='GWR Poverty Coefficient (β)',
                 fraction=0.033, pad=0.04)
    ax.set_title('GWR Local Poverty Coefficient\n(+ve = Higher Poverty → Higher Uninsurance)',
                 fontsize=11, fontweight='semibold')
    savefig('fig_08_gwr_coef_poverty.png')

    # --- 11. Audit ---
    log('\n[STEP 11] Writing audit log')
    log('='*70); log('PHASE 3 COMPLETE'); log('='*70)
    mr_ = moran_out['Uninsurance_Rate_pct']
    log(f"  Moran's I: {mr_['I']:.4f}  z={mr_['z_score']:.3f}  p={mr_['p_value']:.4f}")
    log(f"  Moran's I Poverty: {moran_out['Poverty_Incidence_pct']['I']:.4f}")
    log(f"  Moran's I Illiteracy: {moran_out['Illiteracy_Rate_pct']['I']:.4f}")
    for cat in ['HH','LL','HL','LH','NS']:
        log(f'  LISA {cat}: {int(np.sum(lisa["cluster"]==cat))}')
    log(f'  Gi* hot p<0.05: {int(np.sum(gi_cl=="Hot_95")+np.sum(gi_cl=="Hot_99"))}')
    log(f'  Gi* hot p<0.01: {int(np.sum(gi_cl=="Hot_99"))}')
    log(f'  GWR bw: {best_bw:.3f} deg ({best_bw*111:.0f} km)')
    log(f'  GWR mean local R2: {mlr2:.4f}  OLS R2: {r2_ols:.4f}')

    with open(AUDIT_PATH, 'w') as f:
        f.write('\n'.join(_LOG) + '\n')
    log(f'Audit -> {AUDIT_PATH}')


if __name__ == '__main__':
    main()
