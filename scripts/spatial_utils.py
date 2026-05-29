"""
scripts/spatial_utils.py — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
Reusable spatial analysis utility functions.
Extracted from 02_spatial_analysis.py for use across pipeline stages.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


# ─── SPATIAL WEIGHTS ─────────────────────────────────────────────────────────

def load_geojson(path: str) -> list[dict[str, Any]]:
    """Load GeoJSON features from a file."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    return gj["features"]


def extract_vertices(feature: dict[str, Any], decimals: int = 4) -> set[tuple[float, float]]:
    """Extract coordinate vertex set from a GeoJSON feature (Polygon or MultiPolygon)."""
    geom = feature["geometry"]
    coords: list[list[list[float]]] = []
    if geom["type"] == "Polygon":
        coords = geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for part in geom["coordinates"]:
            coords.extend(part)
    vertices: set[tuple[float, float]] = set()
    for ring in coords:
        for pt in ring:
            vertices.add((round(pt[0], decimals), round(pt[1], decimals)))
    return vertices


def build_bbox(feature: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) bounding box for a GeoJSON feature."""
    geom = feature["geometry"]
    all_pts: list[list[float]] = []
    if geom["type"] == "Polygon":
        for ring in geom["coordinates"]:
            all_pts.extend(ring)
    elif geom["type"] == "MultiPolygon":
        for part in geom["coordinates"]:
            for ring in part:
                all_pts.extend(ring)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return min(xs), min(ys), max(xs), max(ys)


def build_queen(
    features: list[dict[str, Any]],
    decimals: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Queen contiguity spatial weights matrix from GeoJSON features.

    Two districts share a Queen link if their rounded coordinate vertex sets
    share at least one point. Bounding-box pre-filtering applied for efficiency.
    Produces an identical adjacency definition to spdep::poly2nb (Queen=TRUE).

    Parameters
    ----------
    features : list of GeoJSON feature dicts (already filtered to Has_Geometry=True)
    decimals : rounding precision for vertex coordinates (default 4, ≈11 m)

    Returns
    -------
    W_bin : float32 ndarray (n × n), binary adjacency
    W_std : float32 ndarray (n × n), row-standardised
    """
    n = len(features)
    bboxes = [build_bbox(f) for f in features]
    vertices = [extract_vertices(f, decimals) for f in features]

    W_bin = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        minx_i, miny_i, maxx_i, maxy_i = bboxes[i]
        for j in range(i + 1, n):
            minx_j, miny_j, maxx_j, maxy_j = bboxes[j]
            # Bounding-box pre-filter
            if (maxx_i < minx_j or maxx_j < minx_i or
                    maxy_i < miny_j or maxy_j < miny_i):
                continue
            if vertices[i] & vertices[j]:
                W_bin[i, j] = 1.0
                W_bin[j, i] = 1.0

    # Row-standardise
    row_sums = W_bin.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid divide-by-zero (islands)
    W_std = (W_bin / row_sums).astype(np.float32)
    return W_bin, W_std


# ─── GLOBAL MORAN'S I ────────────────────────────────────────────────────────

def moran_i(
    z: np.ndarray,
    W_std: np.ndarray,
    n_perms: int = 999,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute Global Moran's I with permutation inference.

    Parameters
    ----------
    z : standardised outcome vector (n,)
    W_std : row-standardised weights matrix (n, n)
    n_perms : number of permutations
    seed : random seed

    Returns
    -------
    dict with keys: I, EI, z_score, p_value
    """
    rng = np.random.default_rng(seed)
    n = len(z)
    I_obs = float((z @ W_std @ z) / (z @ z))
    EI = -1.0 / (n - 1)

    # Vectorised permutation: (n_perms, n) × (n, n) → (n_perms, n) → dot with z
    Zp = rng.permuted(np.tile(z, (n_perms, 1)), axis=1).astype(np.float32)
    WZp = Zp @ W_std.T  # (n_perms, n)
    I_perm = (WZp * Zp).sum(axis=1) / (Zp * Zp).sum(axis=1)

    z_score = (I_obs - I_perm.mean()) / I_perm.std()
    p_value = float((I_perm >= I_obs).sum() + 1) / (n_perms + 1)
    return {"I": I_obs, "EI": EI, "z_score": float(z_score), "p_value": p_value}


# ─── LOCAL MORAN'S I (LISA) ──────────────────────────────────────────────────

def local_moran(
    z: np.ndarray,
    W_std: np.ndarray,
    n_perms: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute Local Moran's I statistics with cluster classification.

    Returns
    -------
    DataFrame with columns: Ii, p_value, cluster (HH/LL/HL/LH/NS)
    """
    rng = np.random.default_rng(seed)
    n = len(z)
    Ii_obs = z * (W_std @ z)

    Zp = rng.permuted(np.tile(z, (n_perms, 1)), axis=1).astype(np.float32)
    WZp = Zp @ W_std.T
    Ii_perm = Zp * WZp  # (n_perms, n)

    p_vals = ((Ii_perm >= Ii_obs).sum(axis=0) + 1) / (n_perms + 1)

    Wz = W_std @ z
    cluster = np.where(
        p_vals >= 0.05, "NS",
        np.where(z > 0, np.where(Wz > 0, "HH", "HL"),
                          np.where(Wz < 0, "LH", "LL"))
    )

    return pd.DataFrame({"Ii": Ii_obs, "p_value": p_vals, "cluster": cluster})


# ─── BIVARIATE LISA ──────────────────────────────────────────────────────────

def bv_lisa(
    x_std: np.ndarray,
    y_std: np.ndarray,
    W_std: np.ndarray,
    n_perms: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bivariate LISA: Ii = x_i × (W_std[i,:] @ y). Permutes y, fixes x.

    Returns
    -------
    DataFrame with columns: Ii, p_value, cluster
    """
    rng = np.random.default_rng(seed)
    n = len(x_std)
    Wy = W_std @ y_std
    Ii_obs = x_std * Wy

    Yp = rng.permuted(np.tile(y_std, (n_perms, 1)), axis=1).astype(np.float32)
    WYp = Yp @ W_std.T
    Ii_perm = x_std * WYp

    p_vals = ((Ii_perm >= Ii_obs).sum(axis=0) + 1) / (n_perms + 1)

    cluster = np.where(
        p_vals >= 0.05, "NS",
        np.where(x_std > 0, np.where(Wy > 0, "HH", "HL"),
                              np.where(Wy < 0, "LH", "LL"))
    )

    return pd.DataFrame({"Ii": Ii_obs, "p_value": p_vals, "cluster": cluster})


# ─── GETIS-ORD Gi* ───────────────────────────────────────────────────────────

def gistar(x: np.ndarray, W_bin: np.ndarray) -> pd.DataFrame:
    """
    Compute Getis-Ord Gi* with self-inclusive weights.

    Returns
    -------
    DataFrame with columns: z_score, cluster (Hot_99/Hot_95/NS/Cold_95/Cold_99)
    """
    n = len(x)
    W_si = W_bin + np.eye(n, dtype=np.float32)  # self-inclusive

    x_bar = x.mean()
    s = x.std()

    Wi_sum = W_si.sum(axis=1)
    Gi = (W_si @ x - x_bar * Wi_sum) / (
        s * np.sqrt((n * (W_si ** 2).sum(axis=1) - Wi_sum ** 2) / (n - 1))
    )

    cluster = np.where(
        Gi > 2.576, "Hot_99",
        np.where(Gi > 1.960, "Hot_95",
        np.where(Gi < -2.576, "Cold_99",
        np.where(Gi < -1.960, "Cold_95", "NS")))
    )

    return pd.DataFrame({"z_score": Gi, "cluster": cluster})


# ─── GEOGRAPHICALLY WEIGHTED REGRESSION ──────────────────────────────────────

def gaussian_kernel(d: np.ndarray, bw: float) -> np.ndarray:
    """Adaptive Gaussian kernel weights."""
    return np.exp(-(d / bw) ** 2)


def gwr_fit(
    y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    bw: float,
) -> dict[str, np.ndarray]:
    """
    Fit GWR at all n locations. Adaptive Gaussian kernel.

    Parameters
    ----------
    y : outcome vector (n,)
    X : design matrix (n, p) — should include intercept column
    coords : coordinates (n, 2) in decimal degrees
    bw : bandwidth in decimal degrees

    Returns
    -------
    dict with 'betas' (n, p), 'local_r2' (n,), 'y_hat' (n,)
    """
    n, p = X.shape
    betas = np.zeros((n, p))
    y_hat = np.zeros(n)
    local_r2 = np.zeros(n)

    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
        w = gaussian_kernel(d, bw)
        W = np.diag(w)
        XtW = X.T @ W
        try:
            beta_i = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError:
            beta_i = np.linalg.lstsq(XtW @ X, XtW @ y, rcond=None)[0]
        betas[i] = beta_i
        y_hat[i] = X[i] @ beta_i

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    for i in range(n):
        y_hat_local = X @ betas[i]
        ss_res_i = ((y - y_hat_local) ** 2).sum()
        local_r2[i] = 1 - ss_res_i / ss_tot

    return {"betas": betas, "local_r2": local_r2, "y_hat": y_hat}


def gwr_loocv(
    y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    bw: float,
) -> float:
    """LOO-CV score for bandwidth selection (lower = better)."""
    n = X.shape[0]
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        d = np.sqrt(((coords[mask] - coords[i]) ** 2).sum(axis=1))
        w = gaussian_kernel(d, bw)
        W = np.diag(w)
        Xtr = X[mask]
        ytr = y[mask]
        XtW = Xtr.T @ W
        try:
            beta_i = np.linalg.solve(XtW @ Xtr, XtW @ ytr)
        except np.linalg.LinAlgError:
            beta_i = np.linalg.lstsq(XtW @ Xtr, XtW @ ytr, rcond=None)[0]
        errors[i] = (y[i] - X[i] @ beta_i) ** 2
    return float(errors.mean())


def select_bw(
    y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    bw_min: float = 0.3,
    bw_max: float = 5.0,
    nc: int = 10,
) -> float:
    """Select optimal GWR bandwidth by LOO-CV over log-spaced candidates."""
    candidates = np.logspace(np.log10(bw_min), np.log10(bw_max), nc)
    scores = [gwr_loocv(y, X, coords, bw) for bw in candidates]
    return float(candidates[int(np.argmin(scores))])
