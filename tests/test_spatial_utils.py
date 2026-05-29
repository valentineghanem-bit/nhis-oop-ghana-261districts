"""
tests/test_spatial_utils.py — NHIS OOP Ghana 261 Districts | AIPOCH v6.5
Pytest test suite for spatial_utils.py.

Tests cover:
    - build_queen: adjacency structure and row-standardisation
    - moran_i: statistic range, expected value, p-value type
    - local_moran: shape, cluster labels, p-value bounds
    - bv_lisa: shape, cluster labels, permutation inference
    - gistar: z-score computation, cluster classification
    - gwr_fit: output shape, coefficients stability
    - select_bw: returns scalar in [bw_min, bw_max]

All tests use synthetic geometries or random data with fixed seeds.
No external files are required — tests are self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from spatial_utils import (
    bv_lisa,
    build_queen,
    gistar,
    gwr_fit,
    local_moran,
    moran_i,
    select_bw,
)


# ─── SYNTHETIC GEOMETRY FIXTURES ─────────────────────────────────────────────

def _square_feature(x0: float, y0: float, dx: float = 1.0) -> dict:
    """Create a GeoJSON Polygon feature for a square grid cell."""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [x0, y0],
                [x0 + dx, y0],
                [x0 + dx, y0 + dx],
                [x0, y0 + dx],
                [x0, y0],
            ]],
        },
        "properties": {},
    }


def _grid_features(rows: int = 3, cols: int = 3) -> list[dict]:
    """Create a rows×cols grid of unit squares (share edges and corners)."""
    return [_square_feature(float(c), float(r)) for r in range(rows) for c in range(cols)]


@pytest.fixture(scope="module")
def grid3x3():
    """3×3 grid → 9 features; Queen should produce known adjacency structure."""
    return _grid_features(3, 3)


@pytest.fixture(scope="module")
def W_3x3(grid3x3):
    return build_queen(grid3x3)


@pytest.fixture(scope="module")
def rng_data():
    """Random data with fixed seed for reproducibility."""
    rng = np.random.default_rng(0)
    n = 20
    z = rng.standard_normal(n).astype(np.float32)
    return z, n


# ─── build_queen tests ────────────────────────────────────────────────────────

class TestBuildQueen:
    def test_output_shape(self, W_3x3):
        W_bin, W_std = W_3x3
        assert W_bin.shape == (9, 9), "Binary weight matrix must be (n, n)"
        assert W_std.shape == (9, 9), "Standardised weight matrix must be (n, n)"

    def test_binary_values(self, W_3x3):
        W_bin, _ = W_3x3
        unique = np.unique(W_bin)
        assert set(unique).issubset({0.0, 1.0}), "Binary matrix must contain only 0 and 1"

    def test_symmetry(self, W_3x3):
        W_bin, _ = W_3x3
        np.testing.assert_array_equal(W_bin, W_bin.T, err_msg="Adjacency matrix must be symmetric")

    def test_no_self_loops(self, W_3x3):
        W_bin, _ = W_3x3
        assert W_bin.diagonal().sum() == 0, "No district should be adjacent to itself"

    def test_row_standardisation(self, W_3x3):
        _, W_std = W_3x3
        row_sums = W_std.sum(axis=1)
        # All rows with at least one neighbour must sum to 1
        has_neighbour = row_sums > 0
        np.testing.assert_allclose(
            row_sums[has_neighbour], 1.0, atol=1e-5,
            err_msg="Rows with neighbours must sum to 1"
        )

    def test_queen_link_count(self, W_3x3):
        """
        3×3 Queen grid: corner cells → 3 neighbours,
        edge cells → 5, centre cell → 8.
        Total unique links = 20 (from spdep reference).
        """
        W_bin, _ = W_3x3
        total_links = int(W_bin.sum())
        # Each link counted twice (symmetric matrix) → 2 × 20 = 40
        assert total_links == 40, f"Expected 40 directed links (20 undirected), got {total_links}"

    def test_centre_cell_neighbours(self, W_3x3):
        """Centre cell (index 4) has 8 Queen neighbours in 3×3 grid."""
        W_bin, _ = W_3x3
        assert W_bin[4].sum() == 8, "Centre cell must have 8 Queen neighbours"

    def test_corner_cell_neighbours(self, W_3x3):
        """Corner cells (0, 2, 6, 8) each have 3 Queen neighbours."""
        W_bin, _ = W_3x3
        for idx in [0, 2, 6, 8]:
            assert W_bin[idx].sum() == 3, f"Corner cell {idx} must have 3 Queen neighbours"

    def test_dtype(self, W_3x3):
        W_bin, W_std = W_3x3
        assert W_bin.dtype == np.float32
        assert W_std.dtype == np.float32

    def test_single_row_grid(self):
        """Two adjacent squares → exactly one undirected link."""
        feats = [_square_feature(0.0, 0.0), _square_feature(1.0, 0.0)]
        W_bin, W_std = build_queen(feats)
        assert W_bin[0, 1] == 1.0 and W_bin[1, 0] == 1.0
        np.testing.assert_allclose(W_std.sum(axis=1), [1.0, 1.0], atol=1e-5)


# ─── moran_i tests ────────────────────────────────────────────────────────────

class TestMoranI:
    def test_output_keys(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        z9 = z[:9]
        result = moran_i(z9, W_std, n_perms=99, seed=0)
        assert set(result.keys()) == {"I", "EI", "z_score", "p_value"}

    def test_I_range(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        result = moran_i(z[:9], W_std, n_perms=99, seed=0)
        assert -1.0 <= result["I"] <= 1.5, "Moran's I should be near [-1, 1]"

    def test_expected_value(self, W_3x3):
        _, W_std = W_3x3
        n = 9
        EI_expected = -1.0 / (n - 1)
        z = np.random.default_rng(1).standard_normal(n).astype(np.float32)
        result = moran_i(z, W_std, n_perms=99, seed=1)
        assert abs(result["EI"] - EI_expected) < 1e-6

    def test_p_value_bounds(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        result = moran_i(z[:9], W_std, n_perms=99, seed=0)
        assert 0.0 < result["p_value"] <= 1.0

    def test_positive_autocorrelation(self, W_3x3):
        """Perfectly clustered values → Moran's I strongly positive."""
        _, W_std = W_3x3
        # High values in centre, low at corners → spatial autocorrelation
        z = np.array([0, 0, 0, 0, 5, 0, 0, 0, 0], dtype=np.float32)
        z = (z - z.mean()) / (z.std() + 1e-8)
        result = moran_i(z, W_std, n_perms=199, seed=42)
        assert result["I"] > 0, "Clustered pattern should give positive Moran's I"

    def test_reproducibility(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        r1 = moran_i(z[:9], W_std, n_perms=99, seed=7)
        r2 = moran_i(z[:9], W_std, n_perms=99, seed=7)
        assert r1["I"] == r2["I"]
        assert r1["p_value"] == r2["p_value"]


# ─── local_moran tests ────────────────────────────────────────────────────────

class TestLocalMoran:
    def test_output_shape(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        df = local_moran(z[:9], W_std, n_perms=99, seed=0)
        assert df.shape == (9, 3), "Output must have n rows and 3 columns"

    def test_columns(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        df = local_moran(z[:9], W_std, n_perms=99, seed=0)
        assert list(df.columns) == ["Ii", "p_value", "cluster"]

    def test_cluster_labels(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        df = local_moran(z[:9], W_std, n_perms=99, seed=0)
        valid = {"HH", "LL", "HL", "LH", "NS"}
        assert set(df["cluster"].unique()).issubset(valid)

    def test_p_value_bounds(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        df = local_moran(z[:9], W_std, n_perms=99, seed=0)
        assert (df["p_value"] > 0).all() and (df["p_value"] <= 1).all()

    def test_ns_at_p_threshold(self, W_3x3):
        """Districts with p_value ≥ 0.05 must be classified as NS."""
        _, W_std = W_3x3
        rng = np.random.default_rng(99)
        # Near-uniform data → most should be NS
        z = (rng.standard_normal(9) * 0.01).astype(np.float32)
        z = (z - z.mean()) / (z.std() + 1e-8)
        df = local_moran(z, W_std, n_perms=199, seed=99)
        ns_mask = df["p_value"] >= 0.05
        assert (df.loc[ns_mask, "cluster"] == "NS").all()


# ─── bv_lisa tests ────────────────────────────────────────────────────────────

class TestBvLisa:
    def test_output_shape(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        x = z[:9]
        y = np.roll(z[:9], 2)
        df = bv_lisa(x, y, W_std, n_perms=99, seed=0)
        assert df.shape == (9, 3)

    def test_columns(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        df = bv_lisa(z[:9], z[:9], W_std, n_perms=99, seed=0)
        assert list(df.columns) == ["Ii", "p_value", "cluster"]

    def test_cluster_labels(self, W_3x3, rng_data):
        _, W_std = W_3x3
        z, _ = rng_data
        df = bv_lisa(z[:9], np.roll(z[:9], 1), W_std, n_perms=99, seed=0)
        assert set(df["cluster"].unique()).issubset({"HH", "LL", "HL", "LH", "NS"})

    def test_identical_inputs_gives_same_as_univariate(self, W_3x3):
        """BV-LISA with x=y should approximate univariate LISA."""
        _, W_std = W_3x3
        rng = np.random.default_rng(5)
        z = rng.standard_normal(9).astype(np.float32)
        z = (z - z.mean()) / (z.std() + 1e-8)
        df_bv = bv_lisa(z, z, W_std, n_perms=199, seed=5)
        df_uni = local_moran(z, W_std, n_perms=199, seed=5)
        # Ii values should match since Ii = z_i * (W @ z)_i in both cases
        np.testing.assert_allclose(df_bv["Ii"].values, df_uni["Ii"].values, atol=1e-5)


# ─── gistar tests ─────────────────────────────────────────────────────────────

class TestGiStar:
    def test_output_shape(self, W_3x3, rng_data):
        W_bin, _ = W_3x3
        z, _ = rng_data
        df = gistar(np.abs(z[:9]).astype(np.float32), W_bin.astype(np.float32))
        assert df.shape == (9, 2)

    def test_columns(self, W_3x3, rng_data):
        W_bin, _ = W_3x3
        z, _ = rng_data
        df = gistar(np.abs(z[:9]).astype(np.float32), W_bin.astype(np.float32))
        assert list(df.columns) == ["z_score", "cluster"]

    def test_cluster_labels(self, W_3x3, rng_data):
        W_bin, _ = W_3x3
        z, _ = rng_data
        df = gistar(np.abs(z[:9]).astype(np.float32), W_bin.astype(np.float32))
        valid = {"Hot_99", "Hot_95", "NS", "Cold_95", "Cold_99"}
        assert set(df["cluster"].unique()).issubset(valid)

    def test_hotspot_threshold(self, W_3x3):
        """Cell with very high value surrounded by high values → Hot_99."""
        W_bin, _ = W_3x3
        # Centre cell has extreme value
        x = np.ones(9, dtype=np.float32)
        x[4] = 1000.0  # extreme centre value
        df = gistar(x, W_bin.astype(np.float32))
        assert df.loc[4, "cluster"] == "Hot_99", "Extreme centre value should be Hot_99"

    def test_coldspot_threshold(self, W_3x3):
        """Cell with very low value surrounded by low values → Cold_99."""
        W_bin, _ = W_3x3
        x = np.ones(9, dtype=np.float32) * 100.0
        x[4] = 0.0001  # near-zero centre
        df = gistar(x, W_bin.astype(np.float32))
        assert df.loc[4, "cluster"] == "Cold_99", "Near-zero centre should be Cold_99"

    def test_uniform_input_all_ns(self, W_3x3):
        """Uniform input → all z-scores = 0 → all NS."""
        W_bin, _ = W_3x3
        x = np.ones(9, dtype=np.float32)
        df = gistar(x, W_bin.astype(np.float32))
        # With uniform x, Gi* = 0 for all locations
        assert (df["cluster"] == "NS").all()


# ─── gwr_fit tests ────────────────────────────────────────────────────────────

class TestGwrFit:
    @pytest.fixture
    def simple_gwr_data(self):
        """Simple GWR test data: linear relationship with known coefficients."""
        rng = np.random.default_rng(42)
        n = 25  # 5×5 grid
        # Coordinates on unit square grid
        rows = np.repeat(np.arange(5), 5).astype(float)
        cols = np.tile(np.arange(5), 5).astype(float)
        coords = np.column_stack([cols, rows])
        x1 = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x1])
        true_beta = np.array([2.0, 1.5])
        y = X @ true_beta + rng.standard_normal(n) * 0.1
        return y, X, coords

    def test_output_keys(self, simple_gwr_data):
        y, X, coords = simple_gwr_data
        result = gwr_fit(y, X, coords, bw=2.0)
        assert set(result.keys()) == {"betas", "local_r2", "y_hat"}

    def test_output_shapes(self, simple_gwr_data):
        y, X, coords = simple_gwr_data
        n, p = X.shape
        result = gwr_fit(y, X, coords, bw=2.0)
        assert result["betas"].shape == (n, p), "betas must be (n, p)"
        assert result["local_r2"].shape == (n,), "local_r2 must be (n,)"
        assert result["y_hat"].shape == (n,), "y_hat must be (n,)"

    def test_prediction_quality(self, simple_gwr_data):
        """GWR on near-linear data should achieve low RMSE."""
        y, X, coords = simple_gwr_data
        result = gwr_fit(y, X, coords, bw=3.0)
        rmse = np.sqrt(np.mean((y - result["y_hat"]) ** 2))
        assert rmse < 1.0, f"GWR RMSE too high: {rmse:.3f}"

    def test_coefficient_recovery(self, simple_gwr_data):
        """Mean GWR coefficients should approximate true betas."""
        y, X, coords = simple_gwr_data
        result = gwr_fit(y, X, coords, bw=3.0)
        mean_betas = result["betas"].mean(axis=0)
        np.testing.assert_allclose(mean_betas[0], 2.0, atol=0.5,
                                   err_msg="Intercept recovery off")
        np.testing.assert_allclose(mean_betas[1], 1.5, atol=0.5,
                                   err_msg="Slope recovery off")


# ─── select_bw tests ─────────────────────────────────────────────────────────

class TestSelectBw:
    @pytest.fixture
    def bw_data(self):
        rng = np.random.default_rng(10)
        n = 16
        coords = np.column_stack([
            np.tile(np.arange(4), 4).astype(float),
            np.repeat(np.arange(4), 4).astype(float),
        ])
        x = rng.standard_normal(n)
        X = np.column_stack([np.ones(n), x])
        y = 1.5 * x + rng.standard_normal(n) * 0.2
        return y, X, coords

    def test_returns_scalar(self, bw_data):
        y, X, coords = bw_data
        bw = select_bw(y, X, coords, bw_min=0.5, bw_max=3.0, nc=5)
        assert isinstance(bw, float)

    def test_within_range(self, bw_data):
        y, X, coords = bw_data
        bw = select_bw(y, X, coords, bw_min=0.5, bw_max=3.0, nc=5)
        assert 0.5 <= bw <= 3.0, f"Bandwidth {bw} outside search range [0.5, 3.0]"

    def test_reproducibility(self, bw_data):
        """select_bw is deterministic (no random component)."""
        y, X, coords = bw_data
        bw1 = select_bw(y, X, coords, bw_min=0.5, bw_max=3.0, nc=5)
        bw2 = select_bw(y, X, coords, bw_min=0.5, bw_max=3.0, nc=5)
        assert bw1 == bw2
