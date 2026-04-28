"""Tests for SpectrumReconstruction.SpectrumReconstructionBasic module."""

import numpy as np
import pandas as pd
import pytest

from SpectrumReconstruction.SpectrumReconstructionBasic import (
    _clean_pivot_training_data,
    _linear_regression,
    SpectrumReconstructionBasic,
    SpectrumReconstructionBasicHighPerformance,
)


# ========== _clean_pivot_training_data ==========

class TestCleanPivotTrainingData:

    def test_no_missing_values(self):
        """Complete data should pass through unchanged."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val': [1, 2, 3, 4],
        })
        result = _clean_pivot_training_data(df, 'row', 'col', 'val')
        assert result.shape == (2, 2)
        assert not result.isna().any().any()

    def test_drop_column_with_most_missing(self):
        """Should drop the column with the highest missing rate."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
            'col': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'],
            'val': [1, np.nan, 3, 4, np.nan, 6, 7, 8, 9],
        })
        result = _clean_pivot_training_data(df, 'row', 'col', 'val')
        assert 'Y' not in result.columns
        assert result.shape == (3, 2)
        assert not result.isna().any().any()

    def test_drop_row_with_most_missing(self):
        """Should drop the row with the highest missing rate."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'A', 'B', 'B', 'B'],
            'col': ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
            'val': [1, np.nan, 3, np.nan, np.nan, np.nan],
        })
        result = _clean_pivot_training_data(df, 'row', 'col', 'val')
        assert 'B' not in result.index
        assert not result.isna().any().any()

    def test_mixed_removal(self):
        """Should handle cases requiring both row and column removal."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B', 'C', 'C'],
            'col': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'val': [1, np.nan, np.nan, 2, 3, np.nan],
        })
        result = _clean_pivot_training_data(df, 'row', 'col', 'val')
        assert not result.isna().any().any()
        assert not result.empty

    def test_all_missing_raises(self):
        """All-NaN data should raise ValueError."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val': [np.nan, np.nan, np.nan, np.nan],
        })
        with pytest.raises(ValueError, match='No valid data'):
            _clean_pivot_training_data(df, 'row', 'col', 'val')

    def test_pass_in_pivot_data(self):
        """Should accept pre-pivoted data with pass_in_pivot_data=True."""
        pivot = pd.DataFrame(
            {'X': [1, 3], 'Y': [2, 4]},
            index=['A', 'B'],
        )
        result = _clean_pivot_training_data(
            pivot, 'row', 'col', 'val', pass_in_pivot_data=True
        )
        assert result.shape == (2, 2)

    def test_missing_columns_raises(self):
        """Missing column names should raise ValueError."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        with pytest.raises(ValueError, match='Column names not found'):
            _clean_pivot_training_data(df, 'row', 'col', 'val')


# ========== _linear_regression ==========

class TestLinearRegression:

    def setup_method(self):
        # Simple system for OLS/L1 tests (no fit_intercept concern)
        self.x_simple = np.array([[1, 0], [0, 1]], dtype=np.float64)
        self.y_simple = np.array([2, 3], dtype=np.float64)
        self.expected_simple = np.array([2, 3])
        # Larger system for sklearn methods (fit_intercept=True)
        # y ~ X @ true_coef, with enough samples for stable fits
        np.random.seed(42)
        self.true_coef = np.array([3.0, -2.0, 1.5])
        self.x_large = np.random.randn(50, 3)
        self.y_large = self.x_large @ self.true_coef + np.random.randn(50) * 0.01

    def test_normal(self):
        """OLS should recover exact solution for identity matrix."""
        result = _linear_regression(self.x_simple, self.y_simple, 'normal')
        np.testing.assert_allclose(result, self.expected_simple, atol=1e-10)

    def test_l1_zero_regularization(self):
        """L1 with lambda=0 should match OLS."""
        result = _linear_regression(self.x_simple, self.y_simple, 'l1', lambda_reg=0)
        np.testing.assert_allclose(result, self.expected_simple, atol=1e-4)

    def test_l2_recovers_coefficients(self):
        """L2 (Ridge) with small regularization should recover approximate coefficients."""
        result = _linear_regression(self.x_large, self.y_large, 'l2', lambda_reg=0.001)
        np.testing.assert_allclose(result, self.true_coef, atol=0.1)

    def test_ridge_alias(self):
        """'Ridge' should produce same result as 'l2' with same parameters."""
        r_l2 = _linear_regression(self.x_large, self.y_large, 'l2', lambda_reg=0.1)
        r_ridge = _linear_regression(self.x_large, self.y_large, 'Ridge', lambda_reg=0.1)
        np.testing.assert_allclose(r_l2, r_ridge, atol=1e-10)

    def test_l2_high_regularization_shrinks(self):
        """High L2 regularization should shrink coefficients toward zero."""
        result_low = _linear_regression(self.x_large, self.y_large, 'l2', lambda_reg=0.01)
        result_high = _linear_regression(self.x_large, self.y_large, 'l2', lambda_reg=100)
        assert np.linalg.norm(result_high) < np.linalg.norm(result_low)

    def test_elasticnet_recovers_coefficients(self):
        """ElasticNet with small regularization should recover approximate coefficients."""
        result = _linear_regression(
            self.x_large, self.y_large, 'ElasticNet', lambda_reg=0.001, alpha=0.5
        )
        np.testing.assert_allclose(result, self.true_coef, atol=0.2)

    def test_lasso(self):
        """Lasso should return a result with correct shape."""
        result = _linear_regression(self.x_large, self.y_large, 'Lasso')
        assert result.shape == (3,)

    def test_linear_regression_method(self):
        """LinearRegression should recover approximate coefficients."""
        result = _linear_regression(self.x_large, self.y_large, 'LinearRegression')
        np.testing.assert_allclose(result, self.true_coef, atol=0.1)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match='Invalid method'):
            _linear_regression(self.x_simple, self.y_simple, 'invalid')

    def test_overdetermined_system(self):
        """OLS should find least-squares solution for overdetermined system."""
        x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64)
        y = np.array([2, 3, 5.1], dtype=np.float64)
        result = _linear_regression(x, y, 'normal')
        assert result.shape == (2,)
        residual = np.linalg.norm(x @ result - y)
        assert residual < 1.0


# ========== SpectrumReconstructionBasic ==========

class TestSpectrumReconstructionBasic:

    def _make_training_data(self):
        """Create simple training data for gaussian mode."""
        rows, cols, vals = [], [], []
        for bias in [0.0, 1.0, 2.0]:
            for mu in [100.0, 200.0, 300.0]:
                rows.append(bias)
                cols.append(mu)
                vals.append(bias * mu)
        return pd.DataFrame({
            'bias': rows,
            'mu': cols,
            'value': vals,
        })

    def test_init_gaussian(self):
        """Should initialize correctly in gaussian mode."""
        df = self._make_training_data()
        srb = SpectrumReconstructionBasic(
            training_data=df,
            internal_var_col_name='bias',
            external_var_col_name='mu',
            dependent_var_col_name='value',
            base_func='gaussian',
            sigma=1e-6,
        )
        assert srb._base_func_name == 'gaussian'
        assert srb._pivot_training_data.shape == (3, 3)

    def test_init_blackbody(self):
        """Should initialize correctly in blackbody mode."""
        df = self._make_training_data()
        srb = SpectrumReconstructionBasic(
            training_data=df,
            internal_var_col_name='bias',
            external_var_col_name='mu',
            dependent_var_col_name='value',
            base_func='blackbody',
        )
        assert srb._base_func_name == 'blackbody'

    def test_invalid_base_func_raises(self):
        """Invalid base_func should raise ValueError."""
        df = self._make_training_data()
        with pytest.raises(ValueError, match='base_func must be'):
            SpectrumReconstructionBasic(
                training_data=df,
                internal_var_col_name='bias',
                external_var_col_name='mu',
                dependent_var_col_name='value',
                base_func='invalid',
            )

    def test_missing_values_without_verify_raises(self):
        """Missing values without verify_pivot_data should raise ValueError."""
        df = pd.DataFrame({
            'bias': [0.0, 0.0, 1.0],
            'mu': [100.0, 200.0, 100.0],
            'value': [1.0, 2.0, 3.0],
            # Missing: bias=1.0, mu=200.0
        })
        # This will create a pivot with NaN
        with pytest.raises(ValueError, match='Missing values'):
            SpectrumReconstructionBasic(
                training_data=df,
                internal_var_col_name='bias',
                external_var_col_name='mu',
                dependent_var_col_name='value',
                base_func='blackbody',
                verify_pivot_data=False,
            )

    def test_spectrum_before_reconstruct_raises(self):
        """Calling spectrum before reconstruct should raise ValueError."""
        df = self._make_training_data()
        srb = SpectrumReconstructionBasic(
            training_data=df,
            internal_var_col_name='bias',
            external_var_col_name='mu',
            dependent_var_col_name='value',
            base_func='blackbody',
        )
        with pytest.raises(ValueError, match='Spectrum has not been reconstructed'):
            srb.spectrum(np.linspace(0, 1, 10))


# ========== SpectrumReconstructionBasicHighPerformance ==========

class TestSpectrumReconstructionBasicHighPerformance:

    def test_init_gaussian(self):
        """Should initialize correctly with numpy arrays."""
        training = np.random.rand(5, 3)
        srb = SpectrumReconstructionBasicHighPerformance(
            training_data=training,
            internal_var_col_name='bias',
            internal_var_col=np.arange(5, dtype=np.float64),
            external_var_col_name='mu',
            external_var_col=np.array([100.0, 200.0, 300.0]),
            dependent_var_col_name='value',
            base_func='gaussian',
            sigma=1e-6,
        )
        assert srb._base_func_name == 'gaussian'
        assert srb.a is None

    def test_reconstruct_and_spectrum(self):
        """Full cycle: reconstruct then get spectrum."""
        # Build a simple 3x3 identity-like system
        training = np.eye(3)
        testing = np.array([1.0, 0.5, 0.2])

        srb = SpectrumReconstructionBasicHighPerformance(
            training_data=training,
            internal_var_col_name='bias',
            internal_var_col=np.arange(3, dtype=np.float64),
            external_var_col_name='mu',
            external_var_col=np.array([1000e-9, 1200e-9, 1400e-9]),
            dependent_var_col_name='value',
            base_func='gaussian',
            sigma=50e-9,
        )
        srb.reconstruct_spectrum(testing_data=testing, method='normal')
        assert srb.a is not None
        np.testing.assert_allclose(srb.a, testing, atol=1e-8)

        # Spectrum should return an array
        wl = np.linspace(800e-9, 1600e-9, 50)
        spec = srb.spectrum(wl, normalize=True)
        assert spec.shape == (50,)
        assert np.max(spec) == pytest.approx(1.0)

    def test_spectrum_before_reconstruct_raises(self):
        """spectrum() before reconstruct should raise ValueError."""
        srb = SpectrumReconstructionBasicHighPerformance(
            training_data=np.eye(3),
            internal_var_col_name='bias',
            internal_var_col=np.arange(3, dtype=np.float64),
            external_var_col_name='mu',
            external_var_col=np.array([1.0, 2.0, 3.0]),
            dependent_var_col_name='value',
            base_func='blackbody',
        )
        with pytest.raises(ValueError, match='Spectrum has not been reconstructed'):
            srb.spectrum(np.linspace(500e-9, 2000e-9, 10))

    def test_invalid_base_func_raises(self):
        """Invalid base_func should raise ValueError."""
        with pytest.raises(ValueError, match='base_func must be'):
            SpectrumReconstructionBasicHighPerformance(
                training_data=np.eye(3),
                internal_var_col_name='bias',
                internal_var_col=np.arange(3, dtype=np.float64),
                external_var_col_name='mu',
                external_var_col=np.array([1.0, 2.0, 3.0]),
                dependent_var_col_name='value',
                base_func='invalid',
            )
