from typing import Literal, overload

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV,  Ridge, RidgeCV, ElasticNet, ElasticNetCV, LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression
from scipy.optimize import minimize

from .Utility import blackbody, gaussian


def _as_1d_float_array(values: float | np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1)
    else:
        values = values.reshape(-1)
    if values.size == 0:
        raise ValueError(f'{name} array is empty')
    return values


def _normalize_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(values, dtype=np.float64))
    max_value = np.max(values)
    if max_value == 0 or not np.isfinite(max_value):
        return values
    return values / max_value


# Errors may occur in extreme cases.
def _clean_pivot_training_data(data: pd.DataFrame,
                               internal_var_col_name: str,
                               external_var_col_name: str,
                               dependent_var_col_name: str,
                               pass_in_pivot_data: bool = False
                               ) -> pd.DataFrame:
    # First, construct the pivot table
    if pass_in_pivot_data:
        pivot = data
    else:
        if internal_var_col_name in data.columns and external_var_col_name in data.columns and dependent_var_col_name in data.columns:
            pivot = data.pivot(
                index=internal_var_col_name,
                columns=external_var_col_name,
                values=dependent_var_col_name
            )
        else:
            raise ValueError('Column names not found in the training data')

    # Use sets to keep track of rows and columns to be removed
    rows_to_drop = set()
    cols_to_drop = set()

    def current_matrix():
        remaining_rows = pivot.index[~pivot.index.isin(rows_to_drop)]
        remaining_cols = pivot.columns[~pivot.columns.isin(cols_to_drop)]
        return pivot.loc[remaining_rows, remaining_cols]

    sub_mat = current_matrix()
    # Iterate and remove rows or columns with the lowest "completeness" while missing values exist in the submatrix
    while sub_mat.isna().any().any():
        # Compute completeness for each row: non-missing values / total columns
        row_completeness = sub_mat.notna().sum(axis=1) / sub_mat.shape[1]
        # Compute completeness for each column: non-missing values / total rows
        col_completeness = sub_mat.notna().sum(axis=0) / sub_mat.shape[0]

        # Find the row and column with the lowest completeness (the highest missing rate)
        min_row_comp = row_completeness.min() if not row_completeness.empty else 1
        min_col_comp = col_completeness.min() if not col_completeness.empty else 1

        # Compare which side is "worse" and remove that row or column
        if min_row_comp <= min_col_comp:
            # Remove the row with the lowest completeness
            row_to_drop = row_completeness.idxmin()
            rows_to_drop.add(row_to_drop)
            # Optionally, print logs for debugging: print(f"Drop row {row_to_drop} with completeness {min_row_comp:.3f}")
        else:
            # Remove the column with the lowest completeness
            col_to_drop = col_completeness.idxmin()
            cols_to_drop.add(col_to_drop)
            # print(f"Drop column {col_to_drop} with completeness {min_col_comp:.3f}")

        sub_mat = current_matrix()

    # The resulting sub_mat is the "optimal" submatrix without missing values
    if sub_mat.empty:
        raise ValueError('No valid data for spectral reconstruction training process after cleaning')

    return sub_mat


def _linear_regression(x: np.ndarray,
                       y: np.ndarray,
                       method: str,
                       lambda_reg=0.001,
                       alpha=0.5,
                       cv=10
                       ) -> None | np.ndarray:
    # 局部导入SaveMemory变量，避免循环导入
    from . import SaveMemory
    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if SaveMemory:
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
    match method:
        case 'normal':
            result, *_ = np.linalg.lstsq(x, y, rcond=None)
            return np.array(result, dtype=np.float64)

        case 'l1':
            def objective(a):
                return np.sum((np.dot(x, a) - y) ** 2) + lambda_reg * np.sum(np.abs(a))

            initial_guess = np.zeros(x.shape[1])
            result = minimize(objective, initial_guess, method='SLSQP')
            return result.x

        case 'l2':
            ridge = Ridge(alpha=lambda_reg, fit_intercept=True, max_iter=10000, solver='auto')
            ridge.fit(x, y)
            return np.array(ridge.coef_, dtype=np.float64).flatten()

        case 'LinearRegression':
            linear_regression = LinearRegression(fit_intercept=True)
            linear_regression.fit(x, y)
            return np.array(linear_regression.coef_, dtype=np.float64)

        case 'Lasso':
            lasso = Lasso(alpha=lambda_reg, fit_intercept=True, max_iter=10000)
            lasso.fit(x, y)
            return np.array(lasso.coef_, dtype=np.float64)

        case 'LassoCV':
            lasso = LassoCV(cv=cv, fit_intercept=True, max_iter=10000)
            lasso.fit(x, y)
            print("alpha = ", lasso.alpha_)
            return np.array(lasso.coef_, dtype=np.float64)

        case 'LassoLars':
            lasso = LassoLars(alpha=lambda_reg, fit_intercept=True, max_iter=10000)
            lasso.fit(x, y)
            return np.array(lasso.coef_, dtype=np.float64)

        case 'LassoLarsCV':
            lasso = LassoLarsCV(cv=cv, fit_intercept=True, max_iter=10000)
            lasso.fit(x, y)
            return np.array(lasso.coef_, dtype=np.float64)

        case 'LassoLarsIC':
            lasso = LassoLarsIC(criterion='aic', fit_intercept=True, max_iter=10000)
            lasso.fit(x, y)
            return np.array(lasso.coef_, dtype=np.float64)

        case 'Ridge':
            ridge = Ridge(alpha=lambda_reg, fit_intercept=True, max_iter=10000)
            ridge.fit(x, y)
            return np.array(ridge.coef_, dtype=np.float64).flatten()

        case 'RidgeCV':
            ridge = RidgeCV(cv=cv, fit_intercept=True)
            ridge.fit(x, y)
            return np.array(ridge.coef_, dtype=np.float64)

        case 'ElasticNet':
            elastic_net = ElasticNet(alpha=lambda_reg, l1_ratio=alpha, fit_intercept=True, max_iter=10000)
            elastic_net.fit(x, y)
            return np.array(elastic_net.coef_, dtype=np.float64).flatten()

        case 'ElasticNetCV':
            # If lambda_reg is not a list or ndarray, generate a default candidate range for alphas
            if not isinstance(lambda_reg, (list, np.ndarray)):
                alphas = np.logspace(-4, 1, 50)  # default candidate alphas
            else:
                alphas = lambda_reg
            # If alpha is not a list or ndarray, wrap it in a list for candidate l1_ratio values
            if not isinstance(alpha, (list, np.ndarray)):
                l1_ratios = [alpha]
            else:
                l1_ratios = alpha
            elastic_net_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=cv, fit_intercept=True, max_iter=10000)
            elastic_net_cv.fit(x, y)
            print("alpha=", elastic_net_cv.alpha_)
            print("l1_ratio=", elastic_net_cv.l1_ratio_)
            return np.array(elastic_net_cv.coef_, dtype=np.float64)

        case _:
            raise ValueError('Invalid method for linear regression')


class SpectrumReconstructionBasic:
    @overload
    def __init__(self,
                 training_data: pd.DataFrame,
                 internal_var_col_name: str,
                 external_var_col_name: str,
                 dependent_var_col_name: str,
                 base_func: Literal['blackbody'],
                 *args,
                 verify_pivot_data: bool = False,
                 **kwargs):
        ...

    @overload
    def __init__(self,
                 training_data: pd.DataFrame,
                 internal_var_col_name: str,
                 external_var_col_name: str,
                 dependent_var_col_name: str,
                 base_func: Literal['gaussian'],
                 *args,
                 verify_pivot_data: bool = False,
                 sigma: float,
                 **kwargs):
        ...

    def __init__(self,
                 training_data: pd.DataFrame,
                 internal_var_col_name: str,
                 external_var_col_name: str,
                 dependent_var_col_name: str,
                 base_func: Literal['blackbody', 'gaussian'],
                 *args,
                 verify_pivot_data: bool = False,
                 **kwargs):
        self._training_data = training_data
        self._testing_data = None
        self._internal_var_col_name = internal_var_col_name
        self._external_var_col_name = external_var_col_name
        self._dependent_var_col_name = dependent_var_col_name
        self._pivot_training_data = None
        self._pivot_test_data = None
        self._verify_pivot_data = verify_pivot_data
        self._pivot_pass_in_test_data = None
        self._a = None
        # self.spectrum = None

        # def _spectrum_reconstruction_training(self, base_func: str, **kwargs

        if self._verify_pivot_data:
            self._pivot_training_data = _clean_pivot_training_data(
                training_data,
                internal_var_col_name,
                external_var_col_name,
                dependent_var_col_name
            )
        else:
            self._verify_pivot_data = False
            self._pivot_training_data = self._training_data.pivot(
                index=self._internal_var_col_name,
                columns=self._external_var_col_name,
                values=self._dependent_var_col_name
            )
            if self._pivot_training_data.isna().any().any():
                raise ValueError('Missing values found in the pivot table of training data')
            elif self._pivot_training_data.empty:
                raise ValueError('Pivot table of training data is empty')
        self._internal_var_col = self._pivot_training_data.index
        self._external_var_col = self._pivot_training_data.columns

        self._base_func_name = base_func
        match base_func:
            case 'blackbody':
                # self._base_func = _blackbody
                pass
            case 'gaussian':
                # self._base_func = _gaussian
                self._sigma = kwargs.get('sigma', 1e-6)
            case _:
                raise ValueError('base_func must be either "blackbody" or "gaussian"')

    def base_func(self,
                  lambda_: np.ndarray,
                  external_var: np.ndarray
                  ) -> np.ndarray:
        match self._base_func_name:
            case 'blackbody':
                return blackbody(lambda_, external_var)
            case 'gaussian':
                return gaussian(lambda_, external_var, self._sigma)
            case _:
                raise ValueError('self._base_func_name must be either "blackbody" or "gaussian"')

    def reconstruct_spectrum(self,
                             testing_data: pd.DataFrame,
                             method: str,
                             pivot_pass_in_test_data: bool = True,
                             lambda_reg: float = 0.01,
                             alpha: float = 0.5,
                             cv: int = 10,
                             **kwargs) -> np.ndarray:
        self._testing_data = testing_data
        self._pivot_pass_in_test_data = pivot_pass_in_test_data
        if self._pivot_pass_in_test_data:
            pivot = self._testing_data.pivot(
                index=self._internal_var_col_name,
                columns=self._external_var_col_name,
                values=self._dependent_var_col_name
            )
        else:
            pivot = self._testing_data
        if pivot.shape[1] != 1:
            raise ValueError('The pivot table of test data should have only one column')
        elif pivot.isna().any().any():
            raise ValueError('Missing values found in the pivot table of test data')
        elif pivot.empty:
            raise ValueError('Pivot table of test data is empty')
        self._pivot_test_data = pivot

        if self._verify_pivot_data:
            self._pivot_test_data = _clean_pivot_training_data(
                self._pivot_test_data,
                self._internal_var_col_name,
                self._external_var_col_name,
                self._dependent_var_col_name,
                pass_in_pivot_data=True
            )

        # Check if the internal variable of test data matches that of training data
        if not self._internal_var_col.equals(self._pivot_test_data.index):
            raise ValueError('Internal variable of test data does not match that of training data')

        # transfer self._pivot_test_data to a numpy ndarray from pd.DataFrame
        _pivot_training_data_numpy = self._pivot_training_data.values
        _pivot_test_data_numpy = self._pivot_test_data.values

        a = _linear_regression(
            _pivot_training_data_numpy,
            _pivot_test_data_numpy,
            method,
            lambda_reg,
            alpha,
            cv,
            **kwargs
        )

        self._a = a

        return a

    def spectrum(self,
                 lambda_: float | np.ndarray,
                 normalize: bool = True
                 ) -> float | np.ndarray:
        if self._external_var_col is None:
            raise ValueError('External variable not found')

        if self._a is None:
            raise ValueError('Spectrum has not been reconstructed yet')

        scalar_input = np.asarray(lambda_).ndim == 0
        lambda_values = _as_1d_float_array(lambda_, 'lambda_')
        external_var = self._external_var_col.values.astype(np.float64)
        if external_var.size == 0:
            raise ValueError('External variable column is empty')

        result = self.base_func(lambda_values[:, None], external_var[None, :]) @ self._a

        if result.ndim > 1:
            result = result.flatten()

        if normalize:
            result = _normalize_spectrum(result)

        return float(result[0]) if scalar_input else result


class SpectrumReconstructionBasicHighPerformance:
    @overload
    def __init__(self,
                 training_data: np.ndarray,
                 internal_var_col_name: str,
                 internal_var_col: np.ndarray,
                 external_var_col_name: str,
                 external_var_col: np.ndarray,
                 dependent_var_col_name: str,
                 base_func: Literal['blackbody'],
                 *args,
                 verify_pivot_data: bool = False,
                 **kwargs):
        ...

    @overload
    def __init__(self,
                 training_data: np.ndarray,
                 internal_var_col_name: str,
                 internal_var_col: np.ndarray,
                 external_var_col_name: str,
                 external_var_col: np.ndarray,
                 dependent_var_col_name: str,
                 base_func: Literal['gaussian'],
                 *args,
                 verify_pivot_data: bool = False,
                 sigma: float,
                 **kwargs):
        ...

    def __init__(self,
                 training_data: np.ndarray,
                 internal_var_col_name: str,
                 internal_var_col: np.ndarray,
                 external_var_col_name: str,
                 external_var_col: np.ndarray,
                 dependent_var_col_name: str,
                 base_func: Literal['blackbody', 'gaussian'],
                 *args,
                 verify_pivot_data: bool = False,
                 **kwargs):

        self._training_data = training_data
        self._testing_data = None
        self._internal_var_col_name = internal_var_col_name
        self._internal_var_col = internal_var_col
        self._external_var_col_name = external_var_col_name
        self._external_var_col = external_var_col
        self._dependent_var_col_name = dependent_var_col_name
        self._a = None

        self._base_func_name = base_func
        match base_func:
            case 'blackbody':
                pass
            case 'gaussian':
                self._sigma = kwargs.get('sigma', 1e-6)
            case _:
                raise ValueError('base_func must be either "blackbody" or "gaussian"')

    def base_func(self,
                  lambda_: np.ndarray,
                  external_var: np.ndarray
                  ) -> np.ndarray:
        match self._base_func_name:
            case 'blackbody':
                return blackbody(lambda_, external_var)
            case 'gaussian':
                return gaussian(lambda_, external_var, self._sigma)
            case _:
                raise ValueError('self._base_func_name must be either "blackbody" or "gaussian"')

    def reconstruct_spectrum(self,
                             testing_data: np.ndarray,
                             method: Literal['normal', 'l1', 'l2', 'ElasticNet', 'ElasticNetCV'],
                             **kwargs) -> np.ndarray:
        self._testing_data = testing_data

        self._a = _linear_regression(
            self._training_data,
            self._testing_data,
            method,
            **kwargs
        )
        return self._a

    @property
    def a(self):
        return self._a

    def spectrum(self,
                 lambda_: float | np.ndarray,
                 normalize: bool = True
                 ) -> float | np.ndarray:
        if self._external_var_col is None:
            raise ValueError('External variable not found')

        if self._a is None:
            raise ValueError('Spectrum has not been reconstructed yet')

        scalar_input = np.asarray(lambda_).ndim == 0
        lambda_values = _as_1d_float_array(lambda_, 'lambda_')

        if self._external_var_col.size == 0:
            raise ValueError("External variable column is empty")

        result = self.base_func(lambda_values[:, None], self._external_var_col[None, :]) @ self.a

        if result.ndim > 1:
            result = result.flatten()

        if normalize:
            result = _normalize_spectrum(result)

        return float(result[0]) if scalar_input else result
