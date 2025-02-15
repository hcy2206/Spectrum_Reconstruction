from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from SpectrumReconstruction.Utility import blackbody, gaussian


# Programed by o3-mini-high. Errors may occur in extreme cases.
def _clean_pivot_training_data(data: pd.DataFrame,
                               internal_var_col_name: str,
                               external_var_col_name: str,
                               dependent_var_col_name: str,
                               pass_in_pivot_data: bool = False
                               ):  # -> pd.DataFrame:
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
        # Use index.difference/columns.difference to ensure the order remains unchanged
        return pivot.loc[pivot.index.difference(rows_to_drop),
        pivot.columns.difference(cols_to_drop)]

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
                       method: Literal['normal', 'l1', 'l2', 'ElasticNet'],
                       **kwargs
                       ):  # -> np.ndarray:
    match method:
        case 'normal':
            return np.linalg.inv(x.T @ x) @ x.T @ y
        case 'l1':
            lambda_reg = kwargs.get('lambda_reg', 0.001)

            def objective(a):
                return np.sum((x @ a - y) ** 2) + lambda_reg * np.sum(np.abs(a))

            initial_guess = np.zeros(x.shape[1])
            result = minimize(objective, initial_guess, method='SLSQP')
            return result.x
        case 'l2':
            lambda_reg = kwargs.get('lambda_reg', 0.001)

            def objective(a):
                return np.sum((x @ a - y) ** 2) + lambda_reg * np.sum(a ** 2)

            initial_guess = np.zeros(x.shape[1])
            result = minimize(objective, initial_guess, method='SLSQP')
            return result.x
        case 'ElasticNet':
            lambda_reg = kwargs.get('lambda_reg', 0.001)
            alpha = kwargs.get('alpha', 0.5)

            def objective(a):
                return np.sum((x @ a - y) ** 2) + lambda_reg * (
                        alpha * np.sum(np.abs(a)) + (1 - alpha) * np.sum(a ** 2))

            initial_guess = np.zeros(x.shape[1])
            result = minimize(objective, initial_guess, method='SLSQP')
            return result.x


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
        self._pass_in_pivot_test_data = None
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

    def base_func(self, lambda_: np.ndarray, external_var: float):  # -> np.ndarray:
        match self._base_func_name:
            case 'blackbody':
                return blackbody(lambda_, external_var)
            case 'gaussian':
                return gaussian(lambda_, external_var, self._sigma)
            case _:
                raise ValueError('self._base_func_name must be either "blackbody" or "gaussian"')

    @overload
    def reconstruct_spectrum(self,
                             testing_data: pd.DataFrame,
                             method: Literal['normal'],
                             *args,
                             pass_in_pivot_test_data: bool = True,
                             **kwargs):
        ...

    @overload
    def reconstruct_spectrum(self,
                             testing_data: pd.DataFrame,
                             method: Literal['l1', 'l2'],
                             *args,
                             lambda_reg: float,
                             pass_in_pivot_test_data: bool = True,
                             **kwargs):
        ...

    @overload
    def reconstruct_spectrum(self,
                             testing_data: pd.DataFrame,
                             method: Literal['ElasticNet'],
                             *args,
                             lambda_reg: float,
                             alpha: float,
                             pass_in_pivot_test_data: bool = True,
                             **kwargs):
        ...

    def reconstruct_spectrum(self,
                             testing_data: pd.DataFrame,
                             method: Literal['normal', 'l1', 'l2', 'ElasticNet'],
                             *args,
                             pass_in_pivot_test_data: bool = True,
                             **kwargs):  # -> None:
        self._testing_data = testing_data
        self._pass_in_pivot_test_data = pass_in_pivot_test_data
        if self._pass_in_pivot_test_data:
            pivot = self._testing_data.pivot(
                index=self._internal_var_col_name,
                columns=self._external_var_col_name,
                values=self._dependent_var_col_name
            )
        else:
            pivot = self._testing_data
        # print(pivot)
        # print(pivot.shape)
        # print(pivot.values)
        if pivot.shape[1] != 1:
            raise ValueError('The pivot table of test data should have only one row')
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
            **kwargs
        )

        self._a = a

        return a

    def spectrum(self,
                 lambda_: np.ndarray,
                 normalize: bool = True
                 ):  # -> np.ndarray:
        if self._external_var_col is None:
            raise ValueError('External variable not found')

        if self._a is None:
            raise ValueError('Spectrum has not been reconstructed yet')

        external_var = self._external_var_col.values.astype(np.float64)

        result = np.array([np.sum(self._a * self.base_func(l, external_var)) for l in lambda_])

        if normalize:
            result = np.abs(result)
            result = result / np.max(result)

        return result
