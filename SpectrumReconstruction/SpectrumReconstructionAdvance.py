from functools import cached_property
from typing import Callable, Literal, overload

import numpy as np
import pandas as pd
import plotly.express as px
from line_profiler_pycharm import profile
from numba import njit

from SpectrumReconstruction.Utility import smooth_responsivity, gaussian, blackbody

# Constants
q = 1.60217662e-19  # Elementary charge [C]
h = 6.62607015e-34  # Planck constant [J*s]
c = 3.0e8  # Speed of light [m/s]


class IdealSemiconductorPhotoDetector:
    def __init__(self,
                 bias_array: np.ndarray[float],  # Bias array [m]
                 e_g_ev: float,  # Band gap energy [eV]
                 eta: float = 1.0,  # Quantum efficiency
                 delta_lambda: float = 30e-9,  # Transition width [m]
                 base_function: Callable[..., np.ndarray[float]] = smooth_responsivity,  # Base responsivity function
                 wavelength: np.ndarray[float] = np.linspace(0, 2.0e-6, 500),  # Wavelength array [m]
                 visible_blind_cutoff: float = -1,  # Cut-off wavelength for visible blind [m]
                 bias_mode: Literal['normal_move', 'increase_band_gap'] = 'normal_move'
                 ):
        self.base_function = base_function
        self.bias_array = bias_array
        self.e_g = e_g_ev * q
        self.eta = eta
        self.delta_lambda = delta_lambda
        self.wavelength = wavelength
        self.visible_blind_cutoff = visible_blind_cutoff
        self.bias_mode = bias_mode

    def _responsivity_func(self,
                           bias: float,
                           bias_mode: Literal['normal_move', 'increase_band_gap'] = 'normal_move'
                           ) -> np.ndarray:
        match bias_mode:
            case 'normal_move':
                lambda_ = self.wavelength - bias
                e_g = self.e_g
            case 'increase_band_gap':
                lambda_ = self.wavelength
                lambda_g = h * c / self.e_g
                e_g = h * c / (lambda_g + bias)
            case _:
                raise ValueError("Unsupported bias mode.")
        responsivity = np.asarray(self.base_function(lambda_, e_g, self.delta_lambda, self.eta))
        responsivity[responsivity < 0] = 0
        # Apply visible blind cut-off
        if self.visible_blind_cutoff > 0:
            responsivity[(lambda_ + bias) < self.visible_blind_cutoff] = 0
        return responsivity

    # @cached_property
    # @profile
    # def _responsivity(self):
    #     # Initialize an empty DataFrame to store the data
    #     responsivity_data_list = []
    #
    #     for bias in self.bias_array:
    #         responsivity = self._responsivity_func(bias, self.bias_mode)
    #         # Append a new row to the DataFrame
    #         new_data = pd.DataFrame({
    #             'wavelength': self.wavelength,  # Convert to nm for better readability
    #             'responsivity': responsivity,
    #             'bias': bias
    #         })
    #         # new_data = new_data.dropna(axis=1, how='all')  # 删除所有值为NA的列
    #         responsivity_data_list.append(new_data)
    #
    #     return pd.concat(responsivity_data_list)

    @cached_property
    @profile
    def responsivity(self):
        """
        计算 responsivity 的向量化版本：
          - 利用广播同时计算所有 bias 下的 responsivity，生成一个二维矩阵（wavelength x bias）
          - 在计算完成后一次性转换为 DataFrame（长格式），避免循环内的 DataFrame 拼接开销
        """
        num_wl = len(self.wavelength)
        num_bias = len(self.bias_array)

        # 根据 bias_mode 构造 lambda 矩阵和 e_g 矩阵
        if self.bias_mode == 'normal_move':
            # 利用广播：每一列对应一个 bias
            lambda_matrix = self.wavelength[:, None] - self.bias_array[None, :]
            e_g_matrix = np.full((num_wl, num_bias), self.e_g)
        elif self.bias_mode == 'increase_band_gap':
            # wavelength 保持不变，bias 影响 e_g 计算
            lambda_matrix = np.broadcast_to(self.wavelength[:, None], (num_wl, num_bias))
            lambda_g = h * c / self.e_g  # 常数
            e_g_matrix = h * c / (lambda_g + self.bias_array[None, :])
        else:
            raise ValueError("Unsupported bias mode.")

        # 计算 responsivity，要求 base_function 能够接受矩阵输入
        responsivity_matrix = self.base_function(lambda_matrix, e_g_matrix, self.delta_lambda, self.eta)
        responsivity_matrix = np.asarray(responsivity_matrix)
        responsivity_matrix[responsivity_matrix < 0] = 0

        # 应用 visible blind cutoff（若设定了正值）
        if self.visible_blind_cutoff > 0:
            if self.bias_mode == 'normal_move':
                # 对于 normal_move，需将 bias 加回 wavelength
                lambda_matrix_with_bias = lambda_matrix + self.bias_array[None, :]
            elif self.bias_mode == 'increase_band_gap':
                lambda_matrix_with_bias = lambda_matrix  # 或根据具体物理含义调整
            responsivity_matrix[lambda_matrix_with_bias < self.visible_blind_cutoff] = 0

        # 一次性转换为 DataFrame：行索引为 wavelength，列索引为 bias
        df = pd.DataFrame(responsivity_matrix, index=self.wavelength, columns=self.bias_array)
        return df

    @cached_property
    def _responsivity(self):
        # 若下游需要长格式，可一次性转换
        df_long = self.responsivity.reset_index().melt(id_vars='index', var_name='bias', value_name='responsivity')
        df_long.rename(columns={'index': 'wavelength'}, inplace=True)
        return df_long

    def responsivity_figure_show(self):
        # Initialize an empty DataFrame to store the data
        fig_data = self._responsivity
        fig_data['bias'] = (fig_data['bias'] * 1e9).apply(lambda x: f"{x:.0f} nm")  # Convert bias to nm and append "nm"

        # Create a figure using plotly express
        fig = px.line(
            fig_data,
            x='wavelength',
            y='responsivity',
            color='bias',
            labels={
                'wavelength': 'Wavelength (m)',
                'responsivity': 'Responsivity (A/W)',
                'bias': 'Bias'
            },
            title='Photodetector Responsivity vs Wavelength for Different Bias Values'
        )

        return fig


class IncidentSpectrum:
    @overload
    def __init__(self,
                 wavelength: np.ndarray[float],  # Wavelength array [m],
                 base_function_name: Literal['gaussian'],  # Base spectrum function
                 sigma: float,  # Standard deviation for Gaussian
                 mu: np.ndarray  # Mean for Gaussian
                 ):
        ...

    @overload
    def __init__(self,
                 wavelength: np.ndarray[float],  # Wavelength array [m],
                 base_function_name: Literal['blackbody'],  # Base spectrum function
                 T: np.ndarray  # Temperature for Blackbody
                 ):
        ...

    def __init__(self,
                 wavelength: np.ndarray[float],  # Wavelength array [m],
                 base_function_name: Literal['gaussian', 'blackbody'] = 'gaussian',  # Base spectrum function
                 **kwargs
                 ):
        self.wavelength = wavelength
        self.base_function_name = base_function_name
        match self.base_function_name:
            case 'gaussian':
                self.base_function = gaussian
                self.sigma = kwargs.get('sigma', 1e-9)
                self.mu = kwargs.get('mu', None)
                if self.mu is None:
                    raise ValueError("mu must be provided for gaussian base function.")
            case 'blackbody':
                self.base_function = blackbody
                self.T = kwargs.get('T', None)
                if self.T is None:
                    raise ValueError("T must be provided for blackbody base function.")
            case _:
                raise ValueError(f"Unsupported base function: {base_function_name}")

    # @cached_property
    # @profile
    # def _spectrum(self):
    #     spectrum_data_list = []
    #     match self.base_function_name:
    #         case 'gaussian':
    #             for mu in self.mu:
    #                 spectrum = gaussian(self.wavelength, mu, self.sigma)
    #                 new_data = pd.DataFrame({
    #                     'wavelength': self.wavelength,  # Convert to nm for better readability
    #                     'spectrum': spectrum,
    #                     'mu': mu
    #                 })
    #                 spectrum_data_list.append(new_data)
    #         case 'blackbody':
    #             for T in self.T:
    #                 spectrum = blackbody(self.wavelength, T)
    #                 new_data = pd.DataFrame({
    #                     'wavelength': self.wavelength,  # Convert to nm for better readability
    #                     'spectrum': spectrum,
    #                     'T': T
    #                 })
    #                 spectrum_data_list.append(new_data)
    #         case _:
    #             raise ValueError("Unsupported base function.")
    #     return pd.concat(spectrum_data_list)
    #
    # @cached_property
    # def spectrum(self):
    #     match self.base_function_name:
    #         case 'gaussian':
    #             return self._spectrum.pivot(index='wavelength', columns='mu', values='spectrum')
    #         case 'blackbody':
    #             return self._spectrum.pivot(index='wavelength', columns='T', values='spectrum')
    #         case _:
    #             raise ValueError("Unsupported base function.")

    @cached_property
    @profile
    def spectrum(self):
        """
        计算 spectrum 的向量化版本：
          - 利用广播同时计算所有参数（mu 或 T）下的 spectrum，生成一个二维矩阵（wavelength x mu 或 wavelength x T）
          - 一次性转换为 DataFrame（宽格式），避免循环内逐个构建 DataFrame 的开销
        """
        if self.base_function_name == 'gaussian':
            # 对于 gaussian 模式，self.mu 是参数数组
            # 利用广播生成 (num_wl, num_mu) 的矩阵
            spectrum_matrix = gaussian(self.wavelength[:, None], self.mu[None, :], self.sigma)
            # 转换为 DataFrame：行索引为 wavelength，列索引为 mu
            df = pd.DataFrame(spectrum_matrix, index=self.wavelength, columns=self.mu)
            return df

        elif self.base_function_name == 'blackbody':
            # 对于 blackbody 模式，self.T 是参数数组
            spectrum_matrix = blackbody(self.wavelength[:, None], self.T[None, :])
            df = pd.DataFrame(spectrum_matrix, index=self.wavelength, columns=self.T)
            return df

        else:
            raise ValueError("Unsupported base function.")

    @cached_property
    def _spectrum(self):
        # 若下游需要长格式，可一次性转换
        df_long = self.spectrum.reset_index().melt(id_vars='wavelength', var_name=self.base_function_name,
                                                   value_name='spectrum')
        return df_long

    def spectrum_figure_show(self):
        # Initialize an empty DataFrame to store the data
        fig_data = self._spectrum
        match self.base_function_name:
            case 'gaussian':
                color_name = 'mu'
                fig_data[color_name] = (fig_data[color_name] * 1e9).apply(
                    lambda x: f"{x:.0f}nm"
                )  # Convert to string for better readability
            case 'blackbody':
                color_name = 'T'
                fig_data[color_name] = (fig_data[color_name]).apply(
                    lambda x: f"{x:.0f}K"
                )  # Convert to string for better readability
            case _:
                raise ValueError("Unsupported base function.")

        # Create a figure using plotly express
        fig = px.line(
            fig_data,
            x='wavelength',
            y='spectrum',
            color=color_name,
            labels={
                'wavelength': 'Wavelength (m)',
                'spectrum': 'Spectrum',
                self.base_function_name: self.base_function_name
            },
            title=f'{self.base_function_name.capitalize()} Spectrum vs Wavelength'
        )

        return fig


class SimulationSpectrum:
    def __init__(self,
                 wavelength_array: np.ndarray[float],  # Wavelength array [m]
                 spectrum_function: Callable[..., np.ndarray[float]]  # Spectrum function
                 ):
        self._spectrum = None
        self.wavelength = wavelength_array
        self.spectrum_function = spectrum_function

    def set_spectrum(self, **kwargs) -> pd.DataFrame:
        spectrum = self.spectrum_function(self.wavelength, **kwargs)
        spectrum = pd.DataFrame({
            'wavelength': self.wavelength,
            'spectrum': spectrum
        })
        self._spectrum = spectrum
        return spectrum

    @property
    def spectrum(self) -> pd.DataFrame:
        # set wavelength as index
        if self._spectrum is None:
            raise ValueError("Spectrum has not been set. Please call set_spectrum() first.")
        return self._spectrum.set_index('wavelength')

    def spectrum_figure_show(self) -> px.line:
        # Initialize an empty DataFrame to store the data
        if self._spectrum is None:
            raise ValueError("Spectrum has not been set. Please call set_spectrum() first.")
        fig_data = self._spectrum
        # Create a figure using plotly express
        fig = px.line(
            fig_data,
            x='wavelength',
            y='spectrum',
            labels={
                'wavelength': 'Wavelength (m)',
                'spectrum': 'Spectrum'
            },
            title='Spectrum vs Wavelength'
        )

        return fig

    @property
    def spectrum_figure(self) -> px.line:
        # Initialize an empty DataFrame to store the data
        if self._spectrum is None:
            raise ValueError("Spectrum has not been set. Please call set_spectrum() first.")
        fig_data = self._spectrum
        # Create a figure using plotly express
        fig = px.line(
            fig_data,
            x='wavelength',
            y='spectrum',
            labels={
                'wavelength': 'Wavelength (m)',
                'spectrum': 'Spectrum'
            },
            title='Spectrum vs Wavelength'
        )

        return fig


@profile
def simulate_response_matrix(
        photodetector: IdealSemiconductorPhotoDetector,
        incident_spectrum: IncidentSpectrum
) -> pd.DataFrame:
    """
    Calculate the response matrix of the photodetector to the incident spectrum.
    :param photodetector: IdealSemiconductorPhotoDetector object
    :param incident_spectrum: IncidentSpectrum object
    :return: Response matrix DataFrame
    """
    # Check if the wavelength of the incident spectrum matches the responsivity
    if not np.array_equal(photodetector.wavelength, incident_spectrum.wavelength):
        raise ValueError("Wavelengths of the photodetector and incident spectrum do not match.")

    responsivity = photodetector.responsivity
    responsivity = responsivity.values
    spectrum = incident_spectrum.spectrum
    spectrum = spectrum.values

    # response = responsivity.T @ spectrum
    @njit
    def fast_matmul(a, b):
        return a.T @ b

    response = fast_matmul(responsivity, spectrum)

    # Create a DataFrame for the response matrix, index by photodetector.bias_array and columns by incident_spectrum.mu or T
    _response_matrix = pd.DataFrame(
        response,
        columns=incident_spectrum.spectrum.columns,
        index=photodetector.responsivity.columns
    )
    _response_matrix.index.name = 'bias'
    match incident_spectrum.base_function_name:
        case 'gaussian':
            _response_matrix.columns.name = 'mu'
        case 'blackbody':
            _response_matrix.columns.name = 'T'
        case _:
            raise ValueError("Unsupported base function.")
    return _response_matrix


def simulate_unknown_response(
        photodetector: IdealSemiconductorPhotoDetector,
        unknown_spectrum: SimulationSpectrum,
        add_gaussian_noise: bool = False,
        noise_std_ratio: float = 0.01
) -> pd.DataFrame:
    """
    Calculate the response of the photodetector to the unknown spectrum.
    :param photodetector: IdealSemiconductorPhotoDetector object
    :param unknown_spectrum: SimulationSpectrum object
    :param add_gaussian_noise: Add Gaussian noise to the response
    :param noise_std_ratio: Standard deviation ratio for Gaussian noise
    :return: Response DataFrame
    """
    # Check if the wavelength of the unknown spectrum matches the responsivity
    if not np.array_equal(photodetector.wavelength, unknown_spectrum.wavelength):
        raise ValueError("Wavelengths of the photodetector and unknown spectrum do not match.")

    responsivity = photodetector.responsivity.values
    spectrum = unknown_spectrum.spectrum.values

    response = responsivity.T @ spectrum

    if add_gaussian_noise:
        noise_std = noise_std_ratio * np.mean(np.abs(response))
        response += np.random.normal(0, noise_std, response.shape)

    # Create a DataFrame for the response, index by photodetector.bias_array and columns by unknown_spectrum.spectrum.columns
    response_df = pd.DataFrame(
        response,
        columns=['response'],
        index=photodetector.responsivity.columns
    )
    response_df.index.name = 'bias'
    response_df.columns.name = 'response'
    return response_df
