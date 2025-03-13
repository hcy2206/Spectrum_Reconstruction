from typing import Callable, Literal, overload

import numpy as np
import plotly.express as px

from .SpectrumReconstructionAdvance import IdealSemiconductorPhotoDetector, IncidentSpectrum, \
    simulate_response_matrix, SimulationSpectrum, simulate_unknown_response
from .SpectrumReconstructionBasic import SpectrumReconstructionBasicHighPerformance
from .Utility import smooth_responsivity


class SpectrumReconstructionSimulation:

    @overload
    def __init__(
            self,
            bias_array: np.ndarray[float],  # Bias array [m]
            wavelength_array: np.ndarray[float],  # Wavelength array [m]
            base_function_name: Literal["blackbody"],  # Base function name
            black_body_temperature: np.ndarray[float],  # Black body temperature [K]
            photo_detector_bias_mode: Literal["normal_move", "increase_band_gap"] = "normal_move",
            # Photo detector bias mode
            delta_lambda: float = 30e-9,  # Transition width [m]
            e_g_ev: float = 0.75,  # Bandgap energy [eV]
            eta: float = 1.0,  # Quantum efficiency
            visible_blind_cutoff: float = -1,  # Visible blind cut-off wavelength [m]
            **kwargs
    ) -> None:
        ...

    @overload
    def __init__(
            self,
            bias_array: np.ndarray[float],  # Bias array [m]
            wavelength_array: np.ndarray[float],  # Wavelength array [m]
            base_function_name: Literal["gaussian"],  # Base function name
            sigma: float,  # Standard deviation for Gaussian [m]
            mu: np.ndarray[float],  # Mean for Gaussian [m]
            photo_detector_bias_mode: Literal["normal_move", "increase_band_gap"] = "normal_move",
            # Photo detector bias mode
            delta_lambda: float = 30e-9,  # Transition width [m]
            e_g_ev: float = 0.75,  # Bandgap energy [eV]
            eta: float = 1.0,  # Quantum efficiency
            visible_blind_cutoff: float = -1,  # Visible blind cut-off wavelength [m]
            **kwargs
    ) -> None:
        ...

    def __init__(
            self,
            bias_array: np.ndarray[float],  # Bias array [m]
            wavelength_array: np.ndarray[float],  # Wavelength array [m]
            base_function_name: Literal["blackbody", "gaussian"],  # Base function name
            photo_detector_base_function: Callable[..., np.ndarray[float]] = smooth_responsivity,
            # Photodetector base function
            photo_detector_bias_mode: Literal["normal_move", "increase_band_gap"] = "normal_move",
            # Photo detector bias mode
            delta_lambda: float = 30e-9,  # Transition width [m]
            e_g_ev: float = 0.75,  # Band gap energy [eV]
            eta: float = 1.0,  # Quantum efficiency
            visible_blind_cutoff: float = -1,  # Visible blind cut-off wavelength [m]
            **kwargs
    ) -> None:
        self.bias_array = bias_array
        self.wavelength_array = wavelength_array
        self.base_function_name = base_function_name
        self.photo_detector_base_function = photo_detector_base_function
        self.photo_detector_bias_mode = photo_detector_bias_mode
        self.delta_lambda = delta_lambda
        self.e_g_ev = e_g_ev
        self.eta = eta
        self.visible_blind_cutoff = visible_blind_cutoff
        self.kwargs = kwargs
        match base_function_name:
            case "blackbody":
                try:
                    self.black_body_temperature = kwargs["black_body_temperature"]
                except KeyError:
                    raise ValueError("Black body temperature is required for blackbody base function")
                if not isinstance(self.black_body_temperature, np.ndarray):
                    raise ValueError("Black body temperature should be a numpy array")
            case "gaussian":
                try:
                    self.sigma = kwargs["sigma"]
                    self.mu = kwargs["mu"]
                except KeyError:
                    raise ValueError("Sigma and mu are required for gaussian base function")
                if not isinstance(self.sigma, float):
                    raise ValueError("Sigma should be a float")
                if not isinstance(self.mu, np.ndarray):
                    raise ValueError("Mu should be a numpy array")

        # Initialize the photodetector class
        self.photo_detector = IdealSemiconductorPhotoDetector(
            bias_array=self.bias_array,
            e_g_ev=self.e_g_ev,
            eta=self.eta,
            delta_lambda=self.delta_lambda,
            base_function=self.photo_detector_base_function,
            wavelength=self.wavelength_array,
            visible_blind_cutoff=self.visible_blind_cutoff,
            bias_mode=self.photo_detector_bias_mode
        )

        # Initialize the training IncidentSpectrum
        match base_function_name:
            case "blackbody":
                self.incident_spectrum = IncidentSpectrum(
                    wavelength=self.wavelength_array,
                    base_function_name="blackbody",
                    T=self.black_body_temperature
                )
                self.response_pivot_columns = self.black_body_temperature
                self.response_pivot_columns_name = 'T'
            case "gaussian":
                self.incident_spectrum = IncidentSpectrum(
                    wavelength=self.wavelength_array,
                    base_function_name="gaussian",
                    sigma=self.sigma,
                    mu=self.mu
                )
                self.response_pivot_columns = self.mu
                self.response_pivot_columns_name = 'mu'

        # calculate the response
        self.response_pivot = simulate_response_matrix(
            photodetector=self.photo_detector,
            incident_spectrum=self.incident_spectrum
        )
        self.response_pivot_index = self.photo_detector.bias_array
        self.response_pivot_index_name = 'bias'
        # self.response_melted = self.response_pivot.reset_index().melt(
        #     id_vars=self.response_pivot.index.name,
        #     var_name=self.response_pivot.columns.name,
        #     value_name="value"
        # )

        # Initialize the SpectrumReconstructionBasic class
        match base_function_name:
            case "blackbody":
                self.spectrum_reconstruction = SpectrumReconstructionBasicHighPerformance(
                    training_data=self.response_pivot,
                    internal_var_col_name=self.response_pivot_index_name,
                    internal_var_col=self.response_pivot_index,
                    external_var_col_name=self.response_pivot_columns_name,
                    external_var_col=self.response_pivot_columns,
                    dependent_var_col_name="value",
                    base_func="blackbody",
                    verify_pivot_data=True
                )
            case "gaussian":
                self.spectrum_reconstruction = SpectrumReconstructionBasicHighPerformance(
                    training_data=self.response_pivot,
                    internal_var_col_name=self.response_pivot_index_name,
                    internal_var_col=self.response_pivot_index,
                    external_var_col_name=self.response_pivot_columns_name,
                    external_var_col=self.response_pivot_columns,
                    dependent_var_col_name="value",
                    base_func="gaussian",
                    verify_pivot_data=True,
                    sigma=self.sigma
                )

    @property
    def response_mapping_figure(self):
        fig = px.imshow(
            self.response_pivot,
            labels=dict(
                x=self.response_pivot_columns_name,
                y=self.response_pivot_index_name,
                color="Response"
            ),
            x=self.response_pivot_columns,
            y=self.response_pivot_index,
            # color_continuous_scale="viridis",
            aspect="auto",
            title="Response Mapping"
        )
        fig.update_xaxes(side="top")
        return fig

    def reconstruct_spectrum(self,
                             simulation_spectrum: SimulationSpectrum,
                             method: Literal['normal', 'l1', 'l2', 'ElasticNet'],
                             add_gaussian_noise: bool = False,
                             noise_std_ratio: float = 0.001,
                             pivot_pass_in_test_data: bool = False,
                             **kwargs
                             ) -> np.ndarray:

        data_testing = simulate_unknown_response(
            photodetector=self.photo_detector,
            unknown_spectrum=simulation_spectrum,
            add_gaussian_noise=add_gaussian_noise,
            noise_std_ratio=noise_std_ratio
        )

        self.spectrum_reconstruction.reconstruct_spectrum(
            testing_data=data_testing,
            method=method,
            **kwargs
        )
        return self.spectrum_reconstruction.a

    @property
    def reconstruction_spectrum_figure(self):
        fig = px.line(
            x=self.wavelength_array,
            y=self.spectrum_reconstruction.spectrum(
                lambda_=self.wavelength_array,
                normalize=False
            ),
            title="Reconstructed Spectrum",
            labels=dict(
                x="Wavelength",
                y="Response"
            )
        )
        return fig
