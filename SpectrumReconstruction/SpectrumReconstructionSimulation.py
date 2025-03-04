from typing import Callable, Literal, overload

import numpy as np
import plotly.express as px
from line_profiler_pycharm import profile

from SpectrumReconstruction import SpectrumReconstructionBasic
from SpectrumReconstruction.SpectrumReconstructionAdvance import IdealSemiconductorPhotoDetector, IncidentSpectrum, \
    simulate_response_matrix, SimulationSpectrum, simulate_unknown_response
from SpectrumReconstruction.Utility import smooth_responsivity


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

    @profile
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
            case "gaussian":
                self.incident_spectrum = IncidentSpectrum(
                    wavelength=self.wavelength_array,
                    base_function_name="gaussian",
                    sigma=self.sigma,
                    mu=self.mu
                )

        # calculate the response
        self.response_pivot = simulate_response_matrix(
            photodetector=self.photo_detector,
            incident_spectrum=self.incident_spectrum
        )
        self.response_melted = self.response_pivot.reset_index().melt(
            id_vars=self.response_pivot.index.name,
            var_name=self.response_pivot.columns.name,
            value_name="value"
        )

        # Initialize the SpectrumReconstructionBasic class
        match base_function_name:
            case "blackbody":
                self.spectrum_reconstruction = SpectrumReconstructionBasic(
                    training_data=self.response_melted,
                    internal_var_col_name=self.response_pivot.index.name,
                    external_var_col_name=self.response_pivot.columns.name,
                    dependent_var_col_name="value",
                    base_func="blackbody",
                    verify_pivot_data=True
                )
            case "gaussian":
                self.spectrum_reconstruction = SpectrumReconstructionBasic(
                    training_data=self.response_melted,
                    internal_var_col_name=self.response_pivot.index.name,
                    external_var_col_name=self.response_pivot.columns.name,
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
                x=self.response_pivot.columns.name,
                y=self.response_pivot.index.name,
                color="Response"
            ),
            x=self.response_pivot.columns,
            y=self.response_pivot.index,
            # color_continuous_scale="viridis",
            aspect="auto",
            title="Response Mapping"
        )
        fig.update_xaxes(side="top")
        return fig

    # @property
    # def simulation_spectrum(self):
    #     return self.simulation_spectrum
    #
    # @simulation_spectrum.setter
    # def simulation_spectrum(self, simulation_spectrum: SimulationSpectrum):
    #     self.simulation_spectrum = simulation_spectrum
    @profile
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

        a = self.spectrum_reconstruction.reconstruct_spectrum(
            testing_data=data_testing,
            method=method,
            pivot_pass_in_test_data=pivot_pass_in_test_data,
            **kwargs
        )
        return a

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


if __name__ == '__main__':
    factor = 5

    # Set the Parameters
    bias_array = np.linspace(0e-9, 400e-9, 200 * factor + 1)  # Bias array [m]
    wavelength_array = np.linspace(400e-9, 1800e-9, 14000 * factor + 1)  # Wavelength array [m]
    base_function_name = "gaussian"  # Base function name
    # sigma = 1e-9 # Standard deviation for Gaussian [m]
    sigma = 0.5e-9 / 2 / np.sqrt(2 * np.log(2))  # Standard deviation for Gaussian [m] # FWHM = 2*sqrt(2*ln(2))*sigma
    mu = np.linspace(800e-9, 1800e-9, 200 * factor + 1)  # Mean for Gaussian [m]
    black_body_temperature = np.linspace(800, 1400, 25)  # Black body temperature [K]
    photo_detector_bias_mode = "normal_move"  # Photo detector bias mode
    delta_lambda = 10e-9  # Photodetector Smooth Parameter [m]
    e_g_ev = 0.8  # Bandgap energy [eV]
    eta = 1  # Quantum efficiency
    visible_blind_cutoff = -1  # Visible blind cut-off wavelength [m]

    from SpectrumReconstruction import SpectrumReconstructionSimulation
    from SpectrumReconstruction import SpectrumReconstructionAdvance as SRAdvance
    from SpectrumReconstruction import Utility as SRUtility
    import numpy as np

    # Create the Simulation
    srs = SpectrumReconstructionSimulation(
        bias_array=bias_array,
        wavelength_array=wavelength_array,
        base_function_name=base_function_name,
        sigma=sigma,
        mu=mu,
        photo_detector_bias_mode=photo_detector_bias_mode,
        delta_lambda=delta_lambda,
        e_g_ev=e_g_ev,
        eta=eta,
        visible_blind_cutoff=visible_blind_cutoff
    )
    srs.response_mapping_figure.show()

    # Set the Spectrum
    spectrum = SRAdvance.SimulationSpectrum(
        wavelength_array=wavelength_array,
        spectrum_function=SRUtility.gaussian_spectrum_sum
    )
    spectrum.set_spectrum(
        _mu=np.array([1754.0e-9, 1754.5e-9]),
        _sigma=sigma,
        _alpha=np.array([1.0, 1.0])
    )

    spectrum.spectrum_figure.show()

    srs.reconstruct_spectrum(
        simulation_spectrum=spectrum,
        method='l1',
        add_gaussian_noise=True,
        noise_std_ratio=0.01,
        lambda_reg=1e2,
        alpha=0.5
    )
    srs.reconstruction_spectrum_figure.show()
