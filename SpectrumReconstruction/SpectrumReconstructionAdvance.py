from typing import Callable, Literal, overload
from SpectrumReconstruction.SpectrumReconstructionBasic import SpectrumReconstructionBasic
from SpectrumReconstruction.Utility import smooth_responsivity, gaussian, blackbody
import numpy as np
import pandas as pd
import plotly.express as px

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

    @property
    def _responsivity(self):
        # Initialize an empty DataFrame to store the data
        responsivity_data = pd.DataFrame(columns=['wavelength', 'responsivity', 'bias'])

        for bias in self.bias_array:
            responsivity = self._responsivity_func(bias, self.bias_mode)
            # Append a new row to the DataFrame
            new_data = pd.DataFrame({
                'wavelength': self.wavelength,  # Convert to nm for better readability
                'responsivity': responsivity,
                'bias': bias
            })
            # new_data = new_data.dropna(axis=1, how='all')  # 删除所有值为NA的列
            responsivity_data = pd.concat([responsivity_data, new_data], ignore_index=True)

        return responsivity_data

    @property
    def responsivity(self):
        return self._responsivity.pivot(index='wavelength', columns='bias', values='responsivity')

    def responsivity_figure_show(self):
        # Initialize an empty DataFrame to store the data
        fig_data = self._responsivity
        fig_data['bias'] = (fig_data['bias'] * 1e9).apply(lambda x: f"{x:.0f} nm")  # Convert bias to nm and append "nm"

        # Create figure using plotly express
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
                 mu: np.ndarray[float]  # Mean for Gaussian
                 ):
        ...

    @overload
    def __init__(self,
                 wavelength: np.ndarray[float],  # Wavelength array [m],
                 base_function_name: Literal['blackbody'],  # Base spectrum function
                 T: np.ndarray[float] # Temperature for Blackbody
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

    @property
    def _spectrum(self):
        match self.base_function_name:
            case 'gaussian':
                spectrum_data = pd.DataFrame(columns=['wavelength', 'spectrum', 'mu'])
                for mu in self.mu:
                    spectrum = gaussian(self.wavelength, mu, self.sigma)
                    new_data = pd.DataFrame({
                        'wavelength': self.wavelength,  # Convert to nm for better readability
                        'spectrum': spectrum,
                        'mu': mu
                    })
                    spectrum_data = pd.concat([spectrum_data, new_data], ignore_index=True)
                return spectrum_data
            case 'blackbody':
                spectrum_data = pd.DataFrame(columns=['wavelength', 'spectrum', 'T'])
                for T in self.T:
                    spectrum = blackbody(self.wavelength, T)
                    new_data = pd.DataFrame({
                        'wavelength': self.wavelength,  # Convert to nm for better readability
                        'spectrum': spectrum,
                        'T': T
                    })
                    spectrum_data = pd.concat([spectrum_data, new_data], ignore_index=True)
                return spectrum_data
            case _:
                raise ValueError("Unsupported base function.")

    @property
    def spectrum(self):
        match self.base_function_name:
            case 'gaussian':
                return self._spectrum.pivot(index='wavelength', columns='mu', values='spectrum')
            case 'blackbody':
                return self._spectrum.pivot(index='wavelength', columns='T', values='spectrum')
            case _:
                raise ValueError("Unsupported base function.")

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
                 wavelength: np.ndarray[float],  # Wavelength array [m]
                 spectrum_function: Callable[..., np.ndarray[float]]  # Spectrum function
                ):
        self._spectrum = None
        self.wavelength = wavelength
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

    responsivity = photodetector.responsivity.values
    spectrum = incident_spectrum.spectrum.values

    response = responsivity.T @ spectrum

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