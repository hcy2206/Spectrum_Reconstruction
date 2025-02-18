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
                 bias_array: np.ndarray,  # Bias array [m]
                 e_g_ev: float,  # Band gap energy [eV]
                 eta: float = 1.0,  # Quantum efficiency
                 delta_lambda: float = 30e-9,  # Transition width [m]
                 base_function: Callable[..., float] = smooth_responsivity,  # Base responsivity function
                 wavelength: np.ndarray = np.linspace(0, 2.0e-6, 500),  # Wavelength array [m]
                 visible_blind_cutoff: float = -1  # Cut-off wavelength for visible blind [m]
                 ):
        self.base_function = base_function
        self.bias_array = bias_array
        self.e_g = e_g_ev * q
        self.eta = eta
        self.delta_lambda = delta_lambda
        self.wavelength = wavelength
        self.visible_blind_cutoff = visible_blind_cutoff

    def _responsivity_func(self,
                           bias: float,
                           wavelength: np.ndarray = None,
                           ): # -> np.ndarray:
        if wavelength is not None:
            lambda_ = wavelength - bias
        else:
            lambda_ = self.wavelength - bias
        responsivity = np.asarray(self.base_function(lambda_, self.e_g, self.delta_lambda, self.eta))
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
            responsivity = self._responsivity_func(bias)
            # Append a new row to the DataFrame
            new_data = pd.DataFrame({
                'wavelength': self.wavelength * 1e9,  # Convert to nm for better readability
                'responsivity': responsivity,
                'bias': [f'{bias * 1e9:.0f} nm'] * len(self.wavelength)
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

        # Create figure using plotly express
        fig = px.line(
            fig_data,
            x='wavelength',
            y='responsivity',
            color='bias',
            labels={
                'wavelength': 'Wavelength (nm)',
                'responsivity': 'Responsivity (A/W)',
                'bias': 'Bias'
            },
            title='Photodetector Responsivity vs Wavelength for Different Bias Values'
        )

        return fig

class IncidentSpectrum:
    @overload
    def __init__(self,
                 wavelength: np.ndarray,  # Wavelength array [m],
                 base_function_name: Literal['gaussian'],  # Base spectrum function
                 sigma: float,  # Standard deviation for Gaussian
                 mu: np.ndarray  # Mean for Gaussian
                 ):
        ...

    @overload
    def __init__(self,
                 wavelength: np.ndarray,  # Wavelength array [m],
                 base_function_name: Literal['blackbody'],  # Base spectrum function
                 T: np.ndarray # Temperature for Blackbody
                 ):
        ...

    def __init__(self,
                 wavelength: np.ndarray,  # Wavelength array [m],
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
                        'wavelength': self.wavelength * 1e9,  # Convert to nm for better readability
                        'spectrum': spectrum,
                        'mu': [f'{mu * 1e9:.0f} nm'] * len(self.wavelength)
                    })
                    spectrum_data = pd.concat([spectrum_data, new_data], ignore_index=True)
                return spectrum_data
            case 'blackbody':
                spectrum_data = pd.DataFrame(columns=['wavelength', 'spectrum', 'T'])
                for T in self.T:
                    spectrum = blackbody(self.wavelength, T)
                    new_data = pd.DataFrame({
                        'wavelength': self.wavelength * 1e9,  # Convert to nm for better readability
                        'spectrum': spectrum,
                        'T': [f'{T:.0f} K'] * len(self.wavelength)
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
            case 'blackbody':
                color_name = 'T'
            case _:
                raise ValueError("Unsupported base function.")

        # Create figure using plotly express
        fig = px.line(
            fig_data,
            x='wavelength',
            y='spectrum',
            color=color_name,
            labels={
                'wavelength': 'Wavelength (nm)',
                'spectrum': 'Spectrum',
                self.base_function_name: self.base_function_name
            },
            title=f'{self.base_function_name.capitalize()} Spectrum vs Wavelength'
        )

        return fig




