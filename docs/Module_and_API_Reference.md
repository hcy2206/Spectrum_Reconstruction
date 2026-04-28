# SpectrumReconstruction API Reference

**Chenyu Huang**$^{*\dagger}$

$^*$*State Key Laboratory of Infrared Physics, Shanghai Institute of Technical Physics, Chinese Academy of Sciences*

$^\dagger$*University of Chinese Academy of Sciences*

[TOC]

## Overview

The `SpectrumReconstruction` library is designed to simulate the spectrum reconstruction process. It primarily contains `SpectrumReconstructionBasic`, the core class for spectrum reconstruction, and `SpectrumReconstructionSimulation`, the simulation class. The main class and function hierarchy is as follows:

```
SpectrumReconstruction
├── Class: SpectrumReconstructionBasic
├── Class: SpectrumReconstructionSimulation
├── Submodule: SpectrumReconstructionBasic
│   └── Class: SpectrumReconstructionBasic
│       ├── Method: __init__
│       ├── Method: base_func
│       ├── Method: reconstruct_spectrum
│       └── Method: spectrum
├── Submodule: SpectrumReconstructionAdvance
│   ├── Class: IdealSemiconductorPhotoDetector
│   ├── Class: IncidentSpectrum
│   ├── Class: SimulationSpectrum
│   │   ├── Method: set_spectrum
│   │   └── Property: spectrum_figure
│   ├── Function: simulate_response_matrix
│   ├── Function: simulate_unknown_response
│   └── Function: change_visible_blind_cutoff_parameter
├── Submodule: SpectrumReconstructionSimulation
│   └── Class: SpectrumReconstructionSimulation
│       ├── Method: __init__
│       ├── Method: reconstruct_spectrum
│       ├── Property: response_mapping_figure
│       └── Property: reconstruction_spectrum_figure
└── Submodule: Utility
    ├── Function: blackbody
    ├── Function: gaussian
    ├── Function: ideal_responsivity
    ├── Function: smooth_responsivity
    ├── Function: smooth_responsivity_visible_blind
    ├── Function: gaussian_spectrum_sum
    └── Function: blackbody_spectrum_sum
```

## Class `SpectrumReconstructionBasic`

`SpectrumReconstructionBasic` is the core class for spectrum reconstruction, implementing the fundamental reconstruction functionality.

For usage examples, see [eample.ipynb](../examples/eample.ipynb).

## Class `SpectrumReconstructionSimulation`

`SpectrumReconstructionSimulation` is designed for full end-to-end simulation of the spectrum reconstruction pipeline. It is the highest-level abstraction class in the module.

### Constructor `__init__`

The constructor accepts the following parameters:

1. `bias_array`: Spectral responsivity bias array of the photodetector, type `numpy.ndarray`. Simulates the spectral shift of the detector under different operating conditions. Unit: m.
2. `wavelength_array`: Global wavelength array, type `numpy.ndarray`. Defines the spectral range and step size for the entire simulation. Unit: m.
3. `base_function_name`: Base function name, type `str`. Specifies the basis functions used in the simulation. Supported values: `'gaussian'` and `'blackbody'`.
4. `sigma`: Standard deviation of the Gaussian function, type `float`. Determines the shape of the Gaussian; simulates incident light with a Gaussian profile of FWHM equal to `sigma`. Unit: m.
5. `mu`: Mean array of the Gaussian functions, type `numpy.ndarray`. Specifies the center wavelengths of the Gaussian basis functions used during training. Unit: m.
6. `black_body_temperature`: Blackbody temperature array, type `numpy.ndarray`. Specifies the temperatures of the blackbody basis functions used during training. Unit: K.
7. `photo_detector_bias_mode`: Photodetector bias mode, type `str`. Supported values:
   - `'normal_move'`: Shifts the detector's spectral responsivity along the wavelength axis by the values in `bias_array`.
   - `'increase_band_gap'`: Changes the detector's bandgap energy, shifting the cutoff wavelength according to `bias_array`.
8. `delta_lambda`: Smoothing transition width of the photodetector responsivity, type `float`. Unit: m. Default: 30e-9 (30 nm). See `smooth_responsivity` for details.
9. `e_g_ev`: Bandgap energy of the photodetector, type `float`. Unit: eV. Default: 0.75 (corresponding to ~1653 nm, approximately the cutoff wavelength of an InGaAs detector).
10. `eta`: Quantum efficiency of the photodetector, type `float`. Default: 1.
11. `visible_blind_cutoff`: Visible-light cutoff wavelength of the photodetector, used to simulate visible-blind characteristics. Type `float`. Unit: m. Default: -1 (negative value disables the visible-blind cutoff).

### Attribute Summary

#### Primary Input Attributes

1. `bias_array`: Bias array. Unit: m.
2. `wavelength_array`: Wavelength array. Unit: m.
3. `base_function_name`: Base function name; either `"blackbody"` or `"gaussian"`.
4. `photo_detector_base_function`: Photodetector base function; default is `smooth_responsivity`.
5. `photo_detector_bias_mode`: Photodetector bias mode; either `"normal_move"` or `"increase_band_gap"`.
6. `delta_lambda`: Transition width. Default: 30 nm.
7. `e_g_ev`: Bandgap energy. Default: 0.75 eV.
8. `eta`: Quantum efficiency. Default: 1.0.
9. `visible_blind_cutoff`: Visible-light cutoff wavelength. Default: -1 m.

#### Conditional Attributes Based on `base_function_name`

##### Blackbody mode — `base_function_name = 'blackbody'`

1. `black_body_temperature`: Blackbody temperature array. Unit: K.

##### Gaussian mode — `base_function_name = 'gaussian'`

1. `sigma`: Standard deviation of the Gaussian function. Unit: m.
2. `mu`: Mean array of the Gaussian functions. Unit: m.

#### Computed Attributes

1. `photo_detector`: Instance of `IdealSemiconductorPhotoDetector`, used to simulate the photodetector.
2. `incident_spectrum`: Instance of `IncidentSpectrum`, used to simulate the incident training spectra.
3. `response_pivot`: Response matrix representing the detector's response to each incident spectrum.
4. `spectrum_reconstruction`: Instance of `SpectrumReconstructionBasicHighPerformance`, used to perform the spectrum reconstruction.

### Properties (@property)

1. `response_mapping_figure`: Returns a `plotly.express.imshow` heatmap of the response matrix.
2. `reconstruction_spectrum_figure`: Returns a `plotly.express.line` plot of the reconstructed spectrum.

### Method `reconstruct_spectrum`

Simulates the spectrum reconstruction process. Parameters:

1. `simulation_spectrum`: Instance of `SimulationSpectrum`. The unknown input spectrum to be reconstructed.
2. `method`: Reconstruction method. Supported values: `'normal'`, `'l1'`, `'l2'`, `'ElasticNet'`. Default: `'normal'`.
3. `add_gaussian_noise`: Whether to add Gaussian noise to the simulated detector response. Default: `False`.
4. `noise_std_ratio`: Ratio of noise standard deviation to the mean response magnitude. Default: 0.1.

**Returns:** `numpy.ndarray` — the solved spectral coefficient array `a`.

## Class `IdealSemiconductorPhotoDetector`

`IdealSemiconductorPhotoDetector` simulates the spectral responsivity of an ideal semiconductor photodetector. It computes the detector's responsivity across all wavelengths under different bias modes.

### Constructor `__init__`

1. `bias_array`: Bias array, type `numpy.ndarray`. Unit: m. Defines the spectral shift of the detector at each bias state.
2. `e_g_ev`: Bandgap energy, type `float`. Unit: eV. Determines the cutoff wavelength.
3. `eta`: Quantum efficiency, type `float`. Default: 1.0.
4. `delta_lambda`: Transition width near the cutoff, type `float`. Unit: m. Default: 30e-9 (30 nm). Controls the smoothness of the responsivity roll-off.
5. `base_function`: Base responsivity function, type `Callable`. Default: `smooth_responsivity`. Used to compute the underlying responsivity curve.
6. `wavelength`: Wavelength array, type `numpy.ndarray`. Unit: m. Default: `np.linspace(0, 2.0e-6, 500)`.
7. `visible_blind_cutoff`: Visible-light cutoff wavelength, type `float`. Unit: m. Default: -1 (disabled).
8. `bias_mode`: Bias mode, type `str`. Supported values:
   - `'normal_move'`: Shifts the spectral responsivity along the wavelength axis.
   - `'increase_band_gap'`: Shifts the cutoff wavelength by modifying the bandgap energy.
   - `'decrease_eta'`: Shifts the cutoff wavelength while simultaneously reducing quantum efficiency; `eta` decreases as bias increases.

### Properties (@property)

1. `responsivity` (cached_property): Returns a 2-D `numpy.ndarray` of shape `(num_wavelength, num_bias)`. Computed via vectorized broadcasting to obtain responsivity at all bias values simultaneously.
2. `_responsivity`: Converts the responsivity matrix into a long-format `pandas.DataFrame` with columns `wavelength`, `bias`, and `responsivity`.

### Methods

1. `responsivity_figure_show`: Returns a `plotly.express.line` figure showing detector responsivity vs. wavelength for each bias value.

## Class `IncidentSpectrum`

`IncidentSpectrum` simulates a set of training incident spectra. Supports two modes: Gaussian and blackbody radiation.

### Constructor `__init__`

1. `wavelength`: Wavelength array, type `numpy.ndarray`. Unit: m.
2. `base_function_name`: Base function name, type `str`. Supported values: `'gaussian'` and `'blackbody'`.

**Additional parameters for Gaussian mode:**

3. `sigma`: Standard deviation of the Gaussian, type `float`. Unit: m. Default: 1e-9.
4. `mu`: Center wavelength array, type `numpy.ndarray`. Unit: m.

**Additional parameters for Blackbody mode:**

3. `T`: Temperature array, type `numpy.ndarray`. Unit: K.

### Properties (@property)

1. `spectrum`: Returns a 2-D `numpy.ndarray` of shape `(num_wavelength, num_params)`. Computed using broadcasting to evaluate all spectra simultaneously.
2. `_spectrum`: Converts the spectrum matrix into a long-format `pandas.DataFrame` with columns `wavelength`, the parameter name (`mu` or `T`), and `spectrum`.

### Methods

1. `spectrum_figure_show`: Returns a `plotly.express.line` figure showing each incident spectrum vs. wavelength.

## Class `SimulationSpectrum`

`SimulationSpectrum` constructs an arbitrary custom spectrum to serve as the unknown input spectrum for reconstruction. Unlike `IncidentSpectrum`, this class allows the user to define any spectrum shape via a custom function.

### Constructor `__init__`

1. `wavelength_array`: Wavelength array, type `numpy.ndarray`. Unit: m.
2. `spectrum_function`: Spectrum function, type `Callable`. Must accept the wavelength array as its first argument and return the corresponding spectral intensity array.

### Methods

1. `set_spectrum(**kwargs)`: Calls `spectrum_function` to generate and store the spectrum. `**kwargs` are forwarded to the spectrum function. **Returns:** `numpy.ndarray` — the generated spectrum.

### Properties (@property)

1. `spectrum`: Returns the stored spectrum as a `numpy.ndarray`. Raises `ValueError` if `set_spectrum` has not been called.
2. `spectrum_figure`: Returns a `plotly.express.line` figure of the spectrum. Raises `ValueError` if `set_spectrum` has not been called.

## Function `simulate_response_matrix`

Computes the response matrix of the photodetector to a set of incident training spectra, by multiplying the responsivity matrix and the incident spectrum matrix.

**Parameters:**

1. `photodetector`: Instance of `IdealSemiconductorPhotoDetector`.
2. `incident_spectrum`: Instance of `IncidentSpectrum`.

**Returns:** `numpy.ndarray` — response matrix of shape `(num_bias, num_params)`, where rows correspond to bias values and columns to incident spectrum parameters (`mu` or `T`).

**Note:** The wavelength arrays of `photodetector` and `incident_spectrum` must be identical; otherwise a `ValueError` is raised.

## Function `simulate_unknown_response`

Simulates the photodetector's response to an unknown spectrum. Optionally adds Gaussian noise to mimic real measurement conditions.

**Parameters:**

1. `photodetector`: Instance of `IdealSemiconductorPhotoDetector`.
2. `unknown_spectrum`: Instance of `SimulationSpectrum`.
3. `add_gaussian_noise`: Whether to add Gaussian noise to the response. Type `bool`. Default: `False`.
4. `noise_std_ratio`: Ratio of noise standard deviation to the mean absolute response. Type `float`. Default: 0.01.

**Returns:** `numpy.ndarray` — 1-D response vector of length `num_bias`.

**Note:** The wavelength arrays of `photodetector` and `unknown_spectrum` must be identical; otherwise a `ValueError` is raised.

## Function `blackbody`

Computes the spectral radiance of a blackbody according to Planck's law. Calculations are performed in log-space for numerical stability. JIT-compiled with `@njit`.

$$B(\lambda, T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{hc/(\lambda k T)} - 1}$$

**Parameters:**

1. `lambda_`: Wavelength, type `float` or `numpy.ndarray`. Unit: m.
2. `t`: Temperature, type `float` or `numpy.ndarray`. Unit: K.

**Returns:** `float` or `numpy.ndarray` — spectral radiance. Unit: W/(m²·m). Supports broadcasting between scalars and arrays.

## Function `gaussian`

Computes the (unnormalized) Gaussian function. JIT-compiled with `@njit`.

$$G(\lambda, \mu, \sigma) = \exp\left(-\frac{(\lambda - \mu)^2}{2\sigma^2}\right)$$

**Parameters:**

1. `lambda_`: Independent variable (wavelength), type `float` or `numpy.ndarray`. Unit: m.
2. `mu`: Mean (center wavelength), type `float` or `numpy.ndarray`. Unit: m.
3. `sigma`: Standard deviation, type `float`. Unit: m.

**Returns:** `float` or `numpy.ndarray` — Gaussian value (unnormalized; peak = 1). Supports broadcasting.

## Function `ideal_responsivity`

Computes the responsivity of an ideal semiconductor photodetector. Responsivity increases linearly below the cutoff wavelength and is zero above it.

$$R(\lambda) = \begin{cases} \eta \cdot \dfrac{q\lambda}{hc} & \lambda \leq \lambda_g \\ 0 & \lambda > \lambda_g \end{cases}$$

where $\lambda_g = hc/E_g$ is the cutoff wavelength.

**Parameters:**

1. `lambda_`: Wavelength array, type `numpy.ndarray`. Unit: m.
2. `e_g`: Bandgap energy, type `float`. Unit: J.
3. `eta`: Quantum efficiency, type `float`. Default: 1.0.

**Returns:** `numpy.ndarray` — responsivity array. Unit: A/W.

## Function `smooth_responsivity`

Computes a smooth responsivity curve. Compared to `ideal_responsivity`, uses a Sigmoid function near the cutoff wavelength for a smooth roll-off that better approximates real detector behavior. JIT-compiled with `@njit`.

$$R(\lambda) = \eta \cdot \frac{q\lambda}{hc} \cdot \frac{1}{1 + \exp\!\left(\dfrac{\lambda - \lambda_g}{\Delta\lambda}\right)}$$

where $\lambda_g = hc/E_g$ is the cutoff wavelength and $\Delta\lambda$ is the transition width.

**Parameters:**

1. `lambda_`: Wavelength array, type `numpy.ndarray`. Unit: m.
2. `e_g`: Bandgap energy, type `float`. Unit: J.
3. `delta_lambda`: Transition width, type `float`. Unit: m. Default: 30e-9 (30 nm).
4. `eta`: Quantum efficiency, type `float`. Default: 1.0.

**Returns:** `numpy.ndarray` — smooth responsivity array. Unit: A/W.

## Function `smooth_responsivity_visible_blind`

**Deprecated.** No longer in use. Its functionality has been replaced by the `visible_blind_cutoff` parameter of `IdealSemiconductorPhotoDetector`.

## Function `gaussian_spectrum_sum`

Computes the weighted superposition of multiple Gaussian spectra:

$$S(\lambda) = \sum_i \alpha_i \cdot G(\lambda,\, \mu_i,\, \sigma_i)$$

**Parameters:**

1. `_wavelength`: Wavelength array, type `numpy.ndarray`. Unit: m.
2. `_mu`: Center wavelength array for each Gaussian component, type `numpy.ndarray`. Unit: m.
3. `_sigma`: Standard deviation(s), type `float` or `numpy.ndarray`. Unit: m. If `float`, all components share the same sigma; if `numpy.ndarray`, each component uses its own sigma.
4. `_alpha`: Weight coefficient array, type `numpy.ndarray`.

**Returns:** `numpy.ndarray` — superimposed spectrum. Commonly used with `SimulationSpectrum` to construct complex test spectra.

## Function `blackbody_spectrum_sum`

Computes the weighted superposition of multiple blackbody spectra:

$$S(\lambda) = \sum_i \alpha_i \cdot B(\lambda,\, T_i)$$

**Parameters:**

1. `_wavelength`: Wavelength array, type `numpy.ndarray`. Unit: m.
2. `_T`: Temperature array for each blackbody component, type `numpy.ndarray`. Unit: K.
3. `_alpha`: Weight coefficient array, type `numpy.ndarray`.

**Returns:** `numpy.ndarray` — superimposed spectrum. Commonly used with `SimulationSpectrum` to construct complex test spectra.
