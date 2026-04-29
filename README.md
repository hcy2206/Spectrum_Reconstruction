<!-- SpectrumReconstruction -->

<!--suppress HtmlDeprecatedAttribute -->
<div align="center" style="text-align:center">
   <h1> SpectrumReconstruction </h1>
   <p>
      Spectrum Reconstruction &amp; Photodetector Simulation Python Module<br>
      <code><b> v0.2 </b></code>
   </p>
   <p>
      <b>Chenyu Huang</b><sup>*&dagger;</sup>, <b>Weida Hu</b><sup>*&dagger;#</sup><br>
      <sup>*</sup><i>State Key Laboratory of Infrared Physics, Shanghai Institute of Technical Physics, Chinese Academy of Sciences</i><br>
      <sup>&dagger;</sup><i>University of Chinese Academy of Sciences</i><br>
      <sup>#</sup><i>Corresponding author</i>
   </p>
   <p>
      <img alt="GitHub Top Language" src="https://img.shields.io/github/languages/top/hcy2206/Spectrum_Reconstruction?label=Python">
      <img alt="GitHub License" src="https://img.shields.io/github/license/hcy2206/Spectrum_Reconstruction?label=License"/>
   </p>
   <p>
      English | <a href="README_CN.md">中文</a>
   </p>
</div>

## Introduction

**SpectrumReconstruction** is a Python package for simulating the full pipeline of computational spectrum reconstruction based on semiconductor photodetector responses.

It provides:
- **Photodetector modeling** &mdash; simulate ideal semiconductor photodetector responsivity with adjustable bandgap, quantum efficiency, bias modes (`normal_move`, `increase_band_gap`, `decrease_eta`), and visible-blind characteristics.
- **Incident spectrum generation** &mdash; create training spectra using Gaussian or blackbody radiation models.
- **Response matrix computation** &mdash; calculate the detector response matrix to a set of known incident spectra.
- **Spectrum reconstruction** &mdash; recover unknown spectra from detector signals via linear regression with optional regularization (OLS, Lasso, Ridge, ElasticNet, and their cross-validated variants).
- **Noise simulation** &mdash; add Gaussian noise to simulate realistic measurement conditions.
- **Visualization** &mdash; interactive Plotly figures for responsivity curves, incident spectra, response heatmaps, and reconstructed spectra.

For a detailed API reference, see [Module and API Reference](docs/Module_and_API_Reference.md).

## Installation

```bash
pip install -e .
```

**Requirements:** Python >= 3.10

Dependencies: `numpy>=2.0.1,<2.2.0`, `scipy`, `scikit-learn`, `pandas`, `plotly`, `numba`

## Usage

### Quick Start

```python
import numpy as np
from SpectrumReconstruction import SpectrumReconstructionSimulation
from SpectrumReconstruction import SpectrumReconstructionAdvance as SRAdvance
from SpectrumReconstruction import Utility as SRUtility

# Define parameters
bias_array = np.linspace(0e-9, 500e-9, 501)
wavelength_array = np.linspace(400e-9, 1800e-9, 1401)
sigma = 1e-9 / 2 / np.sqrt(2 * np.log(2))  # FWHM = 1nm
mu = np.linspace(400e-9, 1800e-9, 1401)

# Initialize the simulation
srs = SpectrumReconstructionSimulation(
    bias_array=bias_array,
    wavelength_array=wavelength_array,
    base_function_name="gaussian",
    sigma=sigma,
    mu=mu,
    photo_detector_bias_mode="normal_move",
    delta_lambda=30e-9,
    e_g_ev=0.75,
    eta=1.0
)

# Visualize the response matrix
srs.response_mapping_figure.show()

# Create an unknown spectrum to reconstruct
spectrum = SRAdvance.SimulationSpectrum(
    wavelength_array=wavelength_array,
    spectrum_function=SRUtility.gaussian_spectrum_sum
)
spectrum.set_spectrum(
    _mu=np.array([800e-9, 1200e-9]),
    _sigma=np.array([20e-9, 30e-9]),
    _alpha=np.array([1.0, 0.8])
)

# Reconstruct
srs.reconstruct_spectrum(
    simulation_spectrum=spectrum,
    method='ElasticNet',
    add_gaussian_noise=True,
    noise_std_ratio=0.01,
    lambda_reg=0.15,
    alpha=0.5
)

# Visualize the result
srs.reconstruction_spectrum_figure.show()
```

### Module Structure

```
src/
└── SpectrumReconstruction
    ├── SpectrumReconstructionBasic      # Core reconstruction via linear regression
    ├── SpectrumReconstructionSimulation # High-level end-to-end simulation
    ├── SpectrumReconstructionAdvance    # Photodetector & spectrum modeling
    │   ├── IdealSemiconductorPhotoDetector
    │   ├── IncidentSpectrum
    │   ├── SimulationSpectrum
    │   ├── simulate_response_matrix()
    │   └── simulate_unknown_response()
    └── Utility                          # Physics functions
        ├── blackbody()
        ├── gaussian()
        ├── smooth_responsivity()
        ├── ideal_responsivity()
        ├── gaussian_spectrum_sum()
        └── blackbody_spectrum_sum()
```

### Reconstruction Methods

| Method | Description |
|---|---|
| `normal` | Ordinary least squares (no regularization) |
| `l1` | L1-regularized (custom SLSQP optimizer) |
| `l2` / `Ridge` | L2-regularized (Ridge regression) |
| `Lasso` | Lasso regression (scikit-learn) |
| `LassoCV` | Lasso with cross-validated alpha selection |
| `ElasticNet` | ElasticNet (L1 + L2 combined) |
| `ElasticNetCV` | ElasticNet with cross-validated parameter selection |

### Examples

See [example.ipynb](examples/example.ipynb) for a step-by-step walkthrough of the `SpectrumReconstructionBasic` class.

## Notice

- All wavelength and length parameters are in **meters** internally. Convert from nanometers before passing (e.g., `1000e-9` for 1000 nm).
- The `bandgap energy` parameter (`e_g_ev`) is in **eV**.
- Numba JIT compilation (`@njit`) is used to accelerate core physics functions (`blackbody`, `gaussian`, `smooth_responsivity`). The first call may be slower due to compilation.

## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.

This repository contains research code for photodetector-based computational spectrum reconstruction.
It is intended to support reproducibility of the associated manuscript.
