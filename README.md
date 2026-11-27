<!-- SpectrumReconstruction -->

<!--suppress HtmlDeprecatedAttribute -->
<div align="center" style="text-align:center">
   <h1> SpectrumReconstruction </h1>
   <p>
      Spectrum Reconstruction &amp; Photodetector Simulation Python Module<br>
      <code><b> v0.2 </b></code>
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

**SpectrumReconstruction** is a Python package for reconstructing optical spectra from calibrated photodetector signals. It also provides tools for modelling detector responsivity, simulating measurements and evaluating how measurement noise affects the reconstructed spectrum.

The package accompanies the manuscript *End-to-end optoelectronic architecture for event-driven infrared spectroscopy*.

<p>
      <b>Manuscript authors</b><br>
      Chenyu Huang<sup>1,2,4</sup>, Mingxin Deng<sup>3,4</sup>, Hui Xia<sup>1,2,4</sup>, Hangyu Xu<sup>1,2*</sup>, Hailu Wang<sup>1,2</sup>, Zihua Huang<sup>1,2</sup>, Kunyu Wang<sup>1</sup>, Xiao Fu<sup>1,2</sup>, Yinan Wang<sup>3*</sup>, Qingjiang Li<sup>3</sup>, Weida Hu<sup>1,2*</sup><br>
      <sup>1</sup> State Key Laboratory of Infrared Physics, Shanghai Institute of Technical Physics, Chinese Academy of Sciences, Shanghai, 200083, China<br>
      <sup>2</sup> University of Chinese Academy of Sciences, 101408, China<br>
      <sup>3</sup> College of Electronic Science and Technology, National University of Defense Technology, Changsha, 410073, China<br>
      <sup>4</sup> Chenyu Huang, Mingxin Deng and Hui Xia contributed equally.<br>
      <sup>*</sup> Corresponding authors: Hangyu Xu (<a href="mailto:xuhangyu@mail.sitp.ac.cn">xuhangyu@mail.sitp.ac.cn</a>), Yinan Wang (<a href="mailto:wangyinan@nudt.edu.cn">wangyinan@nudt.edu.cn</a>), Weida Hu (<a href="mailto:wdhu@mail.sitp.ac.cn">wdhu@mail.sitp.ac.cn</a>)
   </p>


## Features

- **Photodetector modeling** &mdash; simulate ideal semiconductor photodetector responsivity with adjustable bandgap, quantum efficiency, bias modes (`normal_move`, `increase_band_gap`, `decrease_eta`), and visible-blind characteristics.
- **Incident spectrum generation** &mdash; create training spectra using Gaussian or blackbody radiation models.
- **Response matrix computation** &mdash; calculate the detector response matrix to a set of known incident spectra.
- **Spectrum reconstruction** &mdash; recover unknown spectra from detector signals via linear regression with optional regularization (OLS, Lasso, Ridge, ElasticNet, and their cross-validated variants).
- **Noise simulation** &mdash; add Gaussian noise to simulate realistic measurement conditions.
- **Visualization** &mdash; interactive Plotly figures for responsivity curves, incident spectra, response heatmaps, and reconstructed spectra.

See the [architecture guide](docs/Architecture.md) and [API reference](docs/Module_and_API_Reference.md).

## Installation

```bash
git clone https://github.com/hcy2206/Spectrum_Reconstruction.git
cd Spectrum_Reconstruction
pip install -e .
```

**Requirements:** Python >= 3.10

Dependencies: `numpy>=2.0.1,<2.2.0`, `scipy`, `scikit-learn`, `pandas`, `plotly`, `numba`

## Usage

### Quick Start

#### Reconstruct a spectrum from detector signals

Run the [reconstruction notebook](examples/example.ipynb) in Jupyter with `examples/` as the working directory. The [calibration data](examples/data_training_example.csv) record detector currents at different blackbody temperatures and bias settings. The [test data](examples/data_testing_example.csv) contain measurements of the source to reconstruct. Each row is one measurement, with columns for bias (`V_bias/mV`), current (`Current`) and temperature (`Temperature`). The notebook uses blackbody basis functions and L1 regularization, then plots normalized spectral intensity against wavelength.

#### Simulate a detector and reconstruct a two-peak spectrum

Generate an input spectrum with peaks at 800 nm and 1200 nm, simulate detector signals with Gaussian noise and reconstruct the spectrum with ElasticNet. Wavelengths are in meters, so 800 nm is entered as `800e-9`. The final plot overlays the input and reconstructed spectra so you can compare peak positions and shapes.

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
fig = srs.reconstruction_spectrum_figure
fig.update_traces(name="Reconstructed", showlegend=True)
fig.add_scatter(x=wavelength_array, y=spectrum.spectrum, mode="lines", name="Input")
fig.update_xaxes(title_text="Wavelength (m)")
fig.show()
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

## Notice

- All wavelength and length parameters are in **meters** internally. Convert from nanometers before passing (e.g., `1000e-9` for 1000 nm).
- The `bandgap energy` parameter (`e_g_ev`) is in **eV**.
- Numba JIT compilation (`@njit`) is used to accelerate core physics functions (`blackbody`, `gaussian`, `smooth_responsivity`). The first call may be slower due to compilation.

## License

This project is licensed under the MIT License, see the [LICENSE](LICENSE) file for details.
