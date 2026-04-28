# SpectrumReconstruction Architecture

**Chenyu Huang**$^{*\dagger}$

$^*$*State Key Laboratory of Infrared Physics, Shanghai Institute of Technical Physics, Chinese Academy of Sciences*

$^\dagger$*University of Chinese Academy of Sciences*

[TOC]

---

## 1. Design Goals

`SpectrumReconstruction` provides a complete physical simulation of computational spectrum reconstruction based on **tunable semiconductor photodetectors**. The core idea is:

> By changing the operating state of a detector (bias voltage, bandgap energy, etc.), the same detector exhibits different spectral responsivity at each state. Multiple detector readings can then be solved to recover the incident spectrum.

The module is designed around four principles:

- **Physical accuracy** — responsivity, blackbody radiation, and all other quantities are derived from rigorous physics formulas.
- **High performance** — all critical computation paths use NumPy broadcasting and matrix operations; hot physics functions are JIT-compiled with Numba `@njit`.
- **Layered decoupling** — physical modeling, data flow, and reconstruction solving are separated into distinct layers with no cross-layer coupling.
- **Extensibility** — basis functions, bias modes, and regularization methods are all switchable via parameters, requiring no changes to core code.

---

## 2. Physical Model

### 2.1 Forward Model

Let $R(\lambda, b)$ be the spectral responsivity of the detector at bias state $b$, and $S(\lambda)$ be the incident spectrum. The detector output signal is:

$$y(b) = \int R(\lambda, b) \cdot S(\lambda)\, d\lambda$$

In discretized matrix form:

$$\mathbf{y} = \mathbf{R}^{\top} \mathbf{s}$$

where $\mathbf{R} \in \mathbb{R}^{N_\lambda \times N_b}$ is the responsivity matrix, $\mathbf{s} \in \mathbb{R}^{N_\lambda}$ is the incident spectrum vector, and $\mathbf{y} \in \mathbb{R}^{N_b}$ is the detector output vector.

### 2.2 Spectrum Representation and Training Matrix

The incident spectrum is assumed to be expressible as a linear combination of basis functions $\{f_j(\lambda)\}$:

$$S(\lambda) = \sum_j a_j \cdot f_j(\lambda)$$

Two families of basis functions are supported:

- **Gaussian basis**: $f_j(\lambda) = G(\lambda, \mu_j, \sigma)$, with training set covering a range of center wavelengths $\{\mu_j\}$.
- **Blackbody basis**: $f_j(\lambda) = B(\lambda, T_j)$, with training set covering a range of temperatures $\{T_j\}$.

The detector response to all basis functions is pre-computed to form the training matrix:

$$\mathbf{P}_{ij} = \mathbf{R}^{\top}_{i \cdot} \cdot \mathbf{f}_{j}
\quad \Leftrightarrow \quad
\mathbf{P} = \mathbf{R}^{\top} \mathbf{F}$$

where $\mathbf{F} \in \mathbb{R}^{N_\lambda \times N_j}$ is the basis function matrix and $\mathbf{P} \in \mathbb{R}^{N_b \times N_j}$ is the training response matrix (referred to as `response_pivot` in the code).

### 2.3 Inverse Problem and Regularized Solving

Given the detector output $\mathbf{y}_{\text{test}} \in \mathbb{R}^{N_b}$ for an unknown spectrum, the coefficient vector $\mathbf{a}$ is found by solving:

$$\mathbf{P} \cdot \mathbf{a} \approx \mathbf{y}_{\text{test}}$$

Since this system is typically underdetermined or ill-conditioned, the module provides several regularized solvers:

| Method | Objective |
|---|---|
| `normal` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2$ |
| `l1` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{a}\|_1$ |
| `l2` / `Ridge` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{a}\|_2^2$ |
| `ElasticNet` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2 + \lambda[\alpha\|\mathbf{a}\|_1 + (1-\alpha)\|\mathbf{a}\|_2^2]$ |

The reconstructed spectrum is then $\hat{S}(\lambda) = \sum_j a_j f_j(\lambda)$.

---

## 3. Module Hierarchy

The module is organized into three layers, from lowest to highest:

```
┌─────────────────────────────────────────────────────────┐
│                  Layer 3: Simulation Layer               │
│           SpectrumReconstructionSimulation               │
│  (config → modeling → training → solving → visualization)│
└──────────────────────────┬──────────────────────────────┘
                           │ calls
┌──────────────────────────▼──────────────────────────────┐
│           Layer 2: Physical Modeling & Data Flow         │
│  SpectrumReconstructionAdvance + SpectrumReconstructionBasic │
│  (detector modeling / spectrum modeling / response matrix│
│   computation / linear regression solving)               │
└──────────────────────────┬──────────────────────────────┘
                           │ calls
┌──────────────────────────▼──────────────────────────────┐
│               Layer 1: Physics Function Library          │
│                         Utility                          │
│   (blackbody / gaussian / smooth_responsivity / ...)     │
└─────────────────────────────────────────────────────────┘
```

### 3.1 Layer 1 — Utility (Physics Function Library)

File: `SpectrumReconstruction/Utility.py`

Provides stateless physics primitives. All functions support NumPy array broadcasting:

| Function | Description |
|---|---|
| `blackbody(λ, T)` | Planck blackbody radiance; log-space computation; `@njit` |
| `gaussian(λ, μ, σ)` | Unnormalized Gaussian function; `@njit` |
| `ideal_responsivity(λ, Eg, η)` | Ideal step-function responsivity |
| `smooth_responsivity(λ, Eg, Δλ, η)` | Sigmoid-smoothed responsivity; `@njit` |
| `gaussian_spectrum_sum(λ, μ, σ, α)` | Weighted superposition of Gaussian spectra |
| `blackbody_spectrum_sum(λ, T, α)` | Weighted superposition of blackbody spectra |
| `fast_matmul(A, B)` | Matrix multiplication $A^{\top}B$ |

> **Design note:** High-frequency physics kernels are JIT-compiled with Numba `@njit`. The first call incurs a compilation delay; subsequent calls run at near-native speed. All functions accept mixed scalar/matrix inputs for broadcasting, enabling batch vectorized computation in higher layers.

### 3.2 Layer 2 — Physical Modeling & Data Flow

#### 3.2.1 `SpectrumReconstructionAdvance`

File: `SpectrumReconstruction/SpectrumReconstructionAdvance.py`

Responsible for modeling the photodetector and incident spectra, and computing the response matrix.

**Class relationships:**

```
IdealSemiconductorPhotoDetector          IncidentSpectrum
        │                                       │
        │ responsivity                          │ spectrum
        │ (N_λ × N_b)                           │ (N_λ × N_j)
        └──────────────┬────────────────────────┘
                       │
               simulate_response_matrix
                       │
               response_pivot (N_b × N_j)
                       │
               SpectrumReconstructionBasicHighPerformance
```

**`IdealSemiconductorPhotoDetector`**

The core is the `responsivity` attribute (a `cached_property`), which uses broadcasting to compute the full responsivity matrix in a single call, avoiding Python loops:

```python
# Vectorized computation for normal_move mode
lambda_matrix = wavelength[:, None] - bias_array[None, :]  # (N_λ, N_b)
e_g_matrix    = full((N_λ, N_b), self.e_g)
responsivity  = smooth_responsivity(lambda_matrix, e_g_matrix, delta_lambda, eta)
```

Three bias modes are supported, each modifying different physical quantities:

| Mode | Modified quantity | Effect |
|---|---|---|
| `normal_move` | Effective wavelength $\lambda_{\text{eff}} = \lambda - b$ | Shifts the responsivity curve along the wavelength axis |
| `increase_band_gap` | Bandgap $E_g' = hc/(\lambda_g + b)$ | Shifts the cutoff wavelength to longer wavelengths |
| `decrease_eta` | Bandgap as above; additionally $\eta' = 1/(1+b/\lambda_g)$ | Shifts the cutoff wavelength and reduces the peak responsivity |

**`IncidentSpectrum`**

Also uses broadcasting to generate the full spectrum matrix for all parameter values at once:

```python
# Gaussian mode
spectrum_matrix = gaussian(wavelength[:, None], mu[None, :], sigma)  # (N_λ, N_μ)
# Blackbody mode
spectrum_matrix = blackbody(wavelength[:, None], T[None, :])          # (N_λ, N_T)
```

**`SimulationSpectrum`**

Accepts any callable spectrum function and generates the unknown test spectrum lazily via `set_spectrum(**kwargs)`. Decoupled from `IncidentSpectrum`; dedicated to representing the spectrum to be reconstructed.

#### 3.2.2 `SpectrumReconstructionBasic`

File: `SpectrumReconstruction/SpectrumReconstructionBasic.py`

Provides two implementations:

| Class | Input format | Use case |
|---|---|---|
| `SpectrumReconstructionBasic` | Long-format `pandas.DataFrame` | Experimental data; supports automatic missing-value cleaning |
| `SpectrumReconstructionBasicHighPerformance` | Pre-pivoted `numpy.ndarray` | Simulation pipelines; zero DataFrame overhead |

`SpectrumReconstructionSimulation` uses the high-performance variant internally.

**`_clean_pivot_training_data`** implements a greedy missing-value cleaning algorithm: at each iteration, the row or column with the lowest completeness ratio is removed, until the matrix contains no missing values. This handles incomplete training data caused by equipment failures during real measurements.

**`_linear_regression`** provides a unified interface to 12 solving methods, dispatched via a `match` statement, covering NumPy OLS, SciPy L1 optimization, and the full scikit-learn regularized regression suite.

### 3.3 Layer 3 — `SpectrumReconstructionSimulation` (End-to-End Simulation)

File: `SpectrumReconstruction/SpectrumReconstructionSimulation.py`

Wires all Layer 2 components into a single interface. The internal construction sequence is:

```
__init__()
    │
    ├─ 1. Construct IdealSemiconductorPhotoDetector
    │      → Compute and cache responsivity matrix R  (N_λ × N_b)
    │
    ├─ 2. Construct IncidentSpectrum
    │      → Compute basis function matrix F  (N_λ × N_j)
    │
    ├─ 3. simulate_response_matrix(R, F)
    │      → Training response matrix P = R^T F  (N_b × N_j)
    │         stored as self.response_pivot
    │
    └─ 4. Construct SpectrumReconstructionBasicHighPerformance
           → Initialize solver with P as training data

reconstruct_spectrum(simulation_spectrum, method)
    │
    ├─ simulate_unknown_response(detector, simulation_spectrum)
    │      → Unknown spectrum response vector y_test  (N_b,)
    │
    └─ spectrum_reconstruction.reconstruct_spectrum(y_test, method)
           → Solve for coefficients a, stored in spectrum_reconstruction.a
```

---

## 4. End-to-End Data Flow

```
Input parameters
(bias_array, wavelength_array, e_g_ev, ...)
        │
        ▼
┌───────────────────────┐    ┌──────────────────────┐
│ IdealSemiconductor    │    │   IncidentSpectrum    │
│ PhotoDetector         │    │  (gaussian/blackbody) │
│                       │    │                       │
│ R: (N_λ × N_b)        │    │  F: (N_λ × N_j)       │
└──────────┬────────────┘    └──────────┬────────────┘
           │                            │
           └────────────┬───────────────┘
                        │ fast_matmul: R^T @ F
                        ▼
              ┌──────────────────┐
              │  response_pivot  │
              │  P: (N_b × N_j)  │  ← training matrix
              └────────┬─────────┘
                       │
        ┌──────────────┴──────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐          ┌───────────────────────┐
│ SimulationSpectrum│         │   Linear regression   │
│ s: (N_λ,)        │         │   solver               │
└────────┬─────────┘         │   P · a ≈ y_test       │
         │ fast_matmul        │   method: normal / l1 /│
         │ R^T @ s            │          l2 / ElasticNet│
         ▼                   └───────────┬────────────┘
 y_test: (N_b,) ────────────────────────► a: (N_j,)
                                          │
                                          ▼
                              S(λ) = Σ aⱼ·fⱼ(λ)
                              reconstructed spectrum
```

---

## 5. Key Design Decisions

### 5.1 Full Vectorization — No Python Loops

The responsivity matrix and spectrum matrix are both computed in a single NumPy broadcasting call. Matrix multiplication uses `A.T @ B`. In a typical configuration (1501 bias points × 28001 wavelength points × 2801 Gaussian bases), vectorized computation is approximately 100× faster than column-wise iteration.

### 5.2 `cached_property` for Expensive Computations

`IdealSemiconductorPhotoDetector.responsivity` is a `cached_property`. The responsivity matrix is computed only on first access and cached thereafter. When `reconstruct_spectrum` is called multiple times (e.g., to compare different regularization parameters), this avoids redundant recomputation.

### 5.3 Two Reconstruction Classes for Different Scenarios

| | `SpectrumReconstructionBasic` | `SpectrumReconstructionBasicHighPerformance` |
|---|---|---|
| Training data format | Long-format DataFrame | Pre-pivoted NumPy ndarray |
| Missing-value handling | Supported (greedy cleaning) | Not supported |
| Memory / speed | Higher overhead | Low overhead |
| Typical use | Real experimental data | Simulation pipelines |

### 5.4 Orthogonal Design of Basis Functions and Bias Modes

Basis functions (gaussian / blackbody) and bias modes (normal_move / increase_band_gap / decrease_eta) are two independent dimensions. Any combination of the two is supported through the same interface without modifying core code, yielding 6 valid configurations.

### 5.5 Visible-Blind Simulation

A module-level global parameter `visible_blind_cutoff_parameter` (set via `change_visible_blind_cutoff_parameter()`) acts as a multiplier for detector responsivity at short wavelengths. Combined with the `visible_blind_cutoff` threshold in the constructor, this simulates visible-blind detector behavior without recomputing the entire responsivity matrix.

---

## 6. File Structure

```
SpectrumReconstruction/
├── __init__.py                         # Public interface exports
├── Utility.py                          # Layer 1: physics function library
├── SpectrumReconstructionAdvance.py    # Layer 2: detector & spectrum modeling
├── SpectrumReconstructionBasic.py      # Layer 2: linear regression solver
└── SpectrumReconstructionSimulation.py # Layer 3: end-to-end simulation wrapper

tests/
├── conftest.py                         # Shared fixtures
├── test_utility.py                     # Utility function unit tests
├── test_advance.py                     # Detector & spectrum modeling tests
├── test_reconstruction_basic.py        # Reconstruction solver tests
└── test_simulation.py                  # End-to-end integration tests

docs/
├── architecture.md                     # This document
├── architecture_CN.md                  # Chinese version of this document
├── Module_and_API_Reference.md         # English API reference
└── Module_and_API_Reference_CN.md      # Chinese API reference

examples/
├── eample.ipynb                        # SpectrumReconstructionBasic usage example
├── data_training_example.csv           # Example training data
└── data_testing_example.csv            # Example testing data
```

---

## 7. Extension Guide

### Adding a New Bias Mode

In both `IdealSemiconductorPhotoDetector.responsivity` and `_responsivity_func`, add a new `case` branch that computes the corresponding `lambda_matrix`, `e_g_matrix`, and `eta` (as a **local variable** — never assign to `self.eta` to avoid state mutation).

### Adding a New Regularization Method

In the `match method` statement of `SpectrumReconstructionBasic._linear_regression`, add a new `case` branch that returns a `numpy.ndarray` of coefficients.

### Adding a New Basis Function

1. Implement the new function in `Utility.py` with NumPy broadcasting support.
2. Add the new mode to the `match` statement in `IncidentSpectrum.__init__`.
3. Add the corresponding initialization logic in `SpectrumReconstructionSimulation.__init__`.
