import numpy as np
import pytest


# --- Physical constants (same as in source) ---
Q = 1.60217662e-19
H = 6.62607015e-34
C = 3.0e8
K = 1.38064852e-23


# --- Shared fixtures ---

@pytest.fixture
def wavelength_array():
    """Standard wavelength array covering 400-1800nm, 141 points."""
    return np.linspace(400e-9, 1800e-9, 141)


@pytest.fixture
def bias_array():
    """Standard bias array covering 0-500nm, 51 points."""
    return np.linspace(0e-9, 500e-9, 51)


@pytest.fixture
def mu_array():
    """Gaussian center wavelength array."""
    return np.linspace(600e-9, 1600e-9, 101)


@pytest.fixture
def sigma():
    """Gaussian sigma corresponding to FWHM ~ 1nm."""
    return 1e-9 / 2 / np.sqrt(2 * np.log(2))


@pytest.fixture
def temperature_array():
    """Blackbody temperature array."""
    return np.linspace(800, 1400, 7)
