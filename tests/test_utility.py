"""Tests for SpectrumReconstruction.Utility module."""

import numpy as np
import pytest

from SpectrumReconstruction.Utility import (
    blackbody, gaussian, ideal_responsivity, smooth_responsivity,
    gaussian_spectrum_sum, blackbody_spectrum_sum, fast_matmul,
)

# Physical constants
Q = 1.60217662e-19
H = 6.62607015e-34
C = 3.0e8
K = 1.38064852e-23


# ========== blackbody ==========

class TestBlackbody:
    def test_positive_output(self):
        """Blackbody radiation must be positive for valid inputs."""
        wl = np.linspace(500e-9, 2000e-9, 100)
        result = blackbody(wl, 1000.0)
        assert np.all(result > 0)

    def test_wien_peak(self):
        """Peak wavelength should approximately obey Wien's law: lambda_max * T ~ 2898 um*K."""
        T = 5000.0
        wl = np.linspace(100e-9, 5000e-9, 10000)
        spectrum = blackbody(wl, T)
        peak_wl = wl[np.argmax(spectrum)]
        wien_constant = peak_wl * T
        assert abs(wien_constant - 2.898e-3) / 2.898e-3 < 0.02  # within 2%

    def test_higher_temperature_higher_peak(self):
        """Higher temperature should yield a higher peak intensity."""
        wl = np.linspace(500e-9, 3000e-9, 500)
        peak_low = np.max(blackbody(wl, 800.0))
        peak_high = np.max(blackbody(wl, 1200.0))
        assert peak_high > peak_low

    def test_broadcasting(self):
        """Should support broadcasting: (N,1) wavelength x (1,M) temperature."""
        wl = np.linspace(500e-9, 2000e-9, 50)[:, None]
        T = np.array([800.0, 1000.0, 1200.0])[None, :]
        result = blackbody(wl, T)
        assert result.shape == (50, 3)
        assert np.all(result > 0)


# ========== gaussian ==========

class TestGaussian:
    def test_peak_at_mu(self):
        """Gaussian should peak at mu with value 1.0."""
        mu = 1000e-9
        sigma = 10e-9
        assert gaussian(mu, mu, sigma) == pytest.approx(1.0)

    def test_symmetry(self):
        """Gaussian should be symmetric around mu."""
        mu = 1000e-9
        sigma = 10e-9
        offset = 5e-9
        assert gaussian(mu + offset, mu, sigma) == pytest.approx(
            gaussian(mu - offset, mu, sigma)
        )

    def test_decay(self):
        """Value should decrease with distance from mu."""
        mu = 1000e-9
        sigma = 10e-9
        assert gaussian(mu + sigma, mu, sigma) < gaussian(mu, mu, sigma)

    def test_array_input(self):
        """Should handle array input."""
        wl = np.linspace(900e-9, 1100e-9, 100)
        result = gaussian(wl, 1000e-9, 10e-9)
        assert result.shape == (100,)
        assert np.argmax(result) == pytest.approx(50, abs=1)  # peak near center

    def test_broadcasting(self):
        """Should support broadcasting: (N,1) x (1,M) -> (N,M)."""
        wl = np.linspace(800e-9, 1200e-9, 50)[:, None]
        mu = np.array([900e-9, 1000e-9, 1100e-9])[None, :]
        result = gaussian(wl, mu, 10e-9)
        assert result.shape == (50, 3)


# ========== ideal_responsivity ==========

class TestIdealResponsivity:
    def test_zero_above_cutoff(self):
        """Responsivity should be zero above the cutoff wavelength."""
        e_g = 0.75 * Q  # eV -> J
        lambda_g = H * C / e_g
        wl = np.linspace(lambda_g + 10e-9, lambda_g + 500e-9, 50)
        result = ideal_responsivity(wl, e_g)
        assert np.all(result == 0)

    def test_positive_below_cutoff(self):
        """Responsivity should be positive below the cutoff."""
        e_g = 0.75 * Q
        lambda_g = H * C / e_g
        wl = np.linspace(100e-9, lambda_g - 10e-9, 50)
        result = ideal_responsivity(wl, e_g)
        assert np.all(result > 0)

    def test_linear_growth(self):
        """Below cutoff, responsivity = eta*q*lambda/(h*c), i.e., linear in lambda."""
        e_g = 0.75 * Q
        lambda_g = H * C / e_g
        wl = np.linspace(100e-9, lambda_g - 100e-9, 100)
        result = ideal_responsivity(wl, e_g, eta=1.0)
        expected = Q * wl / (H * C)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_eta_scaling(self):
        """Quantum efficiency should scale the responsivity linearly."""
        e_g = 0.75 * Q
        wl = np.linspace(100e-9, 1000e-9, 50)
        r1 = ideal_responsivity(wl, e_g, eta=1.0)
        r05 = ideal_responsivity(wl, e_g, eta=0.5)
        np.testing.assert_allclose(r05, r1 * 0.5, rtol=1e-10)


# ========== smooth_responsivity ==========

class TestSmoothResponsivity:
    def test_positive_values(self):
        """Smooth responsivity should be non-negative."""
        e_g = 0.75 * Q
        wl = np.linspace(100e-9, 2500e-9, 200)
        result = smooth_responsivity(wl, e_g, delta_lambda=30e-9)
        assert np.all(result >= 0)

    def test_approaches_ideal_far_below_cutoff(self):
        """Well below cutoff, smooth responsivity should approach ideal responsivity."""
        e_g = 0.75 * Q
        wl = np.linspace(100e-9, 500e-9, 50)
        smooth = smooth_responsivity(wl, e_g, delta_lambda=30e-9, eta=1.0)
        ideal = ideal_responsivity(wl, e_g, eta=1.0)
        np.testing.assert_allclose(smooth, ideal, rtol=0.01)

    def test_approaches_zero_far_above_cutoff(self):
        """Well above cutoff, smooth responsivity should approach zero."""
        e_g = 0.75 * Q
        lambda_g = H * C / e_g
        wl = np.linspace(lambda_g + 500e-9, lambda_g + 1000e-9, 50)
        result = smooth_responsivity(wl, e_g, delta_lambda=30e-9)
        assert np.all(result < 1e-6)

    def test_smaller_delta_sharper_transition(self):
        """Smaller delta_lambda should produce a sharper cutoff."""
        e_g = 0.75 * Q
        lambda_g = H * C / e_g
        wl_near_cutoff = np.array([lambda_g + 50e-9])
        r_sharp = smooth_responsivity(wl_near_cutoff, e_g, delta_lambda=5e-9)
        r_smooth = smooth_responsivity(wl_near_cutoff, e_g, delta_lambda=50e-9)
        assert r_sharp[0] < r_smooth[0]


# ========== gaussian_spectrum_sum ==========

class TestGaussianSpectrumSum:
    def test_single_gaussian(self):
        """With one component, should equal alpha * gaussian."""
        wl = np.linspace(800e-9, 1200e-9, 100)
        mu = np.array([1000e-9])
        sigma = 20e-9
        alpha = np.array([2.5])
        result = gaussian_spectrum_sum(wl, mu, sigma, alpha)
        expected = 2.5 * gaussian(wl, 1000e-9, sigma)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_multiple_components(self):
        """Superposition of multiple gaussians."""
        wl = np.linspace(500e-9, 1500e-9, 200)
        mu = np.array([700e-9, 1000e-9, 1300e-9])
        sigma = 30e-9
        alpha = np.array([1.0, 2.0, 0.5])
        result = gaussian_spectrum_sum(wl, mu, sigma, alpha)
        expected = sum(a * gaussian(wl, m, sigma) for a, m in zip(alpha, mu))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_per_component_sigma(self):
        """Should support different sigma for each component."""
        wl = np.linspace(500e-9, 1500e-9, 200)
        mu = np.array([700e-9, 1000e-9])
        sigma = np.array([10e-9, 50e-9])
        alpha = np.array([1.0, 1.0])
        result = gaussian_spectrum_sum(wl, mu, sigma, alpha)
        expected = gaussian(wl, 700e-9, 10e-9) + gaussian(wl, 1000e-9, 50e-9)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ========== blackbody_spectrum_sum ==========

class TestBlackbodySpectrumSum:
    def test_single_temperature(self):
        """With one component, should equal alpha * blackbody."""
        wl = np.linspace(500e-9, 3000e-9, 100)
        T = np.array([1000.0])
        alpha = np.array([3.0])
        result = blackbody_spectrum_sum(wl, T, alpha)
        expected = 3.0 * blackbody(wl, 1000.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_multiple_temperatures(self):
        """Superposition of multiple blackbody spectra."""
        wl = np.linspace(500e-9, 3000e-9, 200)
        T = np.array([800.0, 1000.0, 1200.0])
        alpha = np.array([1.0, 0.5, 2.0])
        result = blackbody_spectrum_sum(wl, T, alpha)
        expected = sum(a * blackbody(wl, t) for a, t in zip(alpha, T))
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ========== fast_matmul ==========

class TestFastMatmul:
    def test_transpose_matmul(self):
        """fast_matmul(a, b) should equal a.T @ b."""
        a = np.random.rand(10, 3)
        b = np.random.rand(10, 5)
        result = fast_matmul(a, b)
        expected = a.T @ b
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_output_shape(self):
        """Output shape should be (cols_a, cols_b)."""
        a = np.random.rand(20, 4)
        b = np.random.rand(20, 7)
        assert fast_matmul(a, b).shape == (4, 7)
