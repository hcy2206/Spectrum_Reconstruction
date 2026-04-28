"""Tests for SpectrumReconstruction.SpectrumReconstructionAdvance module."""

import numpy as np
import pytest

from SpectrumReconstruction.SpectrumReconstructionAdvance import (
    IdealSemiconductorPhotoDetector,
    IncidentSpectrum,
    SimulationSpectrum,
    simulate_response_matrix,
    simulate_unknown_response,
)
from SpectrumReconstruction.Utility import gaussian_spectrum_sum

Q = 1.60217662e-19
H = 6.62607015e-34
C = 3.0e8


# ========== IdealSemiconductorPhotoDetector ==========

class TestIdealSemiconductorPhotoDetector:

    def _make_detector(self, bias_array, wavelength_array, bias_mode='normal_move', **kwargs):
        return IdealSemiconductorPhotoDetector(
            bias_array=bias_array,
            e_g_ev=kwargs.get('e_g_ev', 0.75),
            eta=kwargs.get('eta', 1.0),
            delta_lambda=kwargs.get('delta_lambda', 30e-9),
            wavelength=wavelength_array,
            visible_blind_cutoff=kwargs.get('visible_blind_cutoff', -1),
            bias_mode=bias_mode,
        )

    def test_responsivity_shape(self, wavelength_array, bias_array):
        """Responsivity matrix shape should be (num_wavelength, num_bias)."""
        det = self._make_detector(bias_array, wavelength_array)
        resp = det.responsivity
        assert resp.shape == (len(wavelength_array), len(bias_array))

    def test_responsivity_non_negative(self, wavelength_array, bias_array):
        """All responsivity values should be >= 0."""
        det = self._make_detector(bias_array, wavelength_array)
        assert np.all(det.responsivity >= 0)

    def test_normal_move_shifts_cutoff(self, wavelength_array):
        """In normal_move mode, larger bias extends the effective cutoff to longer wavelengths."""
        bias_array = np.array([0.0, 200e-9])
        det = self._make_detector(bias_array, wavelength_array, bias_mode='normal_move')
        resp = det.responsivity
        # With bias, effective cutoff extends further (lambda_g + bias)
        nonzero_0 = np.sum(resp[:, 0] > 1e-10)
        nonzero_1 = np.sum(resp[:, 1] > 1e-10)
        assert nonzero_1 >= nonzero_0

    def test_increase_band_gap_mode(self, wavelength_array):
        """increase_band_gap mode should produce non-negative responsivity."""
        bias_array = np.linspace(0, 200e-9, 11)
        det = self._make_detector(bias_array, wavelength_array, bias_mode='increase_band_gap')
        resp = det.responsivity
        assert resp.shape == (len(wavelength_array), len(bias_array))
        assert np.all(resp >= 0)

    def test_decrease_eta_does_not_mutate_self_eta(self, wavelength_array):
        """decrease_eta mode should NOT mutate self.eta (bug regression test)."""
        bias_array = np.linspace(0, 200e-9, 11)
        det = self._make_detector(bias_array, wavelength_array, bias_mode='decrease_eta')
        original_eta = det.eta
        _ = det.responsivity
        assert det.eta == original_eta, "self.eta was mutated by decrease_eta mode"

    def test_decrease_eta_responsivity_shape(self, wavelength_array):
        """decrease_eta mode should produce correct shape."""
        bias_array = np.linspace(0, 200e-9, 11)
        det = self._make_detector(bias_array, wavelength_array, bias_mode='decrease_eta')
        resp = det.responsivity
        assert resp.shape == (len(wavelength_array), len(bias_array))
        assert np.all(resp >= 0)

    def test_unsupported_bias_mode_raises(self, wavelength_array, bias_array):
        """Unsupported bias mode should raise ValueError."""
        det = self._make_detector(bias_array, wavelength_array)
        det.bias_mode = 'invalid_mode'
        # Clear cached property so it recomputes
        if 'responsivity' in det.__dict__:
            del det.__dict__['responsivity']
        with pytest.raises(ValueError):
            _ = det.responsivity

    def test_responsivity_figure(self, wavelength_array):
        """responsivity_figure_show should return a plotly figure without error."""
        bias_array = np.linspace(0, 100e-9, 3)
        det = self._make_detector(bias_array, wavelength_array)
        fig = det.responsivity_figure_show()
        assert fig is not None


# ========== IncidentSpectrum ==========

class TestIncidentSpectrum:

    def test_gaussian_spectrum_shape(self, wavelength_array, mu_array, sigma):
        """Gaussian spectrum shape should be (num_wavelength, num_mu)."""
        inc = IncidentSpectrum(
            wavelength=wavelength_array,
            base_function_name='gaussian',
            sigma=sigma,
            mu=mu_array,
        )
        spec = inc.spectrum
        assert spec.shape == (len(wavelength_array), len(mu_array))

    def test_gaussian_spectrum_positive(self, wavelength_array, mu_array, sigma):
        """Gaussian spectrum values should be >= 0."""
        inc = IncidentSpectrum(
            wavelength=wavelength_array,
            base_function_name='gaussian',
            sigma=sigma,
            mu=mu_array,
        )
        assert np.all(inc.spectrum >= 0)

    def test_blackbody_spectrum_shape(self, wavelength_array, temperature_array):
        """Blackbody spectrum shape should be (num_wavelength, num_T)."""
        inc = IncidentSpectrum(
            wavelength=wavelength_array,
            base_function_name='blackbody',
            T=temperature_array,
        )
        spec = inc.spectrum
        assert spec.shape == (len(wavelength_array), len(temperature_array))

    def test_blackbody_spectrum_positive(self, wavelength_array, temperature_array):
        """Blackbody spectrum values should be > 0."""
        inc = IncidentSpectrum(
            wavelength=wavelength_array,
            base_function_name='blackbody',
            T=temperature_array,
        )
        assert np.all(inc.spectrum > 0)

    def test_missing_mu_raises(self, wavelength_array):
        """Gaussian mode without mu should raise ValueError."""
        with pytest.raises(ValueError, match="mu must be provided"):
            IncidentSpectrum(
                wavelength=wavelength_array,
                base_function_name='gaussian',
                sigma=1e-9,
            )

    def test_missing_T_raises(self, wavelength_array):
        """Blackbody mode without T should raise ValueError."""
        with pytest.raises(ValueError, match="T must be provided"):
            IncidentSpectrum(
                wavelength=wavelength_array,
                base_function_name='blackbody',
            )

    def test_unsupported_base_function_raises(self, wavelength_array):
        """Unsupported base function should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported base function"):
            IncidentSpectrum(
                wavelength=wavelength_array,
                base_function_name='invalid',
            )

    def test_spectrum_figure(self, wavelength_array, mu_array, sigma):
        """spectrum_figure_show should return a figure without error."""
        inc = IncidentSpectrum(
            wavelength=wavelength_array,
            base_function_name='gaussian',
            sigma=sigma,
            mu=mu_array,
        )
        fig = inc.spectrum_figure_show()
        assert fig is not None


# ========== SimulationSpectrum ==========

class TestSimulationSpectrum:

    def test_spectrum_before_set_raises(self, wavelength_array):
        """Accessing spectrum before set_spectrum should raise ValueError."""
        sim = SimulationSpectrum(
            wavelength_array=wavelength_array,
            spectrum_function=gaussian_spectrum_sum,
        )
        with pytest.raises(ValueError, match="Spectrum has not been set"):
            _ = sim.spectrum

    def test_set_spectrum_returns_array(self, wavelength_array, sigma):
        """set_spectrum should return a numpy array with correct length."""
        sim = SimulationSpectrum(
            wavelength_array=wavelength_array,
            spectrum_function=gaussian_spectrum_sum,
        )
        result = sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=sigma,
            _alpha=np.array([1.0]),
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == len(wavelength_array)

    def test_spectrum_after_set(self, wavelength_array, sigma):
        """After set_spectrum, spectrum property should return the same data."""
        sim = SimulationSpectrum(
            wavelength_array=wavelength_array,
            spectrum_function=gaussian_spectrum_sum,
        )
        returned = sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=sigma,
            _alpha=np.array([1.0]),
        )
        np.testing.assert_array_equal(sim.spectrum, returned)

    def test_spectrum_figure_before_set_raises(self, wavelength_array):
        """Accessing spectrum_figure before set_spectrum should raise ValueError."""
        sim = SimulationSpectrum(
            wavelength_array=wavelength_array,
            spectrum_function=gaussian_spectrum_sum,
        )
        with pytest.raises(ValueError, match="Spectrum has not been set"):
            _ = sim.spectrum_figure

    def test_spectrum_figure_after_set(self, wavelength_array, sigma):
        """spectrum_figure should return a plotly figure after set_spectrum."""
        sim = SimulationSpectrum(
            wavelength_array=wavelength_array,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=sigma,
            _alpha=np.array([1.0]),
        )
        fig = sim.spectrum_figure
        assert fig is not None


# ========== simulate_response_matrix ==========

class TestSimulateResponseMatrix:

    def test_output_shape(self, wavelength_array, bias_array, mu_array, sigma):
        """Response matrix shape should be (num_bias, num_mu)."""
        det = IdealSemiconductorPhotoDetector(
            bias_array=bias_array,
            e_g_ev=0.75,
            wavelength=wavelength_array,
        )
        inc = IncidentSpectrum(
            wavelength=wavelength_array,
            base_function_name='gaussian',
            sigma=sigma,
            mu=mu_array,
        )
        resp = simulate_response_matrix(det, inc)
        assert resp.shape == (len(bias_array), len(mu_array))

    def test_wavelength_mismatch_raises(self, bias_array, mu_array, sigma):
        """Mismatched wavelength arrays should raise ValueError."""
        wl1 = np.linspace(400e-9, 1800e-9, 100)
        wl2 = np.linspace(400e-9, 1800e-9, 200)
        det = IdealSemiconductorPhotoDetector(
            bias_array=bias_array,
            e_g_ev=0.75,
            wavelength=wl1,
        )
        inc = IncidentSpectrum(
            wavelength=wl2,
            base_function_name='gaussian',
            sigma=sigma,
            mu=mu_array,
        )
        with pytest.raises(ValueError, match="do not match"):
            simulate_response_matrix(det, inc)


# ========== simulate_unknown_response ==========

class TestSimulateUnknownResponse:

    def _make_setup(self, wavelength_array, bias_array, sigma):
        det = IdealSemiconductorPhotoDetector(
            bias_array=bias_array,
            e_g_ev=0.75,
            wavelength=wavelength_array,
        )
        sim = SimulationSpectrum(
            wavelength_array=wavelength_array,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9, 1200e-9]),
            _sigma=np.array([20e-9, 30e-9]),
            _alpha=np.array([1.0, 0.8]),
        )
        return det, sim

    def test_output_shape(self, wavelength_array, bias_array, sigma):
        """Response should be a 1D array of length num_bias."""
        det, sim = self._make_setup(wavelength_array, bias_array, sigma)
        resp = simulate_unknown_response(det, sim)
        assert resp.shape == (len(bias_array),)

    def test_no_noise_deterministic(self, wavelength_array, bias_array, sigma):
        """Without noise, results should be deterministic."""
        det, sim = self._make_setup(wavelength_array, bias_array, sigma)
        r1 = simulate_unknown_response(det, sim, add_gaussian_noise=False)
        # Clear cached property to force recomputation
        r2 = simulate_unknown_response(det, sim, add_gaussian_noise=False)
        np.testing.assert_array_equal(r1, r2)

    def test_noise_adds_variation(self, wavelength_array, bias_array, sigma):
        """With noise, results should differ between runs."""
        det, sim = self._make_setup(wavelength_array, bias_array, sigma)
        np.random.seed(42)
        r1 = simulate_unknown_response(det, sim, add_gaussian_noise=True, noise_std_ratio=0.1)
        np.random.seed(123)
        r2 = simulate_unknown_response(det, sim, add_gaussian_noise=True, noise_std_ratio=0.1)
        assert not np.array_equal(r1, r2)

    def test_wavelength_mismatch_raises(self, bias_array, sigma):
        """Mismatched wavelength arrays should raise ValueError."""
        wl1 = np.linspace(400e-9, 1800e-9, 100)
        wl2 = np.linspace(400e-9, 1800e-9, 200)
        det = IdealSemiconductorPhotoDetector(
            bias_array=bias_array,
            e_g_ev=0.75,
            wavelength=wl1,
        )
        sim = SimulationSpectrum(
            wavelength_array=wl2,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=20e-9,
            _alpha=np.array([1.0]),
        )
        with pytest.raises(ValueError, match="do not match"):
            simulate_unknown_response(det, sim)
