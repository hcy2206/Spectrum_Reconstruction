"""Tests for SpectrumReconstruction.SpectrumReconstructionSimulation (end-to-end)."""

import numpy as np
import pytest

from SpectrumReconstruction import SpectrumReconstructionSimulation
from SpectrumReconstruction.SpectrumReconstructionAdvance import SimulationSpectrum
from SpectrumReconstruction.Utility import gaussian_spectrum_sum, blackbody_spectrum_sum


# Use smaller arrays for faster tests
@pytest.fixture
def small_wavelength():
    return np.linspace(400e-9, 1800e-9, 141)


@pytest.fixture
def small_bias():
    return np.linspace(0e-9, 300e-9, 31)


@pytest.fixture
def small_mu():
    return np.linspace(600e-9, 1600e-9, 101)


@pytest.fixture
def small_sigma():
    return 1e-9 / 2 / np.sqrt(2 * np.log(2))


@pytest.fixture
def small_temperature():
    return np.linspace(800, 1400, 7)


# ========== Gaussian mode ==========

class TestSimulationGaussian:

    @pytest.fixture
    def srs(self, small_wavelength, small_bias, small_mu, small_sigma):
        return SpectrumReconstructionSimulation(
            bias_array=small_bias,
            wavelength_array=small_wavelength,
            base_function_name='gaussian',
            sigma=small_sigma,
            mu=small_mu,
            photo_detector_bias_mode='normal_move',
            delta_lambda=30e-9,
            e_g_ev=0.75,
            eta=1.0,
        )

    def test_init_attributes(self, srs, small_wavelength, small_bias):
        """Should initialize all expected attributes."""
        assert srs.base_function_name == 'gaussian'
        assert srs.photo_detector is not None
        assert srs.incident_spectrum is not None
        assert srs.spectrum_reconstruction is not None
        assert srs.response_pivot.shape == (len(small_bias), len(srs.mu))

    def test_response_matrix_shape(self, srs, small_bias, small_mu):
        """Response pivot should be (num_bias, num_mu)."""
        assert srs.response_pivot.shape == (len(small_bias), len(small_mu))

    def test_response_mapping_figure(self, srs):
        """Should return a plotly figure."""
        fig = srs.response_mapping_figure
        assert fig is not None

    def test_reconstruct_returns_coefficients(self, srs, small_wavelength, small_sigma):
        """reconstruct_spectrum should return coefficient array."""
        sim = SimulationSpectrum(
            wavelength_array=small_wavelength,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=small_sigma,
            _alpha=np.array([1.0]),
        )
        a = srs.reconstruct_spectrum(
            simulation_spectrum=sim,
            method='normal',
        )
        assert a is not None
        assert isinstance(a, np.ndarray)

    def test_reconstruction_spectrum_figure(self, srs, small_wavelength, small_sigma):
        """After reconstruction, spectrum figure should be available."""
        sim = SimulationSpectrum(
            wavelength_array=small_wavelength,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=small_sigma,
            _alpha=np.array([1.0]),
        )
        srs.reconstruct_spectrum(simulation_spectrum=sim, method='normal')
        fig = srs.reconstruction_spectrum_figure
        assert fig is not None

    def test_reconstruct_with_elasticnet(self, srs, small_wavelength, small_sigma):
        """ElasticNet reconstruction should work without error."""
        sim = SimulationSpectrum(
            wavelength_array=small_wavelength,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9, 1200e-9]),
            _sigma=np.array([20e-9, 30e-9]),
            _alpha=np.array([1.0, 0.8]),
        )
        a = srs.reconstruct_spectrum(
            simulation_spectrum=sim,
            method='ElasticNet',
            lambda_reg=0.1,
            alpha=0.5,
        )
        assert a is not None

    def test_reconstruct_with_noise(self, srs, small_wavelength, small_sigma):
        """Reconstruction with noise should still produce results."""
        sim = SimulationSpectrum(
            wavelength_array=small_wavelength,
            spectrum_function=gaussian_spectrum_sum,
        )
        sim.set_spectrum(
            _mu=np.array([1000e-9]),
            _sigma=small_sigma,
            _alpha=np.array([1.0]),
        )
        a = srs.reconstruct_spectrum(
            simulation_spectrum=sim,
            method='normal',
            add_gaussian_noise=True,
            noise_std_ratio=0.01,
        )
        assert a is not None


# ========== Blackbody mode ==========

class TestSimulationBlackbody:

    @pytest.fixture
    def srs_bb(self, small_wavelength, small_bias, small_temperature):
        return SpectrumReconstructionSimulation(
            bias_array=small_bias,
            wavelength_array=small_wavelength,
            base_function_name='blackbody',
            black_body_temperature=small_temperature,
            photo_detector_bias_mode='normal_move',
            delta_lambda=30e-9,
            e_g_ev=0.75,
            eta=1.0,
        )

    def test_init_blackbody(self, srs_bb, small_bias, small_temperature):
        """Should initialize in blackbody mode with correct shapes."""
        assert srs_bb.base_function_name == 'blackbody'
        assert srs_bb.response_pivot.shape == (len(small_bias), len(small_temperature))

    def test_reconstruct_blackbody(self, srs_bb, small_wavelength, small_temperature):
        """Should reconstruct blackbody spectrum."""
        sim = SimulationSpectrum(
            wavelength_array=small_wavelength,
            spectrum_function=blackbody_spectrum_sum,
        )
        sim.set_spectrum(
            _T=np.array([1000.0]),
            _alpha=np.array([1.0]),
        )
        a = srs_bb.reconstruct_spectrum(
            simulation_spectrum=sim,
            method='normal',
        )
        assert a is not None
        assert isinstance(a, np.ndarray)


# ========== Error cases ==========

class TestSimulationErrors:

    def test_missing_sigma_for_gaussian_raises(self, small_wavelength, small_bias, small_mu):
        """Gaussian mode without sigma should raise ValueError."""
        with pytest.raises(ValueError, match="Sigma and mu are required"):
            SpectrumReconstructionSimulation(
                bias_array=small_bias,
                wavelength_array=small_wavelength,
                base_function_name='gaussian',
                mu=small_mu,
            )

    def test_missing_temperature_for_blackbody_raises(self, small_wavelength, small_bias):
        """Blackbody mode without temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Black body temperature is required"):
            SpectrumReconstructionSimulation(
                bias_array=small_bias,
                wavelength_array=small_wavelength,
                base_function_name='blackbody',
            )

    def test_invalid_temperature_type_raises(self, small_wavelength, small_bias):
        """Blackbody temperature must be numpy array."""
        with pytest.raises(ValueError, match="numpy array"):
            SpectrumReconstructionSimulation(
                bias_array=small_bias,
                wavelength_array=small_wavelength,
                base_function_name='blackbody',
                black_body_temperature=[800, 1000, 1200],  # list, not ndarray
            )

    def test_invalid_sigma_type_raises(self, small_wavelength, small_bias, small_mu):
        """Sigma must be a float."""
        with pytest.raises(ValueError, match="Sigma should be a float"):
            SpectrumReconstructionSimulation(
                bias_array=small_bias,
                wavelength_array=small_wavelength,
                base_function_name='gaussian',
                sigma=np.array([1e-9]),  # ndarray, not float
                mu=small_mu,
            )
