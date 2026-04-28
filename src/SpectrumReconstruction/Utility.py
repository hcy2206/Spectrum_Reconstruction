import numpy as np
from numba import njit

# Physical constants
q = 1.60217662e-19  # Elementary charge [C]
h = 6.62607015e-34  # Planck constant [J*s]
c = 3.0e8  # Speed of light [m/s]
k = 1.38064852e-23  # Boltzmann constant [J/K]


def _as_1d_float_array(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f'{name} array is empty')
    return values


@njit
def blackbody(lambda_: float | np.ndarray,
              t: float | np.ndarray
              ) -> float | np.ndarray:
    # Compute the factor: hc/(lambda*k*t)
    factor = h * c / (lambda_ * k * t)

    # Compute the logarithm of the pre-factor: log(2hc^2) - 5*log(lambda)
    log_pre_factor = np.log(2 * h * c ** 2) - 5 * np.log(lambda_)

    # Compute log(exp(factor)-1) in a numerically stable way:
    # log(exp(x)-1) = x + log(1 - exp(-x))
    log_denominator = factor + np.log1p(-np.exp(-factor))

    # Finally, log(I) = log_pre_factor - log_denominator, so I = exp(log(I))
    log_i = log_pre_factor - log_denominator
    return np.exp(log_i)


@njit
def gaussian(lambda_: float | np.ndarray,
             mu: float | np.ndarray,
             sigma: float
             ) -> float | np.ndarray:
    return np.exp(-(lambda_ - mu) ** 2 / (2 * sigma ** 2))


def ideal_responsivity(lambda_: np.ndarray[float],
                       e_g: float,  # Bandgap energy [J]
                       eta: float = 1.0
                       ) -> np.ndarray:
    # Compute the cut-off wavelength
    lambda_g = h * c / e_g

    # Initialize the responsivity array
    R = np.zeros_like(lambda_)

    # Compute the responsivity for wavelengths below the cut-off
    mask = lambda_ <= lambda_g
    R[mask] = eta * (q * lambda_[mask]) / (h * c)

    return R


@njit
def smooth_responsivity(lambda_: np.ndarray[float],
                        e_g: float,  # Bandgap energy [J]
                        delta_lambda: float = 30e-9,
                        eta: float = 1.0
                        ) -> np.ndarray[float]:
    # Compute the cut-off wavelength
    lambda_g = h * c / e_g

    # Compute the basic responsivity
    R = eta * (q * lambda_) / (h * c)

    # Introduce exponential decay for smooth transition
    smooth_factor = 1 / (1 + np.exp((lambda_ - lambda_g) / delta_lambda))

    return R * smooth_factor


def smooth_responsivity_visible_blind(lambda_: np.ndarray[float],
                                      e_g: float,  # Bandgap energy [J]
                                      delta_lambda: float = 30e-9,
                                      blind_wavelength: float = 800e-9
                                      ) -> np.ndarray[float]:
    # Compute the cut-off wavelength
    lambda_g = h * c / e_g

    # Compute the basic responsivity
    R = (q * lambda_) / (h * c)

    R[lambda_ < blind_wavelength] = 0

    # Introduce exponential decay for smooth transition
    smooth_factor = 1 / (1 + np.exp((lambda_ - lambda_g) / delta_lambda))

    return R * smooth_factor


def gaussian_spectrum_sum(
        _wavelength: np.ndarray[float],
        _mu: np.ndarray[float],
        _sigma: float | np.ndarray[float],
        _alpha: np.ndarray[float],
        **kwargs
) -> np.ndarray[float]:
    wavelength = _as_1d_float_array(_wavelength, '_wavelength')
    mu = _as_1d_float_array(_mu, '_mu')
    alpha = _as_1d_float_array(_alpha, '_alpha')
    if mu.size != alpha.size:
        raise ValueError('_mu and _alpha must have the same length')

    sigma = np.asarray(_sigma, dtype=np.float64)
    if sigma.ndim == 0:
        spectrum_matrix = gaussian(wavelength[:, None], mu[None, :], float(sigma))
        return spectrum_matrix @ alpha

    sigma = sigma.reshape(-1)
    if sigma.size != mu.size:
        raise ValueError('_sigma must be scalar or have the same length as _mu')
    result = np.zeros_like(wavelength, dtype=np.float64)
    for i in range(mu.size):
        result += alpha[i] * gaussian(wavelength, mu[i], sigma[i])
    return result


def blackbody_spectrum_sum(
        _wavelength: np.ndarray[float],
        _T: np.ndarray[float],
        _alpha: np.ndarray[float],
        **kwargs
) -> np.ndarray[float]:
    wavelength = _as_1d_float_array(_wavelength, '_wavelength')
    temperature = _as_1d_float_array(_T, '_T')
    alpha = _as_1d_float_array(_alpha, '_alpha')
    if temperature.size != alpha.size:
        raise ValueError('_T and _alpha must have the same length')

    spectrum_matrix = blackbody(wavelength[:, None], temperature[None, :])
    return spectrum_matrix @ alpha


def fast_matmul(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape[0] != b.shape[0]:
        raise ValueError('The first dimension of a and b must match')
    return a.T @ b
