import numpy as np
from numba import njit

# Physical constants
q = 1.60217662e-19  # Elementary charge [C]
h = 6.62607015e-34  # Planck constant [J*s]
c = 3.0e8  # Speed of light [m/s]
k = 1.38064852e-23  # Boltzmann constant [J/K]


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


def gaussian(lambda_: float | np.ndarray,
             mu: float | np.ndarray,
             sigma: float
             ) -> float | np.ndarray:
    # Gaussian function for spectral analysis
    return np.exp(-((lambda_ - mu) ** 2) / (2 * sigma ** 2))


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


@njit(parallel=True)
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
        _sigma: float,
        _alpha: np.ndarray[float],
        **kwargs
) -> np.ndarray[float]:
    result = _alpha[0] * gaussian(_wavelength, _mu[0], _sigma[0])
    for i in range(1, len(_mu)):
        result += _alpha[i] * gaussian(_wavelength, _mu[i], _sigma[i])
    return result


def blackbody_spectrum_sum(
        _wavelength: np.ndarray[float],
        _T: np.ndarray[float],
        _alpha: np.ndarray[float],
        **kwargs
) -> np.ndarray[float]:
    result = _alpha[0] * blackbody(_wavelength, _T[0])
    for i in range(1, len(_T)):
        result += _alpha[i] * blackbody(_wavelength, _T[i])
    return result
