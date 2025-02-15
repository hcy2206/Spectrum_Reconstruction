import numpy as np


# def blackbody(lambda_: np.ndarray,
#               t: float):  # -> np.ndarray:
#     h = 6.62607015e-34
#     c = 3.0e8
#     k = 1.38064852e-23
#     return 2 * h * c ** 2 / lambda_ ** 5 / (np.exp(h * c / (lambda_ * k * t)) - 1)


# def blackbody(lambda_: np.ndarray, t: float) -> np.ndarray:
#     h = 6.62607015e-34
#     c = 3.0e8
#     k = 1.38064852e-23
#
#     # Scale the term h * c / (lambda_ * k * t)
#     factor = h * c / (lambda_ * k * t)
#
#     # Use np.expm1 to avoid overflow; this function computes exp(x) - 1, which is numerically more stable
#     return 2 * h * c ** 2 / lambda_ ** 5 / (np.expm1(factor))

def blackbody(lambda_: np.ndarray, t: float) -> np.ndarray:
    # Physical constants
    h = 6.62607015e-34  # Planck constant [J*s]
    c = 3.0e8  # Speed of light [m/s]
    k = 1.38064852e-23  # Boltzmann constant [J/K]

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


def gaussian(lambda_: np.ndarray,
             mu: float,
             sigma: float):  # -> np.ndarray:
    # Gaussian function for spectral analysis
    return np.exp(-((lambda_ - mu) ** 2) / (2 * sigma ** 2))
