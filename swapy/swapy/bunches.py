"""Distributions for charged bunches of particles

Note: Distributions are normalized to 1C and need to be multiplied by charge units
"""

import numpy as np

def _step(x):
    """A heaviside _step function valid for vectors or scalars"""

    # Handle vector-valued inputs
    if hasattr(x, 'shape'):
        y = np.zeros(x.shape)
        y[x > 0.0] = 1
        y[x == 0.0] = 0.5

    # Handle scalar-valued inputs
    else:
        if x > 0: y = 1
        elif x == 0: y = 0.5
        else: y = 0

    return y

def _uniform(Z, sigma):
  """A uniform bunch with width 2*sigmaz"""

  rho = (-_step(Z - sigma) + _step(Z + sigma)) / (2. * sigma)

  return rho
    
def _linear(Z, sigma):
    """A linearly-ramped bunch with width 2*sigmaz"""

    rho = (Z + sigma) * (-_step(Z-sigma) + _step(Z + sigma)) / (2. * sigma**2)

    return rho
    
def _parabolic(Z, sigma):
    """parabolic bunch with width 2*sigmaz"""

    rho = (sigma**2 - Z**2) * (-_step(Z - sigma) + _step(Z + sigma)) / (4. * sigma**3 / 3.)
    return rho

def _gaussian(Z, sigma):
  """A Gaussian bunch with rms value sigmaz"""

  rho = np.exp(-Z**2 / (2. * sigma**2)) / np.sqrt(2.* np.pi * sigma**2)

  return rho

BUNCHDISTS = {
    "gaussian": _gaussian,
    "parabolic": _parabolic,
    "linear": _linear,
    "_uniform": _uniform
}

def get_bunch(Q, Z, sigma, distribution="gaussian"):
    """Computes longitudinal bunch charge density for a given bunch length & distribution

    Args:
      - Z: longitudinal evaluation points (m)
      - sigma: bunch length (m)
      - distribution: charge distribution profile (default 'gaussian')
        - Valid options are '_uniform', 'linear', 'parabolic', and 'gaussian'
    """

    # Select specified bunch distribution function
    try:
        dist_fun = BUNCHDISTS[distribution.lower()]
    except KeyError:
        raise ValueError("'distribution' must be one of: "+", ".join(BUNCHDISTS))

    # Compute bunch distribution
    dz = (Z.max() - Z.min()) / len(Z)
    deltaZ = Z - np.mean(Z)
    bunch = Q * dist_fun(deltaZ, sigma) * dz

    return bunch
