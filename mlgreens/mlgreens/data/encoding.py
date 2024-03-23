

import numpy as np

def encode(length, dim, n=int(1e4), scale=1.):
    """Creates a positional encoding for the given input
    
    Args:
      X: Data to be encoded
      dim: Dimension of output
      n: Denominator scaling factor (default 10,000)
      scale: Scaling factor for encoding values

    Returns:
      PE: Positional encoding
    """    

    # Compute sinusoid arguments
    num = np.arange(0, length).reshape(length,1)
    den = np.exp(np.arange(0, dim, 2) * -(np.log(n) / dim))    

    # Compute & scale positional encoding
    PE = np.zeros((length, dim))
    PE[:, 0::2] = np.sin(num * den)
    PE[:, 1::2] = np.cos(num * den)
    PE *= scale / PE.max()

    return PE.squeeze()
