

import torch
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
    num = torch.arange(0, length).reshape(length,1)
    den = torch.exp(torch.arange(0, dim, 2) * -(np.log(n) / dim))    

    # Compute even & odd positional encodings (trimming odds if needed)
    evens = torch.sin(num * den)
    odds = torch.cos(num * den)
    if dim%2:
        odds = odds[:, :-1]

    # Combine & scale positional encodings
    PE = torch.zeros((length, dim), dtype=torch.float)
    PE[:, 0::2] = evens
    PE[:, 1::2] = odds
    PE *= scale / PE.max()

    return PE.squeeze()
