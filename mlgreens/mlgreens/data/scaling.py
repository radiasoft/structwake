
import torch
import numpy as np

def scale_minmax(input_tensor):
    """Scales input data stored in a numpy ndarray to the [0, 1] range

    Args:
      input_tensor: Data to normalize
    
    Returns:
      output_tensor: Normalized data
      scales: Scaling factors used to normalize data
    """

    output_tensor = input_tensor.detach().clone()
    scales = [output_tensor.min(),]
    output_tensor -= scales[0]
    scales.append(output_tensor.max())
    output_tensor /= scales[1]

    return output_tensor, scales

def unscale_minmax(input_tensor, scales):
    """Rescales [0,1] normalized input data stored in a numpy ndarray

    Args:
      input_tensor: Normalized data to rescale
      scale: Scale used to normalize original data

    Returns:
      output_tensor: Rescaled data
    """

    output_tensor = input_tensor.detach.clone()
    output_tensor *= scales[1]
    output_tensor += scales[0]

    return output_tensor

def factor_resize(data, factor):
    
    # Do nothing if the scaling factor is unity
    if factor==1:
        return data.clone()

    # Determine output dimensions
    final_size = np.array([1, 1])
    start_size = np.array(data.shape[-2:])
    while any(final_size<start_size):
        final_size *= factor

    #
    sized_data = data.clone()
    sized_data = torch.nn.functional.interpolate(sized_data, list(final_size))
    
    return sized_data
