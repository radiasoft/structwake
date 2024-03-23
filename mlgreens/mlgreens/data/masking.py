
import numpy as np

def patch2D(data, patch_dims):
    """Breaks 2D data into subpatches of specified dimension

    Args:
      data:
      patch_dims:

    Returns:
      patched_data:
    """

    # Determine whether 2D data has channels
    if len(data.shape)==3:
        channel_dims = (data.shape[-1],)
    else:
        channel_dims = ()
    
    # Assign original positions to patches
    dat_dims = np.array(data.shape[:2])
    nx, ny = dat_dims // patch_dims
    patches = np.array([
        [[x*patch_dims[0], (x+1)*patch_dims[0]], [y*patch_dims[1], (y+1)*patch_dims[1]]]
        for y in range(ny) for x in range(nx)
    ])

    # Assign original data to patched data
    patched_data = np.zeros((nx*ny,)+tuple(patch_dims)+channel_dims)
    for p in range(nx*ny):
        pxs, pys = patches[p]
        patched_data[p] = data[pxs[0]:pxs[1], pys[0]:pys[1]]

    return patched_data

def patch3D(data, patch_dims):
    """Breaks 3D data into subpatches of specified dimension

    Args:
      data:
      patch_dims:

    Returns:
      patched_data:
    """

    dat_dims = np.array(data.shape[:3])
    nx, ny, nz = dat_dims // patch_dims

    patches = np.array([
        [[x*nx, (x+1)*nx], [y*ny, (y+1)*ny], [z*nz, (z+1)*nz]] 
        for z in range(nz) for y in range(ny) for x in range(nx)
    ])

    return patches

def unpatch2D(patched_data, full_dims):
    """Reintegrates patched 2D data

    Args:
      patched_data:
      full_dims:

    Returns:
      data:
    """

    patch_dims = np.array(patched_data.shape[-2:])
    nx, ny = full_dims // patch_dims
    patches = np.array([
        [[x*patch_dims[0], (x+1)*patch_dims[0]], [y*patch_dims[1], (y+1)*patch_dims[1]]]
        for y in range(ny) for x in range(nx)
    ])

    # Reconstruct data from patches
    data = np.zeros(full_dims)
    for p in range(nx*ny):
        pxs, pys = patches[p]
        data[pxs[0]:pxs[1], pys[0]:pys[1]] = patched_data[p]

    return data

def unpatch3D(patched_data, patches):
    """Reintegrates patched 3D data

    Args:
      patched_data:
      full_dims:

    Returns:
      data:
    """

    # Determine data shape & create holding array
    n_patches = len(patches)
    nx, ny, nz = patches.ptp(axis=1)
    data = np.zeros((nx,ny))

    # Reconstruct data from patches
    for p in range(n_patches):
        pxs, pys, pzs = patches[p]
        data[pxs[0]:pxs[1], pys[0]:pys[1], pzs[0]:pzs[1]] = patched_data[p]

    return data
    
def mask_data(data, p_mask, mask_val=0., rseed=None):
    """Returns a random mask for input data

    Args:
      data: Data to create a mask for
      p_mask: Percentage of data points to mask
      mask_val: Value to insert in masked positions (default 0.)
        Note - A value of None will result in masked positions being deleted
      rseed: Random seed for selection repeatability (default None)

    Returns:
      masked: Masked data
      imasked: Inverse masked data
    """

    if rseed is not None:
        np.random.seed(rseed)

    # Randomly select data indices to mask
    N = len(data)
    maskIDs = np.random.choice(N, int(p_mask*N), replace=False)
    imaskIDs = np.delete(np.arange(N), maskIDs)

    # Mask by replacing values at indices
    if mask_val is not None:
      masked = data.copy()
      masked[maskIDs] = mask_val
      imasked = data-masked

    # Mask by removing values at indices
    else:
        masked = np.delete(data, maskIDs)
        imasked = np.delete(data, imaskIDs)

    return masked, imasked, (maskIDs, imaskIDs)
