
import torch
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
        channel_dims = (data.shape[0],)
        dat_dims = torch.tensor(data.shape[1:])
    else:
        channel_dims = ()
        dat_dims = torch.tensor(data.shape)
    
    # Assign original positions to patches
    nx, ny = dat_dims // patch_dims
    patches = torch.tensor([[
        [
            [x*patch_dims[0], (x+1)*patch_dims[0]],
            [y*patch_dims[1], (y+1)*patch_dims[1]]
        ] for y in range(ny)] for x in range(nx)
    ])

    # Assign original data to patched data
    patched_data = torch.zeros((nx, ny,)+channel_dims+tuple(patch_dims))
    for x in range(nx):
        for y in range(ny):
          pxs, pys = patches[x, y]
          patched_data[x, y] = data[..., pxs[0]:pxs[1], pys[0]:pys[1]]

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

    patches = torch.tensor([
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

    # Determine whether 2D data has channels
    if len(patched_data.shape)==5:
        channel_dims = (patched_data.shape[2],)
    else:
        channel_dims = ()
    patch_dims = np.array(patched_data.shape[-2:])

    #
    nx, ny = full_dims // patch_dims
    patches = torch.tensor([[
        [
            [x*patch_dims[0], (x+1)*patch_dims[0]],
            [y*patch_dims[1], (y+1)*patch_dims[1]]
        ] for y in range(ny)] for x in range(nx)
    ])

    # Reconstruct data from patches
    data = np.zeros(channel_dims+full_dims)
    for x in range(nx):
        for y in range(ny):
          pxs, pys = patches[x, y]
          data[..., pxs[0]:pxs[1], pys[0]:pys[1]] = patched_data[x, y]
          
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

def random_mask2D(shape, p_mask, sizes=None, rseed=None):
    
    if rseed is not None:
        np.random.seed(rseed)

    mask = torch.zeros(shape, dtype=bool)
    shape = np.array(shape)

    # Do 
    if sizes is None:
        w, h = shape
        n_rand = int(p_mask * shape.prod())
        masked_IDs = np.random.randint(low=[0, 0], high=[w, h], size=(n_rand, 2))
        mask[masked_IDs[:, 0], masked_IDs[:, 1]] = True

    # Do
    else:
        
        # Assign original positions to patches
        nx, ny = shape // sizes
        patches = torch.tensor([[
            [
                [x * sizes[0], (x+1) * sizes[0]],
                [y * sizes[1], (y+1) * sizes[1]]
            ] for y in range(ny)] for x in range(nx)
        ])

        # Randomly select patches to mask
        n_rand = int(p_mask * nx*ny)
        masked_IDs = np.random.randint(low=[0, 0], high=[nx, ny], size=(n_rand, 2))

        #
        for id in masked_IDs:
            pxs, pys = patches[id[0], id[1]]
            mask[pxs[0]:pxs[1], pys[0]:pys[1]] = True

    return mask

def grid_mask2D(shape, grid):
    
    # Randomly select data indices to mask
    w, h = shape
    gstep = shape // grid

    # Mask by replacing values at indices
    mask = np.zeros(shape)
    mask[::gstep[0], ::gstep[1]] = 1.

    return mask
    
def mask_data(data, p_mask, mask_val=None, rseed=None):
    """Returns a random mask for input data

    Args:
      data: Data to create a mask for
      p_mask: Percentage of data points to mask
      mask_val: Value to insert in masked positions (default 0.)
        Note - A value of None will result in masked positions being deleted
      rseed: Random seed for selection repeatability (default None)

    Returns:
      visible: Visible data after applying mask
      hidden: Hidden data with visible data masked
      (visibleIDs, hiddenIDs): Indices for visible & hidden data
    """

    if rseed is not None:
        np.random.seed(rseed)

    # Randomly select data indices to mask
    N = len(data)
    hiddenIDs = np.random.choice(N, int(p_mask*N), replace=False)
    visibleIDs = np.delete(np.arange(N), hiddenIDs, axis=0)

    # Mask by removing values at indices
    if mask_val is None:
      visible = data[list(visibleIDs)]
      hidden = data[list(hiddenIDs)]
    
    # Mask by replacing values at indices
    else:
      visible = data.clone()
      visible[hiddenIDs] = mask_val
      hidden = data-visible

    return visible, hidden, visibleIDs, hiddenIDs
