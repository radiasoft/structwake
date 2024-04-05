

import numpy as np

from mlgreens import data

class GreensGrid(data.GreensData):
    """A set of Green's function data read from an HDF5 file then masked"""

    def __init__(self, file_path, grid_steps, seed=None):
        """Args:
          - file_path: Path to the HDF5 data file
        """

        super().__init__(file_path)

         # Set random seed (if any) before drawing patch sizes
        if seed is not None:
            np.random.seed(seed)

        #
        self.grid_steps = []
        for i in range(len(self)):

            self.grid_steps.append(np.random.choice())

    def __getitem__(self, index):
        """Retrieves an item from the dataset by index"""
        
        # Retrieve raw Green's function data
        G = super().__getitem__(index)

        # Split data into patches
        G_patches = data.patch2D(G, self.patch_sizes[index])

        # Mask data, keeping visible patches
        G_vis, G_hid, vIDs, hIDs = data.mask_data(G_patches, self.p_masks[index], mask_val=None)

        return G_grid
    
    def get_attrs(self, index):
        """Retrieves attributes for item from the dataset by index"""
        
        attrs = super().get_attrs(index).copy()
        attrs.update({"grid_step": self.grid_steps[index]})

        return attrs
