

import torch
import numpy as np

from mlgreens import data

class GreensPatches(torch.utils.data.Dataset):
    """A set of Green's function data read from an HDF5 file then patched & masked"""

    def __init__(self, file_path, p_masks, sizes, max_sets=None, seed=None):
        """Args:
          - file_path: Path to the HDF5 data file
        """

        self.GData = data.GreensData(file_path, max_sets)
        self.attrs = {"p_masks": p_masks, "sizes": sizes}
        self.attrs.update(self.GData.attrs)

        # Set random seed (if any) before drawing patch sizes
        if seed is not None:
            np.random.seed(seed)

        # Set masking ratio & randomly select patch sizes
        self.p_masks = np.random.choice(p_masks, len(self))
        self.patch_sizes = sizes[np.random.choice(len(sizes), len(self))]
        
    def __len__(self):
        """Returns the length of the dataset"""
        
        return len(self.GData)

    def __getitem__(self, index):
        """Retrieves an item from the dataset by index"""
        
        # Retrieve raw Green's function data
        G = self.GData[index]

        # Split data into patches
        G_patches = data.patch2D(G, self.patch_sizes[index])

        # Mask data, keeping visible patches
        G_vis, G_hid, vIDs, hIDs = data.mask_data(G_patches, self.p_masks[index], mask_val=None)

        return (G_vis, G_hid, vIDs, hIDs)
    
    def collator(self, batch):

        return batch
    
    def get_attrs(self, index):
        """Retrieves attributes for item from the dataset by index"""
        
        attrs = {"p_mask":self.p_masks[index], "patch_dims": self.patch_sizes[index]}
        attrs.update(self.GData.get_attrs(index))

        return attrs
