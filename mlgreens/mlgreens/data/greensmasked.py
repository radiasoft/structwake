
import torch
import numpy as np

from mlgreens import data

class GreensMasked(data.GreensData):
    """A set of Green's function data read from an HDF5 file then masked"""

    def __init__(self, file_path, p_masks, sizes, seed=None):
        """Args:
          - file_path: Path to the HDF5 data file
        """

        super().__init__(file_path)

         # Set random seed (if any) before drawing patch sizes
        if seed is not None:
            np.random.seed(seed)

        # Set masking ratio & randomly select patch sizes
        self.p_masks = np.random.choice(p_masks, len(self))
        self.sizes = sizes[np.random.choice(len(sizes), len(self))]

    def __getitem__(self, index):
        """Retrieves an item from the dataset by index"""
        
        # Retrieve raw Green's function data & mask parameters
        G = super().__getitem__(index)
        p = self.p_masks[index]
        size = self.sizes[index]

        # Compute 2D mask for this sample
        shape = G.shape[-2:]
        mask = data.random_mask2D(shape, p, size)

        return G, mask
    
    def collator(self, batch):

        G_batch = torch.stack([sample[0] for sample in batch])
        mask_batch = torch.stack([sample[1] for sample in batch])

        return G_batch, mask_batch
    
    def get_attrs(self, index):
        """Retrieves attributes for item from the dataset by index"""
        
        attrs = {"p_mask":self.p_masks[index], "patch_dims": self.patch_sizes[index]}
        attrs.update(self.GData.get_attrs(index))

        return attrs
