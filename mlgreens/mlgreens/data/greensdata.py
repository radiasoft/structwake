
import h5py
import torch
from mlgreens import data

class GreensData(torch.utils.data.Dataset):
    """A set of Green's function data stored in an HDF5 file"""

    def __init__(self, file_path):
        """Args:
          - file_path: Path to the HDF5 data file
        """

        super().__init__()
        
        # Store the path to the data file
        self.file_path = file_path

        # Create a mapping between indices & data sets
        ind0 = 0
        self._indices = {}
        self.attrs = {}
        with h5py.File(self.file_path) as data_file:
            self.attrs.update(data_file.attrs)
            for group in data_file:
                n_sets = len(data_file[group])
                new_indices = {i+ind0: [group, str(i+1)] for i in range(n_sets)}
                self._indices.update(new_indices)
                ind0 = max(self._indices)+1

    def __len__(self):
        """Returns the length of the dataset"""
        
        return len(self._indices)

    def __getitem__(self, index):
        """Retrieves an item from the dataset by index"""
        
        groupID, setID = self._indices[index]
        with h5py.File(self.file_path) as data_file:
            G = torch.tensor(data_file[groupID][setID][()], dtype=torch.float)

        G, _ = data.scale_minmax(G)

        return G
    
    def get_attrs(self, index):
        """Retrieves attributes for item from the dataset by index"""
        
        groupID, setID = self._indices[index]
        attrs = {"group": groupID, "sample": setID}
        with h5py.File(self.file_path) as data_file:
            attrs.update(data_file[groupID][setID].attrs)

        return attrs
