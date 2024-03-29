
import h5py
import torch
import atexit

class GreensData(torch.utils.data.Dataset):
    """A set of Green's function data stored in an HDF5 file"""

    def __init__(self, file_path, max_sets=None):
        """Args:
          - file_path: Path to the HDF5 data file
        """
        
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
                if max_sets is not None:
                    n_sets = min([max_sets, n_sets])
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
            data = torch.tensor(data_file[groupID][setID][()])

        return data
    
    def get_attrs(self, index):
        """Retrieves attributes for item from the dataset by index"""
        
        groupID, setID = self._indices[index]
        attrs = {"group": groupID, "sample": setID}
        with h5py.File(self.file_path) as data_file:
            attrs.update(data_file[groupID].attrs)

        return attrs
