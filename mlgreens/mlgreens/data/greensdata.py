
import h5py
import torch
import atexit

class GreensData(torch.utils.data.Dataset):
    """A set of Green's function data stored in an HDF5 file"""

    def __init__(self, file_path):
        """Args:
          - file_path: Path to the HDF5 data file
        """
        
        # Create a connection to the HDF5 data file
        self._h5file = h5py.File(file_path)

        # Create a mapping between indices & data sets
        ind0 = 0
        self._indices = {}
        for group in self._h5file:
            n_sets = len(self._h5file[group])
            new_indices = {i+ind0: [group, str(i)] for i in range(n_sets)}
            self._indices.update(new_indices)
            ind0 = max(self._indices)+1

        # Register this instance's cleanup function
        atexit.register(self.close)

    def __len__(self):
        """Returns the length of the dataset"""
        
        return len(self._indices)

    def __getitem__(self, index):
        """Retrieves an item from the dataset by index"""
        
        groupID, setID = self._indices[index]
        data = self._h5file[groupID][setID][()]

        return data
    
    def get_attrs(self, index):
        """Retrieves attributes for item from the dataset by index"""
        
        groupID, setID = self._indices[index]
        attrs = {"group": groupID, "sample": setID}
        attrs.update(self._h5file[groupID].attrs)

        return attrs

    def close(self):
        """Closes the HDF5 data file on cleanup"""

        if self._h5file:
            self._h5file.close()
