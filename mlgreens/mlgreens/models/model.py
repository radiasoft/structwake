
import h5py
import torch

class Model(torch.nn.Module):
    """Base class for all ML models in `mlgreens.models`"""

    def __init__(self, hypers):

        super().__init__()

        #
        self.hypers = list(hypers.keys())
        for h, val in hypers.items():
            setattr(self, h, val)

    def __len__(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @torch.no_grad()
    def fit(self, *args):
        """"""
        
        raise NotImplementedError("This is a vritual method. Stop.")

    def save_model(self, path):
        """Saves model weights and hyperparameters as an HDF5 file

        Args:
          path: path at which to store the HDF5 file
        """

        with h5py.File(path, "w") as model_file:
            for h, hyper in self.hypers.items():
                model_file.attrs[h] = hyper
            for s, state in self.state_dict().items():
                model_file.create_dataset(s, data=state.numpy())      

    @classmethod
    def load_model(cls, path):
        """Loads a stored model from an HDF5 file

        Args:
          path: path from which to load the HDF5 file
          loss: Loss function used in training
          metrics: Dictionary containing pairs of evaluation metric names and functions
        """

        with h5py.File(path, "r") as model_file:
            new_model = cls(**dict(model_file.attrs))
            state_dict = {}
            for d, dataset in model_file.items():
                state_dict[d] = torch.tensor(dataset[()])
            new_model.load_state_dict(state_dict)
        new_model.eval()

        return new_model
    