"""mcae.py

"""

import h5py
import torch
import numpy as np

from mlgreens import data
from mlgreens.models import CNN

class MCAE(torch.nn.Module):
    """A masked CNN-based autoencoder for estimating Green's functions over domain space"""

    def __init__(
            self, n_modes, n_features, p_mask, f_encode, patch_dims=None,
            cnn_layers=3, cnn_filters=16, conv_kernel=3, pool_kernel=2, upsample_kernel=2, 
            dropout=.1, dimension=2, conv_activation="LeakyReLU", output_activation="Sigmoid"
    ):

        super().__init__()

        # Assign patching, encoding, & masking parameters
        self.dimension = dimension
        self.patch_dims = patch_dims
        self.f_encode = f_encode
        self.p_mask = p_mask

        # Construct CNN encoder & decoder models
        CNN_args = {
            "n_layers": cnn_layers,
            "n_filters": cnn_filters,
            "conv_kernel": conv_kernel,
            "conv_activation": conv_activation,
            "dropout": dropout,
            "dimension": dimension
        }
        encoder_args = {
            "in_channels": n_modes,
            "n_out": n_features,
            "sample_kernel": pool_kernel,
            "action": "contract"
        }
        decoder_args = {
            "in_channels": n_features,
            "n_out": n_modes,
            "sample_kernel": upsample_kernel,
            "action": "expand",
            "output_activation": output_activation
        }
        self.encoder = CNN(**encoder_args, **CNN_args)
        self.decoder = CNN(**decoder_args, **CNN_args)

        # Construct a linear layer for producing learning masked tokens
        self.tokenizer = torch.nn.Linear(1, n_features)

    def forward(self, G):

        print(G.shape)

        # Forward batched samples individually
        if len(G.shape)==self.dimension+2:
            return torch.tensor([self(Gb) for Gb in G])
        
        print(G.shape)

        # Break data into patches
        G_dims = np.array(G.shape)
        G_patches = data.patch2D(G, self.patch_dims)
        n_patches = len(G_patches)

        # Compute scaled positional encodings, encode & mask data
        enc_scale = self.f_encode * G_patches.max()
        PE = data.encode(n_patches, 1, scale=enc_scale)
        EncG = np.array([G_patches[i] + PE[i] for i in range(n_patches)])
        G_masked, _, maskIDs = data.mask_data(EncG, .6, mask_val=None)

        # Separate encodings for visible & masked patches
        visible_PE = PE[maskIDs[0]]
        masked_PE = PE[maskIDs[1]]

        # Pass masked, encoded data through encoder CNN & re-encode
        h_visible = [self.encoder(G_masked[i])+visible_PE[i] for i in range(n_patches)]

        # Compute tokens for masked patches & encode
        masked_tokens = self.tokenizer(.5)
        h_masked = [masked_tokens + mPE for mPE in masked_PE]

        # Reconstruct full set of patch feature vector
        hs = h_visible + h_masked

        # Pass feature vectors through decoder & reintegrate patches
        Gh_patches = [self.decoder(h) for h in hs]
        Gh = data.unpatch2D(Gh_patches, G_dims)

        return Gh, hs
    
    @torch.no_grad()
    def process(self, G):
        
        if self.training:
            self.eval()

        G, scales = data.scale_minmax(G)
        Gh, _ = self(G)
        Gh = data.unscale_minmax(Gh, scales)

        return Gh

    def fit(
            self, Gdata, val_split=.1, loss='mse_loss', num_epochs=1000,
            lr=1.e-3, optimizer='Adam', batch_size=25, rseed=None, 
            num_workers=1, verbose=False, print_every=1,
            save_archive=True, archive_path="./MCAE-training.h5", save_model=True, model_path="MCAE.h5"
        ):

        # Create data loader objects for training & validation
        train_data, val_data = data.split_data(Gdata, val_split, seed=rseed)
        loader_args = {"batch_size":batch_size, "shuffle":True, "num_workers":num_workers}
        train_loader = torch.utils.data.DataLoader(train_data, **loader_args)
        val_loader = torch.utils.data.DataLoader(val_data, **loader_args)

        # Initialize an optimizer for training
        opt_class = getattr(torch.optim, optimizer)
        opt = opt_class(self.parameters(), lr=lr)

        # Define loss function & create holding arrays for loss statistics
        loss_fun = getattr(torch.nn.functional, loss)
        losses = {ltype:np.zeros((num_epochs, 2)) for ltype in ["training", "validation"]}

        # Loop over training epochs
        self.train()
        for e in range(num_epochs):

            train_losses = []
            val_losses = []
            
            # Loop over training batches
            for G in train_loader:
                G, _ = data.scale_minmax(G)

                # Compute loss for current model
                Gh, _ = self(G)
                loss = loss_fun(G, Gh)

                # Update model weights
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.detach())

            # Evaluate updated model on validation data
            self.eval()
            for G in val_loader:
                Gh, _ = self(G)
                val_loss = loss_fun(G, Gh)
                val_losses.append(val_loss.detach())
            self.train()

            # Compute & store final epoch loss statistics
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)
            losses["training"][e] = train_losses.mean(), train_losses.std()
            losses["validation"][e] = val_losses.mean(), val_losses.std()

            # Print epoch loss statistics if requested
            if verbose and not (e%print_every):
                ostr = "EPOCH {:d}\n".format(e+1)
                for ltype, lstats in losses.items():
                    ostr += ltype+": {:.3f} (std {:.3f})\n".format(lstats[e])
                print(ostr)

        # Save model & training archive if requested
        if save_model:
            self.save_model(model_path)
        if save_archive:
            with h5py.File(archive_path, "w") as archive_file:
                for ltype, lstats in losses.items():
                    archive_file.create_dataset(ltype, data=lstats)     

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
    