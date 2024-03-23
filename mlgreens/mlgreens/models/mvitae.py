"""gmae.py

Contains a class for describing a GMAE, Green's 
Masked Autoencoder model.

"""

import torch

from mlgreens.models import CNN
from mlgreens.data import encode

class MViTAE(torch.nn.Module):
    """A masked vision transformer (ViT) autoencoder for estimating Green's functions over domain space"""

    def __init__(self, n_features, n_params, cnn_layers=3, cnn_filters=16, n_modes=1, modal=True):

        super().__init__()

        n_hidden = n_modes * n_features
        n_latent = n_modes*n_params if modal else n_params

        self.encoder = GreensEncoder()
        self.decoder = GreensDecoder()

    def mask(self, Gr):

        I = None
        Gm = None

        return Gm, I

    def forward(self, G):

        # Mask input Greens functions & pass through encoder
        Gm, I = self.mask(G)
        z = self.encoder(Gm)

        # Pass latent vector through decoder
        Gp = self.decoder(z, I)

        return Gp, z

    def fit(self, Gdata):
        pass
