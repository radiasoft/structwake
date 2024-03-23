"""mcae.py

"""

import torch

from mlgreens.models import CNN
from mlgreens.data import encode

class MAE(torch.nn.Module):
    """A masked MLP-based autoencoder for estimating Green's functions over domain space"""

    def __init__(self, n_modes, n_features, n_params, cnn_layers=3, cnn_filters=16, , dimension=2
    ):

        super().__init__()

        self, in_channels, n_out, n_layers, n_filters, conv_kernel=3, pool_kernel=2,
        conv_activation='ReLU', output_activation="Sigmoid", dropout=.1, dimension=2, action="contract"

        n_hidden = n_modes * n_features

        self.encoder = CNN(n_modes,,action="contract")
        self.decoder = CNN(n_modes,,action="expand")

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
