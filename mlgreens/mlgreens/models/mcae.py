"""mcae.py

"""

import h5py
import torch
import numpy as np

from mlgreens import data
from mlgreens.models import cnn, Model

class MCAE(Model):
    """A masked CNN-based autoencoder for estimating Green's functions over domain space"""

    def __init__(
            self, n_modes, n_features, cnn_layers=3, conv_kernel=3, sample_kernel=2,
            dropout=.1, dimension=2, conv_activation="LeakyReLU", output_activation="Sigmoid"
    ):
        
        # Store input arguments
        hypers = {arg:val for arg,val in locals().items() if arg not in ['self', '__class__']}
        super().__init__(hypers)

        # Import activation functions & define layer dropouts
        conv_activation = getattr(torch.nn, conv_activation)
        output_activation = getattr(torch.nn, output_activation)
        dropout = list(dropout) if hasattr(dropout, '__len__') else [dropout,]*cnn_layers
        conv_args = {
            "conv_kernel": conv_kernel, 
            "activation": conv_activation,
            "dimension": dimension
        }

        # Add the first convolutional unit
        self.input_layer = cnn.ConvUnit(n_modes, n_modes, **conv_args)

        # Add convolutional layers along the contraction path (with optional dropout layers)
        for l in range(cnn_layers):
            filters = [n_modes*2**l, n_modes*2**(l+1)]
            setattr(
                self, "contraction-layer_{:d}".format(l+1),
                cnn.ContractionLayer(filters[0], filters[1], sample_kernel, **conv_args)
            )
            if dropout[l]:
                setattr(
                    self, "dropout_{:d}".format(l+1),
                    torch.nn.Dropout(p=dropout[l])
                )

        # Define channel gain per convolutional layer & corresponding dense layer dimension
        self.channel_gain = (self.conv_kernel-1) * (self.sample_kernel-1)
        self.dense_dim = n_modes * self.channel_gain**cnn_layers

        # Add dense layers for bottleneck
        self.to_features = torch.nn.Linear(self.dense_dim, n_features)
        self.from_features = torch.nn.Linear(n_features, self.dense_dim)

        # Add convolutional layers along the expansion path (with optional dropout layers)
        for l in range(cnn_layers):
            filters = [n_modes*2**(cnn_layers-l), n_modes*2**(cnn_layers-l-1)]
            setattr(
                self, "expansion-layer_{:d}".format(l+1),
                cnn.ExpansionLayer(filters[0], filters[1], sample_kernel, **conv_args)
            )
            if dropout[l]:
                setattr(
                    self, "dropout_{:d}".format(l+1),
                    torch.nn.Dropout(p=dropout[cnn_layers-l-1])
                )

        # Add final convolutional layer that produces output
        self.output_layer = torch.nn.Sequential(
            cnn.ConvUnit(filters[1], n_modes, **conv_args),
            output_activation()
        )

    def forward(self, G):

        # Compute hidden vectors
        h = self.input_layer(G)
        for l in range(self.cnn_layers):
            h = getattr(self, "contraction-layer_{:d}".format(l+1))(h)

        # Flatten bottleneck vectors
        dense_shape = np.array(h.shape[-2:])
        h = h.flatten(start_dim=1).unsqueeze(1)

        # Reduce flattened bottleneck vector dimensions with pooling
        pool_size = int(dense_shape.prod())
        h = torch.nn.functional.max_pool1d(h, pool_size).squeeze()

        # Compute encoded feature vectors
        X = self.to_features(h) #+ PE_vis

        #
        h = self.from_features(X)[...,None,None]
        h = h * torch.ones((self.dense_dim,)+tuple(dense_shape))

        #
        for l in range(self.cnn_layers):
            h = getattr(self, "expansion-layer_{:d}".format(l+1))(h)

        # Pass feature vectors through decoder & reintegrate patches
        G_pred = self.output_layer(h)

        return G_pred, X
    
    @torch.no_grad()
    def process(self, G):
        
        if self.training:
            self.eval()

        G_rs = data.factor_resize(G, self.channel_gain)
        G_pred, X = self(G_rs)
        G_pred = torch.nn.functional.interpolate(G_pred, G.shape[-2:])

        return G_pred, X

    def fit(
            self, masked_data, val_split=.9, loss='mse_loss', num_epochs=1000, batch_size=20,
            lr=1.e-3, optimizer='Adam', num_workers=1, seed=None, verbose=False, print_every=1,
            save_archive=True, archive_path="./MCAE-training.h5", save_model=True, model_path="MCAE.h5"
        ):
        
        # Set up data loaders for training & validation data
        train_data, val_data = data.split_data(masked_data, val_split, seed=seed)
        loader_args = {
            "shuffle":True,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": masked_data.collator
        }
        train_loader = torch.utils.data.DataLoader(train_data, **loader_args)
        val_loader = torch.utils.data.DataLoader(val_data, **loader_args)

        # Initialize an optimizer for training
        opt_class = getattr(torch.optim, optimizer)
        opt = opt_class(self.parameters(), lr=lr, weight_decay=.01)

        # Define loss function & create holding arrays for loss statistics
        loss_fun = getattr(torch.nn.functional, loss)
        losses = {ltype:np.zeros((num_epochs, 2)) for ltype in ["training", "validation"]}

        # Loop over training epochs
        self.train()
        for e in range(num_epochs):
            train_losses = []
            val_losses = []
            
            # Loop over batches of training data
            for Gs, masks in train_loader:

                if torch.cuda.is_available():
                    Gs.cuda()
                    masks.cuda()

                # Mask data samples
                mode_masks = torch.stack([masks,]*self.n_modes, axis=1)
                Gs[mode_masks] = -1.

                # Pass masked data through network
                Gs = data.factor_resize(Gs, self.channel_gain)
                Gs_pred, _ = self(Gs)

                # Compute loss & update model weights
                batch_loss = loss_fun(Gs, Gs_pred)
                opt.zero_grad()
                batch_loss.backward()
                opt.step()

                train_losses.append(batch_loss.detach())

            # Evaluate updated model on validation data
            self.eval()
            for Gs, masks in val_loader:
                mode_masks = torch.stack([masks,]*self.n_modes, axis=1)
                Gs[mode_masks] = -1.
                Gs = data.factor_resize(Gs, self.channel_gain)
                Gs_pred, _ = self(Gs)
                val_loss = loss_fun(Gs, Gs_pred)
                val_losses.append(val_loss.detach())

            # Compute & store final epoch loss statistics
            train_losses = np.array(train_losses)
            val_losses = np.array(val_losses)
            losses["training"][e] = train_losses.mean(), train_losses.std()
            losses["validation"][e] = val_losses.mean(), val_losses.std()

            # Print epoch loss statistics if requested
            if verbose and not (e%print_every):
                ostr = "EPOCH {:d}\n".format(e+1)
                for ltype, lstats in losses.items():
                    ostr += ltype.capitalize()+" Loss: {:.3f} (std {:.3f})\n".format(*lstats[e])
                print(ostr)

            #
            self.train()

        # Save model & training archive if requested
        if save_model:
            self.save_model(model_path)
        if save_archive:
            with h5py.File(archive_path, "w") as archive_file:
                for ltype, lstats in losses.items():
                    archive_file.create_dataset(ltype, data=lstats)
