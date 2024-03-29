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
            f_encode=.1, dropout=.1, dimension=2, conv_activation="LeakyReLU", output_activation="Sigmoid"
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

        # Add a dense layer for learning masked tokens
        self.tokenizer = torch.nn.Linear(1, n_features)
        self.get_tokens = lambda nh: self.tokenizer(torch.ones(1)).expand(nh, n_features)

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

    def preprocess(self, G_patches):

        # Perform 0/1 scaling on input, leaving room for positional encoding
        G, scales = data.scale_minmax(G_patches)
        G *= (1.-self.f_encode)

        # Resize patches for convolutional encoding/decoding
        G = data.factor_resize(G, self.channel_gain)

        return G, scales
    
    def postprocess(self, G_patches, scales, size):

        # Undo patch resizing & 0/1 scaling
        G = torch.nn.functional.interpolate(G_patches, size)
        G = data.unscale_minmax(G, scales)

        return G

    def forward(self, G_patches, visible_IDs, hidden_IDs):
                
        # Gather information about patch number & dimensions
        n_vis = len(visible_IDs)
        n_hid = len(hidden_IDs)
        n_patches = n_vis + n_hid
        patch_dims = np.array(G_patches.shape[-2:])

        # Positionally encode & rescale visible patches of data
        PE_dims = np.array([n_vis, self.n_modes, *patch_dims])
        PE_in = data.encode(n_vis, PE_dims[1:].prod(), scale=self.f_encode).reshape((*PE_dims,))
        G_enc = G_patches.detach() + PE_in.detach()

        # Create positional encodings for feature space data
        PE_feat = data.encode(n_patches, self.n_features, scale=self.f_encode)
        PE_vis = PE_feat[visible_IDs].detach()
        PE_hid = PE_feat[hidden_IDs].detach()

        # Compute hidden vectors for visible samples
        h_vis = self.input_layer(G_enc)
        for l in range(self.cnn_layers):
            h_vis = getattr(self, "contraction-layer_{:d}".format(l+1))(h_vis)

        # Flatten bottleneck vector for visible samples
        dense_shape = np.array(h_vis.shape[-2:])
        h_vis = h_vis.flatten(start_dim=1).unsqueeze(1)

        # Reduce flattened bottleneck vector dimensions with pooling
        pool_size = int(dense_shape.prod())
        h_vis = torch.nn.functional.max_pool1d(h_vis, pool_size).squeeze()

        # Compute & combine encoded feature vectors for visible & hidden samples
        X_vis = self.to_features(h_vis) + PE_vis
        X_hid = self.get_tokens(n_hid) + PE_hid
        X = torch.zeros((n_patches, self.n_features))
        X[visible_IDs] = X_vis
        X[hidden_IDs] = X_hid

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

        G_proc, scales = self.preprocess(G)
        G_pred, X = self(G_proc)
        G_pred, _ =  data.unscale_minmax(G_pred, scales)
        G_pred = torch.nn.functional.interpolate(G_pred, G.shape)

        return G_pred, X

    def fit(
            self, patch_data, val_split=.9, loss='mse_loss', num_epochs=1000,
            lr=1.e-3, optimizer='Adam', rseed=None, verbose=False, print_every=1,
            save_archive=True, archive_path="./MCAE-training.h5", save_model=True, model_path="MCAE.h5"
        ):
        
        train_data, val_data = data.split_data(patch_data, val_split, seed=rseed)
        loader_args = {"shuffle":True, "collate_fn": patch_data.collator}
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
            
            # Loop over training batches
            s=0
            for sample in train_loader:
                s+=1

                #
                G_vis, G_hid, vIDs, hIDs = sample[0]
                G_vis, _ = self.preprocess(G_vis)
                G_hid, _ = self.preprocess(G_hid)
                
                # Compute loss for current model
                G_pred, _ = self(G_vis.detach(), vIDs, hIDs)
                G_true = torch.zeros(G_pred.shape)
                G_true[vIDs] = G_vis.detach()
                G_true[hIDs] = G_hid.detach()
                loss = loss_fun(G_true, G_pred)

                # Update model weights
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.detach())

            # Evaluate updated model on validation data
            self.eval()
            for sample in val_loader:
                G_vis, G_hid, vIDs, hIDs = sample[0]
                G_vis, _ = self.preprocess(G_vis)
                G_hid, _ = self.preprocess(G_hid)
                
                # Compute loss for current model
                G_pred, _ = self(G_vis, vIDs, hIDs)
                G_true = torch.zeros(G_pred.shape)
                G_true[vIDs] = G_vis.detach()
                G_true[hIDs] = G_hid.detach()
                val_loss = loss_fun(G_true, G_pred)

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
