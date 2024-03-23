

import torch

class ConvUnit(torch.nn.Module):
    """A standard double-convolution unit"""

    def __init__(self, in_channels, out_channels, conv_kernel, activation, dimension=2):
        """Args:
          - in_channels: Number of input channels
          - out_channels: Number of output channels
          - conv_kernel: Convolutional kernel size
          - activation: Activation layer class
          - dimension: Input dimension (1, 2, or 3, default 3)
        """

        super().__init__()

        # Set convolution & batch norm classes for 1, 2, or 3D case
        if dimension==3:
            Conv = torch.nn.Conv3d
            BatchNorm = torch.nn.BatchNorm3d
        elif dimension==2:
            Conv = torch.nn.Conv2d
            BatchNorm = torch.nn.BatchNorm2d
        elif dimension==1:
            Conv = torch.nn.Conv1d
            BatchNorm = torch.nn.BatchNorm1d
        else:
            raise ValueError("\'dimension\' must be 1, 2, or 3")
        
        # Construct convolutional unit
        self.conv = torch.nn.Sequential(
            Conv(in_channels, out_channels, conv_kernel, padding="same", dtype=float),
            activation(),
            BatchNorm(out_channels, dtype=float),
            Conv(out_channels, out_channels, conv_kernel, padding="same", dtype=float),
            activation(),
            BatchNorm(out_channels, dtype=float),
        )            

    def forward(self, x):
        return self.conv(x)
    
class ContractionLayer(torch.nn.Module):
    """A single CNN layer that contracts inputs"""

    def __init__(self, in_channels, out_channels, conv_kernel, activation, pool_kernel, dimension=2):
        """Args:
          - in_channels: Number of input channels
          - out_channels: Number of output channels
          - conv_kernel: Convolutional kernel size
          - activation: Activation layer class
          - pool_kernel: Maxpooling kernel dimensions
        """

        super().__init__()

        # Set max-pool class for 1, 2, or 3D case
        if dimension==3:
            MaxPool = torch.nn.MaxPool3d
        elif dimension==2:
            MaxPool = torch.nn.MaxPool2d
        elif dimension==1:
            MaxPool = torch.nn.MaxPool1d
        else:
            raise ValueError("\'dimension\' must be 1, 2, or 3")

        # Construct max-pool & convolutional unit
        self.pool = MaxPool(pool_kernel)
        self.conv = ConvUnit(in_channels, out_channels, conv_kernel, activation, dimension)

    def forward(self, x):
        return self.conv(self.pool(x))

class ExpansionLayer(torch.nn.Module):
    """A single CNN layer that expands inputs"""

    def __init__(self, in_channels, out_channels, conv_kernel, activation, upsample_kernel, dimension=2):
        """Args:
          - in_channels: Number of input channels
          - out_channels: Number of output channels
          - conv_kernel: Convolutional kernel size
          - activation: Activation layer class
          - upsample_kernel: Maxpooling kernel dimensions
          - dimension: 
        """
        super().__init__()

        # Set transpose convolution class for 1, 2, or 3D case
        if dimension==3:
            ConvTranspose = torch.nn.ConvTranspose3d
        elif dimension==2:
            ConvTranspose = torch.nn.ConvTranspose2d
        elif dimension==1:
            ConvTranspose = torch.nn.ConvTranspose1d
        else:
            raise ValueError("\'dimension\' must be 1, 2, or 3")

        self.upsample = ConvTranspose(in_channels, out_channels, upsample_kernel, stride=upsample_kernel, dtype=float)
        self.conv = ConvUnit(in_channels, out_channels, conv_kernel, activation, dimension)

    def forward(self, x):
        return self.conv(self.upsample(x))

class CNN(torch.nn.Module):
    """A convolutional neural network"""

    def __init__(
        self, in_channels, n_out, n_layers, n_filters, conv_kernel=3, sample_kernel=2,
        conv_activation='LeakyReLU', output_activation="Sigmoid", dropout=.1, dimension=2, action="contract"
    ):
        """Args:
          - in_channels: Number of channels in the input
          - n_out: Number of output variables
          - n_layers: Number of layers in the network
          - n_filters: Base dimension of convolutional filters (doubled at each layer)
          - conv_kernel: Kernel dimensions for regular convolutional layers (default 3)
          - sample_kernel: Kernel dimensions for max-pooling or up-sampling layers (default 2)
          - dropout: Dropout rate (list/tuple or float) between consecutive layers (default .1)
          - conv_activation: Class of activation layer to apply after convolutions (default 'ReLU')
          - output_activation: Class of activation layer to apply after final 2D convolution (default 'Sigmoid')
          - dimensions: Input dimension (1, 2, or 3, default 3)
        """        

        super().__init__()

        #
        conv_activation = getattr(torch.nn, conv_activation)
        output_activation = getattr(torch.nn, output_activation)     

        # Set filter sizing lambda function for expansion or contraction
        action = action.lower()
        if action == "contract":
            filter_size = lambda l: [n_filters*2**l, n_filters*2**(l+1)]
            ConvLayer = ContractionLayer
        elif action == "expand":
            filter_size = lambda l: [n_filters*2**(l+1), n_filters*2**l]
            ConvLayer = ExpansionLayer
        else:
            raise ValueError("'action' must be either 'contract' or 'expand'")
        
        # Define layer dropouts as a list
        if hasattr(dropout, '__len__'):
            dropout = [d for d in dropout]
        else:
            dropout = [dropout,]*n_layers

        # Add the first convolutional unit
        self.input_layer = ConvUnit(in_channels, n_filters, conv_kernel, conv_activation, dimension)

        # Add convolutional layers (with optional dropout layers)
        for l in range(n_layers):
            filters = filter_size(l)
            setattr(
                self, "covolution_{:d}".format(l+1),
                ConvLayer(filters[0], filters[1], conv_kernel, conv_activation, sample_kernel, dimension)
            )
            if dropout[l]:
                setattr(self, "dropout_{:d}".format(l+1), torch.nn.Dropout(p=dropout[l]))

        # Add final convolutional layer that produces output
        self.output_layer = torch.nn.Sequential(
            ConvLayer(filters[1], n_out, (1,1), output_activation, sample_kernel, dimension),
            output_activation()
        )

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
          - x: Input on which computations are performed
        """

        # Compute & store hidden value at the end of the input layer
        h = self.input_layer(x)

        # Compute & store hidden values after each contraction layer
        for l in range(self.hypers["n_layers"]):
            h = getattr(self, "convolution_{:d}".format(l+1))(h)
        
        # Compute & return final output
        return self.output_layer(h)
