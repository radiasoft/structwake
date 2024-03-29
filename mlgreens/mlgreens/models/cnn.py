

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
            Conv(in_channels, out_channels, conv_kernel, padding='same', dtype=torch.float),
            activation(),
            BatchNorm(out_channels, dtype=torch.float),
            Conv(out_channels, out_channels, conv_kernel, padding='same', dtype=torch.float),
            activation(),
            BatchNorm(out_channels, dtype=torch.float),
        )            

    def forward(self, x):
        return self.conv(x)
    
class ContractionLayer(torch.nn.Module):
    """A single CNN layer that contracts inputs"""

    def __init__(self, in_channels, out_channels, pool_kernel, conv_kernel, activation,  dimension=2):
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

    def __init__(self, in_channels, out_channels, upsample_kernel, conv_kernel, activation, dimension=2):
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

        self.conv = ConvUnit(in_channels, out_channels, conv_kernel, activation, dimension)
        self.upsample = ConvTranspose(out_channels, out_channels, upsample_kernel, stride=upsample_kernel, dtype=torch.float)

    def forward(self, x):
        return self.upsample(self.conv(x))
