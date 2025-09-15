# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.utils import *

from . import SS_MPE, LayerNormPermute

# Regular imports
import torch.nn as nn
import torch

__all__ = [
    'PerfectPitch',
    'Encoder',
    'EncoderBlock',
    'ResidualConv2dBlock',
]


class PerfectPitch(SS_MPE):
    """
    Encoder-only variant of modified Timbre-Trap.
    """

    def __init__(self, hcqt_params, n_blocks=4, model_complexity=1):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See SS_MPE class for others...

        n_blocks : int
          Number of blocks for encoder
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        """

        super().__init__(hcqt_params)

        # Extract HCQT parameters to infer dimensionality of input features
        n_bins, n_harmonics = hcqt_params['n_bins'], len(hcqt_params['harmonics'])

        #self.encoder = Encoder(feature_size=n_bins, n_harmonics=n_harmonics, n_blocks=n_blocks, latent_size=n_bins, model_complexity=model_complexity)
        self.encoder = Encoder2(n_bins=n_bins, n_harmonics=n_harmonics, n_blocks=n_blocks, model_complexity=model_complexity)
        #self.encoder = Encoder3(n_bins=n_bins, n_harmonics=n_harmonics, n_blocks=n_blocks, model_complexity=model_complexity)
        #self.encoder = Encoder4(n_bins=n_bins, n_harmonics=n_harmonics, n_blocks=n_blocks, model_complexity=model_complexity)

    def forward(self, features):
        """
        Process spectral features to obtain pitch salience logits (for training/evaluation).

        Parameters
        ----------
        features : Tensor (B x H x F x T)
          Batch of HCQT spectral features

        Returns
        ----------
        output : Tensor (B x F X T)
          Batch of (implicit) pitch salience logits
        """

        # Process features with the encoder
        output, _, _ = self.encoder(features)

        debug_nans(output, 'encoder output')

        return output

    def decoder_parameters(self):
        """
        Return an empty generator to indicate no decoder.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Return empty generator
        for p in list():
            yield p


class Encoder(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, feature_size, n_harmonics, n_blocks=4, latent_size=None, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        n_harmonics : int
          Number of input harmonic channels
        n_blocks : int
          Number of blocks for encoder
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = [2 ** (i + 1) * 2 ** (model_complexity - 1) for i in range(n_blocks + 1)]

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = channels[-1]

        embedding_sizes = [feature_size]

        for i in range(n_blocks):
            # Dimensionality after strided convolutions
            embedding_sizes.append(embedding_sizes[-1] // 2 - 1)

        self.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, channels[0], kernel_size=3, padding='same'),
            LayerNormPermute(normalized_shape=[channels[0], feature_size]),
            #nn.BatchNorm2d(channels[0]),
            nn.ELU(inplace=True)
        )

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(
                EncoderBlock(channels[i], channels[i + 1], embedding_sizes[i + 1], stride=2)
            ))

        self.convlat = nn.Sequential(
            nn.Conv2d(channels[-1], latent_size, kernel_size=(embedding_sizes[-1], 1))
        )

    def forward(self, coefficients):
        """
        Encode a batch of input spectral coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of spectral coefficients

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Initialize a list to hold features for skip connections
        embeddings = list()

        # Encode features into embeddings
        embeddings.append(self.convin(coefficients))

        for block in self.blocks:
            # Feed embeddings through next encoder block
            embeddings.append(block(embeddings[-1]))

        # Compute latent vectors from embeddings
        latents = self.convlat(embeddings[-1]).squeeze(-2)

        # No encoder losses
        loss = dict()

        return latents, embeddings, loss


class EncoderBlock(nn.Module):
    """
    Implements a chain of residual convolutional blocks with progressively
    increased dilation, followed by down-sampling via strided convolution.
    """

    def __init__(self, in_channels, out_channels, embedding_size, stride=2):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        embedding_size : int
          Number of features along the frequency dimension
        stride : int
          Stride for the final convolutional layer
        """

        nn.Module.__init__(self)

        self.block1 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=3)

        self.hop = stride
        self.win = 2 * stride

        self.sconv = nn.Sequential(
            # Down-sample along frequency (height) dimension via strided convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1)),
            #nn.BatchNorm2d(out_channels),
            LayerNormPermute(normalized_shape=[out_channels, embedding_size]),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        """
        Feed features through the encoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)

        # Down-sample
        y = self.sconv(y)

        return y


class ResidualConv2dBlock(nn.Module):
    """
    Implements a 2D convolutional block with dilation, no down-sampling, and a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        """
        Initialize the convolutional block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        kernel_size : int
          Kernel size for convolutions
        dilation : int
          Amount of dilation for first convolution
        """

        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=dilation),
            #nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            #nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        """
        Feed features through the convolutional block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.conv1(x)
        y = self.conv2(y)

        # Residual connection
        y = y + x

        return y


class Encoder2(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, n_bins, n_harmonics, n_blocks=4, model_complexity=1):
        """
        """

        nn.Module.__init__(self)

        channels = [2 ** (i + 1) * 2 ** (model_complexity - 1) for i in reversed(range(n_blocks + 1))]

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        self.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, channels[0], kernel_size=5, padding='same'),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            kernel_size = 5 if i < n_blocks / 4 else 3

            self.blocks.append(nn.Sequential(
                EncoderBlock2(channels[i], channels[i + 1], kernel_size)
            ))

        self.convout = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1, padding='same')
        )

        # TODO - final 3x3 convolution with dropout?

    def forward(self, coefficients):
        """
        """

        # Initialize a list to hold features for skip connections
        embeddings = list()

        # Encode features into embeddings
        embeddings.append(self.convin(coefficients))

        for block in self.blocks:
            # Feed embeddings through next encoder block
            embeddings.append(block(embeddings[-1]))

        # Obtain final output logits
        logits = self.convout(embeddings[-1]).squeeze(-3)

        # No encoder losses
        loss = dict()

        return logits, embeddings, loss


class EncoderBlock2(nn.Module):
    """
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        """
        """

        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),#, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        """

        # Process features
        y = self.conv1(x)

        return y


class Encoder4(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, n_bins, n_harmonics, n_blocks=4, model_complexity=1):
        """
        """

        nn.Module.__init__(self)

        channels = [2 ** (i + 1) * 2 ** (model_complexity - 1) for i in reversed(range(n_blocks + 1))]

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        self.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, channels[0], kernel_size=5, padding='same'),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            kernel_size = 5 if i < n_blocks / 4 else 3

            self.blocks.append(nn.Sequential(
                EncoderBlock2(channels[i], channels[i + 1], kernel_size)
            ))

        self.convout = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1, padding='same')
        )

        self.fc1 = nn.Linear(n_bins, n_bins, bias=False)

        # TODO - final 3x3 convolution with dropout?

    def forward(self, coefficients):
        """
        """

        # Initialize a list to hold features for skip connections
        embeddings = list()

        # Encode features into embeddings
        embeddings.append(self.convin(coefficients))

        for block in self.blocks:
            # Feed embeddings through next encoder block
            embeddings.append(block(embeddings[-1]))

        # Obtain final output logits
        logits = self.convout(embeddings[-1]).squeeze(-3)

        logits = self.fc1(logits.transpose(-1, -2)).transpose(-1, -2)

        # No encoder losses
        loss = dict()

        return logits, embeddings, loss


class Encoder3(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, n_bins, n_harmonics, n_blocks=4, model_complexity=1):
        """
        """

        nn.Module.__init__(self)

        channels = [2 ** (i + 1) * 2 ** (model_complexity - 1) for i in reversed(range(n_blocks + 1))]

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        self.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, channels[0], kernel_size=5, padding='same'),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            kernel_size = 5 if i < n_blocks / 4 else 3

            self.blocks.append(nn.Sequential(
                EncoderBlock3(n_bins, channels[i], channels[i + 1], kernel_size)
            ))

        self.convout = nn.Sequential(
            nn.Conv2d(channels[-1], 1, kernel_size=1, padding='same')
        )

        # TODO - final 3x3 convolution with dropout?

    def forward(self, coefficients):
        """
        """

        # Initialize a list to hold features for skip connections
        embeddings = list()

        # Encode features into embeddings
        embeddings.append(self.convin(coefficients))

        for block in self.blocks:
            # Feed embeddings through next encoder block
            embeddings.append(block(embeddings[-1]))

        # Obtain final output logits
        logits = self.convout(embeddings[-1]).squeeze(-3)

        # No encoder losses
        loss = dict()

        return logits, embeddings, loss


class EncoderBlock3(nn.Module):
    """
    """

    def __init__(self, n_bins, in_channels, out_channels, kernel_size, n_heads=4, dropout=0.1):
        """
        """

        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),#, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.positional = nn.Parameter(torch.randn(n_bins, out_channels))
        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(out_channels)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        """
        """

        B, _, F, T = x.size()

        #x = x + self.dp(self.conv1(x))
        x = self.conv1(x)

        # Flatten for self-attention
        x_flat = x.permute(0, 3, 2, 1).reshape(B * T, F, -1)

        x_flat = x_flat + self.positional

        # Self-attention with LayerNorm
        #x_norm = self.layer_norm_attn(x_flat)  # Normalize before attention
        #attn_out, _ = self.attention(x_norm, x_norm, x_norm)  # Self-attention
        x_flat, _ = self.attn(x_flat, x_flat, x_flat)  # Self-attention
        #attn_out = self.layer_norm_attn(attn_out + x_flat)  # Residual connection with LayerNorm

        # Reshape back to original shape
        #attn_out = attn_out.reshape(batch_size, time_steps, freq_bins, channels).permute(0, 3, 1, 2)
        x_flat = x_flat.reshape(B, T, F, -1).permute(0, 3, 2, 1)

        # Final residual connection
        #y = x + self.dropout(attn_out)

        return x_flat
