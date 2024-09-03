# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.framework import *
from timbre_trap.utils import *

from . import SS_MPE

# Regular imports
import torch.nn as nn
import torch


all = ['TT_Base',
       'EncoderNorm',
       'DecoderNorm',
       'LayerNormPermute']


class TT_Base(SS_MPE):
    """
    Implements base model from Timbre-Trap (https://arxiv.org/abs/2309.15717).
    """

    def __init__(self, hcqt_params, latent_size=None, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See SS_MPE class for others...

        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        super().__init__(hcqt_params)

        # Extract HCQT parameters to infer dimensionality of input features
        n_bins, n_harmonics = hcqt_params['n_bins'], len(hcqt_params['harmonics'])

        if latent_size is None:
            # Set default dimensionality of latents
            latent_size = 32 * 2 ** (model_complexity - 1)

        self.encoder = EncoderNorm(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity)
        self.decoder = DecoderNorm(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity)

        convin_out_channels = self.encoder.convin[0].out_channels
        convout_in_channels = self.decoder.convout[0].in_channels

        self.encoder.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, convin_out_channels, kernel_size=3, padding='same'),
            *self.encoder.convin[1:]
        )

        self.decoder.convout = nn.Sequential(
            nn.Conv2d(convout_in_channels, 1, kernel_size=3, padding='same'),
        )

        latent_channels = self.decoder.convin[0].out_channels
        latent_kernel = self.decoder.convin[0].kernel_size

        self.decoder.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size, latent_channels, kernel_size=latent_kernel),
            *self.decoder.convin[1:]
        )

        if skip_connections:
            # Start by adding encoder features with identity weighting
            self.skip_weights = torch.nn.Parameter(torch.ones(5))
        else:
            # No skip connections
            self.skip_weights = None

    def decoder_parameters(self):
        """
        Obtain parameters for decoder part of network.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Obtain parameters corresponding to decoder
        parameters = list(super().decoder_parameters())

        if self.skip_weights is not None:
            # Append skip connection parameters
            parameters.append(self.skip_weights)

        # Return generator type
        for p in parameters:
            yield p

    def apply_skip_connections(self, embeddings):
        """
        Apply skip connections to encoder embeddings, or discard the embeddings if skip connections do not exist.

        Parameters
        ----------
        embeddings : list of [Tensor (B x C x H x T)]
          Embeddings produced by encoder at each level

        Returns
        ----------
        embeddings : list of [Tensor (B x C x H x T)]
          Encoder embeddings scaled with learnable weight
        """

        if self.skip_weights is not None:
            # Apply a learnable weight to the embeddings for the skip connection
            embeddings = [self.skip_weights[i] * e for i, e in enumerate(embeddings)]
        else:
            # Discard embeddings from encoder
            embeddings = None

        return embeddings

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
        latents, embeddings, _ = self.encoder(features)

        debug_nans(latents, 'encoder output')

        # Apply skip connections if applicable
        embeddings = self.apply_skip_connections(embeddings)

        # Process latents with the decoder
        output = self.decoder(latents, embeddings)

        debug_nans(latents, 'decoder output')

        # Collapse channel dimension
        output = output.squeeze(-3)

        return output


class EncoderNorm(nn.Module):
    """
    Implements the 2D convolutional encoder from Timbre-Trap with layer normalization.
    """

    def __init__(self, feature_size, latent_size=None, n_blocks=4, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        n_blocks : int
          Number of blocks for encoder

        See Encoder class for others...
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
            nn.Conv2d(2, channels[0], kernel_size=3, padding='same'),
            nn.ELU(inplace=True),
            LayerNormPermute(normalized_shape=[channels[0], embedding_sizes[0]])
        )

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(
                EncoderBlock(channels[i], channels[i + 1], stride=2),
                LayerNormPermute(normalized_shape=[channels[i + 1], embedding_sizes[i + 1]])
            ))

        self.convlat = nn.Sequential(
            nn.Conv2d(channels[-1], latent_size, kernel_size=(embedding_sizes[-1], 1)),
            LayerNormPermute(normalized_shape=[latent_size, 1])
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


class DecoderNorm(nn.Module):
    """
    Implements the 2D convolutional decoder from Timbre-Trap with layer normalization.
    """

    def __init__(self, feature_size, latent_size=None, n_blocks=4, model_complexity=1):
        """
        Initialize the decoder.

        Parameters
        ----------
        n_blocks : int
          Number of blocks for encoder

        See Decoder class for others...
        """

        nn.Module.__init__(self)

        channels = [2 ** (n_blocks + 1 - i) * 2 ** (model_complexity - 1) for i in range(n_blocks + 1)]

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = channels[-1]

        padding = list()

        embedding_sizes = [feature_size]

        for i in range(n_blocks):
            # Padding required for expected output size
            padding.append(embedding_sizes[-1] % 2)
            # Dimensionality after strided convolutions
            embedding_sizes.append(embedding_sizes[-1] // 2 - 1)

        # Reverse order
        padding.reverse()
        embedding_sizes.reverse()

        self.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size + 1, channels[0], kernel_size=(embedding_sizes[0], 1)),
            nn.ELU(inplace=True),
            LayerNormPermute(normalized_shape=[channels[0], embedding_sizes[0]])
        )

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(
                DecoderBlock(channels[i], channels[i + 1], stride=2, padding=padding[i]),
                LayerNormPermute(normalized_shape=[channels[i + 1], embedding_sizes[i + 1]])
            ))

        self.convout = nn.Sequential(
            nn.Conv2d(channels[-1], 2, kernel_size=3, padding='same'),
        )

    def forward(self, latents, encoder_embeddings=None):
        """
        Decode a batch of input latent codes.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        encoder_embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level

        Returns
        ----------
        output : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Restore feature dimension
        latents = latents.unsqueeze(-2)

        # Process latents with decoder blocks
        embeddings = self.convin(latents)

        if encoder_embeddings is not None:
            # Add encoder embeddings through skip connection
            embeddings = embeddings + encoder_embeddings[-1]

        for i, block in enumerate(self.blocks):
            # Feed embeddings through next decoder block
            embeddings = block(embeddings)

            if encoder_embeddings is not None:
                # Add encoder embeddings through skip connection
                embeddings = embeddings + encoder_embeddings[-2 - i]

        # Decode embeddings into spectral logits
        output = self.convout(embeddings)

        return output


class LayerNormPermute(nn.LayerNorm):
    """
    Layer normalization with required axis permutation.
    """

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

        # Bring channel and feature axis to back
        x = x.permute(0, -1, -3, -2)
        # Perform layer normalization
        y = super().forward(x)
        # Restore original dimensionality
        y = y.permute(0, -2, -1, -3)

        return y
