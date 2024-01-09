# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.models import Encoder, EncoderBlock, Decoder, DecoderBlock

from . import SS_MPE, ProjectionHead

# Regular imports
import torch.nn as nn
import torch


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
        convout_in_channels = self.decoder.convout.in_channels

        self.encoder.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, convin_out_channels, kernel_size=3, padding='same'),
            nn.ELU(inplace=True)
        )

        self.decoder.convout = nn.Conv2d(convout_in_channels, 1, kernel_size=3, padding='same')

        latent_channels = self.decoder.convin[0].out_channels
        latent_kernel = self.decoder.convin[0].kernel_size

        self.decoder.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size, latent_channels, kernel_size=latent_kernel),
            *self.decoder.convin[1:]
        )

        #self.projection = ProjectionHead(latent_size, latent_size // 4)
        self.projection = ProjectionHead(n_bins, n_bins // 4)

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
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        losses : dict containing
          ...
        """

        # Process features with the encoder
        latents, embeddings, losses = self.encoder(features)

        # Apply skip connections if applicable
        embeddings = self.apply_skip_connections(embeddings)

        # Process latents with the decoder
        output = self.decoder(latents, embeddings)

        # Collapse channel dimension
        output = output.squeeze(-3)

        return output, latents, losses


class EncoderNorm(Encoder):
    """
    Implements the 2D convolutional encoder from Timbre-Trap with layer normalization.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (2 * 2 ** (model_complexity - 1),
                    4 * 2 ** (model_complexity - 1),
                    8 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    32 * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)

        embedding_sizes = [feature_size]

        for i in range(4):
            # Dimensionality after strided convolutions
            embedding_sizes.append(embedding_sizes[-1] // 2 - 1)

        self.convin = nn.Sequential(
            nn.Conv2d(2, channels[0], kernel_size=3, padding='same'),
            nn.ELU(inplace=True),
            LayerNormPermute(normalized_shape=[channels[0], embedding_sizes[0]])
        )

        self.block1 = nn.Sequential(
            EncoderBlock(channels[0], channels[1], stride=2),
            LayerNormPermute(normalized_shape=[channels[1], embedding_sizes[1]])
        )
        self.block2 = nn.Sequential(
            EncoderBlock(channels[1], channels[2], stride=2),
            LayerNormPermute(normalized_shape=[channels[2], embedding_sizes[2]])
        )
        self.block3 = nn.Sequential(
            EncoderBlock(channels[2], channels[3], stride=2),
            LayerNormPermute(normalized_shape=[channels[3], embedding_sizes[3]])
        )
        self.block4 = nn.Sequential(
            EncoderBlock(channels[3], channels[4], stride=2),
            LayerNormPermute(normalized_shape=[channels[4], embedding_sizes[4]])
        )

        self.convlat = nn.Sequential(
            nn.Conv2d(channels[4], latent_size, kernel_size=(embedding_sizes[-1], 1)),
            LayerNormPermute(normalized_shape=[latent_size, 1])
        )


class DecoderNorm(Decoder):
    """
    Implements the 2D convolutional decoder from Timbre-Trap with layer normalization.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the decoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (32 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1),
                    4  * 2 ** (model_complexity - 1),
                    2  * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)

        padding = list()

        embedding_sizes = [feature_size]

        for i in range(4):
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

        self.block1 = nn.Sequential(
            DecoderBlock(channels[0], channels[1], stride=2, padding=padding[0]),
            LayerNormPermute(normalized_shape=[channels[1], embedding_sizes[1]])
        )
        self.block2 = nn.Sequential(
            DecoderBlock(channels[1], channels[2], stride=2, padding=padding[1]),
            LayerNormPermute(normalized_shape=[channels[2], embedding_sizes[2]])
        )
        self.block3 = nn.Sequential(
            DecoderBlock(channels[2], channels[3], stride=2, padding=padding[2]),
            LayerNormPermute(normalized_shape=[channels[3], embedding_sizes[3]])
        )
        self.block4 = nn.Sequential(
            DecoderBlock(channels[3], channels[4], stride=2, padding=padding[3]),
            LayerNormPermute(normalized_shape=[channels[4], embedding_sizes[4]])
        )

        # TODO - add layer normalization after final convolution?
        self.convout = nn.Conv2d(channels[4], 2, kernel_size=3, padding='same')


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
