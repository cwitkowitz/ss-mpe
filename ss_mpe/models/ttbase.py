# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.models import Encoder, Decoder

from . import SS_MPE

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

        self.encoder = Encoder(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity)
        self.decoder = Decoder(feature_size=n_bins, latent_size=latent_size, model_complexity=model_complexity)

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
            nn.ELU(inplace=True)
        )

        if skip_connections:
            # Start by adding encoder features with identity weighting
            self.skip_weights = torch.nn.Parameter(torch.ones(5))
        else:
            # No skip connections
            self.skip_weights = None

        self.layer_norm = nn.LayerNorm(normalized_shape=[latent_size])

    def encoder_parameters(self):
        """
        Obtain parameters for encoder part of network.

        Returns
        ----------
        parameters : generator
          Layer-wise iterator over parameters
        """

        # Obtain parameters corresponding to encoder
        parameters = list(super().encoder_parameters())
        # Append layer normalization parameters
        parameters += list(self.layer_norm.parameters())

        # Return generator type
        for p in parameters:
            yield p

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

        # Normalize features of latent codes w.r.t. one another
        latents = self.layer_norm(latents.transpose(-1, -2)).transpose(-1, -2)

        # Process latents with the decoder
        output = self.decoder(latents, embeddings)

        # Collapse channel dimension
        output = output.squeeze(-3)

        return output, latents, losses
