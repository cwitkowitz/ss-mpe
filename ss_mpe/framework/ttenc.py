# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.utils import *

from . import SS_MPE, EncoderNorm

# Regular imports
import torch.nn as nn


class TT_Enc(SS_MPE):
    """
    Encoder-only variant of modified Timbre-Trap (see TT_Base...).
    """

    def __init__(self, hcqt_params, n_blocks=4, model_complexity=1):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See SS_MPE class for others...

        latent_size : int or None (Optional)
          Dimensionality of latent space
        n_blocks : int
          Number of blocks for encoder
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        """

        super().__init__(hcqt_params)

        # Extract HCQT parameters to infer dimensionality of input features
        n_bins, n_harmonics = hcqt_params['n_bins'], len(hcqt_params['harmonics'])

        self.encoder = EncoderNorm(feature_size=n_bins, latent_size=n_bins, n_blocks=n_blocks, model_complexity=model_complexity)

        convin_out_channels = self.encoder.convin[0].out_channels

        self.encoder.convin = nn.Sequential(
            nn.Conv2d(n_harmonics, convin_out_channels, kernel_size=3, padding='same'),
            *self.encoder.convin[1:]
        )

        # Remove final layer normalization
        self.encoder.convlat = self.encoder.convlat[:-1]

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
