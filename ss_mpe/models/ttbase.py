# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.models import Encoder, Decoder

from . import HCQT

# Regular imports
import torch.nn as nn
import torch


class TT_Base(nn.Module):
    """
    Implements base model from Timbre-Trap (https://arxiv.org/abs/2309.15717).
    """

    def __init__(self, hcqt_params, latent_size=None, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        hcqt_params : dict
          Parameters for feature extraction module
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        nn.Module.__init__(self)

        self.hcqt_params = hcqt_params.copy()

        hcqt_params.pop('weights')
        self.hcqt = HCQT(**hcqt_params)

        self.encoder = Encoder(feature_size=self.hcqt.n_bins, latent_size=latent_size, model_complexity=model_complexity)
        self.decoder = Decoder(feature_size=self.hcqt.n_bins, latent_size=latent_size, model_complexity=model_complexity)

        n_harmonics = len(hcqt_params['harmonics'])

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

    def get_all_features(self, audio):
        """
        Compute all possible features.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio

        Returns
        ----------
        features : dict
          Various sets of spectral features
        """

        # Compute features for audio
        features_amp = self.hcqt(audio)

        # Convert raw HCQT spectral features to decibels [-80, 0] dB
        features_dec = self.hcqt.to_decibels(features_amp, rescale=False)
        # Convert decibels to linear probability-like values [0, 1]
        features_lin = self.hcqt.decibels_to_amplitude(features_dec)
        # Scale decibels to represent probability-like values [0, 1]
        features_log = self.hcqt.rescale_decibels(features_dec)

        # Extract relevant parameters
        harmonics = self.hcqt_params['harmonics']
        harmonic_weights = self.hcqt_params['weights']

        # Determine first harmonic index
        h_idx = harmonics.index(1)

        # Obtain first harmonic spectral features
        features_lin_1 = features_lin[:, h_idx]
        features_log_1 = features_log[:, h_idx]

        # Compute a weighted sum of features to obtain a rough salience estimate
        features_lin_h = torch.sum(features_lin * harmonic_weights, dim=-3)
        features_log_h = torch.sum(features_log * harmonic_weights, dim=-3)

        features = {
            'amp'   : features_lin,
            'dec'   : features_log,
            'amp_1' : features_lin_1,
            'dec_1' : features_log_1,
            'amp_h' : features_lin_h,
            'dec_h' : features_log_h
        }

        return features

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
        Perform all model functions efficiently (for training/evaluation).

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

    def transcribe(self, audio):
        """
        Helper function to transcribe audio directly.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio

        Returns
        ----------
        salience : Tensor (B x F X T)
          Batch of pitch salience activations
        """

        # Compute HCQT features (dB) and re-scale to probability-like
        features = self.hcqt.to_decibels(self.hcqt(audio), rescale=True)

        # Process features and convert to activations
        salience = torch.sigmoid(self(features)[0])

        return salience
