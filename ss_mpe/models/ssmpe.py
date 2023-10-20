# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import HCQT

# Regular imports
import torch.nn as nn
import torch


class SS_MPE(nn.Module):
    """
    Basic functionality for self-supervised (SS) multi-pitch estimation (MPE).
    """

    def __init__(self, hcqt_params):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        hcqt_params : dict
          Parameters for feature extraction module
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        nn.Module.__init__(self)

        self.hcqt_params = hcqt_params.copy()

        hcqt_params.pop('weights')
        # TODO - don't save with HCQT
        self.hcqt = HCQT(**hcqt_params)

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
        ...
        """

        return NotImplementedError

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
