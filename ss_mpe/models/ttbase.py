# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.models import Encoder, Decoder

from . import HCQT

# Regular imports
import torch.nn as nn
import torch


class TT_Base(nn.Module):
    """
    Implements a 2D convolutional U-Net architecture based loosely on SoundStream.
    """

    def __init__(self, sample_rate, n_octaves, bins_per_octave, secs_per_block=3, latent_size=None, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        sample_rate : int
          Expected sample rate of input
        n_octaves : int
          Number of octaves below Nyquist frequency to represent
        bins_per_octave : int
          Number of frequency bins within each octave
        secs_per_block : float
          Number of seconds to process at once with sliCQ
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        nn.Module.__init__(self)

        self.sliCQ = CQT(n_octaves=n_octaves,
                         bins_per_octave=bins_per_octave,
                         sample_rate=sample_rate,
                         secs_per_block=secs_per_block)

        self.encoder = Encoder(feature_size=self.sliCQ.n_bins, latent_size=latent_size, model_complexity=model_complexity)
        self.decoder = Decoder(feature_size=self.sliCQ.n_bins, latent_size=latent_size, model_complexity=model_complexity)

        if skip_connections:
            # Start by adding encoder features with identity weighting
            self.skip_weights = torch.nn.Parameter(torch.ones(5))
        else:
            # No skip connections
            self.skip_weights = None

    def encode(self, audio):
        """
        Encode a batch of raw audio into latent codes.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Batch of input raw audio

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x H x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Compute CQT spectral features
        coefficients = self.sliCQ(audio)

        # Encode features into latent vectors
        latents, embeddings, losses = self.encoder(coefficients)

        return latents, embeddings, losses

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

    def decode(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x H x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for performing transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Create binary values to indicate function decoder should perform
        indicator = (not transcribe) * torch.ones_like(latents[..., :1, :])

        # Concatenate indicator to final dimension of latents
        latents = torch.cat((latents, indicator), dim=-2)

        # Decode latent vectors into real/imaginary coefficients
        coefficients = self.decoder(latents, embeddings)

        return coefficients

    def forward(self, audio, consistency=False):
        """
        Perform all model functions efficiently (for training/evaluation).

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Batch of input raw audio
        consistency : bool
          Whether to perform computations for consistency loss

        Returns
        ----------
        reconstruction : Tensor (B x 2 x F X T)
          Batch of reconstructed spectral coefficients
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        transcription : Tensor (B x 2 x F X T)
          Batch of transcription spectral coefficients
        transcription_rec : Tensor (B x 2 x F X T)
          Batch of reconstructed spectral coefficients for transcription coefficients input
        transcription_scr : Tensor (B x 2 x F X T)
          Batch of transcription spectral coefficients for transcription coefficients input
        losses : dict containing
          ...
        """

        # Encode raw audio into latent vectors
        latents, embeddings, losses = self.encode(audio)

        # Apply skip connections if they are turned on
        embeddings = self.apply_skip_connections(embeddings)

        # Decode latent vectors into spectral coefficients
        reconstruction = self.decode(latents, embeddings)

        # Estimate pitch using transcription switch
        transcription = self.decode(latents, embeddings, True)

        if consistency:
            # Encode transcription coefficients for samples with ground-truth
            latents_trn, embeddings_trn, _ = self.encoder(transcription)

            # Apply skip connections if they are turned on
            embeddings_trn = self.apply_skip_connections(embeddings_trn)

            # Attempt to reconstruct transcription spectral coefficients
            transcription_rec = self.decode(latents_trn, embeddings_trn)

            # Attempt to transcribe audio pertaining to transcription coefficients
            transcription_scr = self.decode(latents_trn, embeddings_trn, True)
        else:
            # Return null for both sets of coefficients
            transcription_rec, transcription_scr = None, None

        return reconstruction, latents, transcription, transcription_rec, transcription_scr, losses
