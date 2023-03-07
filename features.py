# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>


from nnAudio.features import CQT2010v2 as CQT
import numpy as np
import torch


class HCQT(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, sample_rate=22050, hop_length=512, fmin=32.70, harmonics=None,
                 n_bins=84, bins_per_octave=12,
                 filter_scale=None, norm=None, basis_norm=None, window=None, pad_mode=None, earlydownsample=None): # TODO - see if there is a need for these
        """
        TODO
        """

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Default the harmonics to convention
        if harmonics is None:
            harmonics = [0.5, 1, 2, 3, 4, 5]
        self.harmonics = sorted(harmonics)

        self.modules = []
        # Construct a list of CQT modules for the harmonic transform
        for h in self.harmonics:
            # Center frequency for the harmonic
            fmin_h = h * fmin
            # Add a module for this harmonic's CQT
            self.modules += [CQT(sr=sample_rate,
                                 hop_length=hop_length,
                                 fmin=fmin_h,
                                 n_bins=n_bins,
                                 bins_per_octave=bins_per_octave,
                                 verbose=False)]

    def get_expected_frames(self, audio):
        """
        TODO

        TODO - will this actually be needed anywhere?
        """

        num_frames = None

        return num_frames

    def get_sample_range(self, num_frames):
        """
        TODO

        TODO - will this actually be needed anywhere?
        """

        sample_range = None

        return sample_range

    def get_num_samples_required(self):
        """
        Determine the number of samples required to extract one full frame of features.

        TODO - will this actually be needed anywhere?

        Returns
        ----------
        num_samples_required : int
            Number of samples
        """

        # Maximum number of samples which still produces one frame
        num_samples_required = self.get_sample_range(1)[-1]

        return num_samples_required

    @staticmethod
    def divisor_pad(audio, divisor):
        """
        Pad audio such that it is divisible by the specified divisor.

        TODO - will this actually be needed anywhere?

        Parameters
        ----------
        audio : ndarray
            Mono-channel audio
        divisor : int
            Number by which the amount of audio samples should be divisible

        Returns
        ----------
        audio : ndarray
            Padded audio
        """

        # Determine how many samples would be needed such that the audio is evenly divisible
        pad_amt = divisor - (audio.shape[-1] % divisor)

        if pad_amt > 0 and pad_amt != divisor:
            # Pad the audio for divisibility
            audio = np.append(audio, np.zeros(pad_amt).astype(tools.FLOAT32), axis=-1)

        return audio

    def frame_pad(self, audio):
        """
        Pad the audio to fill out the final frame.

        TODO - will this actually be needed anywhere?

        Parameters
        ----------
        audio : ndarray
            Mono-channel audio

        Returns
        ----------
        audio : ndarray
            Padded audio
        """

        # We need at least this many samples
        divisor = self.get_num_samples_required()

        if audio.shape[-1] > divisor:
            # If above is satisfied, just pad for one extra hop
            divisor = self.hop_length

        # Pad the audio so it is divisible by the divisor
        audio = self.divisor_pad(audio, divisor)

        return audio

    def forward(self, audio):
        """
        TODO
        """

        feats = None

        # TODO - to decibels, 0/1 scaling

        return feats

    def get_times(self, audio, at_start=False):
        """
        TODO
        """

        times = None

        return times

    def get_sample_rate(self):
        """
        Helper function to access sampling rate.

        TODO - will this actually be needed anywhere?

        Returns
        ----------
        sample_rate : int or float
            Presumed sampling rate for incoming audio
        """

        sample_rate = self.sample_rate

        return sample_rate

    def get_hop_length(self):
        """
        Helper function to access hop length.

        TODO - will this actually be needed anywhere?

        Returns
        ----------
        hop_length : int or float
            Number of samples between frames
        """

        hop_length = self.hop_length

        return hop_length

    def get_num_channels(self):
        """
        Helper function to access number of channels.

        TODO - will this actually be needed anywhere?

        Returns
        ----------
        num_channels : int
            Number of independent channels
        """

        num_channels = len(self.modules)

        return num_channels

    def get_feature_size(self):
        """
        TODO

        TODO - will this actually be needed anywhere?
        """

        feature_size = None

        return feature_size