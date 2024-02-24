# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt import LHVQT, torch_amplitude_to_db

# Regular imports
import numpy as np
import librosa


class HCQT(LHVQT):
    """
    Wrapper which adds some basic functionality to the LHVQT module.
    """

    def __init__(self, sample_rate, hop_length, fmin, bins_per_octave, n_bins, harmonics, **kwargs):
        """
        Instantiate the LHVQT module and wrapper.

        Parameters
        ----------
        sample_rate : int or float
          Number of samples per second of audio
        hop_length : int
          Number of samples between frames
        fmin : float
          First center frequency (MIDI) of geometric progression
        bins_per_octave : int
          Number of bins allocated to each octave
        n_bins : int
          Number of frequency bins per channel
        harmonics : list of float
          Harmonics to stack along channels
        kwargs : dict
          Additional LVQT parameters
        """

        super().__init__(fmin=librosa.midi_to_hz(fmin),
                         harmonics=harmonics,
                         fs=sample_rate,
                         hop_length=hop_length,
                         n_bins=n_bins,
                         bins_per_octave=bins_per_octave,
                         update=False,
                         to_db=False,
                         db_to_prob=False,
                         batch_norm=False,
                         **kwargs)

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.n_bins = n_bins

        # Minimum MIDI frequency of each harmonic
        fmins = fmin + 12 * np.log2([harmonics]).T

        # Determine center frequency (MIDI) associated with each bin of module
        self.midi_freqs = fmins + np.arange(self.n_bins) / (bins_per_octave / 12)

    @staticmethod
    def to_decibels(amplitude, rescale=True):
        """
        Convert a set of amplitude coefficients to decibels.

        Parameters
        ----------
        amplitude : Tensor (B x F X T)
          Batch of amplitude coefficients (amplitude)
        rescale : bool
          Rescale decibels to the range [0, 1]

        Returns
        ----------
        decibels : Tensor (B x F X T)
          Batch of power coefficients (dB)
        """

        # Initialize a differentiable conversion to decibels
        decibels = torch_amplitude_to_db(amplitude, to_prob=rescale)

        return decibels

    @staticmethod
    def decibels_to_amplitude(decibels, negative_infinity_dB=-80):
        """
        Convert a tensor of decibel values to amplitudes between 0 and 1.

        Parameters
        ----------
        decibels : ndarray or Tensor
          Tensor of decibel values with a ceiling of 0
        negative_infinity_dB : float
          Decibel cutoff beyond which is considered negative infinity

        Returns
        ----------
        gain : ndarray or Tensor
          Tensor of values linearly scaled between 0 and 1
        """

        # Make sure provided lower boundary is negative
        negative_infinity_dB = -abs(negative_infinity_dB)

        # Convert decibels to a gain between 0 and 1
        gain = 10 ** (decibels / 20)
        # Set gain of values below -∞ to 0
        gain[decibels <= negative_infinity_dB] = 0

        return gain

    @staticmethod
    def rescale_decibels(decibels, negative_infinity_dB=-80):
        """
        Log-scale a tensor of decibel values between 0 and 1.

        Parameters
        ----------
        decibels : ndarray or Tensor
          Tensor of decibel values with a ceiling of 0
        negative_infinity_dB : float
          Decibel cutoff beyond which is considered negative infinity

        Returns
        ----------
        scaled : ndarray or Tensor
          Decibel values scaled logarithmically between 0 and 1
        """

        # Make sure provided lower boundary is positive
        negative_infinity_dB = abs(negative_infinity_dB)

        # Scale decibels to be between 0 and 1
        scaled = 1 + (decibels / negative_infinity_dB)

        return scaled

    def get_expected_samples(self, t):
        """
        Determine the number of samples corresponding to a specified amount of time.

        Parameters
        ----------
        t : float
          Amount of time

        Returns
        ----------
        num_samples : int
          Number of audio samples expected
        """

        # Compute number of samples and round down
        num_samples = int(max(0, t) * self.sample_rate)

        return num_samples

    def get_expected_frames(self, num_samples):
        """
        Determine the number of frames the module will return for a given number of samples.

        Parameters
        ----------
        num_samples : int
          Number of audio samples available

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        # One plus the number frames of hops available
        num_frames = 1 + num_samples // self.hop_length

        return num_frames

    def get_times(self, n_frames):
        """
        Determine the time associated with each frame of coefficients.

        Parameters
        ----------
        n_frames : int
          Number of frames available

        Returns
        ----------
        times : ndarray (T)
          Time (seconds) associated with each frame
        """

        # Compute times as cumulative hops in seconds
        times = np.arange(n_frames) * self.hop_length / self.sample_rate

        return times

    def get_midi_freqs(self):
        """
        Obtain the MIDI frequencies associated with the 1st harmonic.

        Returns
        ----------
        midi_freqs : ndarray (F)
          Center frequency of each bin
        """

        # Determine first harmonic index
        h_idx = self.harmonics.index(1)

        # Extract pre-computed frequencies
        midi_freqs = self.midi_freqs[h_idx]

        return midi_freqs
