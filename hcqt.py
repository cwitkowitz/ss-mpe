# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt import LHVQT as _LHVQT

# Regular imports
import numpy as np
import librosa


class LHVQT(_LHVQT):
    """
    Wrapper which adds ability to obtain time associated with
    each frame for the LHVQT module and access certain fields.
    """

    def __init__(self, fs, hop_length, n_bins, bins_per_octave, **kwargs):
        """
        TODO.

        Parameters
        ----------
        fs : int
          TODO
        hop_length : int
          TODO
        kwargs : TODO
          TODO
        """

        self.sample_rate = fs
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

        super().__init__(fs=fs, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, **kwargs)

    def get_times(self, audio):
        """
        TODO

        Parameters
        ----------
        audio : TODO
          TODO

        Returns
        ----------
        times : TODO
          TODO
        """

        # Determine number of frames as the number of hops
        n_frames = 1 + audio.size(-1) // self.hop_length

        # Compute time associated with center of each frame
        times = librosa.frames_to_time(np.arange(n_frames),
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length)

        return times
