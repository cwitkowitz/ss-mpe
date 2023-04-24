# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from NSynth import NSynth
from common import EvalSet

# Regular imports
import warnings
import random


class ToyNSynthEval(EvalSet, NSynth):
    """
    TODO
    """

    def __init__(self, n_tracks=None, **kwargs):
        """
        TODO.

        Parameters
        ----------
        n_tracks : int
          TODO
        kwargs : TODO
          TODO
        """

        self.n_tracks = n_tracks

        super().__init__(**kwargs)

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string (unused)
          TODO

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Obtain the standard track list
        tracks = super().get_tracks(split)

        # Filter out tracks with no positive ground-truth activations
        tracks = [t for t in tracks if self.get_pitch(t) in self.center_freqs]

        if self.n_tracks is not None:
            # Shuffle the tracks
            random.shuffle(tracks)

            # Trim tracks to selected amount
            tracks = tracks[:self.n_tracks]

        return tracks

    def get_pitch(self, track):
        """
        Determine the pitch associated with a track.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        pitch : TODO
          TODO
        """

        # Extract the pitch from track name
        pitch = int(track.split('-')[-2])

        return pitch

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        try:
            # Obtain the index of the pitch of the sample from the track name
            pitch_idx = int(self.res_func_freq(self.get_pitch(track)).item())

            # Obtain time indices corresponding to pitch activity
            time_idcs = (times >= 0) & (times <= 3)

            # Make the pitch active for the entire duration
            ground_truth[pitch_idx, time_idcs] = 1
        except ValueError:
            warnings.warn('Cannot represent ground-truth '
                          f'for track \'{track}\'.', RuntimeWarning)

        return ground_truth

    @classmethod
    def name(cls):
        """
        Simple helper function to get the class name.
        """

        name = NSynth.name()

        return name
