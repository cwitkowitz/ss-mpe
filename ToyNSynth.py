# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from NSynth import NSynth
from common import EvalSet

# Regular imports
import random


class ToyNSynthTrain(NSynth):
    """
    TODO
    """

    def __init__(self, n_tracks, **kwargs):
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

        # Shuffle the tracks
        random.shuffle(tracks)

        # Trim tracks to selected amount
        tracks = tracks[:self.n_tracks]

        return tracks

    @classmethod
    def name(cls):
        """
        Simple helper function to get the class name.
        """

        name = NSynth.name()

        return name


class ToyNSynthEval(EvalSet, ToyNSynthTrain):
    """
    TODO
    """

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
            pitch_idx = int(self.res_func_freq(track.split('-')[-2]).item())

            # Obtain time indices corresponding to pitch activity
            time_idcs = (times >= 0) & (times <= 3)

            # Make the pitch active for the entire duration
            ground_truth[pitch_idx, time_idcs] = 1
        except Exception as e:
            pass

        return ground_truth
