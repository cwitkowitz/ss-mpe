# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import EvalSet

# Regular imports
import numpy as np
import scipy
import os


class MedleyDB(EvalSet):
    """
    Implements a wrapper for the MedleyDB Pitch Tracking subset (https://zenodo.org/record/2620624).
    """

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

        # TODO
        tracks = None

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = None

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        TODO : string
          TODO
        """

        # TODO

        return mat_path

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # TODO

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # TODO

        return ground_truth

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        return NotImplementedError
