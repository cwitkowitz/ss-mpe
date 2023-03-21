# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import EvalSet

# Regular imports
import numpy as np
import os


class Bach10(EvalSet):
    """
    Implements a wrapper for the Bach10 dataset (TODO).
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

        # Obtain all track names as the numbered directories
        tracks = sorted([d for d in os.listdir(self.base_dir)
                         if d.split('-')[0].isdigit()])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, track, f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        txt_path : string
          TODO
        """

        # Get the path to the ground-truth text annotations
        txt_path = os.path.join(self.base_dir, track, f'{track}.txt')

        return txt_path

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain the path of the track's ground_truth
        txt_path = self.get_ground_truth_path(track)

        # Open the txt file in reading mode
        with open(txt_path) as txt_file:
            # Read all notes into an array
            notes = np.array([n.split() for n in txt_file.readlines()], dtype='uint')

        # Split apart the note attributes
        onsets, offsets, pitches, _ = notes.transpose()

        # Convert onsets and offsets to seconds
        onsets, offsets = 0.001 * onsets, 0.001 * offsets

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # TODO - insert notes into ground-truth

        return ground_truth
