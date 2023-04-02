# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from NSynth import NSynth
from common import TrainSet, EvalSet

# Regular imports
import os


class ToyNSynthTrain(TrainSet):
    """
    Implements a minimal wrapper for the NSynth dataset (https://magenta.tensorflow.org/datasets/nsynth).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset for different stages of pipeline
        """

        splits = ['all']

        return splits

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

        # Pre-define a few tracks
        tracks = ['guitar_acoustic_010-072-127', 'guitar_acoustic_010-055-127']

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, 'nsynth-valid', 'audio', track + '.wav')

        return wav_path

    @classmethod
    def name(cls):
        """
        Simple helper function to get the class name.
        """

        name = NSynth.name()

        return name

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        return NotImplementedError



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

        # Obtain the index of the pitch of the sample from the track name
        pitch_idx = int(self.res_func_freq(track.split('-')[-2]).item())

        # Obtain time indices corresponding to pitch activity
        time_idcs = (times >= 0) & (times <= 3)

        # Make the pitch active for the entire duration
        ground_truth[pitch_idx, time_idcs] = 1

        return ground_truth
