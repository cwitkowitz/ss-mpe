# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import EvalSetFrameLevel

# Regular imports
import numpy as np
import librosa
import scipy
import os


class Bach10(EvalSetFrameLevel):
    """
    Implements a wrapper for the Bach10 dataset (https://labsites.rochester.edu/air/resource.html).
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
        mat_path : string
          TODO
        """

        # Get the path to the ground-truth multi-pitch annotations
        mat_path = os.path.join(self.base_dir, track, f'{track}-GTF0s.mat')

        return mat_path

    def get_ground_truth(self, track):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the path to the track's ground_truth
        mat_path = self.get_ground_truth_path(track)

        # Extract the frame-level multi-pitch annotations
        multi_pitch = scipy.io.loadmat(mat_path)['GTF0s']

        # Determine how many frames were provided
        num_frames = multi_pitch.shape[-1]

        # Compute the original times for each frame
        times = 0.023 + 0.010 * np.arange(num_frames)

        # Obtain ground-truth as a list of pitch observations in Hertz
        multi_pitch = [librosa.midi_to_hz(p[p != 0]) for p in multi_pitch.T]

        return times, multi_pitch

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        return NotImplementedError
