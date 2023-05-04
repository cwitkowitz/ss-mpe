# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import EvalSetFrameLevel

# Regular imports
import numpy as np
import scipy
import os


class Su(EvalSetFrameLevel):
    """
    Implements a wrapper for the Su dataset (TODO).
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

        # Construct a path to the directory containing MIDI
        midi_dir = os.path.join(self.base_dir, 'midi')

        # Obtain all track names as the directories containing MIDI files
        tracks = sorted([d for d in os.listdir(midi_dir)
                         if os.path.isdir(os.path.join(midi_dir, d))])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, 'audio', f'{track}_audio.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        txt_path : string
          TODO
        """

        # Get the path to the ground-truth multi-pitch annotations
        txt_path = os.path.join(self.base_dir, 'gt_F0', f'{track}_F0.txt')

        return txt_path

    @staticmethod
    def resample_multi_pitch(_times, _multi_pitch, times):
        """
        Protocol for resampling the ground-truth annotations, if necessary.

        Some tracks in this dataset have pitches in the last frame, so
        it is not a good idea to set the final frame as the fill value.

        Parameters
        ----------
        _times : ndarray (T)
          Original times
        _multi_pitch : list of ndarray (T x [...])
          Multi-pitch annotations corresponding to original times
        times : ndarray (K)
          Target times for resampling

        Returns
        ----------
        multi_pitch : list of ndarray (K x [...])
          Multi-pitch annotations corresponding to target times
        """

        # Create array of frame indices
        original_idcs = np.arange(len(_times))

        # Out-of-range times will be set to first time (always silent)
        fill_values = (original_idcs[0], original_idcs[0])

        # Obtain a function to resample annotation times
        res_func_time = scipy.interpolate.interp1d(x=_times,
                                                   y=original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)

        # Resample the multi-pitch annotations using above function
        multi_pitch = [_multi_pitch[t] for t in res_func_time(times).astype('uint')]

        return multi_pitch

    def get_ground_truth(self, track):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the path to the track's ground_truth
        txt_path = self.get_ground_truth_path(track)

        # Open annotations in reading mode
        with open(txt_path) as txt_file:
            # Read frame-level annotations into a list
            frames = [f.split() for f in txt_file.readlines()]

        # Extract the original times for each frame
        times = np.array([f.pop(0) for f in frames]).astype('float')

        # Obtain ground-truth as a list of pitch observations
        multi_pitch = [np.array(p).astype('float') for p in frames]

        return times, multi_pitch

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        return NotImplementedError
