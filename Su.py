# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import EvalSet

# Regular imports
import numpy as np
import librosa
import scipy
import os


class Su(EvalSet):
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

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain the path to the track's ground_truth
        txt_path = self.get_ground_truth_path(track)

        # Open annotations in reading mode
        with open(txt_path) as txt_file:
            # Read frame-level annotations into a list
            frames = [f.split() for f in txt_file.readlines()]

        # Create array of frame indices
        original_idcs = np.arange(len(frames))

        # Extract the original times for each frame
        original_times = np.array([f.pop(0) for f in frames]).astype('float')

        # Out-of-range times will be set to first time (always silent)
        fill_values = (original_idcs[0], original_idcs[0])

        # Obtain a function to resample annotation times
        res_func_time = scipy.interpolate.interp1d(x=original_times,
                                                   y=original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)

        # Resample the multi-pitch annotations using above function
        multi_pitch = [np.array(frames[i]).astype('float')
                       for i in res_func_time(times).astype('uint')]

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # Determine the frames corresponding to pitch observations
        time_idcs = np.concatenate([[i] * len(multi_pitch[i])
                                    for i in range(len(times))])

        # Flatten multi pitch annotations and convert to MIDI
        multi_pitch = librosa.hz_to_midi(np.concatenate(multi_pitch))

        # Determine the closest frequency bin for each pitch observation
        multi_pitch_idcs = self.res_func_freq(multi_pitch)

        # Insert pitch activity into the ground-truth
        ground_truth[multi_pitch_idcs.astype('uint'), time_idcs.astype('uint')] = 1

        return ground_truth

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        return NotImplementedError
