# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from common import EvalSet

# Regular imports
import numpy as np
import scipy
import os


class Bach10(EvalSet):
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

        # Obtain the path to the track's ground_truth
        mat_path = self.get_ground_truth_path(track)

        # Extract the frame-level multi-pitch annotations
        multi_pitch = scipy.io.loadmat(mat_path)['GTF0s']

        # Determine how many frames were provided
        num_frames = multi_pitch.shape[-1]

        # Create array of frame indices
        original_idcs = np.arange(num_frames)

        # Compute the original times for each frame
        original_times = 0.023 + 0.010 * original_idcs

        # Clamp resampled indices within the valid range
        fill_values = (original_idcs[0], original_idcs[-1])

        # Obtain a function to resample annotation times
        res_func_time = scipy.interpolate.interp1d(x=original_times,
                                                   y=original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)

        # Resample the multi-pitch annotations using above function
        multi_pitch = multi_pitch[..., res_func_time(times).astype('uint')]

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # Determine the frames corresponding to pitch observations
        _, frame_idcs = multi_pitch.nonzero()

        # Determine the closest frequency bin for each pitch observation
        multi_pitch_idcs = self.res_func_freq(multi_pitch[multi_pitch != 0.])

        # Insert pitch activity into the ground-truth
        ground_truth[multi_pitch_idcs.astype('uint'), frame_idcs] = 1

        return ground_truth

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        return NotImplementedError
