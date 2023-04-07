# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove, change_base_dir
from common import EvalSet

# Regular imports
import numpy as np
import librosa
import scipy
import os


class URMP(EvalSet):
    """
    Implements a wrapper for the URMP dataset (https://labsites.rochester.edu/air/projects/URMP.html).
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
                         if d.split('_')[0].isdigit()])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          URMP track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, track, f'AuMix_{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the paths to a track's ground truth.

        Parameters
        ----------
        track : string
          URMP track name

        Returns
        ----------
        txt_paths : string
          TODO
        """

        # Obtain a list of all files under the track's directory
        track_files = os.listdir(os.path.join(self.base_dir, track))

        # Get the path for the F0 annotations of each instrument in the mixture
        txt_paths = [os.path.join(self.base_dir, track, f) for f in track_files
                      if f.split('_')[0] == 'F0s']

        return txt_paths

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          URMP track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # TODO

        # Obtain the paths to the track's ground_truth
        txt_paths = self.get_ground_truth_path(track)

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # Loop through files
        for t in txt_paths:
            # Read annotation file
            with open(t) as txt_file:
                # Read frame-level annotations into a list
                annotations = [f.split() for f in txt_file.readlines()]

            # Break apart frame time and pitch observations from the annotations
            original_times, pitches = np.array(annotations).astype('float').T

            print(pitches[0], pitches[-1])

            # Create array of frame indices
            original_idcs = np.arange(len(original_times))

            # Clamp resampled indices within the valid range
            fill_values = (original_idcs[0], original_idcs[-1])

            # Obtain a function to resample annotation times
            res_func_time = scipy.interpolate.interp1d(x=original_times,
                                                       y=original_idcs,
                                                       kind='nearest',
                                                       bounds_error=False,
                                                       fill_value=fill_values,
                                                       assume_sorted=True)

            # Resample the pitch annotations using above function
            pitches = pitches[..., res_func_time(times).astype('uint')]

            # Determine the frames corresponding to pitch observations
            frame_idcs = pitches.nonzero()[0]

            # Determine the closest frequency bin for each pitch observation
            pitch_idcs = self.res_func_freq(librosa.hz_to_midi(pitches[pitches != 0.]))

            # Insert pitch activity into the ground-truth
            ground_truth[pitch_idcs.astype('uint'), frame_idcs] = 1

        return ground_truth

    @classmethod
    def download(cls, save_dir):
        """
        Download the URMP dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of URMP
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the tar file containing data for all tracks
        url = 'https://datadryad.org/stash/downloads/file_stream/99348'

        # Construct a path for saving the file
        tar_path = os.path.join(save_dir, 'URMP.tar.gz')

        # Download the tar file
        stream_url_resource(url, tar_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(tar_path, tar=True)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'Dataset'))
