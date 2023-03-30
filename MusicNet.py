# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove, change_base_dir
from common import EvalSet

# Regular imports
import numpy as np
import librosa
import scipy
import os


class MusicNet(EvalSet):
    """
    Implements a wrapper for the MusicNet dataset (https://zenodo.org/record/5120004).
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

        splits = ['train', 'test']

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

        # Obtain tracks names as files under the split's data directory
        tracks = os.listdir(os.path.join(self.base_dir, f'{split}_data'))

        # Remove the file extension for all tracks
        tracks = sorted([os.path.join(split, os.path.splitext(t)[0]) for t in tracks])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, f'{split}_data', f'{name}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        csv_path : string
          TODO
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the ground-truth note annotations
        csv_path = os.path.join(self.base_dir, f'{split}_labels', f'{name}.csv')

        return csv_path

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain the path to the track's ground_truth
        csv_path = self.get_ground_truth_path(track)

        # TODO

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # TODO

        return ground_truth

    @classmethod
    def download(cls, save_dir):
        """
        Download the MusicNet dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MusicNet
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the tar file containing data for all tracks
        url = 'https://zenodo.org/record/5120004/files/musicnet.tar.gz'

        # Construct a path for saving the file
        tar_path = os.path.join(save_dir, os.path.basename(url))

        # Download the tar file
        stream_url_resource(url, tar_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(tar_path, tar=True)

        # Move contents of untarred directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'musicnet'))
