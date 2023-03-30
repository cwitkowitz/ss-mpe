# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove, change_base_dir
from common import TrainSet

# Regular imports
import numpy as np
import os


class FreeMusicArchive(TrainSet):
    """
    Implements a wrapper for the FreeMusicArchive dataset (https://github.com/mdeff/fma).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Top-level directories
        """

        # All numbered directories with leading zeros
        splits = [str(i).zfill(3) for i in np.arange(156)]

        return splits

    def get_tracks(self, split):
        """
        Get the track names associated with dataset partitions.

        Parameters
        ----------
        split : string
          TODO

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Construct a path to the dataset split
        split_path = os.path.join(self.base_dir, split)

        # Obtain a sorted list of all files in the split's directory
        tracks = sorted([os.path.splitext(f)[0] for f in os.listdir(split_path)])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          FreeMusicArchive track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the MP3 file
        mp3_path = os.path.join(self.base_dir, track[:3], f'{track}.mp3')

        return mp3_path

    @classmethod
    def download(cls, save_dir):
        """
        Download the FreeMusicArchive dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of FreeMusicArchive
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing excerpts for all tracks
        url = 'https://os.unil.cloud.switch.ch/fma/fma_large.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'fma_large'))
