# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove
from common import TrainSet

# Regular imports
import shutil
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
          TODO
        """

        # TODO
        splits = []

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

        # TODO
        tracks = sorted([])

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

        # TODO
        wav_path = None

        return wav_path

    @classmethod
    def download(cls, save_dir):
        """
        Download the FreeMusicArchive dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of FreeMusicArchive
        """

        # If the directory already exists, remove it
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        # Create the base directory
        os.makedirs(save_dir)

        print(f'Downloading {cls.__name__}')

        # URL pointing to the zip file containing excerpts for all tracks
        url = f'https://os.unil.cloud.switch.ch/fma/fma_large.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)