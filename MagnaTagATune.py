# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove
from common import TrainSet

# Regular imports
import shutil
import os


class MagnaTagATune(TrainSet):
    """
    Implements a wrapper for the MagnaTagATune dataset (https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset).
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

        splits = ['0', '1', '2', '3', '4', '5',
                  '6', '7', '8', '9', 'a', 'b',
                  'c', 'd', 'e', 'f']

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

        # Combine the parent directory of the split with each file
        tracks = sorted([os.path.join(split, os.path.splitext(f)[0])
                         for f in os.listdir(split_path)
                         if os.path.getsize(os.path.join(split_path, f)) > 0])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MagnaTagATune track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the MP3 file
        mp3_path = os.path.join(self.base_dir, split, f'{name}.mp3')

        return mp3_path

    @classmethod
    def download(cls, save_dir):
        """
        Download the MagnaTagATune dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MagnaTagATune
        """

        # If the directory already exists, remove it
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        # Create the base directory
        os.makedirs(save_dir)

        print(f'Downloading {cls.__name__}')

        for part in [1, 2, 3]:
            # URL pointing to the zip partition file
            url = f'https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.00{part}'

            # Construct a path for saving the file
            zip_path = os.path.join(save_dir, os.path.basename(url))

            # Download the zip file
            stream_url_resource(url, zip_path, 1000 * 1024)

        # Construct a path to the top-level zip file
        save_path = os.path.join(save_dir, 'mp3.zip')

        # Combine zip file partitions
        os.system(f'cat {save_path}* > {save_path}')

        # Unzip the downloaded file and remove it
        unzip_and_remove(save_path)

        # Remove remaining zip file partitions
        os.system(f'rm {save_path}*')
