# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.utils import stream_url_resource, unzip_and_remove
from timbre_trap.datasets import AudioDataset

# Regular imports
import os


class MagnaTagATune(AudioDataset):
    """
    Implements a wrapper for the MagnaTagATune dataset
    (https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Top-level directories from download
        """

        splits = ['0', '1', '2', '3', '4', '5',
                  '6', '7', '8', '9', 'a', 'b',
                  'c', 'd', 'e', 'f']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Top-level directory

        Returns
        ----------
        tracks : list of strings
          List containing the songs under the selected directory
        """

        # Construct a path to the dataset split
        split_dir = os.path.join(self.base_dir, split)

        # Ignore tracks corresponding to empty files
        valid_files = [f for f in os.listdir(split_dir)
                       if os.path.getsize(os.path.join(split_dir, f)) > 0]

        # Obtain a sorted list of all valid tracks within the split directory
        tracks = sorted([os.path.join(split, os.path.splitext(f)[0]) for f in valid_files])

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
        mp3_path : string
          Path to audio for the specified track
        """

        # Break apart split and track name
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

        # Create top-level directory
        super().download(save_dir)

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
