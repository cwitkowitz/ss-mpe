# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import AudioDataset
from timbre_trap.utils.data import *

# Regular imports
import pandas as pd
import os


class ESC_50(AudioDataset):
    """
    Implements a wrapper for the ESC-50 dataset
    (https://github.com/karolpiczak/ESC-50).
    """

    @classmethod
    def name(cls):
        """
        Obtain a string representing the dataset.

        Returns
        ----------
        name : string
          Dataset name with dashes
        """

        # Obtain class name and replace underscores with dashes
        tag = super().name().replace('_', '-')

        return tag

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Names of originally proposed splits
        """

        # Animals
        splits = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog',
                  'Cat', 'Hen', 'Insects', 'Sheep', 'Crow']

        # Natural soundscapes & water sounds
        splits += ['Rain', 'Sea Waves', 'Crackling Fire', 'Crickets', 'Chirping Birds',
                   'Water Drops', 'Wind', 'Pouring Water', 'Toilet Flush', 'Thunderstorm']

        # Human, non-speech sounds
        splits += ['Crying Baby', 'Sneezing', 'Clapping', 'Breathing', 'Coughing',
                   'Footsteps', 'Laughing', 'Brushing Teeth', 'Snoring', 'Drinking Sipping']

        # Interior/domestic sounds
        splits += ['Door Wood Knock', 'Mouse Click', 'Keyboard Typing', 'Door Wood Creaks', 'Can Opening',
                   'Washing Machine', 'Vacuum Cleaner', 'Clock Alarm', 'Clock Tick', 'Glass Breaking']

        # Exterior/urban noises
        splits += ['Helicopter', 'Chainsaw', 'Siren', 'Car Horn', 'Engine',
                   'Train', 'Church Bells', 'Airplane', 'Fireworks', 'Hand Saw']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating environmental sound class

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """

        # Make sure the split matches the csv file format
        split = split.replace(' ', '_').lower()

        # Load tabulated metadata from the csv file under the top-level directory
        csv_data = pd.read_csv(os.path.join(self.base_dir, 'meta', 'esc50.csv'))

        # Obtain a list of the track names and their corresponding splits
        names, categories = csv_data['filename'], csv_data['category']
        # Filter out tracks that do not belong to the specified data split
        tracks = [t for t, a in zip(names, categories) if a == split]
        # Remove the file extensions from each track and sort the list
        tracks = sorted([os.path.splitext(track)[0] for track in tracks])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          E-GMD track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Construct the path to the audio
        wav_path = os.path.join(self.base_dir, 'audio', f'{track}.wav')

        return wav_path

    @classmethod
    def download(cls, save_dir):
        """
        Download the ESC-50 dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of ESC-50
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        url = f'https://github.com/karoldvl/ESC-50/archive/master.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'ESC-50-master'))