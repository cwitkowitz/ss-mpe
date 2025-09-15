# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import AudioDataset
from timbre_trap.utils.data import *

# Regular imports
import pandas as pd
import os


class E_GMD(AudioDataset):
    """
    Implements a wrapper for the expanded Groove MIDI Dataset
    (https://magenta.tensorflow.org/datasets/e-gmd).
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

        splits = ['train', 'validation', 'test']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating train, validation, or test split

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """

        # Load tabulated metadata from the csv file under the top-level directory
        csv_data = pd.read_csv(os.path.join(self.base_dir, 'e-gmd-v1.0.0.csv'))

        # Obtain a list of the track names and their corresponding splits
        names, associations = csv_data['audio_filename'], csv_data['split']
        # Filter out tracks that do not belong to the specified data split
        tracks = [t for t, a in zip(names, associations) if a == split]
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
        mp3_path : string
          Path to audio for the specified track
        """

        # Construct the path to the audio
        wav_path = os.path.join(self.base_dir, f'{track}.wav')

        return wav_path

    @classmethod
    def download(cls, save_dir):
        """
        Download the E-GMD dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of E-GMD
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing MIDI for all tracks
        midi_url = 'https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip'

        # Construct a path for saving the MIDI
        midi_path = os.path.join(save_dir, os.path.basename(midi_url))

        # Download the MIDI zip file
        stream_url_resource(midi_url, midi_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(midi_path)

        # URL pointing to the zip file containing audio for all tracks
        audio_url = f'https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip'

        # Construct a path for saving the audio
        audio_path = os.path.join(save_dir, os.path.basename(audio_url))

        # Download the audio zip file
        stream_url_resource(audio_url, audio_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(audio_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, f'e-gmd-v1.0.0'))
