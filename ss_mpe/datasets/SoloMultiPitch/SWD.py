# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import AMTDataset
from timbre_trap.utils.data import *

# Regular imports
import pandas as pd
import numpy as np
import os


class SWD(AMTDataset):
    """
    Implements a wrapper for the SWD dataset
    (https://zenodo.org/record/5139893).
    """

    @staticmethod
    def get_splits_by_song():
        """
        Return a list of indices corresponding to each song in the cycle.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset based on the 24 songs of the cycle
        """

        return NotImplementedError

    @staticmethod
    def get_splits_by_performance():
        """
        Return a list of individual performances of the cycle.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset based on performances of the 24-song cycle
        """

        splits = ['AL98', 'FI55', 'FI66', 'FI80', 'HU33',
                  'OL06', 'QU98', 'SC06', 'TR99']

        return splits

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Individual performances of 24-song cycle
        """

        splits = SWD.get_splits_by_performance()

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating dataset partition (song or performance)

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """

        # Construct a path to the directory containing all audio
        audio_dir = os.path.join(self.base_dir, '01_RawData', 'audio_wav')

        # Obtain names of tracks corresponding to the specified performance
        tracks = sorted([os.path.splitext(t)[0] for t in os.listdir(audio_dir) if split in t])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          SWD track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, '01_RawData', 'audio_wav', f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          SWD track name

        Returns
        ----------
        csv_path : string
          Path to ground-truth for the specified track
        """

        # Get the path to the ground-truth note annotations
        csv_path = os.path.join(self.base_dir, '02_Annotations', 'ann_audio_note', f'{track}.csv')

        return csv_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          SWD track name

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        # Obtain the path to the track's ground_truth
        csv_path = self.get_ground_truth_path(track)

        # Load tabulated note data from the csv file
        note_entries = pd.read_csv(csv_path, sep=';').to_numpy()

        # Unpack the relevant note attributes and convert them to floats
        onsets, offsets, pitches = note_entries[:, (0, 1, 2)].T.astype(float)

        # Combine onsets and offsets to obtain intervals
        intervals = np.concatenate(([onsets], [offsets])).T

        return pitches, intervals

    @classmethod
    def download(cls, save_dir):
        """
        Download the SWD dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of SWD
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        url = 'https://zenodo.org/record/5139893/files/Schubert_Winterreise_Dataset_v2-0.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)
