# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove
from common import EvalSet

# Regular imports
import pandas as pd
import numpy as np
import librosa
import os


class SWD(EvalSet):
    """
    Implements a wrapper for the SWD dataset (https://zenodo.org/record/5139893).
    """

    @staticmethod
    def get_splits_by_song():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset based on the 24 songs of the cycle
        """

        splits = 1 + np.arange(24)

        return splits

    @staticmethod
    def get_splits_by_performance():
        """
        Obtain a list of pre-defined dataset splits.

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
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset
        """

        splits = SWD.get_splits_by_performance()

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset with audio available.

        Parameters
        ----------
        split : string (unused)
          TODO

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Construct a path to the audio contained in the dataset
        audio_dir = os.path.join(self.base_dir, '01_RawData', 'audio_wav')

        # Obtain names of tracks corresponding to the specified performance
        tracks = sorted([os.path.splitext(t)[0] for t in os.listdir(audio_dir)
                         if split in t])

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
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, '01_RawData', 'audio_wav', f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          SWD track name

        Returns
        ----------
        csv_path : string
          TODO
        """

        # Get the path to the ground-truth note annotations
        csv_path = os.path.join(self.base_dir, '02_Annotations', 'ann_audio_note', f'{track}.csv')

        return csv_path

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          SWD track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain the path to the track's ground_truth
        csv_path = self.get_ground_truth_path(track)

        # Load tabulated note data from the csv file
        note_entries = pd.read_csv(csv_path, sep=';').to_numpy()

        # Unpack the relevant note attributes
        onsets, offsets, pitches = note_entries[:, (0, 1, 2)].T

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # Convert onsets and offsets to frame indices
        onsets = librosa.time_to_frames(onsets, sr=self.sample_rate, hop_length=self.hop_length)
        offsets = librosa.time_to_frames(offsets, sr=self.sample_rate, hop_length=self.hop_length)

        # Clip offsets occurring after number of frames of audio
        offsets = np.clip(offsets, a_min=0, a_max=len(times) - 1)

        # Compute durations in frames
        durations = 1 + offsets - onsets

        # Determine the closest frequency bin for each pitch
        pitch_idcs = self.res_func_freq(pitches.astype('uint'))
        # Repeat each pitch index for the number of frames it is active
        pitch_idcs = np.concatenate([[p] * d for p, d in zip(pitch_idcs, durations)])
        # Create time indices corresponding to the full duration of each note
        time_idcs = np.concatenate([np.arange(i, i + d) for i, d in zip(onsets, durations)])

        # Insert pitch activity into the ground-truth
        ground_truth[pitch_idcs.astype('uint'), time_idcs] = 1

        return ground_truth

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
