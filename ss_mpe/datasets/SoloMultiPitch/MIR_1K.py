# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import MPEDataset
from timbre_trap.utils.data import *

# Regular imports
import numpy as np
import torchaudio
import librosa
import os


class MIR_1K(MPEDataset):
    """
    Implements a wrapper for the MIR-1K dataset.
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
          Singer identifications across all tracks
        """

        splits = ['Ani', 'Kenshin', 'abjones', 'amy', 'annar', 'ariel',
                  'bobon', 'bug', 'davidson', 'fdps', 'geniusturtle',
                  'heycat', 'jmzen', 'khair', 'leon', 'stool', 'tammy',
                  'titon', 'yifen']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Singer identifier

        Returns
        ----------
        tracks : list of strings
          List containing the excerpts with the specified singer
        """

        # Extract the names of all the files in audio directory
        audio_files = os.listdir(os.path.join(self.base_dir, 'Wavfile'))

        # Filter out files with a mismatching singer and remove WAV extension
        tracks = [os.path.splitext(t)[0] for t in audio_files if t.startswith(split)]

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MIR-1K track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, 'Wavfile', f'{track}.wav')

        return wav_path

    def get_audio(self, track):
        """
        Extract the audio for the specified track.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        audio : Tensor (1 x N)
          Audio data read for the track
        """

        # Obtain the path to the track's audio
        audio_path = self.get_audio_path(track)

        # Load the audio with torchaudio
        audio, fs = torchaudio.load(audio_path)
        # Resample audio from right channel to the specified sampling rate
        audio = torchaudio.functional.resample(audio[1:], fs, self.sample_rate)

        if audio.abs().max():
            # Normalize the audio using the infinity norm
            audio /= audio.abs().max()

        return audio

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          MIR-1K track name

        Returns
        ----------
        pv_path : string
          Path to ground-truth for the specified track
        """

        # Get the path to the F0 annotations
        pv_path = os.path.join(self.base_dir, 'PitchLabel', f'{track}.pv')

        return pv_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          MIR-1K track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        pitches : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the path to the track's ground_truth
        pv_path = self.get_ground_truth_path(track)

        # Open annotations in reading mode
        with open(pv_path) as pv_file:
            # Read F0 annotations into an array
            pitches = np.array([float(p.strip()) for p in pv_file.readlines()])

        # Determine how many frames were provided
        num_frames = len(pitches)

        # Compute the original times for each frame
        times = 0.020 + 0.020 * np.arange(num_frames)

        # Obtain ground-truth as a list of pitch observations in Hertz
        pitches = [librosa.midi_to_hz(p[p != 0]) for p in pitches]

        return times, pitches

    @classmethod
    def download(cls, save_dir):
        """
        Download the MIR-1K dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MIR-1K
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file for the MIR-1K dataset
        url = f'http://mirlab.org/dataset/public/MIR-1K.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'MIR-1K'))
