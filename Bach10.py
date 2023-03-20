# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import normalize

# Regular imports
from torch.utils.data import Dataset

import soundfile as sf
import numpy as np
import warnings
import shutil
import json
import os


class Bach10(Dataset):
    """
    Implements a wrapper for the Bach10 dataset (TODO).
    """

    def __init__(self, base_dir):
        """
        TODO.

        Parameters
        ----------
        base_dir : TODO
          TODO
        """

        self.base_dir = base_dir

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset if it is missing and if a procedure exists
            self.download(self.base_dir)

        # Aggregate all the track names
        self.tracks = self.get_tracks()

    def __len__(self):
        """
        Defines the notion of length for the dataset - used by PyTorch Dataset class.

        Returns
        ----------
        length : int
          TODO
        """

        # TODO
        length = len(self.tracks)

        return length

    def __getitem__(self, index):
        """
        TODO.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        track : TODO
          TODO
        """

        # TODO
        track = self.tracks[index]

        return track

    def get_tracks(self):
        """
        Get the names of the tracks in the dataset.

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Obtain all track names as the numbered directories
        tracks = sorted([d for d in os.listdir(self.base_dir)if d.split('-')[0].isdigit()])

        return tracks

    def get_wav_path(self, track):
        """
        Get the path for a track's audio.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, track, f'{track}.wav')

        return wav_path

    def get_audio(self, track):
        """
        Get the audio for a track.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        audio : TODO
          TODO
        """

        # Obtain the path of the track's audio
        wav_path = self.get_wav_path(track)

        # TODO
        audio, _ = sf.read(wav_path)

        # Normalize the audio between the range [-1, 1]
        audio = normalize(audio)

        return audio

    def get_txt_path(self, track):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        txt_path : string
          TODO
        """

        # Get the path to the ground-truth text annotations
        txt_path = os.path.join(self.base_dir, track, f'{track}.txt')

        return txt_path

    def get_ground_truth(self, track, times, bins):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain the path of the track's ground_truth
        txt_path = self.get_txt_path(track)

        # Open the txt file in reading mode
        with open(txt_path) as txt_file:
            # Read all notes into an array
            notes = np.array([n.split() for n in txt_file.readlines()], dtype='uint')

        # Split apart the note attributes
        onsets, offsets, pitches, _ = notes.transpose()

        # Convert onsets and offsets to seconds
        onsets, offsets = 0.001 * onsets, 0.001 * offsets

        ground_truth = None

        return ground_truth

    @classmethod
    def download(cls, save_dir):
        """
        TODO
        """

        # TODO
        raise NotImplementedError()
