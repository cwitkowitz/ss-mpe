# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import normalize

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod

import numpy as np
import warnings
import librosa
import torch
import os


DEFAULT_LOCATION = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets')


class TrainSet(Dataset):
    """
    Implements a wrapper for an MPE dataset intended for training.
    """

    def __init__(self, base_dir=None, splits=None, sample_rate=16000, seed=0, device='cpu'):
        """
        TODO.

        Parameters
        ----------
        base_dir : TODO
          TODO
        splits : TODO
          TODO
        seed : TODO
          TODO
        """

        if base_dir is None:
            base_dir = os.path.join(DEFAULT_LOCATION, self.__class__.__name__)

        self.base_dir = base_dir
        self.sample_rate = sample_rate

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset if it is missing
            self.download(self.base_dir)

        if splits is None:
            # Choose all available dataset splits
            splits = self.available_splits()

        # Initialize a random number generator for the dataset
        self.rng = np.random.RandomState(seed)

        self.tracks = []
        # Aggregate all the track names from the selected splits
        for split in splits:
            self.tracks += self.get_tracks(split)

        self.device = device

    @staticmethod
    @abstractmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset
        """

        splits = ['all']

        return splits

    @abstractmethod
    def get_tracks(self, split):
        """
        Get the track names associated with dataset partitions.

        Parameters
        ----------
        split : string
          TODO
        """

        return NotImplementedError

    def __len__(self):
        """
        Defines the number of samples in dataset.

        Returns
        ----------
        length : int
          TODO
        """

        # TODO
        length = len(self.tracks)

        return length

    @abstractmethod
    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Track name
        """

        return NotImplementedError

    def get_audio(self, track):
        """
        Get the audio for a track.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        audio : TODO
          TODO
        """

        # Obtain the path of the track's audio
        audio_path = self.get_audio_path(track)

        # TODO
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Normalize the audio between the range [-1, 1]
        audio = normalize(audio)

        return audio

    @abstractmethod
    def __getitem__(self, index):
        """
        TODO.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        audio : TODO
          TODO
        """

        # TODO
        audio = self.get_audio(self.tracks[index])

        # Add the audio to the appropriate GPU
        audio = torch.from_numpy(audio).to(self.device).float()

        # Add a channel dimension to the audio
        audio = audio.unsqueeze(-2)

        return audio

    @classmethod
    @abstractmethod
    def download(cls, save_dir):
        """
        Download the dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the dataset
        """

        return NotImplementedError


class EvalSet(TrainSet):
    """
    Implements a wrapper for an MPE dataset intended for evaluation.
    """

    @abstractmethod
    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          Track name
        """

        return NotImplementedError

    @abstractmethod
    def get_ground_truth(self, track):
        """
        TODO
        """

        return NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """
        TODO.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        audio : TODO
          TODO
        """

        # TODO
        audio = super().__getitem__(index)

        # TODO - get ground_truth
        ground_truth = None

        return audio, ground_truth
