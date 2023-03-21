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
        sample_rate : TODO
          TODO
        seed : TODO
          TODO
        device : TODO
          TODO
        """

        if base_dir is None:
            base_dir = os.path.join(DEFAULT_LOCATION, self.get_name())

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

    @classmethod
    def get_name(cls):
        """
        Simple helper function to get the class name.
        """

        name = cls.__name__

        return name

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

    def __init__(self, hop_length, n_bins, bins_per_octave, fmin=None, **kwargs):
        """
        TODO.

        Parameters
        ----------
        hop_length : TODO
          TODO
        n_bins : TODO
          TODO
        bins_per_octave : TODO
          TODO
        fmin : TODO
          TODO
        kwargs : TODO
          TODO
        """

        super().__init__(**kwargs)

        self.hop_length = hop_length

        if fmin is None:
            # Default minimum frequency to note C1
            fmin = librosa.note_to_midi('C1')

        # Determine the MIDI frequency of the highest bin
        fmax = fmin + (n_bins - 1) / (bins_per_octave / 12)

        # Compute the center frequencies for all bins
        self.center_freqs = np.linspace(fmin, fmax, n_bins)

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
    def get_ground_truth(self, track, times):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string (unused)
          Track name
        times : TODO (unused)
          TODO

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Construct an empty array of relevant size by default
        ground_truth = np.zeros((len(self.center_freqs), len(times)))

        return ground_truth

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

        # Compute the number of frames as the number of hops
        num_frames = 1 + audio.size(-1) // self.hop_length

        # Determine the time associated with each frame (center)
        times = (self.hop_length / self.sample_rate) * np.arange(num_frames)

        # TODO
        ground_truth = self.get_ground_truth(self.tracks[index], times)

        # Convert ground-truth to tensor format and add to appropriate device
        ground_truth = torch.from_numpy(ground_truth).to(self.device).float()

        return audio, ground_truth
