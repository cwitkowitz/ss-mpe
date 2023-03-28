# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import normalize

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod

import numpy as np
#import torchaudio # TODO - audio/ground-truth loading w/ torchaudio?
import warnings
import librosa
import scipy
import torch
import os


DEFAULT_LOCATION = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets')


class TrainSet(Dataset):
    """
    Implements a wrapper for an MPE dataset intended for training.
    """

    def __init__(self, base_dir=None, splits=None, sample_rate=16000, n_secs=None, seed=0):
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
        n_secs : TODO
          TODO
        seed : TODO
          TODO
        """

        if base_dir is None:
            base_dir = os.path.join(DEFAULT_LOCATION, self.name())

        self.base_dir = base_dir

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset if it is missing
            self.download(self.base_dir)

        if splits is None:
            # Choose all available dataset splits
            splits = self.available_splits()

        self.tracks = []
        # Aggregate all the track names from the selected splits
        for split in splits:
            self.tracks += self.get_tracks(split)

        self.sample_rate = sample_rate
        self.n_secs = n_secs

        # Initialize a random number generator for the dataset
        self.rng = np.random.RandomState(seed)

    @classmethod
    def name(cls):
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
        #audio, fs = torchaudio.load(audio_path)
        #audio = torch.mean(audio, dim=0, keepdim=True)
        #audio = torchaudio.functional.resample(audio, fs, self.sample_rate)

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

        # Determine corresponding track
        track = self.tracks[index]

        try:
            # Attempt to read the track
            audio = self.get_audio(track)
        except Exception as e:
            # Print offending track to console
            print(f'Error loading track \'{track}\'...')

            # Default audio to silence
            audio = np.empty(0)
            #audio = torch.empty((1, 0))

        if self.n_secs is not None:
            # Determine the required sequence length
            n_samples = int(self.n_secs * self.sample_rate)

            if len(audio) >= n_samples:
            #if audio.size(-1) >= n_samples:
                # Sample a random starting index for the trim
                start = self.rng.randint(0, len(audio) - n_samples + 1)
                #start = self.rng.randint(0, audio.size(-1) - n_samples + 1)
                # Trim audio to the sequence length
                audio = audio[start : start + n_samples]
                #audio = audio[..., start : start + n_samples]
            else:
                # Determine how much padding is required
                pad_total = n_samples - len(audio)
                #pad_total = n_samples - audio.size(-1)
                # Randomly distributed between both sides
                pad_left = self.rng.randint(0, pad_total)
                # Pad the audio with zeros
                audio = np.pad(audio, (pad_left, pad_total - pad_left))
                #audio = torch.nn.functional.pad(audio, (pad_left, pad_total - pad_left))

        # Convert audio to tensor with float type
        audio = torch.from_numpy(audio).float()

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

    def __init__(self, hop_length, fmin, n_bins, bins_per_octave, **kwargs):
        """
        TODO.

        Parameters
        ----------
        hop_length : TODO
          TODO
        fmin : TODO
          TODO
        n_bins : TODO
          TODO
        bins_per_octave : TODO
          TODO
        kwargs : TODO
          TODO
        """

        super().__init__(**kwargs)

        self.hop_length = hop_length

        # Determine the MIDI frequency of the highest bin
        fmax = fmin + (n_bins - 1) / (bins_per_octave / 12)

        # Compute the center frequencies for all bins
        self.center_freqs = np.linspace(fmin, fmax, n_bins)

        # Obtain a function to resample annotation frequencies
        self.res_func_freq = scipy.interpolate.interp1d(x=self.center_freqs,
                                                        y=np.arange(n_bins),
                                                        kind='nearest',
                                                        bounds_error=False,
                                                        assume_sorted=True)

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
        n_frames = 1 + audio.size(-1) // self.hop_length

        # Determine the time associated with each frame (center)
        times = librosa.frames_to_time(np.arange(n_frames),
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length)

        # TODO
        ground_truth = self.get_ground_truth(self.tracks[index], times)

        # Convert ground-truth to tensor with float type
        ground_truth = torch.from_numpy(ground_truth).float()

        return audio, ground_truth


class ComboSet(TrainSet):
    """
    Support for training with multiple datasets.
    """

    def __init__(self, datasets):
        """
        TODO.

        Parameters
        ----------
        datasets : list of TrainSets
          Pre-initialized datasets from which to sample
        """

        self.datasets = datasets

    def __len__(self):
        """
        Number of samples across all datasets.

        Returns
        ----------
        length : int
          TODO
        """

        # Add together length of all constituent datasets
        length = sum([d.__len__() for d in self.datasets])

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
        audio : TODO
          TODO
        """

        local_idx, dataset_idx = index, 0

        while local_idx >= self.datasets[dataset_idx].__len__():
            # Remove the index offset from this dataset
            local_idx -= self.datasets[dataset_idx].__len__()
            # Check the next dataset
            dataset_idx += 1

        # Get the audio at the sampled dataset's local index
        audio = self.datasets[dataset_idx].__getitem__(local_idx)

        return audio
