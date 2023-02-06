# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, untar_and_remove

# Regular imports
from torch.utils.data import Dataset

import soundfile as sf
import numpy as np
import warnings
import shutil
import json
import os

# TODO - define transcription-relevant and transcription-irrelevant transformations


class NSynth(Dataset):
    """
    Implements a wrapper for the NSynth dataset (https://magenta.tensorflow.org/datasets/nsynth).
    """

    def __init__(self, base_dir, splits=None, seed=0):
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

        self.base_dir = base_dir

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset if it is missing and if a procedure exists
            self.download(self.base_dir)

        # Choose all available dataset splits if none were provided
        if splits is None:
            splits = self.available_splits()

        # Initialize a random number generator for the dataset
        self.rng = np.random.RandomState(seed)

        self.tracks = []
        # Aggregate all the track names from the selected splits
        for split in splits:
            self.tracks += self.get_tracks(split)

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
        audio : TODO
          TODO
        """

        # TODO
        audio, _ = sf.read(self.get_wav_path(self.tracks[index]))

        return audio

    def get_tracks(self, split):
        """
        Get the track names associated with a partition of the dataset.

        Parameters
        ----------
        split : string
          TODO

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Construct a path to the JSON annotations for the partition
        json_path = os.path.join(self.base_dir, f'nsynth-{split}', 'examples.json')

        with open(json_path) as f:
            # Read JSON data
            tracks = json.load(f)

        # Retain the names of the tracks
        tracks = list(tracks.keys())

        # Append the split name to all tracks
        tracks = [os.path.join(f'nsynth-{split}', t) for t in tracks]

        return tracks

    def get_wav_path(self, track):
        """
        Get the path a track's audio.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, split, 'audio', name + '.wav')

        return wav_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset for different stages of pipeline
        """

        splits = ['train', 'valid', 'test']

        return splits

    @classmethod
    def download(cls, save_dir):
        """
        Download the NSynth dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of NSynth
        """

        # If the directory already exists, remove it
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        # Create the base directory
        os.makedirs(save_dir)

        print(f'Downloading {cls.__name__}')

        for split in cls.available_splits():
            # URL pointing to the zip file for the split
            url = f'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{split}.jsonwav.tar.gz'

            # Construct a path for saving the file
            save_path = os.path.join(save_dir, os.path.basename(url))

            # Download the zip file
            stream_url_resource(url, save_path, 1000 * 1024)

            # Unzip the downloaded file and remove it
            untar_and_remove(save_path)
