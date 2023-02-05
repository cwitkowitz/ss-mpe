# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, untar_and_remove

# Regular imports
from torch.utils.data import Dataset

import shutil
import os


class NSynth(Dataset):
    """
    Implements a wrapper for the NSynth dataset (https://magenta.tensorflow.org/datasets/nsynth).
    """

    def __init__(self, base_dir, seed):
        """
        TODO.

        Parameters
        ----------
        base_dir : TODO
          TODO
        seed : TODO
          TODO
        """

        # TODO
        pass

    def __len__(self):
        """
        Defines the notion of length for the dataset - used by PyTorch Dataset class.

        Returns
        ----------
        length : int
          TODO
        """

        # TODO
        length = None

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
        data : TODO
          TODO
        """

        # TODO
        data = None

        return data

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
