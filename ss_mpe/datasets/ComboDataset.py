from torch.utils.data import Dataset
from random import shuffle

import numpy as np


class BalancedComboDataset(Dataset):
    """
    Wrapper to train with multiple pre-instantiated datasets.
    """

    def __init__(self, datasets, ratios=None):
        """
        Instantiate the combination wrapper.

        Parameters
        ----------
        datasets : list of BaseDataset
          Pre-instantiated datasets from which to sample
        ratios : list of float
          Relative ratios for sampling from each dataset
        """

        self.datasets = datasets

        #if len(datasets):
        if ratios is None:
            # Default to ratios for balanced sampling
            #ratios = [1 / len(datasets)] * len(datasets)
            ratios = [1.] * len(datasets)

        # Normalize sampling ratios to sum to 1
        #ratios = [r / sum(ratios) for r in ratios]
        ratios = [r / min(ratios) for r in ratios]

        self.ratios = ratios

    def shuffle_datasets(self):
        """
        Shuffle tracks in each dataset.
        """

        # Loop through each dataset
        for dataset in self.datasets:
            # Shuffle list of tracks
            shuffle(dataset.tracks)

    def __len__(self):
        """
        Number of effective samples per epoch.

        Returns
        ----------
        length : int
          Number of tracks per epoch
        """

        # Determine total number of tracks implied by ratio for each dataset
        tracks_by_ratio = [len(d) / r for d, r in zip(self.datasets, self.ratios)]

        # Choose minimum and normalize by sum of ratios
        length = int(min(tracks_by_ratio) * sum(self.ratios)) if len(tracks_by_ratio) else 0

        return length

    def __getitem__(self, index):
        """
        Extract the data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict containing
          track : string
            Identifier for the track
          ...
        """

        # Keep track of relative index
        local_idx, dataset_idx = index, 0

        # Determine number of tracks allocated per epoch to each dataset
        track_counts = len(self) * np.array(self.ratios) / sum(self.ratios)

        while local_idx >= int(track_counts[dataset_idx]):
            # Subtract number of tracks from global index
            local_idx -= int(track_counts[dataset_idx])
            # Check next dataset
            dataset_idx += 1

        # Sample data at the local index of selected dataset
        data = self.datasets[dataset_idx].__getitem__(local_idx)

        return data
