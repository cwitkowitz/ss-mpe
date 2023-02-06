# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import seed_everything, normalize
from model import PlaceHolderNet
from NSynth import NSynth

# Regular imports
from torch.utils.data import DataLoader

import torch
import os


# TODO - try using apex for faster training?

seed = 0
seed_everything(seed)

dataset_base_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets', 'NSynth')

nsynth = NSynth(base_dir=dataset_base_dir, seed=seed)

loader = DataLoader(nsynth, batch_size=10, shuffle=True, num_workers=0, drop_last=False)

model = PlaceHolderNet()

while True:
    for batch in loader:
        isolated_embeddings = model(batch)

        mixtures = batch.unsqueeze(0).repeat(10, 1, 1) + \
                   batch.unsqueeze(1).repeat(1, 10, 1)

        mixture_idcs_r = torch.arange(10).unsqueeze(0).repeat(10, 1).reshape(10 * 10)
        mixture_idcs_c = torch.arange(10).unsqueeze(1).repeat(1, 10).reshape(10 * 10)

        # TODO - try random mixtures instead of pairwise?

        mixtures = mixtures.reshape(10 * 10, -1)

        mixtures = mixtures / 2

        mixture_embeddings = model(mixtures)

        # TODO - compute loss between isolated and mixed embeddings

        pass
