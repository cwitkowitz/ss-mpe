# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import seed_everything
from NSynth import NSynth

# Regular imports
from torch.utils.data import DataLoader

import os

seed = 0
seed_everything(seed)

dataset_base_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets', 'NSynth')

nsynth = NSynth(base_dir=dataset_base_dir, seed=seed)

loader = DataLoader(nsynth, batch_size=10, shuffle=True, num_workers=0, drop_last=False)

# TODO - initialize model

while True:
    for batch in loader:
        # TODO - make (pairwise or random?) mixtures
        #      - compute loss between embeddings
        pass
