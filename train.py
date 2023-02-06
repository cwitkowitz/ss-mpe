# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import seed_everything, normalize
from model import PlaceHolderNet
from NSynth import NSynth

# Regular imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torch
import os


# TODO - try using apex for faster training?

seed = 0
seed_everything(seed)

dataset_base_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets', 'NSynth')

nsynth = NSynth(base_dir=dataset_base_dir, seed=seed)

batch_size = 10
loader = DataLoader(nsynth, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

model = PlaceHolderNet()

# Initialize an optimizer for the model parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)

writer = SummaryWriter()

iter = 0

while True:
    for batch in loader:
        isolated_embeddings = model(batch)

        mixtures = batch.unsqueeze(0).repeat(batch_size, 1, 1) + \
                   batch.unsqueeze(1).repeat(1, batch_size, 1)

        mixture_idcs_r = torch.arange(batch_size).unsqueeze(0).repeat(batch_size, 1).flatten()
        mixture_idcs_c = torch.arange(batch_size).unsqueeze(1).repeat(1, batch_size).flatten()

        # TODO - try random mixtures instead of pairwise?

        mixtures = mixtures.reshape(batch_size ** 2, -1)

        mixtures = mixtures / 2

        mixture_embeddings = model(mixtures)

        pair_weights = 1 - torch.eye(batch_size).flatten()

        target_embeddings = isolated_embeddings[mixture_idcs_r] + \
                            isolated_embeddings[mixture_idcs_c]

        pair_losses = torch.nn.functional.mse_loss(mixture_embeddings, target_embeddings, reduction='none')

        loss = torch.mean(pair_weights * pair_losses.sum(-1).mean(-1)) / 2

        writer.add_scalar('train/loss', loss, iter)

        print(f'iter: {iter} | loss: {loss}')

        # Zero the accumulated gradients
        optimizer.zero_grad()
        # Compute gradients based on total loss
        loss.backward()
        # Perform an optimization step
        optimizer.step()

        iter += 1
