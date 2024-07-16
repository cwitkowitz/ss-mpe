# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.utils import constants

# Regular imports
from torch.utils.data import DataLoader
from copy import deepcopy

import torch.nn.functional as F
import torch


__all__ = [
    'mix_random_percussion',
    'compute_percussion_loss'
]


def mix_random_percussion(audio, percussive_set_combo, max_volume=1.0, n_workers=0):
    # Determine batch size
    B = audio.size(0)

    # Create copy of datasets to avoid overwriting attributes
    percussive_set_combo = deepcopy(percussive_set_combo)

    for d in percussive_set_combo.datasets:
        # TODO - add function to ComboDataset?
        # Update sequence length to match audio
        d.n_secs = audio.size(-1) / d.sample_rate

    # Initialize a PyTorch dataloader for percussion data
    loader_pc = DataLoader(dataset=percussive_set_combo,
                           batch_size=B,
                           shuffle=True,
                           num_workers=n_workers,
                           pin_memory=True,
                           drop_last=True)

    # Sample a batch of percussive data
    percussive_data = next(iter(loader_pc))
    # Extract audio from the sampled data and add to the appropriate device
    percussive_audio = percussive_data[constants.KEY_AUDIO].to(audio.device)

    # Sample random volumes for percussion audio
    volumes = max_volume * torch.rand((B, 1, 1), device=audio.device)

    # Mix sampled percussive audio with original audio
    mixtures = audio + volumes * percussive_audio

    return mixtures


def compute_percussion_loss(model, audio, targets, **pc_kwargs):
    # Superimpose random percussive audio onto original audio
    percussive_audio = mix_random_percussion(audio, **pc_kwargs)

    # Compute spectral features for percussive audio mixture
    percussive_features = model.hcqt.to_decibels(model.hcqt(percussive_audio))

    # Process percussive features with provided model
    percussive_embeddings = model(percussive_features)

    # Compute percussion loss as BCE of embeddings computed from percussive features with respect to original targets
    percussion_loss = F.binary_cross_entropy_with_logits(percussive_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    percussion_loss = percussion_loss.sum(-2).mean(-1).mean(-1)

    return percussion_loss
