# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.utils import constants

# Regular imports
from torch.utils.data import DataLoader
from copy import deepcopy

import torch.nn.functional as F
import torch


__all__ = [
    'drop_random_channels',
    'compute_channel_loss'
]


def drop_random_channels(features):
    # Perform channel-wise dropout and correct values
    dropped_features = F.dropout2d(0.5 * features)

    return dropped_features


def compute_channel_loss(model, features, targets):
    # Randomly drop harmonic channels of features
    dropped_features = drop_random_channels(features)

    # Process dropped features with provided model
    dropped_embeddings = model(dropped_features)[0]

    # Compute channel loss as BCE of embeddings computed from dropped features with respect to original targets
    channel_loss = F.binary_cross_entropy_with_logits(dropped_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    channel_loss = channel_loss.sum(-2).mean(-1).mean(-1)

    return channel_loss
