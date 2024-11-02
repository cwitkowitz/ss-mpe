# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F


__all__ = [
    'drop_random_channels',
    'compute_channel_loss'
]


def drop_random_channels(features):
    # Perform dropout and correct values
    #dropped_features = F.dropout(0.5 * features) # Time-frequency bin -wise
    #dropped_features = F.dropout2d(0.5 * features.transpose(-2, -3)).transpose(-2, -3) # Frequency bin -wise
    dropped_features = F.dropout2d(0.5 * features) # Channel (harmonic) -wise

    return dropped_features


def compute_channel_loss(model, features, targets):
    # Randomly drop harmonic channels of features
    dropped_features = drop_random_channels(features)

    # Process dropped features with provided model
    dropped_embeddings = model(dropped_features)

    # Compute channel loss as BCE of embeddings computed from dropped features with respect to original targets
    channel_loss = F.binary_cross_entropy_with_logits(dropped_embeddings, targets.detach(), reduction='none')

    # Sum across frequency bins and average across time and batch
    channel_loss = channel_loss.sum(-2).mean(-1).mean(-1)

    return channel_loss
