# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F


__all__ = [
    'drop_random_features',
    'compute_feature_loss'
]


def drop_random_features(features, mode=0):
    if mode == 0:
        # Perform dropout channel (harmonic) -wise
        dropped_features = F.dropout2d(0.5 * features)
    elif mode == 1:
        # Perform dropout frequency bin -wise
        dropped_features = F.dropout2d(0.5 * features.transpose(-2, -3)).transpose(-2, -3)
    else:
        # Perform dropout time-frequency bin -wise
        dropped_features = F.dropout(0.5 * features)

    return dropped_features


def compute_feature_loss(model, features, targets, **dp_kwargs):
    # Randomly drop channels, frequencies, or bins of features
    dropped_features = drop_random_features(features, **dp_kwargs)

    # Process dropped features with provided model
    dropped_embeddings = model(dropped_features)

    # Compute feature loss as BCE of embeddings computed from dropped features with respect to original targets
    feature_loss = F.binary_cross_entropy_with_logits(dropped_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    feature_loss = feature_loss.sum(-2).mean(-1).mean(-1)

    return feature_loss
