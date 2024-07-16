# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'compute_supervised_loss'
]


def compute_supervised_loss(embeddings, ground_truth, weight_positive_class=False):
    # Compute supervised loss as BCE of activations with respect to ground-truth
    supervised_loss = F.binary_cross_entropy_with_logits(embeddings, ground_truth, reduction='none')

    if weight_positive_class:
        # Sum ground-truth and its complement for weights
        positive_weight = torch.sum(ground_truth, dim=-2, keepdim=True)
        negative_weight = torch.sum(1 - ground_truth, dim=-2, keepdim=True)
        # Compute multi-class imbalance ratio for each frame
        positive_scaling = negative_weight / (positive_weight + torch.finfo().eps)
        # Determine scaling for each loss element
        scaling = positive_scaling * (ground_truth == 1)
        # Correct scaling for negative activations
        scaling[scaling == 0] = 1
        # Scale transcription loss
        supervised_loss *= scaling

    # Sum across frequency bins and average across time and batch
    supervised_loss = supervised_loss.sum(-2).mean(-1).mean(-1)

    return supervised_loss
