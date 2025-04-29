# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .utils import gate_and_average_loss

# Regular imports
import torch.nn.functional as F
import torch


__all__ = [
    'compute_supervised_loss'
]


def compute_supervised_loss(embeddings, ground_truth, weight_positive_class=False, rms_vals=None, rms_thr=0.01):
    # Normalize activations for probabilities
    total_energy = ground_truth.sum(-2)
    ground_truth.transpose(-1, -2)[total_energy > 0] /= total_energy[total_energy > 0].unsqueeze(-1)

    ground_truth = torch.cat(((total_energy == 0).unsqueeze(-2), ground_truth), dim=-2)

    # Compute supervised loss as BCE of activations with respect to ground-truth
    #supervised_loss = F.binary_cross_entropy_with_logits(embeddings, ground_truth, reduction='none')
    supervised_loss = F.cross_entropy(embeddings, ground_truth, reduction='none')

    if weight_positive_class:
        # Sum binarized ground-truth and its complement for weights
        positive_weight = torch.sum(ground_truth, dim=-2, keepdim=True)
        #positive_weight = torch.sum(ground_truth.int(), dim=-2, keepdim=True)
        negative_weight = torch.sum(1 - ground_truth, dim=-2, keepdim=True)
        #negative_weight = torch.sum(1 - ground_truth.int(), dim=-2, keepdim=True)
        # Compute multi-class imbalance ratio for each frame
        positive_scaling = negative_weight / (positive_weight + torch.finfo().eps)
        # Determine scaling for each loss element
        scaling = positive_scaling * (ground_truth == 1)
        # Correct scaling for negative activations
        scaling[scaling == 0] = 1
        # Scale transcription loss
        supervised_loss *= scaling

    # Sum across frequency bins
    #supervised_loss = supervised_loss.sum(-2)

    if rms_vals is not None:
        # Gate based on RMS values and average across time and batch
        supervised_loss = gate_and_average_loss(supervised_loss, rms_vals, rms_thr)
    else:
        # Average across time and batch
        supervised_loss = supervised_loss.mean(-1).mean(-1)

    return supervised_loss
