# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'compute_energy_loss',
    'compute_support_loss',
    'compute_harmonic_loss',
    'compute_sparsity_loss',
    'compute_entropy_loss',
    'compute_content_loss'
]


def compute_energy_loss(embeddings, salience):
    # Compute energy loss as BCE of activations with respect to features
    energy_loss = F.binary_cross_entropy_with_logits(embeddings, salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    energy_loss = energy_loss.sum(-2).mean(-1).mean(-1)

    return energy_loss


def compute_support_loss(embeddings, h1_features):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute support loss as BCE of activations with respect to features (negative activations only)
    support_loss = F.binary_cross_entropy_with_logits(embeddings, h1_features, reduction='none', pos_weight=pos_weight)

    # Sum across frequency bins and average across time and batch
    support_loss = support_loss.sum(-2).mean(-1).mean(-1)

    return support_loss


def compute_harmonic_loss(embeddings, salience):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute harmonic loss as BCE of activations with respect to salience estimate (positive activations only)
    harmonic_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - salience), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    harmonic_loss = harmonic_loss.sum(-2).mean(-1).mean(-1)

    return harmonic_loss


def compute_sparsity_loss(activations):
    # Compute sparsity loss as the L1 norm of the activations
    sparsity_loss = torch.norm(activations, 1, dim=-2)

    # Average loss across time and batch
    sparsity_loss = sparsity_loss.mean(-1).mean(-1)

    return sparsity_loss


def compute_entropy_loss(embeddings):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute entropy as BCE of activations with respect to themselves (positive activations only)
    entropy_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - torch.sigmoid(embeddings)), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    entropy_loss = entropy_loss.sum(-2).mean(-1).mean(-1)

    return entropy_loss


# TODO - can scale by amount of energy in input
def compute_content_loss(activations, lmbda=5):
    # Determine the maximum activation within each frame
    max_activations = torch.max(activations, dim=-2)[0]

    # Compute content loss as likelihood of exponential distribution of the maximum activations
    content_loss = lmbda * torch.exp(-lmbda * max_activations)

    # Average loss across time and batch
    content_loss = content_loss.mean(-1).mean(-1)

    return content_loss
