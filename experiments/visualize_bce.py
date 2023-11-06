# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import *

# Regular imports
import torch.nn.functional as F
import torch


def bce_full(est, ref):
    # Compute full binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(est, ref, reduction='none')

    return loss


def bce_neg(est, ref):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute negative-activation half of binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(est, ref, reduction='none', pos_weight=pos_weight)

    return loss


def bce_pos(est, ref):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute positive-activation half of binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(-est, (1 - ref), reduction='none', pos_weight=neg_weight)

    return loss


# Set number of steps
n_steps = 2000

# Create an array of potential ground-truth activations
ref = torch.linspace(0, 1, steps=n_steps + 1)
# Compute corresponding logits for valid estimates
est = torch.logit(ref.clone())[1:-1]

# Superimpose all combinations of activations
est, ref = torch.meshgrid(est, ref, indexing='xy')

# Compute each BCE loss variation
loss_full = to_array(bce_full(est, ref))
loss_neg = to_array(bce_neg(est, ref))
loss_pos = to_array(bce_pos(est, ref))

# Plot each loss landscape
fig_full = plot_bce_loss(loss_full, 'Full')
fig_neg = plot_bce_loss(loss_neg, 'Negative')
fig_pos = plot_bce_loss(loss_pos, 'Positive')

# Wait for user input
input('Press ENTER to finish...')
