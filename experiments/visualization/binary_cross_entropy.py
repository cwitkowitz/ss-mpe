# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import sys
import os


def bce_full(est, ref):
    # Compute full binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(est, ref, reduction='none')

    #loss -= F.binary_cross_entropy(ref, ref, reduction='none')

    #loss = loss[1:-1]
    #loss += loss.clone().t()

    #est, gt = torch.tensor(0.001), torch.tensor(0.)
    #-(gt * torch.log(est) + (1 - gt) * torch.log(1 - est))

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


# Import utilities from parent directory
sys.path.insert(0, os.path.join('..'))
from utils import *

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

# Initialize a new figure with subplots if one was not given
(fig, ax) = plt.subplots(nrows=1, ncols=3, figsize=(9.5, 3), width_ratios=[1, 1, 1.25])

# Add a global title above all sub-plots
#fig.suptitle('BCE Loss Variants')

# Plot each BCE loss landscape
fig.sca(ax[0])
plot_bce_loss(loss_pos, fig=fig)
fig.sca(ax[1])
plot_bce_loss(loss_neg, fig=fig)
fig.sca(ax[2])
plot_bce_loss(loss_full, colorbar=True, fig=fig)
# Remove 2nd and 3rd y-axis labels
ax[1].set_ylabel('')
ax[2].set_ylabel('')

# Minimize free space
fig.tight_layout()

# Open the figure manually
plt.show(block=False)

# Wait for keyboard input
while plt.waitforbuttonpress() != True:
    continue

# Prompt user to save figure
save = input('Save figure? (y/n)')

if save == 'y':
    # Create a directory for saving visualized samples
    save_dir = os.path.join('..', '..', 'generated', 'visualization')
    # Construct path under visualization directory
    save_path = os.path.join(save_dir, f'BCE.pdf')
    # Save the figure with minimal whitespace
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

# Close figure
plt.close(fig)
