# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.utils import *

# Regular imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# Experiment directories with loss values to plot
experiments = ['SS-MPE', 'Timbre', 'Geometric', 'Energy']

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiments
if path_layout == 1:
    experiments_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch')
else:
    experiments_dir = os.path.join('..', '..', 'generated', 'experiments')

# Open a new figure
fig = initialize_figure(figsize=(12, 6))
ax = fig.gca()

for exp in experiments:
    # Construct the path to the model losses to plot
    losses_path = os.path.join(experiments_dir, exp, 'models', f'{exp}.csv')

    # Load the CSV data into an array
    steps, losses = np.array(pd.read_csv(losses_path, usecols=['Step', 'Value'])).T

    # Clip at end of third epoch
    losses = losses[steps <= 43000]
    steps  = steps[steps <= 43000]

    # Draw a line plot
    fig.gca().plot(steps, losses, label=exp)


# Axis management
ax.set_xlabel('Epochs')
ax.set_ylabel('Total Loss')
checkpoint_interval = 250
iter_per_epoch = 14355
ax.set_xlim([0, 3 * iter_per_epoch + checkpoint_interval])
ax.set_xticks([0, iter_per_epoch, 2 * iter_per_epoch, 3 * iter_per_epoch])
ax.set_xticklabels([0, 1, 2, 3])
ax.set_ylim([150, 350])
ax.legend()

# Open the figure manually
plt.show(block=False)

# Wait for keyboard input
while plt.waitforbuttonpress() != True:
    continue

# Prompt user to save figure
save = input('Save figure? (y/n)')

if save == 'y':
    # Create a directory for saving visualized loss curves
    save_dir = os.path.join('..', '..', 'generated', 'visualization')
    # Construct save path under visualization directory
    save_path = os.path.join(save_dir, f'loss_curves.pdf')
    # Save the figure with minimal whitespace
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

# Close figure
plt.close(fig)
