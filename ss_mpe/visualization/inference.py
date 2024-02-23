# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import Bach10, URMP, Su, TRIOS
from timbre_trap.datasets.SoloMultiPitch import GuitarSet

from timbre_trap.datasets.utils import constants
from ss_mpe.models import SS_MPE

# Regular imports
from scipy.signal import convolve2d
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os


# Experiment dictionary [tag, checkpoint]
experiments = {
    'SS-MPE' : ['SS-MPE', 37000],
    'Timbre' : ['Timbre-Only', 41750],
    'Geometric' : ['Geometric-Only', 37000],
    'Energy' : ['Energy-Only', 43000]
}

# Choose the GPU on which to perform evaluation
gpu_id = None

# Flag to print results for each track separately
verbose = True

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiments
if path_layout == 1:
    experiments_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch')
else:
    experiments_dir = os.path.join('..', '..', 'generated', 'experiments')

# Import utilities from parent directory
sys.path.insert(0, os.path.join('..'))
from utils import *


###########
## MODEL ##
###########

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Initialize a dictionary to hold the models
models = dict()

for (name, details) in experiments.items():
    # Construct the path to the model checkpoint to evaluate
    model_path = os.path.join(experiments_dir, name, 'models', f'model-{details[1]}.pt')

    # Load the model at the specified checkpoint
    model = SS_MPE.load(model_path, device=device)
    model.eval()

    # Add the loaded model to dictionary
    models.update({details[0] : model})

########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050

# Get HCQT of last model (should all be the same)
hcqt = model.hcqt

# Extract necessary parameters
fmin = hcqt.get_midi_freqs()[0]
fmax = fmin + hcqt.n_bins / (hcqt.bins_per_octave / 12)


#############
## DATASET ##
#############

# Point to the datasets within the storage drive containing them or use the default location
bch10_base_dir = os.path.join('/', 'storage', 'frank', 'Bach10') if path_layout else None
urmp_base_dir  = os.path.join('/', 'storage', 'frank', 'URMP') if path_layout else None
su_base_dir    = os.path.join('/', 'storage', 'frank', 'Su') if path_layout else None
trios_base_dir = os.path.join('/', 'storage', 'frank', 'TRIOS') if path_layout else None
gset_base_dir  = os.path.join('/', 'storage', 'frank', 'GuitarSet') if path_layout else None

# Instantiate Bach10 dataset mixtures for inference
bch10_test = Bach10(base_dir=bch10_base_dir,
                    splits=None,
                    sample_rate=sample_rate,
                    cqt=hcqt)

# Instantiate URMP dataset mixtures for inference
urmp_test = URMP(base_dir=urmp_base_dir,
                 splits=None,
                 sample_rate=sample_rate,
                 cqt=hcqt)

# Instantiate Su dataset for inference
su_test = Su(base_dir=su_base_dir,
             splits=None,
             sample_rate=sample_rate,
             cqt=hcqt)

# Instantiate TRIOS dataset for inference
trios_test = TRIOS(base_dir=trios_base_dir,
                   splits=None,
                   sample_rate=sample_rate,
                   cqt=hcqt)

# Instantiate GuitarSet dataset for inference
gset_test = GuitarSet(base_dir=gset_base_dir,
                      splits=None,
                      sample_rate=sample_rate,
                      cqt=hcqt)


################
## EVALUATION ##
################

# Create a directory for saving inference visualization
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'inference')

# Create the visualization directory
os.makedirs(save_dir, exist_ok=True)

# Loop through inference datasets
for eval_set in [bch10_test]:
    # Loop through all tracks in the test set
    for i, data in enumerate(tqdm(eval_set)):
        # Determine which track is being processed
        track = data[constants.KEY_TRACK]
        # Extract audio and add to the appropriate device
        audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)

        # Compute full set of spectral features
        features = model.get_all_features(audio)

        # Extract first harmonic of input HCQT
        features_db_1 = features['db_1'].squeeze(0)
        features_db_h = features['db_h'].squeeze(0)

        # Extract ground-truth pitch salience activations
        gt_activations = data[constants.KEY_GROUND_TRUTH]

        # Widen the activations for easier visualization
        gt_activations = convolve2d(gt_activations,
                                    np.array([[1, 1, 1]]).T, 'same')

        # Make sure there are enough rows
        n_rows = round(0.5 * (len(models) + 2))

        # Initialize a new figure with subplots
        (fig, ax) = plt.subplots(nrows=n_rows, ncols=2, figsize=(18, 3 * n_rows))

        # Determine track's attributes
        #name, pitch, vel = track.split('-')
        # Add a global title above all sub-plots
        #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

        # Define extent for magnitude plots
        extent_midi = [0, audio.size(-1) / sample_rate, fmin, fmax]

        # Plot first-harmonic CQT features
        fig.sca(ax[0, 0])
        plot_magnitude(features_db_1, extent=extent_midi, fig=fig)
        # Axis management
        ax[0, 0].set_title('$X_1$')
        ax[0, 0].axis('on')
        ax[0, 0].set_xlabel('')
        ax[0, 0].set_ylabel('Frequency (MIDI)')
        ax[0, 0].set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
        #ax[0, 0].grid()

        """
        # Plot weighted average of HCQT features
        fig.sca(ax[0, 1])
        plot_magnitude(features_db_h, extent=extent_midi, fig=fig)
        # Axis management
        ax[0, 1].set_title('$X_{h_w}$')
        ax[0, 1].axis('on')
        ax[0, 1].set_xlabel('')
        ax[0, 1].set_ylabel('')
        ax[0, 1].set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
        ax[0, 1].set_yticklabels(['' for t in ax[0, 1].get_yticks()])
        ax[0, 1].grid()
        """

        # Plot ground-truth activations
        fig.sca(ax[0, 1])
        plot_magnitude(gt_activations, extent=extent_midi, fig=fig)
        # Axis management
        ax[0, 1].set_title('Ground-Truth')
        ax[0, 1].axis('on')
        ax[0, 1].set_xlabel('')
        ax[0, 1].set_ylabel('')
        ax[0, 1].set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
        ax[0, 1].set_yticklabels(['' for t in ax[0, 1].get_yticks()])
        #ax[0, 1].grid()

        for i, (tag, model) in enumerate(models.items()):
            # Transcribe the audio using the SS-MPE model
            ss_activations = to_array(model.transcribe(audio).squeeze(0))

            # Offset subplot
            k = i
            #k = i + 1
            # Obtain a reference to current axis
            ax_curr = ax[1 + k // 2, k % 2]
            # Plot multi-pitch salience-gram
            fig.sca(ax_curr)
            plot_magnitude(ss_activations, extent=extent_midi, fig=fig)
            # Axis management
            ax_curr.set_title(f'{tag}')
            ax_curr.axis('on')
            ax_curr.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
            if k % 2 == 0:
                ax_curr.set_ylabel('Frequency (MIDI)')
            else:
                ax_curr.set_ylabel('')
                ax_curr.set_yticklabels(['' for t in ax_curr.get_yticks()])
            if (k // 2) == n_rows - 2:
                ax_curr.set_xlabel('Time (s)')
            else:
                ax_curr.set_xlabel('')
            #ax_curr.grid()

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
            # Construct path under visualization directory
            save_path = os.path.join(save_dir, f'{track}.pdf')
            # Save the figure with minimal whitespace
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # Close figure
        plt.close(fig)
