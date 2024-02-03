# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ss_mpe.datasets.SoloMultiPitch import NSynth

from timbre_trap.datasets.utils import constants
from ss_mpe.models import SS_MPE

# Regular imports
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import sys
import os


# Name of the model to evaluate
ex_name = 'Energy'

# Choose the model checkpoint to compare
checkpoint = 43000

# Choose the GPU on which to perform evaluation
gpu_id = None

# File layout of system (0 - desktop | 1 - lab)
path_layout = 0

# Construct the path to the top-level directory of the experiment
if path_layout == 1:
    experiment_dir = os.path.join('/', 'storage', 'frank', 'self-supervised-pitch', ex_name)
else:
    experiment_dir = os.path.join('..', '..', 'generated', 'experiments', ex_name)

# Import utilities from parent directory
sys.path.insert(0, os.path.join('..'))
from utils import *

# Set randomization seed
seed = 0

# Seed everything with the same seed
seed_everything(seed)


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050


###########
## MODEL ##
###########

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the SS-MPE model
ss_mpe = SS_MPE.load(model_path, device=device)
ss_mpe.eval()


#############
## DATASET ##
#############

# Instantiate NSynth validation split
nsynth_val = NSynth(base_dir=None,
                    splits=['valid'],
                    n_tracks=200,
                    sample_rate=sample_rate,
                    cqt=ss_mpe.hcqt,
                    seed=seed)

# Sample from which to start NSynth analysis
nsynth_start_idx = 0

# Slice NSynth dataset
nsynth_val.tracks = nsynth_val.tracks[nsynth_start_idx:]


###################
## Visualization ##
###################

# Create a directory for saving visualized harmonics
vis_dir = os.path.join('..', '..', 'generated', 'visualization', 'harmonics')

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(nsynth_val)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract relevant feature sets
    features_db   = features['db'].squeeze(0)
    features_db_1 = features['db_1'].squeeze(0)
    features_db_h = features['db_h'].squeeze(0)

    # Transcribe the audio using the SS-MPE model
    ss_activations = to_array(ss_mpe.transcribe(audio).squeeze())

    # Extract ground-truth pitch salience activations
    gt_activations = data[constants.KEY_GROUND_TRUTH]

    # Determine track's attributes
    name, pitch, vel = track.split('-')
    # Add a global title above all sub-plots
    #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

    # Replace / in the track name
    track = track.replace('/', '-')
    # Create a directory for saving visualized harmonics
    save_dir = os.path.join(vis_dir, track)
    # Create the root directory
    os.makedirs(save_dir, exist_ok=True)

    # Loop through harmonic CQT features
    for (h, cqt) in zip(ss_mpe.hcqt.harmonics, features_db):
        # Open a new figure
        fig = initialize_figure(figsize=(4, 3))
        # Plot spectral features
        fig = plot_magnitude(cqt, fig=fig)
        # Minimize free space
        fig.tight_layout()
        # Construct path under visualization directory
        save_path = os.path.join(save_dir, f'{track}_h{h}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Loop through custom features, output, and ground-truth
    for (h, cqt) in zip(['h1\'', 'wavg', 'out', 'pgt'],
                        [features_db_1, features_db_h,
                         ss_activations, gt_activations]):
        # Open a new figure
        fig = initialize_figure(figsize=(4, 3))
        # Plot spectral features
        fig = plot_magnitude(cqt, fig=fig)
        # Minimize free space
        fig.tight_layout()
        # Construct path under visualization directory
        save_path = os.path.join(save_dir, f'{track}_{h}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    """
    # Open the figure manually
    plt.show(block=False)

    # Wait for keyboard input
    while plt.waitforbuttonpress() != True:
        continue
    """

    # Close figure
    plt.close('all')
