# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ss_mpe.datasets.SoloMultiPitch import NSynth

from ss_mpe.models.objectives import sample_gaussian_equalization
from timbre_trap.datasets.utils import constants
from ss_mpe.models import TT_Base

# Regular imports
import matplotlib.pyplot as plt
from tqdm import tqdm

import librosa
import torch
import math
import sys
import os


# Set number of curves per methodology
n_curves = 3

# Set randomization seed
seed = 0

# Import utilities from parent directory
sys.path.insert(0, os.path.join('..'))
from utils import *

# Seed everything with the same seed
seed_everything(seed)


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050

# Number of samples between frames
hop_length = 256

# First center frequency (MIDI) of geometric progression
fmin = librosa.note_to_midi('A0')

# Number of bins in a single octave
bins_per_octave = 60

# Number of frequency bins per CQT
n_bins = 440

# Harmonics to stack along channel dimension of HCQT
harmonics = [0.5, 1, 2, 3, 4, 5]
# Create weighting for harmonics (harmonic loss)
harmonic_weights = 1 / torch.Tensor(harmonics) ** 2
# Apply zero weight to sub-harmonics (harmonic loss)
harmonic_weights[harmonic_weights > 1] = 0
# Normalize the harmonic weights
harmonic_weights /= torch.sum(harmonic_weights)
# Add frequency and time dimensions for broadcasting
harmonic_weights = harmonic_weights.unsqueeze(-1).unsqueeze(-1)

# Pack together HCQT parameters for readability
hcqt_params = {'sample_rate': sample_rate,
               'hop_length': hop_length,
               'fmin': fmin,
               'bins_per_octave': bins_per_octave,
               'n_bins': n_bins,
               'gamma': None,
               'harmonics': harmonics,
               'weights': harmonic_weights}

# Determine maximum supported MIDI frequency
fmax = fmin + n_bins / (bins_per_octave / 12)


###########
## MODEL ##
###########

# Initialize autoencoder model
ss_mpe = TT_Base(hcqt_params,
                latent_size=128,
                model_complexity=2,
                skip_connections=False)


##############
## DATASETS ##
##############

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

# Create a directory for saving visualized equalizations
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'equalization')

# Create the visualization directory
os.makedirs(save_dir, exist_ok=True)

# Maximum amplitude for Gaussian equalization
max_A = 0.375

# Maximum standard deviation for Gaussian equalization
max_std_dev = 2 * bins_per_octave

# Determine how many octaves have been covered
n_octaves = int(math.ceil(n_bins / bins_per_octave))

# Cover the full octave for proper interpolation
n_out = n_octaves * bins_per_octave

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(nsynth_val)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract first harmonic CQT spectral features
    features_db_1 = to_array(features['db_1'][0])

    # Sample parametric Gaussian equalization curves
    gaussian_curves = sample_gaussian_equalization(n_out,
                                                   batch_size=n_curves,
                                                   max_A=max_A,
                                                   max_std_dev=max_std_dev)

    # Initialize a new figure with subplots
    (fig, ax) = plt.subplots(nrows=n_curves, ncols=3, width_ratios=[2, 1, 2], figsize=(6.666, 3 * n_curves))

    # Determine track's attributes
    name, pitch, vel = track.split('-')
    # Add a global title above all sub-plots
    #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

    # Define extent for magnitude plots
    extent_midi = [0, 4, fmin, fmax]
    # Define ticks for frequencies in MIDI
    midi_ticks = [30, 40, 50, 60, 70, 80, 90, 100]

    for i, curve in enumerate(to_array(gaussian_curves)):
        # Plot equalized features and curve
        plot_equalization(features_db_1, curve[:n_bins], ax[i])
        # Ticks and labels
        ax[i, 0].set_title('')
        ax[i, 1].set_title('')
        ax[i, 2].set_title('')
        ax[i, 0].axis('on')
        ax[i, 0].set_ylabel('Frequency (MIDI)')
        ax[i, 0].get_images()[0].set_extent(extent_midi)
        ax[i, 0].set_yticks(midi_ticks)
        ax[i, 1].lines[0].set_linewidth(2)
        ax[i, 2].axis('on')
        ax[i, 2].set_ylabel('')
        ax[i, 2].get_images()[0].set_extent(extent_midi)
        ax[i, 2].set_yticks(midi_ticks)
        ax[i, 2].set_yticklabels(['' for t in midi_ticks])
    ax[i, 0].set_xlabel('Time (s)')
    ax[i, 1].set_xlabel('Scaling')
    ax[i, 2].set_xlabel('Time (s)')

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
        # Replace / in the track name
        track = track.replace('/', '-')
        # Construct path under visualization directory
        save_path = os.path.join(save_dir, f'{track}_{seed}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Close figure
    plt.close(fig)
