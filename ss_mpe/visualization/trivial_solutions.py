# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures
from ss_mpe.datasets.AudioMixtures import E_GMD

from ss_mpe.objectives import *
from ss_mpe.framework import TT_Base
from timbre_trap.utils import *

# Regular imports
from scipy.signal import convolve2d
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import os


# Choose the GPU on which to perform evaluation
gpu_id = None

# Set randomization seed
seed = 0

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

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Initialize autoencoder model
ss_mpe = TT_Base(hcqt_params,
                latent_size=128,
                model_complexity=2,
                skip_connections=False)


#############
## DATASET ##
#############

# Set the URMP validation set in accordance with the MT3 paper
urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

# Allocate remaining tracks to URMP training set
urmp_train_splits = URMP_Mixtures.available_splits()

for t in urmp_val_splits:
    # Remove validation tracks
    urmp_train_splits.remove(t)

# Instantiate URMP dataset mixtures for training
urmp_mixes_train = URMP_Mixtures(base_dir=None,
                                 splits=urmp_train_splits,
                                 sample_rate=sample_rate,
                                 cqt=ss_mpe.hcqt,
                                 n_secs=4,
                                 seed=seed)

# Sample from which to start NSynth analysis
urmp_start_idx = 20 # higher polyphony

# Slice NSynth dataset
urmp_mixes_train.tracks = urmp_mixes_train.tracks[urmp_start_idx:]


###################
## Visualization ##
###################

# Create a directory for saving visualized harmonics
vis_dir = os.path.join('..', '..', 'generated', 'visualization', 'trivial')

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(urmp_mixes_train)):
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

    if i == 0:
        # Seed everything with the same seed
        seed_everything(0)

    # Determine training sequence length in frames
    n_frames = int(4 * sample_rate / ss_mpe.hcqt.hop_length)

    # Extract ground-truth pitch salience activations
    gt_activations = data[constants.KEY_GROUND_TRUTH]

    # Widen the activations for easier visualization
    #gt_activations = convolve2d(gt_activations,
    #                            np.array([[0.5, 1, 0.5]]).T, 'same')

    trivial_zeros = np.zeros_like(gt_activations)
    trivial_ones = np.ones_like(gt_activations)

    trivial_pattern = np.zeros_like(gt_activations)
    spacing = 20
    for i in range(n_bins // spacing):
        constant_energy = np.copy(trivial_zeros)
        constant_energy[spacing // 2 + i * spacing] = 1
        constant_energy = convolve2d(constant_energy, np.array([[0.75, 1, 0.75]]).T, 'same')
        trivial_pattern += constant_energy

    # Create the root directory
    os.makedirs(vis_dir, exist_ok=True)

    # Define extent for magnitude plots
    extent_midi = [0, 4, fmin, fmax]
    # Define ticks for frequencies in MIDI
    midi_ticks = [30, 40, 50, 60, 70, 80, 90, 100]

    # Loop through custom features, output, and ground-truth
    for (trivial, salience) in zip(['zeros', 'pattern', 'ones'],
                        [trivial_zeros, trivial_pattern, trivial_ones]):
        # Open a new figure
        fig = initialize_figure(figsize=(4, 3))
        # Plot spectral features
        fig = plot_magnitude(salience, extent=extent_midi, fig=fig)
        ax = fig.gca()
        ax.axis('on')
        ax.set_yticks(midi_ticks)
        ax.set_ylabel('Frequency (MIDI)')
        ax.set_xlabel('Time (s)')
        # Minimize free space
        fig.tight_layout()
        # Construct path under visualization directory
        save_path = os.path.join(vis_dir, f'{trivial}.jpg')
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

    break
