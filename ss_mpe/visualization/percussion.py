# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures
from ss_mpe.datasets.SoloMultiPitch import NSynth
from ss_mpe.datasets.AudioMixtures import E_GMD
from timbre_trap.datasets import ComboDataset

from ss_mpe.objectives import mix_random_percussion
from ss_mpe.framework import TT_Base
from timbre_trap.utils import *

# Regular imports
import matplotlib.pyplot as plt
from tqdm import tqdm

import librosa
import torch
import math
import os


# Set number of curves per methodology
n_examples = 2

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

# Instantiate E-GMD audio for percussion-invariance
egmd = E_GMD(base_dir=None,
             splits=['train'],
             sample_rate=sample_rate,
             seed=seed)


###################
## Visualization ##
###################

# Create a directory for saving visualized equalizations
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'percussion')

# Create the visualization directory
os.makedirs(save_dir, exist_ok=True)

# Maximum volume of percussion relative to original audio
max_volume = 1.0

# Create a combination dataset of percussive audio
percussive_set_combo = ComboDataset([egmd])

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(urmp_mixes_train)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract first harmonic CQT spectral features
    features_db_1 = to_array(features['db_1'][0])

    """
    # Initialize a new figure with subplots
    (fig, ax) = plt.subplots(nrows=n_examples, ncols=2, figsize=(5, 3 * n_examples))
    """

    # Initialize a new figure
    fig = initialize_figure(figsize=(6.666, 3 * n_examples))
    fig.set_layout_engine('constrained')
    # Create sub-figures within the figure
    subfigs = fig.subfigures(nrows=n_examples, ncols=3, width_ratios=[2.35, 1.325, 1.925],
                                                          height_ratios=[1] * (n_examples - 1) + [1.065])

    # Determine track's attributes
    #name, pitch, vel = track.split('-')
    # Add a global title above all sub-plots
    #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

    # Define extent for magnitude plots
    extent_midi = [0, 4, fmin, fmax]
    # Define ticks for frequencies in MIDI
    midi_ticks = [30, 40, 50, 60, 70, 80, 90, 100]

    for i in range(n_examples):
        # Create axes for plotting
        ax_orig = subfigs[i, 0].subplots(nrows=1, ncols=1)
        ax_mixt = subfigs[i, 2].subplots(nrows=1, ncols=1)
        ax_param = subfigs[i, 1].subplots(nrows=2, ncols=1, height_ratios=[3, 1])

        # Plot original magnitude features as an image
        ax_orig.imshow(features_db_1, vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)

        # Superimpose random percussive audio onto original audio
        mixtures, volumes, percussive_audio = mix_random_percussion(audio.repeat(n_examples, 1, 1), percussive_set_combo, max_volume)

        # Compute full set of spectral features for mixed audio
        mixture_features = ss_mpe.get_all_features(mixtures)

        # Extract first harmonic CQT spectral features for mixed audio
        mixture_features_db_1 = to_array(mixture_features['db_1'][0])

        # Compute full set of spectral features for percussive audio
        percussive_features = ss_mpe.get_all_features(percussive_audio)

        # Extract first harmonic CQT spectral features for percussive audio
        percussive_features_db_1 = to_array(percussive_features['db_1'][0])

        # Plot percussive magnitude features as an image
        ax_mixt.imshow(mixture_features_db_1, vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)

        # Ticks and labels
        ax_orig.set_ylabel('Frequency (MIDI)')
        ax_orig.set_yticks(midi_ticks)
        ax_mixt.set_yticks(midi_ticks)
        ax_mixt.set_yticklabels(['' for t in midi_ticks])

        ax_param[0].imshow(percussive_features_db_1, vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)
        ax_param[0].set_xticks([0, 1, 2, 3, 4])
        ax_param[0].set_xticklabels(['' for t in range(5)])
        ax_param[0].set_yticks(midi_ticks)
        ax_param[0].set_yticklabels(['' for t in midi_ticks])
        ax_param[0].set_xlabel('Percussion')
        ax_param[1].plot([volumes[i].item(), volumes[i].item()], [0, 1], linewidth=2)
        ax_param[1].set_ylim([0, 1])
        ax_param[1].set_xlim([0, max_volume])
        ax_param[1].set_xticks([0, max_volume / 2, max_volume])
        ax_param[1].set_xticklabels(['0x', '0.5x', '1x'])
        ax_param[1].get_yaxis().set_visible(False)
        ax_param[1].set_xlabel('Volume')

    ax_orig.set_xlabel('Time (s)')
    ax_mixt.set_xlabel('Time (s)')

    fig.suptitle('Example Percussive Transformations $t_{iv-p}$')

    # Minimize free space
    #fig.tight_layout()

    """
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
    """
    # Construct path under visualization directory
    save_path = os.path.join(save_dir, f'{track}_c{n_examples}_s{seed}.jpg')
    # Save the figure with minimal whitespace
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Close figure
    plt.close(fig)
