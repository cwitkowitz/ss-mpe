# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures
from ss_mpe.datasets.SoloMultiPitch import NSynth

from ss_mpe.objectives import apply_random_transformations
from ss_mpe.framework import TT_Base
from timbre_trap.utils import *

# Regular imports
import matplotlib.pyplot as plt
from tqdm import tqdm

import librosa
import torch
import os


# Set number of curves per methodology
n_transforms = 2

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


###################
## Visualization ##
###################

# Create a directory for saving visualized equalizations
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'geometric')

# Create the visualization directory
os.makedirs(save_dir, exist_ok=True)

# Determine training sequence length in frames
n_frames = int(4 * sample_rate / hop_length)

# Define maximum time and frequency shift
max_shift_v = 2 * bins_per_octave
max_shift_h = n_frames // 4

# Maximum rate by which audio can be sped up or slowed down
max_stretch_factor = 2

# Compute inverse of maximum stretch factor
min_stretch_factor = 1 / max_stretch_factor

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(urmp_mixes_train)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract first harmonic CQT spectral features and repeat across transforms
    features_db = torch.cat([features['db'] for t in range(n_transforms)])

    features_db_transformed, (vs, hs, sfs) = apply_random_transformations(features_db, max_shift_v, max_shift_h, max_stretch_factor)
    features_db = to_array(features_db[0, harmonics.index(1)])
    features_db_transformed = to_array(features_db_transformed[:, harmonics.index(1)])

    # Initialize a new figure
    fig = initialize_figure(figsize=(6.666, 3 * n_transforms))
    fig.set_layout_engine('constrained')
    # Create sub-figures within the figure
    subfigs = fig.subfigures(nrows=n_transforms, ncols=3, width_ratios=[2.35, 1.325, 1.925],
                                                          height_ratios=[1] * (n_transforms - 1) + [1.065])

    # Determine track's attributes
    #name, pitch, vel = track.split('-')
    # Add a global title above all sub-plots
    #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

    # Define extent for magnitude plots
    extent_midi = [0, 4, fmin, fmax]
    # Define ticks for frequencies in MIDI
    midi_ticks = [30, 40, 50, 60, 70, 80, 90, 100]

    for i in range(n_transforms):
        # Create axes for plotting
        ax_orig = subfigs[i, 0].subplots(nrows=1, ncols=1)
        ax_trns = subfigs[i, 2].subplots(nrows=1, ncols=1)
        ax_param = subfigs[i, 1].subplots(nrows=3, ncols=1)
        # Plot original and transformed features as images
        ax_orig.imshow(features_db, vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)
        ax_trns.imshow(features_db_transformed[i], vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)
        # Ticks and labels
        ax_orig.set_ylabel('Frequency (MIDI)')
        ax_orig.set_yticks(midi_ticks)
        ax_trns.set_ylabel('')
        ax_trns.set_yticks(midi_ticks)
        ax_trns.set_yticklabels(['' for t in midi_ticks])
        # Plot sampled parameter values
        ax_param[0].plot([vs[i], vs[i]], [0, 1], linewidth=2)
        ax_param[0].set_ylim([0, 1])
        ax_param[0].set_xlim([-max_shift_v, max_shift_v])
        ax_param[0].set_xticks([-max_shift_v, 0, max_shift_v])
        ax_param[0].set_xticklabels(['-2 oct.', '0', '2 oct.'])
        ax_param[0].get_yaxis().set_visible(False)
        ax_param[0].set_xlabel('Freq. Shift (bins)')
        ax_param[1].plot([hs[i], hs[i]], [0, 1], linewidth=2)
        ax_param[1].set_ylim([0, 1])
        ax_param[1].set_xlim([-max_shift_h, max_shift_h])
        ax_param[1].set_xticks([-max_shift_h, 0, max_shift_h])
        ax_param[1].set_xticklabels(['-1 s', '0', '1 s'])
        ax_param[1].get_yaxis().set_visible(False)
        ax_param[1].set_xlabel('Time Shift')
        ax_param[2].plot([sfs[i], sfs[i]], [0, 1], linewidth=2)
        ax_param[2].set_ylim([0, 1])
        ax_param[2].set_xlim([min_stretch_factor, max_stretch_factor])
        ax_param[2].set_xticks([min_stretch_factor, min_stretch_factor + (max_stretch_factor - min_stretch_factor) / 2, max_stretch_factor])
        ax_param[2].set_xticklabels([f'{min_stretch_factor}x', '1x', f'{max_stretch_factor}x'])
        ax_param[2].get_yaxis().set_visible(False)
        ax_param[2].set_xlabel('Time Stretch')
        # Resize spectral plots
        #ax_orig.set_aspect(ax_trns.get_aspect())
    ax_orig.set_xlabel('Time (s)')
    ax_trns.set_xlabel('Time (s)')

    fig.suptitle('Example Geometric Transformations $t_{ev-g}$')

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
    save_path = os.path.join(save_dir, f'{track}_t{n_transforms}_s{seed}.jpg')
    # Save the figure with minimal whitespace
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Close figure
    plt.close(fig)
