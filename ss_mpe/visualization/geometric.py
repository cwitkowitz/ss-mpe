# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ss_mpe.datasets.SoloMultiPitch import NSynth

from ss_mpe.models.objectives import apply_translation, apply_distortion
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
n_transforms = 2

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
for i, data in enumerate(tqdm(nsynth_val)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract first harmonic CQT spectral features and repeat across transforms
    features= torch.cat([features['db_1'].unsqueeze(0) for t in range(n_transforms)])

    # Sample a random vertical / horizontal shift for each sample in the batch
    v_shifts = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(n_transforms,))
    h_shifts = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(n_transforms,))

    # Sample a random stretch factor for each sample in the batch, starting at minimum
    stretch_factors, stretch_factors_ = min_stretch_factor, torch.rand(size=(n_transforms,))
    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()
    # Scale stretch factor evenly across effective range
    stretch_factors += neg_perc * (1 - min_stretch_factor)
    stretch_factors += pos_perc * (max_stretch_factor - 1)

    # Apply vertical and horizontal translations, inserting zero at empties
    transform = apply_translation(features, v_shifts, axis=-2, val=0)
    transform = apply_translation(transform, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    transform = apply_distortion(transform, stretch_factors)

    # Remove the temporary batch and channel dimensions
    features = to_array(features.squeeze(1)[0])
    transform = to_array(transform.squeeze(1))

    # Initialize a new figure
    fig = initialize_figure(figsize=(6.666, 3 * n_transforms))
    fig.set_layout_engine('constrained')
    # Create sub-figures within the figure
    subfigs = fig.subfigures(nrows=n_transforms, ncols=3, width_ratios=[2.35, 1.325, 1.925],
                                                          height_ratios=[1] * (n_transforms - 1) + [1.065])

    # Determine track's attributes
    name, pitch, vel = track.split('-')
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
        ax_orig.imshow(features, vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)
        ax_trns.imshow(transform[i], vmin=0, vmax=1, aspect='auto', origin='lower', extent=extent_midi)
        # Ticks and labels
        ax_orig.set_ylabel('Frequency (MIDI)')
        ax_orig.set_yticks(midi_ticks)
        ax_trns.set_ylabel('')
        ax_trns.set_yticks(midi_ticks)
        ax_trns.set_yticklabels(['' for t in midi_ticks])
        # Plot sampled parameter values
        ax_param[0].plot([v_shifts[i], v_shifts[i]], [0, 1], linewidth=2)
        ax_param[0].set_ylim([0, 1])
        ax_param[0].set_xlim([-max_shift_v, max_shift_v])
        ax_param[0].set_xticks([-max_shift_v, 0, max_shift_v])
        ax_param[0].set_xticklabels(['-2 oct.', '0', '2 oct.'])
        ax_param[0].get_yaxis().set_visible(False)
        ax_param[0].set_xlabel('Freq. Shift (bins)')
        ax_param[1].plot([h_shifts[i], h_shifts[i]], [0, 1], linewidth=2)
        ax_param[1].set_ylim([0, 1])
        ax_param[1].set_xlim([-max_shift_h, max_shift_h])
        ax_param[1].set_xticks([-max_shift_h, 0, max_shift_h])
        ax_param[1].set_xticklabels(['-1 s', '0', '1 s'])
        ax_param[1].get_yaxis().set_visible(False)
        ax_param[1].set_xlabel('Time Shift')
        ax_param[2].plot([stretch_factors_[i], stretch_factors_[i]], [0, 1], linewidth=2)
        ax_param[2].set_ylim([0, 1])
        ax_param[2].set_xlim([0, 1])
        ax_param[2].set_xticks([0, 0.5, 1.0])
        ax_param[2].set_xticklabels([f'{min_stretch_factor}x', '1x', f'{max_stretch_factor}x'])
        ax_param[2].get_yaxis().set_visible(False)
        ax_param[2].set_xlabel('Time Stretch')
        # Resize spectral plots
        #ax_orig.set_aspect(ax_trns.get_aspect())
    ax_orig.set_xlabel('Time (s)')
    ax_trns.set_xlabel('Time (s)')

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
        save_path = os.path.join(save_dir, f'{track}_t{n_transforms}_s{seed}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Close figure
    plt.close(fig)
