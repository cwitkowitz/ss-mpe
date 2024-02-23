# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ss_mpe.datasets.SoloMultiPitch import NSynth

from timbre_trap.datasets.utils import constants
from ss_mpe.models import TT_Base

# Regular imports
from tqdm import tqdm

import matplotlib.pyplot as plt
import librosa
import torch
import sys
import os


# Set randomization seed
seed = 0

# Import utilities from parent directory
sys.path.insert(0, os.path.join('..'))
from utils import plot_magnitude


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 16000

# Number of samples between frames
hop_length = 512

# First center frequency (MIDI) of geometric progression
fmin = librosa.note_to_midi('A0')

# Number of bins in a single octave
bins_per_octave = 36

# Number of frequency bins per CQT
n_bins = 264

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

# Create a directory for saving visualized samples
save_dir = os.path.join('..', '..', 'generated', 'visualization', 'samples')

# Create the visualization directory
os.makedirs(save_dir, exist_ok=True)

# Loop through all tracks in the test set
for i, data in enumerate(tqdm(nsynth_val)):
    # Determine which track is being processed
    track = data[constants.KEY_TRACK]
    # Extract audio and add a batch dimension
    audio = data[constants.KEY_AUDIO].unsqueeze(0)

    # Compute full set of spectral features
    features = ss_mpe.get_all_features(audio)

    # Extract scaled power features
    features_db_1 = features['db_1']

    # Initialize a new figure with subplots
    (fig, ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), width_ratios=[0.8, 1])

    # Determine track's attributes
    name, pitch, vel = track.split('-')
    # Add a global title above all sub-plots
    #fig.suptitle(f'Track: {name} | Pitch: {pitch} | Velocity: {vel}')

    # Compute tick intervals
    tick_interval_wav = audio.size(-1) // 4
    tick_interval_mag = features_db_1.size(-1) / 4

    # Plot raw waveform
    ax[0].plot(audio.squeeze())
    ax[0].set_xlim([0, audio.size(-1) - 1])
    ax[0].set_ylim([-1, 1])
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlabel('Time (s)')
    # Ticks and labels
    ax[0].axis('on')
    ax[0].set_xticks([0, tick_interval_wav, 2 * tick_interval_wav,
                      3 * tick_interval_wav, 4 * tick_interval_wav])
    ax[0].set_xticklabels([0, 1, 2, 3, 4])
    #ax[0].set_xlabel(f'Time-Domain')

    # Define extent for magnitude plots
    extent_midi = [0, features_db_1.size(-1), fmin, fmax]

    # Define ticks for frequencies in Hz
    midi_ticks = [oct * 12 + librosa.note_to_midi('A0') for oct in range(0, 8)]
    hz_labels = librosa.midi_to_hz(midi_ticks).tolist()
    hz_labels = [int(l) if not l % 1 else l for l in hz_labels]

    # Plot 1st harmonic (log) activations
    fig.sca(ax[1])
    plot_magnitude(features_db_1[0], extent=extent_midi, colorbar=True, fig=fig)
    # Ticks and labels
    ax[1].axis('on')
    ax[1].set_yticks(midi_ticks)
    ax[1].set_yticklabels(hz_labels)
    ax[1].set_ylabel('Frequency (Hz)')
    #ax[1].get_yaxis().set_visible(False)
    #ax[1].set_xlabel(f'Frequency-Domain')
    ax[1].set_xticks([0, tick_interval_mag, 2 * tick_interval_mag,
                      3 * tick_interval_mag, 4 * tick_interval_mag])
    ax[1].set_xticklabels([0, 1, 2, 3, 4])
    cbar = ax[1].get_images()[0].colorbar
    cbar.ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(['-80 dB', '-60 dB', '-40 dB', '-20 dB', '+0 dB'])

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
        save_path = os.path.join(save_dir, f'{track}.pdf')
        # Save the figure with minimal whitespace
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Close figure
    plt.close(fig)
