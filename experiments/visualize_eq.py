# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ss_mpe.models.objectives import (sample_random_equalization,
                                      sample_parabolic_equalization)
from utils import *

# Regular imports
import numpy as np
import librosa
import math


# Set randomization seed
seed = 0

# Set number of curves per methodology
n_curves = 3

# Load an example piece of audio from librosa
audio, sample_rate = librosa.load(librosa.ex('trumpet'))

# Number of samples between frames
hop_length = 256

# First center frequency (MIDI) of geometric progression
fmin = librosa.note_to_midi('A0')

# Number of bins in a single octave
bins_per_octave = 60

# Number of frequency bins per CQT
n_bins = 440

# Number of random points to sample per octave
points_per_octave = 2

# Standard deviation of boost/cut
std_dev = 0.10

# Pointiness of parabolic equalization
pointiness = 10


# Compute VQT features
features = librosa.vqt(audio,
                       sr=sample_rate,
                       hop_length=hop_length,
                       fmin=fmin,
                       n_bins=n_bins,
                       bins_per_octave=bins_per_octave)

# Obtain magnitude VQT features in decibels
features = librosa.amplitude_to_db(np.abs(features), ref=np.max)
# Rescale decibels to range [0, 1]
features = 1 + features / 80


# Seed everything with the same seed
seed_everything(seed)

# Determine how many octaves have been covered
n_octaves = int(math.ceil(n_bins / bins_per_octave))

# Determine the number of cut/boost points to sample
n_points = 1 + points_per_octave * n_octaves

# Cover the full octave for proper interpolation
n_out = n_octaves * bins_per_octave

# Sample equalization curves based on uniform random boosts/cuts
random_curves = sample_random_equalization(n_points, n_curves, n_out, std_dev)

for curve in to_array(random_curves):
    # Plot equalized features and curve
    plot_equalization(features, curve[:n_bins])

# Sample parametric parabolic equalivation curves
parabolic_curves = sample_parabolic_equalization(n_out, n_curves, pointiness)

for curve in to_array(parabolic_curves):
    # Plot equalized features and curve
    plot_equalization(features, curve[:n_bins])

# Wait for user input
input('Press ENTER to finish...')
