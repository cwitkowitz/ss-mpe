# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import *

# Regular imports
import numpy as np
import os


def hz_to_midi(freqs):
    # Convert frequencies provided in Hz to MIDI
    midi = 12 * (np.log2(freqs) - np.log2(440.0)) + 69

    return midi


def evaluate(model, hcqt, eval_set, writer=None):
    # Extract the CQT for the first harmonic
    cqt = hcqt.get_modules()[hcqt.harmonics.index(1)]

    # Extract signal processing parameters from CQT
    sample_rate, hop_length = cqt.fs, cqt.hop_length

    # Compute the number of bins per semitone
    bins_per_semitone = (cqt.bins_per_octave / 12)

    # Determine the frequency of the lowest and highest bin
    fmin = hz_to_midi(cqt.fmin)
    fmax = fmin + (cqt.n_bins - 1) / bins_per_semitone

    # center_freqs = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
    # center_freqs = 32.7 * (2.0 ** (np.arange(216) / 36))
    # center_freqs = 12 * (np.log2(center_freqs) - np.log2(440.0)) + 69
    # center_freqs = np.linspace(24, 96, 217)[:-1]
    # Compute center frequencies of each bin
    center_freqs = np.linspace(fmin, fmax, cqt.n_bins)

    # Loop through each track name
    for track in eval_set:
        # Obtain the audio for the track
        # TODO - resample
        audio = torch.Tensor(eval_set.get_audio(track))
        # Add a batch and channel dimension to the audio
        # TODO - parameterize device
        audio = audio.unsqueeze(0).unsqueeze(0).to(0)
        # Obtain features for the audio
        features = decibels_to_linear(hcqt(audio))
        # Determine the time associated with each frame (center)
        times = (hop_length / sample_rate) * np.arange(features.shape[-1])
        # Extract the ground-truth pitch salience for the track
        eval_set.get_ground_truth(track, times, bins=center_freqs)

        # Compute the pitch salience of the features
        # TODO - need to train with longer sizes
        salience = torch.sigmoid(model(features).squeeze())

        # TODO - evaluate pitch salience (https://github.com/cwitkowitz/amt-tools/blob/b41ead77a348157caaeec57243f30be8f5536330/amt_tools/evaluate.py#L794)

        # TODO - log results to writer
