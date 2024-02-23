# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import AMTDataset
from timbre_trap.utils.data import *

# Regular imports
from mir_eval.multipitch import MIN_FREQ, MAX_FREQ
from torchaudio import functional as F

import numpy as np
import librosa
import torch
import json
import os


class NSynth(AMTDataset):
    """
    Implements a wrapper for the NSynth dataset
    (https://magenta.tensorflow.org/datasets/nsynth).
    """

    def __init__(self, n_tracks=None, midi_range=librosa.hz_to_midi([MIN_FREQ, MAX_FREQ]), **kwargs):
        """
        Introduce extra parameters for evaluation.

        Parameters
        ----------
        n_tracks : int
          Number of tracks to retain within instance
        midi_range : bool
          Supported range of (MIDI) frequencies
        """

        self.n_tracks = n_tracks
        self.midi_range = midi_range

        super().__init__(**kwargs)

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Names of originally proposed splits
        """

        splits = ['train', 'valid', 'test']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating train, validation, or test split

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """

        # Construct a path to the JSON metadata for the specified partition
        json_path = os.path.join(self.base_dir, f'nsynth-{split}', 'examples.json')

        with open(json_path) as f:
            # Read JSON metadata
            metadata = json.load(f)

        # Extract track names
        tracks = list(metadata.keys())

        # Append name of split to all track names
        tracks = [os.path.join(split, t) for t in tracks]

        if self.midi_range is not None:
            # Ignore tracks with out-of-bounds pitches
            tracks = [t for t in tracks if
                      (self.get_pitch(t) >= self.midi_range[0]) &
                      (self.get_pitch(t) <= self.midi_range[1])]

        if self.n_tracks is not None:
            # Reduce tracks to specified amount
            tracks = tracks[:self.n_tracks]

        # Sort track list
        tracks = sorted(tracks)

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Break apart split and track name
        split, name = os.path.split(track)

        # Get the path to the synthesized audio
        wav_path = os.path.join(self.base_dir, f'nsynth-{split}', 'audio', f'{name}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.
        """

        return NotImplementedError

    def get_pitch(self, track):
        """
        Determine the pitch associated with an NSynth track.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        pitch : int
          Pitch of the note sample in the track
        """

        # Extract nominal pitch from name
        pitch = int(track.split('-')[-2])

        return pitch

    def get_ground_truth(self, track, w_length_t=0.10, a_perc=0.10):
        """
        Construct the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          NSynth track name
        w_length_t : float
          Size of RMS analysis window in seconds
        a_perc : float
          Percentage of envelope maximum used as activity threshold

        Returns
        ----------
        pitches : ndarray (1)
          Array of note pitches
        intervals : ndarray (1 x 2)
          Array of corresponding onset-offset time pair
        """

        # Extract nominal pitch
        pitch = self.get_pitch(track)

        # Load the track's audio
        audio = self.get_audio(track).squeeze()

        # Determine safe lower-bound for frequency content
        cutoff_low = pitch / 3

        if self.midi_range is not None:
            # Determine minimum allowable frequency in Hertz
            cutoff_low_global = librosa.midi_to_hz(self.midi_range)[0]
            # Choose the higher of the two cutoffs for filtering
            cutoff_low = max(cutoff_low, cutoff_low_global)

        # Low-pass filter to remove artifacts before inferring pitch activity
        audio_highpass = F.highpass_biquad(audio, self.sample_rate, cutoff_low)

        # Determine window length in number of samples
        w_length = round(w_length_t * self.sample_rate)

        # Compute RMS window
        w = torch.ones(w_length)
        # Compute discrete RMS values for filtered signal
        rms = torch.sqrt((1 / w_length) * F.convolve(audio_highpass ** 2, w, 'same'))
        # Low-pass filter RMS values to obtain amplitude envelope
        amplitude = F.lowpass_biquad(rms, self.sample_rate, cutoff_freq=30)

        # Compute time corresponding to each sample
        times = np.arange(len(audio)) / self.sample_rate
        # Determine where amplitude is within specified % of maximum
        activity = (amplitude >= a_perc * amplitude.max()).long()

        # Determine which samples correspond to onsets and offsets
        onsets = torch.cat([activity[:1], activity[1:] - activity[:-1]]).relu()
        offsets = torch.cat([activity[-1:], activity[:-1] - activity[1:]]).relu()

        #import matplotlib.pyplot as plt
        #plt.plot(audio)
        #plt.title(track)
        #plt.plot(audio_highpass)
        #plt.plot(amplitude)
        #plt.plot(onsets)
        #plt.plot(offsets)

        # Convert samples to times
        onsets = times[onsets.bool()]
        offsets = times[offsets.bool()]

        # Create an array for detected note intervals
        intervals = np.concatenate(([onsets], [offsets])).T

        # Create a corresponding array for pitches
        pitches = np.array([pitch] * len(intervals))

        return pitches, intervals

    @classmethod
    def download(cls, save_dir):
        """
        Download the NSynth dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of NSynth
        """

        # Create top-level directory
        super().download(save_dir)

        for split in cls.available_splits():
            # URL pointing to the zip file for the split
            url = f'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{split}.jsonwav.tar.gz'

            # Construct a path for saving the file
            save_path = os.path.join(save_dir, os.path.basename(url))

            # Download the zip file
            stream_url_resource(url, save_path, 1000 * 1024)

            # Unzip the downloaded file and remove it
            unzip_and_remove(save_path, tar=True)
