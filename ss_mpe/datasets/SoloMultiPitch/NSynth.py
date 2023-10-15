# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets.utils import stream_url_resource, unzip_and_remove
from timbre_trap.datasets import AMTDataset

# Regular imports
from mir_eval.multipitch import MIN_FREQ, MAX_FREQ

import numpy as np
import librosa
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

    def get_ground_truth(self, track):
        """
        Construct the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          NSynth track name

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
        # Compute time corresponding to each sample
        times = np.arange(len(audio)) / self.sample_rate

        # Determine where amplitude rises above and falls below 10% of maximum
        active_times = times[audio.abs() >= 0.10 * audio.max()]

        # Select onset and offset as time boundaries of activity
        onset, offset = np.min(active_times), np.max(active_times)

        # Create arrays of expected dimensions for the pitch and interval
        pitches, intervals = np.array([pitch]), np.array([[onset, offset]])

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
