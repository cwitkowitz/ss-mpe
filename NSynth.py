# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove
from common import TrainSet, EvalSetNoteLevel

# Regular imports
import numpy as np
import librosa
import random
import json
import os


class NSynth(TrainSet):
    """
    Implements a wrapper for the NSynth dataset (https://magenta.tensorflow.org/datasets/nsynth).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset for different stages of pipeline
        """

        splits = ['train', 'valid', 'test']

        return splits

    def get_tracks(self, split):
        """
        Get the track names associated with dataset partitions.

        Parameters
        ----------
        split : string
          TODO

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Construct a path to the JSON annotations for the partition
        json_path = os.path.join(self.base_dir, f'nsynth-{split}', 'examples.json')

        with open(json_path) as f:
            # Read JSON data
            tracks = json.load(f)

        # Retain the names of the tracks
        tracks = sorted(list(tracks.keys()))

        # Append the split name to all tracks
        tracks = [os.path.join(f'nsynth-{split}', t) for t in tracks]

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
          Path to the specified track's audio
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, split, 'audio', name + '.wav')

        return wav_path

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


class NSynthValidation(EvalSetNoteLevel, NSynth):
    """
    TODO
    """

    def __init__(self, n_tracks=None, midi_range=None, **kwargs):
        """
        TODO.

        Parameters
        ----------
        n_tracks : int
          TODO
        midi_range : bool
          TODO
        kwargs : TODO
          TODO
        """

        self.n_tracks = n_tracks
        self.midi_range = midi_range

        super().__init__(**kwargs)

    @classmethod
    def name(cls):
        """
        Simple helper function to get the class name.
        """

        name = NSynth.name()

        return name

    def get_pitch(self, track):
        """
        Determine the pitch associated with a track.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        pitch : int
          Pitch of the note sample in the track
        """

        # Extract the pitch from track name
        pitch = int(track.split('-')[-2])

        return pitch

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string (unused)
          TODO

        Returns
        ----------
        tracks : list of strings
          TODO
        """

        # Obtain the standard track list
        tracks = super().get_tracks(split)

        if self.midi_range is not None:
            # Filter out tracks with out-of-bounds ground-truth activations
            tracks = [t for t in tracks if
                      (self.get_pitch(t) >= self.midi_range[0]) &
                      (self.get_pitch(t) <= self.midi_range[1])]

        if self.n_tracks is not None:
            # Shuffle the tracks
            random.shuffle(tracks)

            # Trim tracks to selected amount
            tracks = tracks[:self.n_tracks]

        return tracks

    def get_ground_truth(self, track, times):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string
          NSynth track name
        times : ndarray (T)
          Frame times to use when constructing ground-truth

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain nominal pitch in Hertz from the track name
        pitch = librosa.midi_to_hz(self.get_pitch(track))

        # Obtain ground-truth as the nominal pitch across time
        multi_pitch = [np.array([pitch]) if (t >= 0) & (t <= 3)
                       else np.empty(0) for t in times]

        return times, multi_pitch
