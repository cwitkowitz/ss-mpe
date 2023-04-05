# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove
from common import TrainSet, EvalSet

# Regular imports
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


class NSynthEval(EvalSet, NSynth):
    """
    TODO
    """

    @classmethod
    def name(cls):
        """
        Simple helper function to get the class name.
        """

        name = NSynth.name()

        return name

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          NSynth track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        try:
            # Obtain the index of the pitch of the sample from the track name
            pitch_idx = int(self.res_func_freq(track.split('-')[-2]).item())

            # Obtain time indices corresponding to pitch activity
            time_idcs = (times >= 0) & (times <= 3)

            # Make the pitch active for the entire duration
            ground_truth[pitch_idx, time_idcs] = 1

        except ValueError as e:
            # Print warning message
            print(f'{repr(e)}')

        return ground_truth
