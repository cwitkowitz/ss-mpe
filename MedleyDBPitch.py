# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove, change_base_dir
from common import EvalSet

# Regular imports
import pandas as pd
import numpy as np
import warnings
import librosa
import scipy
import os


class MedleyDB_Pitch(EvalSet):
    """
    Implements a wrapper for the MedleyDB Pitch Tracking subset (https://zenodo.org/record/2620624).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset
        """

        # TODO - splits can be genre/instrument
        splits = ['all']

        return splits

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

        # Construct a path to the audio directory
        audio_dir = os.path.join(self.base_dir, 'audio')

        # Obtain all track names under the audio directory
        tracks = sorted([os.path.splitext(d)[0] for d in os.listdir(audio_dir)])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, 'audio', f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        csv_path : string
          TODO
        """

        # Get the path to the ground-truth pitch annotations
        csv_path = os.path.join(self.base_dir, 'pitch', f'{track}.csv')

        return csv_path

    def get_ground_truth(self, track, times):
        """
        Get the path for a track's ground_truth.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        ground_truth : TODO
          TODO
        """

        # Obtain the path to the track's ground_truth
        csv_path = self.get_ground_truth_path(track)

        # Load tabulated pitch data from the csv file and unpack
        original_times, pitches = pd.read_csv(csv_path, header=None).to_numpy().T

        # Create array of frame indices
        original_idcs = np.arange(len(original_times))

        # Out-of-range times will be set to first time (always silent)
        fill_values = (original_idcs[0], original_idcs[0])

        # Obtain a function to resample annotation times
        res_func_time = scipy.interpolate.interp1d(x=original_times,
                                                   y=original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)

        # Resample the pitch annotations using above function
        pitches = pitches[res_func_time(times).astype('uint')]

        # Convert non-zero pitch annotations to MIDI frequencies
        pitches[pitches != 0.] = librosa.hz_to_midi(pitches[pitches != 0.])

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        if np.min(pitches[pitches != 0.]) < np.min(self.center_freqs) or \
            np.max(pitches[pitches != 0.]) > np.max(self.center_freqs):
            warnings.warn('Cannot fully represent ground-truth '
                          f'for track \'{track}\'.', RuntimeWarning)
            # Set any out-of-bounds pitch observations to zero
            pitches[np.logical_or(pitches < np.min(self.center_freqs),
                                  pitches > np.max(self.center_freqs))] = 0.

        # Determine the closest frequency bin for each pitch observation
        pitch_idcs = self.res_func_freq(pitches[pitches != 0.])

        # Insert non-zero pitch activity into the ground-truth
        ground_truth[pitch_idcs.astype('uint'), pitches.nonzero()] = 1

        return ground_truth

    @classmethod
    def name(cls):
        """
        Obtain a string representing the dataset.

        Returns
        ----------
        tag : str
          Dataset name with dashes
        """

        # Obtain class name and replace underscore with dash
        name = super().name().replace('_', '-')

        return name

    @classmethod
    def download(cls, save_dir):
        """
        Download the MedleyDB pitch-tracking subset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MedleyDB-Pitch
        """

        # Create top-level directory
        #super().download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        #url = 'https://zenodo.org/record/2620624/files/MedleyDB-Pitch.zip'

        # Construct a path for saving the file
        #zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        #stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        #unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        #change_base_dir(save_dir, os.path.join(save_dir, MedleyDB_Pitch.name(), MedleyDB_Pitch.name()))

        warnings.warn('MedleyDB must be downloaded manually. Request access '
                      'here: https://zenodo.org/record/2620624', RuntimeWarning)

        return NotImplementedError
