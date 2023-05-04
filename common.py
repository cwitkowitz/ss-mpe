# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import *

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod

import numpy as np
import torchaudio
import warnings
import librosa
import shutil
import scipy
import torch
import os


DEFAULT_LOCATION = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets')


class TrainSet(Dataset):
    """
    Implements a wrapper for an MPE dataset intended for training.
    """

    def __init__(self, base_dir=None, splits=None, sample_rate=16000, n_secs=None, seed=0):
        """
        TODO.

        Parameters
        ----------
        base_dir : TODO
          TODO
        splits : TODO
          TODO
        sample_rate : TODO
          TODO
        n_secs : TODO
          TODO
        seed : TODO
          TODO
        """

        if base_dir is None:
            base_dir = os.path.join(DEFAULT_LOCATION, self.name())

        self.base_dir = base_dir

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset if it is missing
            self.download(self.base_dir)

        if splits is None:
            # Choose all available dataset splits
            splits = self.available_splits()

        self.tracks = []
        # Aggregate all the track names from the selected splits
        for split in splits:
            self.tracks += self.get_tracks(split)

        self.sample_rate = sample_rate
        self.n_secs = n_secs

        # Initialize a random number generator for the dataset
        self.rng = np.random.RandomState(seed)

    @classmethod
    def name(cls):
        """
        Simple helper function to get the class name.
        """

        name = cls.__name__

        return name

    @staticmethod
    @abstractmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset
        """

        splits = ['all']

        return splits

    @abstractmethod
    def get_tracks(self, split):
        """
        Get the track names associated with dataset partitions.

        Parameters
        ----------
        split : string
          TODO
        """

        return NotImplementedError

    def __len__(self):
        """
        Defines the number of samples in dataset.

        Returns
        ----------
        length : int
          TODO
        """

        # TODO
        length = len(self.tracks)

        return length

    @abstractmethod
    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Track name
        """

        return NotImplementedError

    def get_audio(self, track):
        """
        Get the audio for a track.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        audio : TODO
          TODO
        """

        # Obtain the path of the track's audio
        audio_path = self.get_audio_path(track)

        # Load the audio
        audio, fs = torchaudio.load(audio_path)
        # Average channels to obtain mono-channel
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample audio to appropriate sampling rate
        audio = torchaudio.functional.resample(audio, fs, self.sample_rate)

        if audio.abs().max():
            # Normalize the audio using the infinity norm
            audio /= audio.abs().max()

        return audio

    @abstractmethod
    def __getitem__(self, index):
        """
        TODO.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        audio : TODO
          TODO
        """

        # Determine corresponding track
        track = self.tracks[index]

        try:
            # Attempt to read the track
            audio = self.get_audio(track)
        except Exception as e:
            # Print offending track to console
            print(f'Error loading track \'{track}\': {repr(e)}')

            # Default audio to silence
            audio = torch.empty((1, 0))

        if self.n_secs is not None:
            # Determine the required sequence length
            n_samples = int(self.n_secs * self.sample_rate)

            if audio.size(-1) >= n_samples:
                # Sample a random starting index for the trim
                start = self.rng.randint(0, audio.size(-1) - n_samples + 1)
                # Trim audio to the sequence length
                audio = audio[..., start : start + n_samples]
            else:
                # Determine how much padding is required
                pad_total = n_samples - audio.size(-1)
                # Randomly distributed between both sides
                pad_left = self.rng.randint(0, pad_total)
                # Pad the audio with zeros
                audio = torch.nn.functional.pad(audio, (pad_left, pad_total - pad_left))

        return audio

    @classmethod
    @abstractmethod
    def download(cls, save_dir):
        """
        Create the top-level directory for a dataset.

        Parameters
        ----------
        save_dir : string
          Directory under which to save dataset contents
        """

        if os.path.isdir(save_dir):
            # Remove directory if it already exists
            shutil.rmtree(save_dir)

        # Create the base directory
        os.makedirs(save_dir)

        print(f'Downloading {cls.__name__}...')


class EvalSetFrameLevel(TrainSet):
    """
    Implements a wrapper for an MPE dataset with frame-level annotations intended for evaluation.
    """

    @staticmethod
    @abstractmethod
    def has_frame_level_annotations():
        """
        Helper function to determine if ground-truth times are available.
        """

        return True

    @abstractmethod
    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          Track name
        """

        return NotImplementedError

    @staticmethod
    @abstractmethod
    def resample_multi_pitch(_times, _multi_pitch, times):
        """
        Protocol for resampling the ground-truth annotations, if necessary.

        Parameters
        ----------
        _times : ndarray (T)
          Original times
        _multi_pitch : list of ndarray (T x [...])
          Multi-pitch annotations corresponding to original times
        times : ndarray (K)
          Target times for resampling

        Returns
        ----------
        multi_pitch : list of ndarray (K x [...])
          Multi-pitch annotations corresponding to target times
        """

        # Create array of frame indices
        original_idcs = np.arange(len(_times))

        # Clamp resampled indices within the valid range
        fill_values = (original_idcs[0], original_idcs[-1])

        # Obtain a function to resample annotation times
        res_func_time = scipy.interpolate.interp1d(x=_times,
                                                   y=original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)

        # Resample the multi-pitch annotations using above function
        multi_pitch = [_multi_pitch[t] for t in res_func_time(times).astype('uint')]

        return multi_pitch

    @abstractmethod
    def get_ground_truth(self, track):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        return NotImplementedError

    @staticmethod
    def activations_to_multi_pitch(activations, midi_freqs, thr=0.5):
        """
        Convert an array of discrete pitch activations into a list of active pitches.

        Parameters
        ----------
        activations : ndarray (F x T)
          Discrete activations corresponding to MIDI pitches
        midi_freqs : ndarray (F)
          MIDI frequency corresponding to each bin
        thr : float [0, 1]
          Threshold value

        Returns
        ----------
        multi_pitch : list of ndarray (T x [...])
          Array of active pitches (in Hertz) across time
        """

        # Initialize empty pitch arrays for each frame
        multi_pitch = [np.empty(0)] * activations.shape[-1]

        # Make sure the activations are binarized
        activations = threshold(activations, thr)

        # Determine which frames contain pitch activity
        non_silent_frames = np.where(np.sum(activations, axis=-2) > 0)[-1]

        # Loop through these frames
        for i in list(non_silent_frames):
            # Determine the active pitches within the frame and insert into the list
            multi_pitch[i] = librosa.midi_to_hz(midi_freqs[np.where(activations[..., i])[-1]])

        return multi_pitch

    @staticmethod
    def multi_pitch_to_activations(multi_pitch, midi_freqs):
        """
        Convert a list of active pitches into an array of discrete pitch activations.

        Parameters
        ----------
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        midi_freqs : ndarray (F)
          MIDI frequency corresponding to each bin

        Returns
        ----------
        activations : ndarray (F x T)
          Discrete activations corresponding to MIDI pitches
        """

        # Obtain a function to resample to discrete frequencies
        res_func_freq = scipy.interpolate.interp1d(x=midi_freqs,
                                                   y=np.arange(len(midi_freqs)),
                                                   kind='nearest',
                                                   bounds_error=True,
                                                   assume_sorted=True)

        # Construct an empty array of relevant size by default
        activations = np.zeros((len(midi_freqs), len(multi_pitch)))

        # Make sure zeros are filtered out and convert to MIDI
        multi_pitch = [librosa.hz_to_midi(p[p != 0]) for p in multi_pitch]

        # Obtain frame indices corresponding to pitch activity
        frame_idcs = np.concatenate([[i] * len(multi_pitch[i])
                                     for i in range(len(multi_pitch)) if len(multi_pitch[i])])

        # Determine the closest frequency bin for each pitch observation
        multi_pitch_idcs = np.concatenate([res_func_freq(multi_pitch[i])
                                           for i in sorted(set(frame_idcs))])

        # Insert pitch activity into the ground-truth
        activations[multi_pitch_idcs.astype('uint'), frame_idcs] = 1

        return activations


class EvalSetNoteLevel(EvalSetFrameLevel):
    """
    Implements a wrapper for an MPE dataset with note-level annotations intended for evaluation.
    """

    @staticmethod
    def has_frame_level_annotations():
        """
        Helper function to determine if ground-truth times are available.
        """

        return False

    @staticmethod
    def resample_multi_pitch(_times, _multi_pitch, times):
        """
        This is not necessary due to parameterization of times when generating ground-truth.

        Parameters
        ----------
        _times : ndarray (T)
          Original times
        _multi_pitch : list of ndarray (T x [...])
          Multi-pitch annotations corresponding to original times
        times : ndarray (K)
          Target times for resampling

        Returns
        ----------
        multi_pitch : list of ndarray (K x [...])
          Multi-pitch annotations corresponding to target times
        """

        return NotImplementedError

    @abstractmethod
    def get_ground_truth(self, track, times):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string (unused)
          Track name
        times : ndarray (T)
          Frame times to use when constructing ground-truth

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        return NotImplementedError

    @staticmethod
    def notes_to_multi_pitch(pitches, intervals, times):
        """
        Convert a collection of notes into a list of active pitches.

        Parameters
        ----------
        pitches : ndarray (N)
          Array of pitches corresponding to notes in MIDI
        intervals : ndarray (N x 2)
          Array of onset-offset time pairs corresponding to notes
        times : ndarray (T)
          Frame times to use when constructing multi-pitch annotations

        Returns
        ----------
        multi_pitch : list of ndarray (T x [...])
          Array of active pitches (in Hertz) across time
        """

        # Initialize empty pitch arrays for each frame
        multi_pitch = [np.empty(0)] * times.shape[-1]

        # Loop through the attributes of each note
        for p, (on, off) in zip(pitches, intervals):
            # Obtain indices corresponding to note's pitch activity
            time_idcs = np.where((times >= on) & (times < off))[0]

            # Convert pitch to Hertz
            p_hz = librosa.midi_to_hz(p)

            # Loop through relevant frames
            for t in time_idcs:
                # Append the new pitch observation to the frame
                multi_pitch[t] = np.append(multi_pitch[t], p_hz)

        return multi_pitch


class ComboSet(TrainSet):
    """
    Support for training with multiple datasets.
    """

    def __init__(self, datasets):
        """
        TODO.

        Parameters
        ----------
        datasets : list of TrainSets
          Pre-initialized datasets from which to sample
        """

        self.datasets = datasets

    def __len__(self):
        """
        Number of samples across all datasets.

        Returns
        ----------
        length : int
          TODO
        """

        # Add together length of all constituent datasets
        length = sum([d.__len__() for d in self.datasets])

        return length

    def __getitem__(self, index):
        """
        TODO.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        audio : TODO
          TODO
        """

        local_idx, dataset_idx = index, 0

        while local_idx >= self.datasets[dataset_idx].__len__():
            # Remove the index offset from this dataset
            local_idx -= self.datasets[dataset_idx].__len__()
            # Check the next dataset
            dataset_idx += 1

        # Get the audio at the sampled dataset's local index
        audio = self.datasets[dataset_idx].__getitem__(local_idx)

        return audio
