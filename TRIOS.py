# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import stream_url_resource, unzip_and_remove, change_base_dir
from common import EvalSetNoteLevel

# Regular imports
import numpy as np
import pretty_midi
import librosa
import os


class TRIOS(EvalSetNoteLevel):
    """
    Implements a wrapper for the TRIOS dataset (https://zenodo.org/record/6797837).
    """

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

        # Obtain track names as the directories under the top-level
        tracks = sorted([d for d in os.listdir(self.base_dir)
                         if os.path.isdir(os.path.join(self.base_dir, d))])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          TRIOS track name

        Returns
        ----------
        wav_path : string
          Path to the specified track's audio
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, track, 'mix.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the paths to a track's ground truth.

        Parameters
        ----------
        track : string
          TRIOS track name

        Returns
        ----------
        midi_paths : string
          TODO
        """

        # Obtain a list of all files under the track's directory
        track_files = os.listdir(os.path.join(self.base_dir, track))

        # Get the path for the MIDI annotations of (unpitched) instruments in the mixture
        midi_paths = [os.path.join(self.base_dir, track, f) for f in track_files
                      if '.mid' in f and not ('kick' in f or 'ride' in f or 'snare' in f)]

        return midi_paths

    def get_ground_truth(self, track, times):
        """
        Get the ground-truth for a track.

        Parameters
        ----------
        track : string
          TRIOS track name
        times : ndarray (T)
          Frame times to use when constructing ground-truth

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the paths to the track's ground_truth
        midi_paths = self.get_ground_truth_path(track)

        # Obtain an empty array for inserting ground-truth
        ground_truth = super().get_ground_truth(track, times)

        # Initialize arrays to hold note attributes for all instruments
        onsets, offsets, pitches = np.empty(0), np.empty(0), np.empty(0)

        # Loop through instruments
        for m in midi_paths:
            # Extract the notes from the MIDI file
            notes = pretty_midi.PrettyMIDI(m).instruments[0].notes
            # Determine relevant attributes of each note
            _onsets, _offsets, _pitches = np.array([(n.start, n.end, n.pitch)
                                                    for n in notes]).transpose()

            # Append the note attributes for this instrument
            onsets = np.append(onsets, _onsets)
            offsets = np.append(offsets, _offsets)
            pitches = np.append(pitches, _pitches)

        # Combine onsets and offsets to obtain note intervals
        intervals = np.concatenate(([onsets], [offsets])).T

        # Convert MIDI notes to multi-pitch annotations aligned with provided times
        multi_pitch = self.notes_to_multi_pitch(pitches, intervals, times)

        return times, multi_pitch

    @classmethod
    def download(cls, save_dir):
        """
        Download the TRIOS dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of TRIOS
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        url = 'https://zenodo.org/record/6797837/files/TRIOS Dataset.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'TRIOS Dataset'))
