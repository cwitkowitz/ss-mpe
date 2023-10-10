# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt import torch_amplitude_to_db

from utils import *

# Regular imports
from scipy.stats import hmean
from copy import deepcopy

import numpy as np
import mir_eval
import librosa
import sys


EPSILON = sys.float_info.epsilon


class MultipitchEvaluator(object):
    """
    TODO
    """

    def __init__(self, tolerance=0.5):
        """
        TODO
        """

        self.tolerance = tolerance

        # Initialize dictionary to track results
        self.results = None
        self.reset_results()

    def reset_results(self):
        """
        Reset tracked results to empty dictionary.
        """

        self.results = {}

    def append_results(self, results):
        """
        TODO.

        Parameters
        ----------
        results : TODO
          TODO
        """

        # Loop through all keys
        for key in results.keys():
            if key in self.results.keys():
                # Add the provided score to the pre-existing array
                self.results[key] = np.append(self.results[key], results[key])
            else:
                # Initialize a new array for the metric
                self.results[key] = np.array([results[key]])

    def average_results(self):
        """
        TODO.

        Returns
        ----------
        mean : TODO
          TODO
        std_dev : TODO
          TODO
        """

        # Clone all current scores
        mean = deepcopy(self.results)
        std_dev = deepcopy(self.results)

        # Loop through all metrics
        for key in self.results.keys():
            # Compute statistics for the metric
            mean[key] = round(np.mean(mean[key]), 5)
            std_dev[key] = round(np.std(std_dev[key]), 5)

        return mean, std_dev

    def evaluate(self, times_est, multi_pitch_est, times_ref, multi_pitch_ref):
        """
        TODO.

        Parameters
        ----------
        times_est : TODO
          TODO
        multi_pitch_est : TODO
          TODO
        times_ref : TODO
          TODO
        multi_pitch_ref : TODO
          TODO

        Returns
        ----------
        results : TODO
          TODO
        """

        # Use mir_eval to compute multi-pitch results at specified tolerance
        results = mir_eval.multipitch.evaluate(times_ref, multi_pitch_ref,
                                               times_est, multi_pitch_est,
                                               window=self.tolerance)

        # Make keys lowercase and switch to regular dict type
        results = {k.lower(): results[k] for k in results.keys()}

        # Calculate the f1-score using the harmonic mean formula
        f_measure = hmean([results['precision'] + EPSILON,
                           results['recall'] + EPSILON]) - EPSILON

        # Add f1-score to the mir_eval results
        results.update({'f1-score' : f_measure})

        return results


def evaluate(model, hcqt, eval_set, writer=None, i=0, device='cpu'):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # Unwrap the HCQT module from DataParallel to access attributes
    h = hcqt.module if isinstance(hcqt, torch.nn.DataParallel) else hcqt

    # Determine the MIDI frequency associated with each bin of the input/output
    midi_freqs = librosa.hz_to_midi(h.fmin) + np.arange(h.n_bins) / (h.bins_per_octave / 12)

    # Place model in evaluation mode
    model.eval()

    with torch.no_grad():
        # Loop through each testing track
        for track in eval_set.tracks:
            # Extract the audio for this track and add to appropriate device
            audio = eval_set.get_audio(track).to(device).unsqueeze(0)

            # Obtain spectral features in decibels
            features_dec = torch_amplitude_to_db(hcqt(audio))
            # Obtain amplitude features for the audio
            features_lin = decibels_to_amplitude(features_dec)
            # Obtain log-scale features for the audio
            features_log = rescale_decibels(features_dec)
            # Compute the pitch salience of the features
            salience = torch.sigmoid(model(features_log).squeeze())
            # Peak-pick and threshold salience to obtain binarized activations
            activations = np.round(filter_non_peaks(salience.cpu().numpy()))

            # Determine the times associated with predictions
            times_est = h.get_times(audio)
            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, midi_freqs)

            if eval_set.has_frame_level_annotations():
                # Extract the ground-truth multi-pitch annotations for this track
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)
            else:
                # Construct the ground-truth multi-pitch annotations for this track
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track, times_est)

            # Compute results for this track using mir_eval multi-pitch metrics
            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)
            # Track the computed results
            evaluator.append_results(results)

        # Compute the average for all scores
        average_results, _ = evaluator.average_results()

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'val-{eval_set.name()}/{key}', average_results[key], i)

            if eval_set.has_frame_level_annotations():
                # Resample the multi-pitch annotations to align with predicted times
                multi_pitch_ref = eval_set.resample_multi_pitch(times_ref, multi_pitch_ref, times_est)
            # Convert the multi-pitch annotations to activations for visualization
            ground_truth = torch.Tensor(eval_set.multi_pitch_to_activations(multi_pitch_ref, midi_freqs))

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'val-{eval_set.name()}/log-scaled CQT', features_log.squeeze()[1: 2].flip(-2), i)
            writer.add_image(f'val-{eval_set.name()}/amplitude CQT', features_lin.squeeze()[1: 2].flip(-2), i)
            writer.add_image(f'val-{eval_set.name()}/pitch salience', salience.unsqueeze(0).flip(-2), i)
            writer.add_image(f'val-{eval_set.name()}/ground-truth', ground_truth.unsqueeze(0).flip(-2), i)

    return average_results
