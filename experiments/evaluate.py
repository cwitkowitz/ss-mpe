# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.datasets import NoteDataset, constants
from ss_mpe.models import filter_non_peaks, threshold
from ss_mpe.models.objectives import *
from utils import *

# Regular imports
from scipy.stats import hmean
from copy import deepcopy

import numpy as np
import mir_eval
import librosa
import torch
import sys


EPSILON = sys.float_info.epsilon


class MultipitchEvaluator(object):
    """
    A simple tracker to store results and compute statistics across an entire test set.
    """

    def __init__(self, tolerance=0.5):
        """
        Initialize the tracker.

        Parameters
        ----------
        tolerance : float
          Semitone tolerance for correct predictions
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
        Append the results for a test sample.

        Parameters
        ----------
        results : dict of {str : float} entries
          Numerical results for a single sample
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
        Compute the mean and standard deviation for each metric across currently tracked results.

        Returns
        ----------
        mean : dict of {str : float} entries
          Average scores across currently tracked results
        std_dev : dict of {str : float} entries
          Standard deviation of scores across currently tracked results
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
        Compute MPE results for a set of predictions using mir_eval.

        Parameters
        ----------
        times_est : ndarray (T)
          Times corresponding to multi-pitch estimates
        multi_pitch_est : list of ndarray (T x [...])
          Frame-level multi-pitch estimates
        times_ref : ndarray (K)
          Times corresponding to ground-truth multi-pitch
        multi_pitch_ref : list of ndarray (K x [...])
          Frame-level multi-pitch ground-truth

        Returns
        ----------
        results : dict of {str : float} entries
          Numerical MPE results for a set of predictions
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

        for k in deepcopy(results).keys():
            # Prepend tag to indicate MPE metric
            results[f'mpe/{k}'] = results.pop(k)

        return results


def evaluate(model, eval_set, multipliers, writer=None, i=0, device='cpu', eq_fn=None, **eq_kwargs):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # Add model to selected device and switch to evaluation mode
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Loop through tracks
        for data in eval_set:
            # Determine which track is being processed
            track = data[constants.KEY_TRACK]
            # Extract audio and add to the appropriate device
            audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)
            # Extract ground-truth targets as a Tensor
            ground_truth = torch.Tensor(data[constants.KEY_GROUND_TRUTH])

            if isinstance(eval_set, NoteDataset):
                # Extract frame times of ground-truth targets as reference
                times_ref = data[constants.KEY_TIMES]
                # Obtain the ground-truth note annotations
                pitches, intervals = eval_set.get_ground_truth(track)
                # Convert note pitches to Hertz
                pitches = librosa.midi_to_hz(pitches)
                # Convert the note annotations to multi-pitch annotations
                multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)
            else:
                # Obtain the ground-truth multi-pitch annotations
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)

            # Compute full set of spectral features
            features = model.get_all_features(audio)

            # Extract relevant feature sets
            features_log   = features['dec']
            features_log_1 = features['dec_1']
            features_log_h = features['dec_h']

            # Process features to obtain logits
            logits, _, losses = model(features_log)
            # Convert to (implicit) pitch salience activations
            transcription = torch.sigmoid(logits)

            # Determine the times associated with predictions
            times_est = model.hcqt.get_times(model.hcqt.get_expected_frames(audio.size(-1)))
            # Perform peak-picking and thresholding on the activations
            activations = threshold(filter_non_peaks(to_array(transcription)), 0.5).squeeze(0)

            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, model.hcqt.get_midi_freqs())

            # Compute results for this track using mir_eval multi-pitch metrics
            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)
            # Store the computed results
            evaluator.append_results(results)

            # Compute support loss w.r.t. first harmonic for the track
            support_loss = compute_support_loss(logits, features_log_1)
            # Compute harmonic loss w.r.t. weighted harmonic sum for the track
            harmonic_loss = compute_harmonic_loss(logits, features_log_h)
            # Compute sparsity loss for the track
            sparsity_loss = compute_sparsity_loss(transcription)
            # Compute the total loss for the track
            total_loss = multipliers['support'] * support_loss + \
                         multipliers['harmonic'] * harmonic_loss + \
                         multipliers['sparsity'] * sparsity_loss

            if eq_fn is not None:
                # Compute timbre loss for the track using specified equalization
                timbre_loss = compute_timbre_loss(model, features_log, logits, eq_fn, **eq_kwargs)
                # Store the timbre loss for the track
                evaluator.append_results({'loss/timbre' : timbre_loss.item()})
                # Add the timbre loss to the total loss
                total_loss += multipliers['timbre'] * timbre_loss

            for key_loss, val_loss in losses.items():
                # Store the model loss for the track
                evaluator.append_results({f'loss/{key_loss}' : val_loss.item()})
                # Add the model loss to the total loss
                total_loss += multipliers.get(key_loss, 1) * val_loss

            # Store all losses for the track
            evaluator.append_results({'loss/support' : support_loss.item(),
                                      'loss/harmonic' : harmonic_loss.item(),
                                      'loss/sparsity' : sparsity_loss.item(),
                                      'loss/total' : total_loss.item()})

        # Compute the average for all scores
        average_results, _ = evaluator.average_results()

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'{eval_set.name()}/{key}', average_results[key], i)

            # Add channel dimension to input/outputs
            ground_truth = ground_truth.unsqueeze(-3)
            transcription = transcription.unsqueeze(-3)
            features_log_1 = features_log_1.unsqueeze(-3)
            features_log_h = features_log_h.unsqueeze(-3)

            # Remove batch dimension from inputs
            transcription = transcription.squeeze(0)
            features_log_1 = features_log_1.squeeze(0)
            features_log_h = features_log_h.squeeze(0)

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'{eval_set.name()}/ground-truth', ground_truth.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/transcription', transcription.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/CQT (dB)', features_log_1.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/W.Avg. HCQT', features_log_h.flip(-2), i)

    return average_results
