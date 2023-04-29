# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from lhvqt import torch_amplitude_to_db

from utils import *

# Regular imports
from torch.utils.data import DataLoader
from scipy.stats import hmean
from copy import deepcopy

import numpy as np
import sys


EPSILON = sys.float_info.epsilon


class MultipitchEvaluator(object):
    """
    TODO
    """

    def __init__(self):
        """
        TODO
        """

        # Initialize dictionary to track results
        self.results = None

        self.reset_results()

    def reset_results(self):
        """
        Reset tracked results to empty dictionary.
        """

        self.results = {
            'precision' : np.empty(0),
            'recall' : np.empty(0),
            'f1-score' : np.empty(0),
            'accuracy' : np.empty(0)
        }

    def append_results(self, results):
        """
        TODO.

        Parameters
        ----------
        results : TODO
          TODO
        """

        # Loop through all keys in the array
        for key in results.keys():
            # Add the provided score to the pre-existing array
            self.results[key] = np.append(self.results[key], results[key])

    def average_results(self):
        """
        TODO.
        """

        # Clone all current scores
        results = deepcopy(self.results)

        # Loop through all keys in the array
        for key in results.keys():
            # Average entries for the metric
            results[key] = round(np.mean(results[key]), 5)

        return results

    @staticmethod
    def evaluate(multi_pitch_est, multi_pitch_ref):
        """
        TODO.

        Parameters
        ----------
        multi_pitch_est : TODO
          TODO
        multi_pitch_ref : TODO
          TODO

        Returns
        ----------
        results : TODO
          TODO
        """

        # Flatten the estimated and reference data
        flattened_multi_pitch_est = multi_pitch_est.round().flatten()
        flattened_multi_pitch_ref = multi_pitch_ref.round().flatten()

        # Determine the number of correct predictions, where estimated activation lines up with reference
        num_correct = np.sum(flattened_multi_pitch_est * flattened_multi_pitch_ref, axis=-1)

        # Count the total number of positive activations predicted
        num_predicted = np.sum(flattened_multi_pitch_est, axis=-1)
        # Count the total number of positive activations referenced
        num_ground_truth = np.sum(flattened_multi_pitch_ref, axis=-1)

        # Calculate precision and recall
        precision = num_correct / (num_predicted + EPSILON)
        recall = num_correct / (num_ground_truth + EPSILON)

        # Calculate the f1-score using the harmonic mean formula
        f_measure = hmean([precision + EPSILON, recall + EPSILON]) - EPSILON

        # Calculate the accuracy of the predictions
        accuracy = num_correct / (num_predicted + num_ground_truth - num_correct + EPSILON)

        results = {
            'precision' : precision,
            'recall' : recall,
            'f1-score' : f_measure,
            'accuracy' : accuracy
        }

        return results


def evaluate(model, hcqt, eval_set, writer=None, i=0, device='cpu'):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # Initialize a PyTorch dataloader for the data
    loader = DataLoader(dataset=eval_set,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)

    # Place model in evaluation mode
    model.eval()

    with torch.no_grad():
        # Loop through each testing track
        for audio, ground_truth in loader:
            # Obtain spectral features in decibels
            features_dec = torch_amplitude_to_db(hcqt(audio.to(device)))
            # Obtain amplitude features for the audio
            features_lin = decibels_to_amplitude(features_dec)
            # Obtain log-scale features for the audio
            features_log = rescale_decibels(features_dec)
            # Compute the pitch salience of the features
            salience = torch.sigmoid(model(features_log).squeeze())
            # Threshold the salience to obtain multi pitch predictions
            multi_pitch = np.round(salience.cpu().numpy())
            # Bring the ground-truth to the cpu
            ground_truth = ground_truth.squeeze().cpu()
            # Compute results for this track
            results = evaluator.evaluate(multi_pitch, ground_truth.numpy())
            # Track the computed results
            evaluator.append_results(results)

        # Compute the average for all scores
        average_results = evaluator.average_results()

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'val-{eval_set.name()}/{key}', average_results[key], i)

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'val-{eval_set.name()}/log-scaled CQT', features_log.squeeze()[1: 2].flip(-2), i)
            writer.add_image(f'val-{eval_set.name()}/amplitude CQT', features_lin.squeeze()[1: 2].flip(-2), i)
            writer.add_image(f'val-{eval_set.name()}/pitch salience', salience.unsqueeze(0).flip(-2), i)
            writer.add_image(f'val-{eval_set.name()}/ground-truth', ground_truth.unsqueeze(0).flip(-2), i)

    return average_results
