# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
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
            results[key] = np.mean(results[key])

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

        # Calculate the total percentage of correct predictions
        accuracy = num_correct / len(flattened_multi_pitch_est)

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
            # Obtain features for the audio
            features = decibels_to_linear(hcqt(audio.to(device)))
            # Compute the pitch salience of the features
            salience = torch.sigmoid(model(features).squeeze())
            # Threshold the salience to obtain multi pitch predictions
            multi_pitch = np.round(salience.cpu().numpy())
            # Bring the ground-truth to the cpu
            ground_truth = ground_truth.squeeze().cpu()
            # Compute results for this track
            # TODO - upper bound when raw CQT is fed in?
            #results = evaluator.evaluate(features[0, 1].cpu().detach().numpy(), ground_truth)
            results = evaluator.evaluate(multi_pitch, ground_truth.numpy())
            # Track the computed results
            evaluator.append_results(results)

        # Compute the average for all scores
        average_results = evaluator.average_results()

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'val/{eval_set.name()}/{key}', average_results[key], i)

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'val/{eval_set.name()}/CQT', features.squeeze()[1: 2].flip(-2), i)
            writer.add_image(f'val/{eval_set.name()}/salience', salience.unsqueeze(0).flip(-2), i)
            writer.add_image(f'val/{eval_set.name()}/ground-truth', ground_truth.unsqueeze(0).flip(-2), i)
