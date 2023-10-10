# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import numpy as np
import random
import scipy
import torch
import math


def seed_everything(seed):
    """
    Set all necessary seeds for PyTorch at once.
    WARNING: the number of workers in the training loader affects behavior:
             this is because each sample will inevitably end up being processed
             by a different worker if num_workers is changed, and each worker
             has its own random seed

    Parameters
    ----------
    seed : int
      Seed to use for random number generation
    """

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def cosine_anneal(i, n_steps, start=0, floor=0.):
    """
    Obtain a decaying scaling factor based on cosine annealing.

    Parameters
    ----------
    i : int
      Current step
    n_steps : int
      Number of steps across which annealing occurs
    start : int (optional)
      Step where annealing begins
    floor : float [0, 1] (optional)
      Percentage floor

    Returns
    ----------
    scaling : float [0, 1]
      Scaling factor for the current step
    """

    # Compute scaling within range [0, n_steps]
    x = max(0, min(i - start, n_steps))

    # Compute scaling factor for the current iteration
    scaling = 0.5 * (1 + math.cos(x * math.pi / n_steps))

    # Compress the scaling between [floor, 1]
    scaling = (1 - floor) * scaling + floor

    return scaling


def normalize(_arr):
    """
    Normalize an array such that values fall within the range [-1, 1].

    Parameters
    ----------
    _arr : ndarray
      Original data

    Returns
    ----------
    arr : ndarray
      Normalized data
    """

    # Identify the element with the highest magnitude
    max = np.max(np.abs(_arr))

    if max > 0:
        # Divide by this value
        arr = _arr / max

    return arr


def decibels_to_amplitude(decibels, negative_infinity_dB=-80):
    """
    Convert a tensor of decibel values to amplitudes between 0 and 1.

    Parameters
    ----------
    decibels : ndarray or Tensor
      Tensor of decibel values with a ceiling of 0
    negative_infinity_dB : float
      Decibel cutoff beyond which is considered negative infinity

    Returns
    ----------
    gain : ndarray or Tensor
      Tensor of values linearly scaled between 0 and 1
    """

    # Make sure provided lower boundary is negative
    negative_infinity_dB = -abs(negative_infinity_dB)

    # Convert decibels to a gain between 0 and 1
    gain = 10 ** (decibels / 20)
    # Set gain of values below -∞ to 0
    gain[decibels <= negative_infinity_dB] = 0

    return gain


def rescale_decibels(decibels, negative_infinity_dB=-80):
    """
    Log-scale a tensor of decibel values between 0 and 1.

    Parameters
    ----------
    decibels : ndarray or Tensor
      Tensor of decibel values with a ceiling of 0
    negative_infinity_dB : float
      Decibel cutoff beyond which is considered negative infinity

    Returns
    ----------
    scaled : ndarray or Tensor
      Decibel values scaled logarithmically between 0 and 1
    """

    # Make sure provided lower boundary is positive
    negative_infinity_dB = abs(negative_infinity_dB)

    # Scale decibels to be between 0 and 1
    scaled = 1 + (decibels / negative_infinity_dB)

    return scaled


def threshold(_arr, t=0.5):
    """
    Binarize data based on a given threshold.

    Parameters
    ----------
    _arr : ndarray
      Original data
    t : float [0, 1]
      Threshold value

    Returns
    ----------
    arr : ndarray
      Binarized data
    """

    # Initialize an array to hold binarized data
    arr = np.zeros(_arr.shape)
    # Set values above threshold to one
    arr[_arr >= t] = 1

    return arr


# TODO - can this function be sped up?
def translate_batch(batch, shifts, dim=-1, val=0):
    """
    TODO
    """

    # Determine the dimensionality of the batch
    dimensionality = batch.size()

    # Combine the original tensor with tensor filled with zeros such that no wrapping will occur
    rolled_batch = torch.cat([batch, val * torch.ones(dimensionality, device=batch.device)], dim=dim)

    # Roll each sample in the batch independently and reconstruct the tensor
    rolled_batch = torch.cat([x.unsqueeze(0).roll(i, dim) for x, i in zip(rolled_batch, shifts)])

    # Trim the rolled tensor to its original dimensionality
    translated_batch = rolled_batch.narrow(dim, 0, dimensionality[dim])

    return translated_batch


# TODO - can this function be sped up?
def stretch_batch(batch, stretch_factors):
    """
    TODO
    """

    # Determine height and width of the batch
    H, W = batch.size(-2), batch.size(-1)

    # Inserted stretched values to a copy of the original tensor
    stretched_batch = batch.clone()

    # Loop through each sample and stretch factor in the batch
    for i, (sample, factor) in enumerate(zip(batch, stretch_factors)):
        # Reshape the sample to B x H x W
        original = sample.reshape(-1, H, W)
        # Stretch the sample by the specified amount
        stretched_sample = F.interpolate(original,
                                         scale_factor=factor,
                                         mode='linear',
                                         align_corners=True)

        # Patch upsampled -∞ values that end up being NaNs (a little hacky)
        stretched_sample[stretched_sample.isnan()] = -torch.inf

        if factor < 1:
            # Determine how much padding is necessary
            pad_amount = W - stretched_sample.size(-1)
            # Pad the stretched sample to fit original width
            stretched_sample = F.pad(stretched_sample, (0, pad_amount))

        # Insert the stretched sample back into the batch
        stretched_batch[i] = stretched_sample[..., :W].view(sample.shape)

    return stretched_batch
