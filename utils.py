# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from tqdm import tqdm

import torch.nn.functional as F
import numpy as np
import requests
import tarfile
import zipfile
import shutil
import random
import scipy
import torch
import math
import os


def stream_url_resource(url, save_path, chunk_size=1024):
    """
    Download a file at a URL by streaming it.

    Parameters
    ----------
    url : string
      URL pointing to the file
    save_path : string
      Path to the save location (including the file name)
    chunk_size : int
      Number of bytes to download at a time
    """

    # Create an HTTP GET request
    r = requests.get(url, stream=True)

    # Determine the total number of bytes to be downloaded
    total_length = int(r.headers.get('content-length'))

    # Open the target file in write mode
    with open(save_path, 'wb') as file:
        # Iteratively download chunks of the file,
        # displaying a progress bar in the console
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),
                          total=int(total_length/chunk_size+1)):
            # If a chunk was successfully downloaded,
            if chunk:
                # Write the chunk to the file
                file.write(chunk)


def unzip_and_remove(zip_path, target=None, tar=False):
    """
    Unzip a zip file and remove it.

    Parameters
    ----------
    zip_path : string
      Path to the zip file
    target : string or None
      Directory to extract the zip file contents into
    tar : bool
      Whether the compressed file is in TAR format
    """

    print(f'Unzipping {os.path.basename(zip_path)}')

    # Default the save location as the same directory as the zip file
    if target is None:
        target = os.path.dirname(zip_path)

    if tar:
        # Open the tar file in read mode
        with tarfile.open(zip_path, 'r') as tar_ref:
            # Extract the contents into the target directory
            tar_ref.extractall(target)
    else:
        # Open the zip file in read mode
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract the contents into the target directory
            zip_ref.extractall(target)

    # Delete the zip file
    os.remove(zip_path)


def change_base_dir(new_dir, old_dir):
    """
    Change the top-level directory from the path chain of each file.

    Parameters
    ----------
    new_dir : string
      New top-level directory
    old_dir : string
      Old top-level directory
    """

    # Loop through all contents of the old directory
    for content in os.listdir(old_dir):
        # Construct the old path to the contents
        old_path = os.path.join(old_dir, content)
        # Construct the new path to the contents
        new_path = os.path.join(new_dir, content)
        # Move all files and subdirectories recursively
        shutil.move(old_path, new_path)

    # Remove the (now empty) old top-level directory
    os.rmdir(old_dir)


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
      Tensor of values log-scale scaled between 0 and 1
    """

    # Make sure provided lower boundary is positive
    negative_infinity_dB = abs(negative_infinity_dB)

    # Scale decibels to be between 0 and 1
    scaled = 1 + (decibels / negative_infinity_dB)

    return scaled


def filter_non_peaks(_arr):
    """
    Remove any values that are not peaks along the vertical axis.

    Parameters
    ----------
    _arr : ndarray (... x H x W)
      Original data

    Returns
    ----------
    arr : ndarray (... x H x W)
      Data with non-peaks removed
    """

    # Create an extra row to all for edge peaks
    extra_row = np.zeros((1, _arr.shape[-1]))

    # Pad the given array with extra rows
    padded_arr = np.concatenate((extra_row, _arr, extra_row))

    # Initialize an array to hold filtered data
    arr = np.zeros(padded_arr.shape)

    # Determine which indices correspond to peaks
    peaks = scipy.signal.argrelmax(padded_arr, axis=-2)

    # Transfer peaks to new array
    arr[peaks] = padded_arr[peaks]

    # Remove padded rows
    arr = arr[..., 1 : -1, :]

    return arr


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
