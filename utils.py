# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from tqdm import tqdm

import numpy as np
import requests
import tarfile
import random
import torch
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


def untar_and_remove(tar_path, target=None):
    """
    Untar a tar file and remove it.

    Parameters
    ----------
    tar_path : string
      Path to the tar file
    target : string or None
      Directory to extract the tar file contents into
    """

    print(f'Untarring {os.path.basename(tar_path)}')

    # Default the save location as the same directory as the tar file
    if target is None:
        target = os.path.dirname(tar_path)

    # Open the tar file in read mode
    with tarfile.open(tar_path, 'r') as tar_ref:
        # Extract the contents of the tar file into the target directory
        tar_ref.extractall(target)

    # Delete the tar file
    os.remove(tar_path)


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


def normalize(_arr):
    """
    Normalize an array such that values fall within the range [-1, 1].

    Parameters
    ----------
    _arr : np.ndarray
      Original data

    Returns
    ----------
    arr : np.ndarray
      Normalized data
    """

    # Identify the element with the highest magnitude
    max = np.max(np.abs(_arr))

    if max > 0:
        # Divide by this value
        arr = _arr / max

    return arr
