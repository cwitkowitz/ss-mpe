# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import math


__all__ = [
    'cosine_anneal'
]


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
