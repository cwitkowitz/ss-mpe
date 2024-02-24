# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch
import time


__all__ = [
    'get_current_time',
    'print_time_difference'
]


def get_current_time(decimals=3):
    """
    Determine the current system time.

    Parameters
    ----------
    decimals : int
      Number of digits to keep when rounding

    Returns
    ----------
    current_time : float
      Current system time
    """

    # Get current time rounded to specified number of digits
    current_time = round(time.time(), decimals)

    return current_time


def print_time_difference(start_time, label=None, decimals=3, device=None):
    """
    Print the time elapsed since the given system time.

    Parameters
    ----------
    start_time : float
      Arbitrary system time
    decimals : int
      Number of digits to keep when rounding
    label : string or None (Optional)
      Label for the optional print statement
    device : string
      CUDA device currently in use

    Returns
    ----------
    elapsed_time : float
      Time elapsed since specified time
    """

    # Wait until CUDA processes finish
    torch.cuda.synchronize(device)

    # Take rounded difference between current time and given time
    elapsed_time = round(get_current_time(decimals) - start_time, decimals)

    # Initialize string to print
    message = 'Time'

    if label is not None:
        # Add label if it was specified
        message += f' ({label})'

    # Add the time to the string
    message += f' : {elapsed_time}'

    # Print constructed string
    print(message)
