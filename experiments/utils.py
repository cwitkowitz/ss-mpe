# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import matplotlib.pyplot as plt
import random
import torch
import math
import time


__all__ = [
    'seed_everything',
    'to_array',
    'print_and_log',
    'get_current_time',
    'print_time_difference',
    'initialize_figure',
    'plot_magnitude',
    'track_gradient_norms',
    'cosine_anneal',
    'CosineWarmup',
]


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


def to_array(tensor):
    """
    Convert a PyTorch Tensor to a Numpy ndarray.

    Parameters
    ----------
    tensor : Tensor
      Arbitrary tensor data

    Returns
    ----------
    arr : ndarray
      Same data as Numpy ndarray
    """

    # Move to CPU, detach gradient, and convert to ndarray
    arr = tensor.cpu().detach().numpy()

    return arr


def print_and_log(text, path=None):
    """
    Print a string to the console and optionally log it to a specified file.

    Parameters
    ----------
    text : str
      Text to print/log
    path : str (None to bypass)
      Path to file to write text
    """

    # Print text to the console
    print(text)

    if path is not None:
        with open(path, 'a') as f:
            # Append the text to the file
            print(text, file=f)


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


def print_time_difference(start_time, label=None, decimals=3):
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

    Returns
    ----------
    elapsed_time : float
      Time elapsed since specified time
    """

    # Wait until CUDA processes finish
    torch.cuda.synchronize()

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


def initialize_figure(figsize=(9, 3), interactive=False):
    """
    Create a new figure and display it.

    Parameters
    ----------
    figsize : tuple (x, y) or None (Optional)
      Size of plot window in inches - if unspecified set to default
    interactive : bool
      Whether to set turn on matplotlib interactive mode

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the created figure
    """

    if interactive and not plt.isinteractive():
        # Make sure pyplot is in interactive mode
        plt.ion()

    # Create a new figure with the specified size
    fig = plt.figure(figsize=figsize, tight_layout=True)

    if not interactive:
        # Open the figure manually if interactive mode is off
        plt.show(block=False)

    return fig


def plot_magnitude(magnitude, extent=None, fig=None, save_path=None):
    """
    Plot magnitude coefficients within range [0, 1].

    Parameters
    ----------
    magnitude : ndarray (F x T)
      Magnitude coefficients [0, 1]
      F - number of frequency bins
      T - number of frames
    extent : list [l, r, b, t] or None (Optional)
      Boundaries of horizontal and vertical axis
    fig : matplotlib Figure object
      Preexisting figure to use for plotting
    save_path : string or None (Optional)
      Save the figure to this path

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the TFR
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if extent is not None:
        # Swap position of bottom and top
        extent = [extent[0], extent[1],
                  extent[3], extent[2]]

    # Plot magnitude as an image
    ax.imshow(magnitude, vmin=0, vmax=1, extent=extent)
    # Flip the axis for ascending pitch
    ax.invert_yaxis()
    # Make sure the image fills the figure
    ax.set_aspect('auto')

    if extent is not None:
        # Add axis labels
        ax.set_ylabel('Frequency (MIDI)')
        ax.set_xlabel('Time (s)')
    else:
        # Hide the axes
        ax.axis('off')

    if save_path is not None:
        # Save the figure
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return fig


def track_gradient_norms(module, writer=None, i=0, prefix='gradients'):
    """
    Compute the cumulative gradient norm of a network across all layers.

    Parameters
    ----------
    module : torch.nn.Module
      Network containing gradients to track
    writer : SummaryWriter
      Results logger for tensorboard
    i : int
      Current iteration for logging
    prefix : str
      Tag prefix for logging

    Returns
    ----------
    cumulative_norm : float
      Summed norms across all layers
    """

    # Initialize the cumulative norm
    cumulative_norm = 0.

    for layer, values in module.named_parameters():
        if values.grad is not None:
            # Compute the L2 norm of the gradients
            grad_norm = values.grad.norm(2).item()

            if writer is not None:
                # Log the norm of the gradients for this layer
                writer.add_scalar(f'{prefix}/{layer}', grad_norm, i)

            # Accumulate the norm of the gradients
            cumulative_norm += grad_norm

    return cumulative_norm


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


class CosineWarmup(torch.optim.lr_scheduler.LRScheduler):
    """
    A simple wrapper to implement reverse cosine annealing as a PyTorch LRScheduler.
    """

    def __init__(self, optimizer, n_steps, last_epoch=-1, verbose=False):
        """
        Initialize the scheduler and set the duration of warmup.

        Parameters
        ----------
        See LRScheduler class...
        """

        self.n_steps = max(1, n_steps)

        super().__init__(optimizer, last_epoch, verbose)

    def is_active(self):
        """
        Helper to determine when to stop stepping.
        """

        active = self.last_epoch < self.n_steps

        return active

    def get_lr(self):
        """
        Obtain scheduler learning rates.
        """

        # Simply use closed form expression
        lr = self._get_closed_form_lr()

        return lr

    def _get_closed_form_lr(self):
        """
        Compute the learning rates for the current step.
        """

        # Clamp the current step at the chosen number of steps
        curr_step = max(0, min(self.last_epoch, self.n_steps))
        # Compute scaling corresponding to current step
        scaling = 1 - 0.5 * (1 + math.cos(curr_step * math.pi / self.n_steps))
        # Apply the scaling to each learning rate
        lr = [scaling * base_lr for base_lr in self.base_lrs]

        return lr
