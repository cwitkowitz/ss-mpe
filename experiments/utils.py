# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from matplotlib.pyplot import Figure

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import math


__all__ = [
    'seed_everything',
    'to_array',
    'print_and_log',
    'initialize_figure',
    'plot_magnitude',
    'plot_bce_loss',
    'plot_equalization',
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
      A handle for the figure used to plot TFR
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
    # Flip y-axis for ascending pitch
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


def plot_bce_loss(loss, title=None, colorbar=False, fig=None, save_path=None):
    """
    Plot BCE loss landscape over entire range of ground-truth (y) / estimates (x).

    Parameters
    ----------
    loss : ndarray (N + 1 x N - 1)
      Loss over potential combinations [0, ?]
    title : string or None (Optional)
      Title to add above image
    colorbar : bool
      Whether to include a colorbar for reference
    fig : matplotlib Figure object
      Preexisting figure to use for plotting
    save_path : string or None (Optional)
      Save the figure to this path

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot loss landscape
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(figsize=(6, 5), interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    # Activation boundaries
    extent = [0, 1, 1, 0]

    # Plot loss landscape as an image
    img = ax.imshow(loss, vmin=0, vmax=np.max(loss), extent=extent)
    # Flip y-axis for ascending activations
    ax.invert_yaxis()
    # Make sure the image fills the figure
    ax.set_aspect('auto')

    if colorbar:
        # Add a legend to image
        fig.colorbar(img)

    # Add axis labels
    ax.set_ylabel('Ground-Truth')
    ax.set_xlabel('Estimated')

    if title is not None:
        # Add title to plot
        ax.set_title(title)

    if save_path is not None:
        # Save the figure
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return fig


def plot_equalization(original, curve, fig=None, save_path=None):
    """
    Plot original and equalized (normalized decibel) magnitude features, along with equalization curve.

    Parameters
    ----------
    original : ndarray (F x T)
      Magnitude coefficients [0, 1]
      F - number of frequency bins
      T - number of frames
    curve : ndarray (F)
      Equalization curve to apply
    fig : matplotlib Figure object
      Preexisting figure to use for plotting
    save_path : string or None (Optional)
      Save the figure to this path

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot equalization
    """

    # Equalized features with provided curve
    equalized = np.expand_dims(curve, -1) * original
    # Ensure equalized features remain within bounds
    equalized = np.clip(equalized, a_min=0, a_max=1)

    if fig is None:
        # Initialize a new figure with subplots if one was not given
        (fig, ax) = plt.subplots(nrows=1, ncols=3, width_ratios=[2, 1, 2], figsize=(12, 4), tight_layout=True)
        # Open the figure manually
        plt.show(block=False)
    elif isinstance(fig, Figure):
        # Obtain a handle for the figure's current axis
        ax = fig.gca()
    else:
        # Axis was provided
        ax = fig

    # Plot original magnitude features as an image
    ax[0].imshow(original, vmin=0, vmax=1, aspect='auto', origin='lower')
    # Remove both axes
    ax[0].axis('off')
    # Add title to subplot
    ax[0].set_title('Original')

    # Plot upright equalization curve
    ax[1].plot(curve, np.arange(curve.shape[-1]))
    # Compress x-axis to valid range
    ax[1].set_xlim(0.5, 1.5)
    # Remove vertical axis
    ax[1].get_yaxis().set_visible(False)
    # Add title to subplot
    ax[1].set_title('Curve')

    # Plot equalized magnitude features as an image
    ax[2].imshow(equalized, vmin=0, vmax=1, aspect='auto', origin='lower')
    # Remove both axes
    ax[2].axis('off')
    # Add title to subplot
    ax[2].set_title('Equalized')

    if save_path is not None:
        # Save the figure
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return fig


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
