# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from timbre_trap.utils.visualization import initialize_figure

# Regular imports
from matplotlib.pyplot import Figure

import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'plot_bce_loss',
    'plot_equalization'
]


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
    ax.set_ylabel('Target')
    ax.set_xlabel('Estimate')

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

    # Number of frequency bins
    F = curve.shape[-1]

    # Plot upright equalization curve
    ax[1].plot(curve, np.arange(F))
    # Compress axes to valid range
    ax[1].set_xlim(0.5, 1.5)
    ax[1].set_ylim(0, F)
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
