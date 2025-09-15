# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch


__all__ = [
    'filter_non_peaks_torch',
    'gate_and_average_loss'
]


def filter_non_peaks_torch(_tnsr, fill_val=0.):
    """
    Remove any values that are not peaks along the vertical axis.

    Parameters
    ----------
    tnsr : Tensor (... x H x W)
      Original data

    Returns
    ----------
    tnsr : Tensor (... x H x W)
      Data with non-peaks removed
    """

    # Initialize array to hold filtered data
    tnsr = fill_val * torch.ones_like(_tnsr)

    # Create a tensor for positive boolean values
    bound = torch.BoolTensor([True]).to(_tnsr.device)
    # Extend boolean tensor along additional dimensions
    bound = bound.repeat(tuple(_tnsr.shape[:-2]) + (1, _tnsr.shape[-1]))

    # Determine which frequency bins have increasing activations
    increasing_activations = _tnsr[..., 1:, :] > _tnsr[..., :-1, :]
    # Add boundary for first bin of activations assuming increase
    increasing_activations = torch.cat([bound, increasing_activations], dim=-2)

    # Determine which frequency bins have decreasing activations
    decreasing_activations = _tnsr[..., :-1, :] > _tnsr[..., 1:, :]
    # Add boundary for last bin of activations assuming decrease
    decreasing_activations = torch.cat([decreasing_activations, bound], dim=-2)

    # Determine which frequency bins have activation peaks
    peaks = torch.logical_and(increasing_activations, decreasing_activations)

    # Insert activations at peaks
    tnsr[peaks] = _tnsr[peaks]

    return tnsr

def gate_and_average_loss(loss, vals, thr):
    """
    Remove any values that are not peaks along the vertical axis.

    Parameters
    ----------
    loss : Tensor (B x T)
      Frame-level loss for a batch
    vals : Tensor (B x T)
      Values associated with each frame
    thr : float
      Threshold for gating

    Returns
    ----------
    loss : tensor ()
      Loss averaged over valid frames
    """

    # Ignore loss for frames below threshold for each track
    loss_ = [c[vals[i] >= thr] for i, c in enumerate(loss)]
    # Compute average across valid frames for each track and repack
    loss = torch.stack([c.mean() for c in loss_])
    # Compute average across batch for valid tracks only
    loss = torch.mean(loss[torch.logical_not(loss.isnan())])

    return loss
