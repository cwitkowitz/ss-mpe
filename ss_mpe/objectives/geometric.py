# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'apply_geometric_transformations',
    'apply_random_transformations',
    'compute_geometric_loss'
]


def apply_translation(tensor, shifts, axis=-1, val=None):
    """
    Perform an independent translation on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x ...)
      Input tensor with at least 2 dimensions
    shifts : Tensor (B)
      Independent translations to perform
    axis : int
      Axis to translate
    val : float or None (optional)
      Value to insert or None to wrap original data

    Returns
    ----------
    translated : Tensor (B x ...)
      Original tensor translated as specified
    """

    # Determine dimensionality and device of input tensor
    dimensionality, device = tensor.size(), tensor.device

    # Copy original tensor
    tensor = tensor.clone()

    if val is not None:
        # Initialize hidden data to replace wrapped elements
        hidden_data = torch.full(dimensionality, val, device=device)
        # Combine original and hidden data to disable wrapping
        tensor = torch.cat([tensor, hidden_data], dim=axis)

    # Translate each entry by specified amount at specified axis
    translated = torch.cat([x.unsqueeze(0).roll(k.item(), axis)
                            for x, k in zip(tensor, shifts)])

    # Trim translated tensor to original dimensionality
    translated = translated.narrow(axis, 0, dimensionality[axis])

    return translated


def apply_distortion(tensor, stretch_factors):
    """
    Perform an independent distortion on each entry of a tensor.

    Parameters
    ----------
    tensor : Tensor (B x C x H x W)
      Input tensor with standard 4 dimensions
    stretch_factors : Tensor (B)
      Independent distortions to perform

    Returns
    ----------
    distorted : Tensor (B x ...)
      Original tensor distorted as specified
    """

    # Initialize list for distortions
    distorted = list()

    # Loop through each entry and scale
    for x, t in zip(tensor, stretch_factors):
        # Stretch entry by specified factor using linear interpolation
        distorted_ = F.interpolate(x, scale_factor=t.item(), mode='linear')

        if t >= 1:
            # Determine starting index to center distortion
            start_idx = (distorted_.size(-1) - x.size(-1)) // 2
            # Center distorted tensor and trim to original width
            distorted_ = distorted_.narrow(-1, start_idx, x.size(-1))
        else:
            # Determine total padding necessary
            pad_t = x.size(-1) - distorted_.size(-1)
            # Distribute padding between both sides
            pad_l, pad_r = pad_t // 2, pad_t - pad_t // 2
            # Pad distorted tensor to match original width
            distorted_ = F.pad(distorted_, (pad_l, pad_r))

        # Append distorted entry to distortion list
        distorted.append(distorted_.unsqueeze(0))

    # Combine all distorted entries
    distorted = torch.cat(distorted)

    return distorted


def apply_geometric_transformations(tensor, vs, hs, sfs):
    # Apply vertical and horizontal translations
    transformed_features = apply_translation(tensor, vs, axis=-2, val=0)
    transformed_features = apply_translation(transformed_features, hs, axis=-1, val=0)
    # Apply horizontal stretching while keeping original dimensionality
    transformed_features = apply_distortion(transformed_features, sfs)

    return transformed_features


def apply_random_transformations(features, max_shift_v, max_shift_h, max_stretch_factor):
    # Determine batch size
    B = features.size(0)

    # Sample a random vertical / horizontal shift for each sample in the batch
    vs = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(B,))
    hs = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(B,))

    # Compute inverse of maximum stretch factor
    min_stretch_factor = 1 / max_stretch_factor

    # Sample a random stretch factor for each sample in the batch
    sfs, stretch_factors_ = min_stretch_factor, torch.rand(size=(B,))

    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()

    # Scale stretch factors evenly across range
    sfs += neg_perc * (1 - min_stretch_factor)
    sfs += pos_perc * (max_stretch_factor - 1)

    # Apply the sampled geometric transformations to the provided features
    transformed_features = apply_geometric_transformations(features, vs, hs, sfs)

    return transformed_features, (vs, hs, sfs)


def compute_geometric_loss(model, features, targets, **gm_kwargs):
    # Perform random geometric transformations on batch of features
    transformed_features, (vs, hs, sfs) = apply_random_transformations(features, **gm_kwargs)

    # Process transformed features with provided model
    transformation_embeddings, _ = model(transformed_features)

    # Add a temporary channel dimension
    targets = targets.unsqueeze(-3)

    # Apply parallel geometric transformation to provided targets
    transformed_targets = apply_geometric_transformations(targets, vs, hs, sfs)

    # Remove temporarily added channel dimension
    transformed_targets = transformed_targets.squeeze(-3)

    # Compute geometric loss as BCE of embeddings computed from transformed features with respect to transformed targets
    geometric_loss = F.binary_cross_entropy_with_logits(transformation_embeddings, transformed_targets, reduction='none')
    #geometric_loss = F.cross_entropy(transformation_embeddings, transformed_targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    geometric_loss = geometric_loss.sum(-2).mean(-1).mean(-1)
    #geometric_loss = geometric_loss.mean(-1).mean(-1)

    powers = 1 + torch.arange(model.hcqt_params['n_bins'])

    alpha = 1.019440644  # 2 ** (1/36)
    tau = 0.122462048  # 2 ** (1/6) - 1

    alphas = (alpha ** powers).unsqueeze(-1).to(targets.device)

    projected_targets = alphas * targets.squeeze(-3)
    projected_transformation = alphas * torch.sigmoid(transformation_embeddings)

    projected_difference = projected_targets / projected_transformation - (alpha ** vs).unsqueeze(-1).unsqueeze(-1).to(targets.device)

    x = projected_difference.abs()

    equivariance_loss = torch.where(x.le(tau), x ** 2 / 2, tau ** 2 / 2 + tau * (x - tau))

    equivariance_loss = equivariance_loss.sum(-2).mean(-1).mean(-1)

    return geometric_loss
