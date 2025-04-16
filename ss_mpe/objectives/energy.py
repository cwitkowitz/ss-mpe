# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from torch.distributions.categorical import Categorical

import torch.nn.functional as F
import torch


__all__ = [
    'compute_energy_loss',
    'compute_support_loss',
    'compute_harmonic_loss',
    'compute_sparsity_loss',
    'compute_entropy_loss',
    'compute_content_loss',
    'compute_contrastive_loss'
]


def compute_energy_loss(embeddings, salience):
    # Compute energy loss as BCE of activations with respect to features
    energy_loss = F.binary_cross_entropy_with_logits(embeddings, salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    energy_loss = energy_loss.sum(-2).mean(-1).mean(-1)

    return energy_loss


def compute_support_loss(embeddings, h1_features):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute support loss as BCE of activations with respect to features (negative activations only)
    support_loss = F.binary_cross_entropy_with_logits(embeddings, h1_features, reduction='none', pos_weight=pos_weight)

    # Sum across frequency bins and average across time and batch
    support_loss = support_loss.sum(-2).mean(-1).mean(-1)

    return support_loss


def compute_harmonic_loss(embeddings, salience):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute harmonic loss as BCE of activations with respect to salience estimate (positive activations only)
    harmonic_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - salience), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    harmonic_loss = harmonic_loss.sum(-2).mean(-1).mean(-1)

    return harmonic_loss


def compute_sparsity_loss(activations):
    # Compute sparsity loss as the L1 norm of the activations
    sparsity_loss = torch.norm(activations, 1, dim=-2)

    # Average loss across time and batch
    sparsity_loss = sparsity_loss.mean(-1).mean(-1)

    return sparsity_loss


def compute_entropy_loss(embeddings):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute entropy as BCE of activations with respect to themselves (positive activations only)
    entropy_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - torch.sigmoid(embeddings)), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    entropy_loss = entropy_loss.sum(-2).mean(-1).mean(-1)

    return entropy_loss


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


def compute_content_loss(embeddings, k=100, rms_vals=None, rms_thr=0.01, n_bins_blur_decay=2.5):
    with torch.no_grad():
        # Initialize targets for content loss
        targets = torch.zeros_like(embeddings)
        #targets = torch.zeros_like(embeddings).repeat(k, 1, 1, 1)

        # Filter out all non-peaks (along frequency) from logits
        embeddings_peak = filter_non_peaks_torch(embeddings, fill_val=-torch.inf)

        # Determine indices of maximum activation within each frame
        topk_idcs = torch.topk(embeddings_peak, dim=-2, k=k).indices

        # Sample indices for positive activations from given logits
        #idcs = Categorical(logits=embeddings_peak.transpose(-1, -2)).sample((k,)).unsqueeze(-2)
        #idcs = Categorical(logits=embeddings_peak.transpose(-1, -2)).sample((k,)).transpose(-2, -3)

        # Insert unit activations at chosen bin(s)
        targets.scatter_(-2, topk_idcs, 1)
        #targets.scatter_(-2, idcs, 1)

        if n_bins_blur_decay:
            # Compute standard deviation for kernel
            std_dev = (2 * n_bins_blur_decay) / 5
            # Truncate kernel at 4 deviations
            kernel_size = int(8 * std_dev + 1)
            # Initialize indices for the kernel
            idcs = torch.arange(kernel_size) - kernel_size // 2
            # Compute weights for a Gaussian kernel
            kernel = torch.exp(-0.5 * (idcs / std_dev) ** 2)
            # Determine number of frames
            out_channels = targets.size(-1)
            #out_channels = embeddings.size(0)
            # Give kernel a batch and channel dimension
            kernel = kernel.view(1, 1, -1).to(targets.device)
            #kernel = kernel.view(1, 1, -1, 1).to(targets.device)
            # Extend kernel along channels
            kernel = kernel.repeat(out_channels, 1, 1)
            #kernel = kernel.repeat(out_channels, 1, 1, 1)
            # Blur activations along the frequency axis with the filter
            targets = F.conv1d(targets.transpose(-1, -2), kernel, padding='same', groups=out_channels).transpose(-1, -2)
            #targets = F.conv2d(targets, kernel, padding='same', groups=out_channels)
            # Clamp superimposed activations to maximum probability
            targets = targets.clip(min=0.0, max=1.0)

    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute content loss as BCE of activations with respect to binarized and blurred maximum activations (positive activations only)
    content_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - targets), reduction='none', pos_weight=neg_weight)
    #content_loss = F.binary_cross_entropy_with_logits(embeddings, targets, reduction='none')
    #content_loss = F.binary_cross_entropy_with_logits(-embeddings.repeat(k, 1, 1, 1), (1 - targets), reduction='none', pos_weight=neg_weight)

    # Sum loss across frequency bins
    content_loss = content_loss.sum(-2)

    # Average across samples
    #content_loss = content_loss.mean(0)

    if rms_vals is not None:
        # Ignore content loss for frames with energy below RMS threshold for each track
        content_loss_ = [c[rms_vals[i] >= rms_thr] for i, c in enumerate(content_loss)]
        # Compute average across valid frames for each track and repack
        content_loss = torch.stack([c.mean() for c in content_loss_])
        # Compute average across batch for valid (non-silent) tracks only
        content_loss = torch.mean(content_loss[torch.logical_not(content_loss.isnan())])
    else:
        # Average loss across time and batch
        content_loss = content_loss.mean(-1).mean(-1)

    return content_loss


def compute_contrastive_loss(activations, tau=0.1):
    # Determine batch size and number of frames
    (B, _, T) = activations.size()

    # Fold out time dimension of activations
    activations = activations.permute(2, 0, 1)
    # Compute the L2 norm for each frame
    norms = activations.norm(p=2, dim=-1, keepdim=True)
    # Normalize activations using L2 norm
    activations = activations / norms.clamp(torch.finfo(activations.dtype).eps)
    # Compute frame-level cosine similarity between activations across batch
    similarities = torch.bmm(activations, activations.transpose(-1, -2))
    # Apply temperature scaling
    similarities = similarities / tau

    # Create labels for contrastive learning as identity mapping
    labels = torch.arange(B, device=activations.device).repeat(T, 1)

    # Compute contrastive loss as CCE of similarities with respect to identity mapping
    contrastive_loss = F.cross_entropy(similarities, labels, reduction='none')

    # Average across batch and time
    contrastive_loss = contrastive_loss.mean(-1).mean(-1)

    return contrastive_loss
