# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'compute_energy_loss',
    'compute_support_loss',
    'compute_harmonic_loss',
    'compute_sparsity_loss',
    'compute_entropy_loss',
    'compute_content_loss',
    'compute_content_loss2',
    'compute_content_loss3'
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


# TODO - can modulate by amount of energy in input (rms threshold 0.01)
def compute_content_loss(activations, lmbda=5):
    # Determine the maximum activation within each frame
    max_activations = torch.max(activations, dim=-2)[0]

    # Compute content loss as likelihood of exponential distribution of the maximum activations
    content_loss = lmbda * torch.exp(-lmbda * max_activations)

    # Average loss across time and batch
    content_loss = content_loss.mean(-1).mean(-1)

    return content_loss


# TODO - should pick peaks before taking max if topk > 1
def compute_content_loss2(embeddings, n_bins_blur_decay=2.5):
    with torch.no_grad():
        # Initialize targets for content loss
        targets = torch.zeros_like(embeddings)

        # Determine index of maximum activation within each frame
        max_idcs = torch.argmax(embeddings, dim=-2, keepdim=True)

        # Insert unit activations at maximum
        targets.scatter_(-2, max_idcs, 1)

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
            # Give kernel a batch and channel dimension
            kernel = kernel.view(1, 1, -1).to(targets.device)
            # Extend kernel along channels
            kernel = kernel.repeat(out_channels, 1, 1)
            # Blur activations along the frequency axis with the filter
            targets = F.conv1d(targets.transpose(-1, -2), kernel, padding='same', groups=out_channels).transpose(-1, -2)
            # Clamp superimposed activations to maximum probability
            targets = targets.clip(min=0.0, max=1.0)

    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute content loss as BCE of activations with respect to binarized and blurred maximum activations (positive activations only)
    content_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - targets), reduction='none', pos_weight=neg_weight)

    # Sum across frequency bins and average across time and batch
    content_loss = content_loss.sum(-2).mean(-1).mean(-1)

    return content_loss


def compute_content_loss3(embeddings, tau=0.1):
    # Determine batch size and number of frames
    (B, _, T) = embeddings.size()

    # Fold out time dimension of embeddings
    embeddings = embeddings.permute(2, 0, 1)
    # Compute the L2 norm for each frame
    norms = embeddings.norm(p=2, dim=-1, keepdim=True)
    # Normalize embeddings using L2 norm
    embeddings = embeddings / norms.clamp(torch.finfo(embeddings.dtype).eps)
    # Compute frame-level cosine similarity between embeddings across batch
    similarities = torch.bmm(embeddings, embeddings.transpose(-1, -2))
    # Apply temperature scaling
    similarities = similarities / tau

    # Create labels for contrastive learning as identity mapping
    labels = torch.arange(B, device=embeddings.device).repeat(T, 1)

    # Compute content loss as CCE of similarities with respect to identity mapping
    content_loss = F.cross_entropy(similarities, labels, reduction='none')

    # Average across batch and time
    content_loss = content_loss.mean(-1).mean(-1)

    return content_loss
