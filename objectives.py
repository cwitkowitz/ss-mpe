# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>


import torch.nn.functional as F
import torch


def compute_reconstruction_loss(features, activations):
    # Compute the reconstruction loss as BCE of activations with respect to features
    reconstruction_loss = F.binary_cross_entropy(activations, features, reduction='none')

    # Sum across frequency bins and average across time and batch
    reconstruction_loss = reconstruction_loss.sum(-2).mean(-1).mean(-1)

    return reconstruction_loss


def compute_content_loss(features, activations):
    # TODO - can likely intelligently combine content/reconstruction loss

    # Compute the total energy (averaged across channels) in each frame
    energy_features = torch.sum(torch.mean(features, dim=-3), dim=-2)
    energy_activations = torch.sum(torch.mean(activations, dim=-3), dim=-2)

    # Compute magnitude difference between total energy of features and activations
    content_loss = torch.abs(energy_features - energy_activations)

    # Average loss across time and batch
    content_loss = content_loss.mean(-1).mean(-1)

    return content_loss


def get_random_mixtures(audio, p=0.8):
    # Keep track of the original dimensionality of the audio
    dimensionality = audio.size()

    # Collapse the channel dimension if it exists
    audio = audio.squeeze()

    # Determine the amount of tracks in the batch
    N = dimensionality[0]

    # Randomly sample a matrix for mixing
    legend = torch.rand((N, N), device=audio.device)

    # Threshold to determine which tracks should be mixed
    legend = torch.threshold(legend, threshold=p, value=0)
    # Include the original track in each mix (diagonal)
    legend = torch.logical_or(torch.eye(N, device=legend.device), legend)

    # Determine how many tracks will be included in each mixture
    n_mix = torch.sum(legend, dim=-1).unsqueeze(-1)

    # Randomly sample mixture weights
    legend = torch.rand((N, N), device=legend.device) * legend

    # Mix the tracks based on the legend
    mixtures = torch.matmul(legend, audio)
    # Apply the mixture weights
    mixtures /= n_mix

    # Restore the original dimensionality
    mixtures = mixtures.view(dimensionality)

    return mixtures, legend


def compute_linearity_loss(activations, mixture_activations, legend):
    # Keep track of original dimensionality
    dimensionality = activations.size()

    # Superimpose thresholded activations for mixture targets
    pseudo_ground_truth = torch.matmul(torch.ceil(legend),
                                       torch.round(activations).flatten(-3))

    # Ignore overlapping activations and restore dimensionality
    pseudo_ground_truth = torch.clip(pseudo_ground_truth, max=1).view(dimensionality)

    # Compute BCE loss to push activations of mixture toward linear combination of individual activations
    linearity_loss = F.binary_cross_entropy(mixture_activations, pseudo_ground_truth, reduction='none')

    # Sum across frequency bins and average across time and batch
    linearity_loss = linearity_loss.sum(-2).mean(-1).mean()

    return linearity_loss


def compute_contrastive_loss(original_embeddings, augment_embeddings, temperature=0.07):
    # TODO - support for more than one augmentation?

    # Determine which device to use for processing
    device = original_embeddings.device

    # Keep track of original dimensionality
    B, T, E = original_embeddings.shape

    # Concatenate both sets of embeddings along the batch dimension
    all_embeddings = torch.cat((original_embeddings, augment_embeddings), dim=0)

    # Normalize both sets of embeddings
    all_embeddings = F.normalize(all_embeddings, dim=-1)

    # Switch the batch and frame dimensions for the embeddings
    all_embeddings = all_embeddings.transpose(0, 1)

    # Compute cosine similarity between every embedding across each frame
    similarities = torch.bmm(all_embeddings, all_embeddings.transpose(-1, -2))

    # Construct a matrix indicating same-sample membership for each embedding
    labels = (torch.eye(2 * B) + torch.eye(2 * B).roll(B, dims=-1)).to(device)

    # Create mask to indicate which elements belong to diagonal
    diagonal = torch.eye(2 * B, dtype=torch.bool).to(device)

    # Discard labels indicating identity
    labels = labels[~diagonal].view(2 * B, -1)
    # Discard similarities for identical pairs across each frame
    similarities = similarities[:, ~diagonal].view(T, 2 * B, -1)

    # Obtain the similarity measures for positive pairs
    positives = similarities[:, labels.bool()].view(T, 2 * B, -1)

    # Obtain the similarity measures for negative pairs
    negatives = similarities[:, ~labels.bool()].view(T, 2 * B, -1)

    # Combine all similarities, ordering the positive pair similarities first
    logits = torch.cat([positives, negatives], dim=-1) / temperature

    # Construct labels indicating first index as pertaining to ground-truth class
    targets = torch.zeros(logits.shape[:-1], dtype=torch.long).to(device)

    # Compute loss based on similarity of embeddings originating from same sample
    contrastive_loss = F.cross_entropy(logits.view(2 * B * T, -1),
                                       targets.flatten(), reduction='none')

    # Restore the original dimensions to the computed losses
    contrastive_loss = contrastive_loss.view(T, 2 * B).t()

    # Average the loss across frames and then across the batch
    contrastive_loss = contrastive_loss.mean(-1).mean(-1)

    return contrastive_loss


def compute_translation_loss():
    # TODO
    pass
