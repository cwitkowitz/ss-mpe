# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import translate_batch, stretch_batch

# Regular imports
import torch.nn.functional as F
import torch


def compute_support_loss(embeddings, h1_features):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute support loss as BCE of activations with respect to features (negative activations only)
    support_loss = F.binary_cross_entropy_with_logits(embeddings, h1_features, reduction='none', pos_weight=pos_weight)

    # Sum across frequency bins and average across time and batch
    support_loss = support_loss.sum(-2).mean(-1).mean(-1)

    return support_loss


def compute_content_loss(activations, h1_features):
    # Sum the features across all frequency bins
    total_energy = torch.sum(h1_features, dim=-2)

    # Sum the activations across all frequency bins
    total_activations = torch.sum(activations, dim=-2)

    # Compute content loss as squared difference between total power
    content_loss = (total_energy - total_activations) ** 2

    # Average loss across time and batch
    content_loss = content_loss.mean(-1).mean(-1)

    return content_loss


def compute_harmonic_loss(embeddings, features, weights=None):
    # Determine the number of CQT channels
    n_channels = features.size(-3)

    if weights is None:
        # Default to weighting each channel equally
        weights = torch.ones(n_channels)

    # Normalize the harmonic weights
    weights /= torch.sum(weights)

    # Make sure weights are on appropriate device
    weights = weights.to(features.device)

    # Compute a weighted sum of the features to obtain a rough salience estimate
    salience = torch.sum(features * weights.unsqueeze(-1).unsqueeze(-1), dim=-3)

    # Compute harmonic loss as BCE of activations with respect to salience estimate (positive activations only)
    harmonic_loss = -salience * torch.log(torch.sigmoid(embeddings)) # TODO - log sum exp trick implementation
    #support_loss = F.binary_cross_entropy_with_logits(embeddings, salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    harmonic_loss = harmonic_loss.sum(-2).mean(-1).mean(-1)

    return harmonic_loss


def compute_geometric_loss(model, features, activations, max_shift_f=12,
                           max_shift_t=25, min_stretch=0.5, max_stretch=2):
    # Determine the number of samples in the batch
    B = features.size(0)

    # Sample a random frequency and time shift for each sample in the batch
    freq_shifts = torch.randint(low=0, high=max_shift_f + 1, size=(B,)).tolist()
    time_shifts = torch.randint(low=-max_shift_t, high=max_shift_t + 1, size=(B,)).tolist()

    # Sample a random stretch factor for each sample in the batch
    stretch_factors = (torch.rand(size=(B,)) * (max_stretch - min_stretch) + min_stretch).tolist()

    with torch.no_grad():
        # Translate the features by the sampled number of bins and frames
        shifted_features = translate_batch(features, freq_shifts, -2)
        shifted_features = translate_batch(shifted_features, time_shifts)

        # Translate the activations by the sampled number of bins and frames
        shifted_activations = translate_batch(activations, freq_shifts, -2)
        shifted_activations = translate_batch(shifted_activations, time_shifts)

        # Stretch the features and activations by the sampled stretch factors
        shifted_features = stretch_batch(shifted_features, stretch_factors)
        shifted_activations = stretch_batch(shifted_activations, stretch_factors)

    # Process the shifted features with the model
    embeddings = model(shifted_features).squeeze()

    # Compute geometric loss as BCE of embeddings computed from shifted features with respect to shifted activations
    geometric_loss = F.binary_cross_entropy_with_logits(embeddings, shifted_activations, reduction='none')

    # Sum across frequency bins and average across time and batch
    geometric_loss = geometric_loss.sum(-2).mean(-1).mean(-1)

    return geometric_loss


def compute_timbre_loss(model, embeddings, features, n_bins, bins_per_octave, points_per_octave=2):
    # Determine the number of samples in the batch
    B = features.size(0)

    # Determine the number of cut/boost points to sample
    n_points = 1 + points_per_octave * (n_bins // bins_per_octave)

    # Sample a random stretch factor for each sample in the batch
    equalization_curves = 0.5 + torch.rand(size=(B, 1, n_points))
    # Upsample the equalization curve to the number of frequency bins via linear interpolation
    equalization_curves = F.interpolate(equalization_curves, size=(n_bins), mode='linear', align_corners=True)

    with torch.no_grad():
        # TODO - apply equalization curves
        pass

        # Process the augmented features with the model
        # TODO - place into eval mode? do this the other way around?
        augmented_activations = model(shifted_features).squeeze()

    # Compute timbre loss as BCE of embeddings with respect to activations computed from augmented features
    timbre_loss = F.binary_cross_entropy_with_logits(embeddings, augmented_activations, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss


def get_random_mixtures(audio, p=0.8, seed=None):
    # Keep track of the original dimensionality of the audio
    dimensionality = audio.size()

    # Collapse the channel dimension if it exists
    audio = audio.squeeze()

    # Determine the amount of tracks in the batch
    N = dimensionality[0]

    # TODO - random seed

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


def compute_linearity_loss(mixture_embeddings, activations, legend):
    # Keep track of original dimensionality
    dimensionality = activations.size()

    # Superimpose thresholded activations for mixture targets
    pseudo_ground_truth = torch.matmul(torch.ceil(legend), activations.flatten(-2))

    # Ignore overlapping activations and restore dimensionality
    pseudo_ground_truth = torch.clip(pseudo_ground_truth, max=1).view(dimensionality)

    # Compute BCE loss to push activations of mixture toward linear combination of individual activations
    linearity_loss = F.binary_cross_entropy_with_logits(mixture_embeddings, pseudo_ground_truth, reduction='none')

    # Sum across frequency bins and average across time and batch
    linearity_loss = linearity_loss.sum(-2).mean(-1).mean(-1)

    return linearity_loss


# TODO - support for more than one augmentation?
def compute_contrastive_loss(original_embeddings, augment_embeddings, temperature=0.07):
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

    # Sum the loss across embeddings and then average across frames
    contrastive_loss = contrastive_loss.sum(0).mean(-1)

    return contrastive_loss
