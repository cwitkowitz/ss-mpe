# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from utils import *

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


def get_random_mixtures(audio, mix_probability=0.5):
    # Keep track of the original dimensionality of the audio
    dimensionality = audio.size()

    # Collapse the channel dimension if it exists
    audio = audio.squeeze()

    # Determine the amount of tracks in the batch
    N = dimensionality[0]

    # Randomly sample a matrix for mixing
    legend = torch.rand((N, N), device=audio.device)

    # Threshold to determine which tracks should be mixed
    legend = torch.threshold(legend, threshold=(1 - mix_probability), value=0)
    # Include the original track in each mix (diagonal)
    legend = torch.logical_or(torch.eye(N, device=legend.device), legend)

    # Determine how many tracks will be included in each mixture
    n_mix = torch.sum(legend, dim=-1).unsqueeze(-1)

    # Randomly sample mixture weights
    legend = torch.rand((N, N), device=legend.device) * legend

    # Mix the tracks based on the legend
    mixtures = torch.sparse.mm(legend, audio)
    # Apply the mixture weights
    mixtures /= n_mix

    # Restore the original dimensionality
    mixtures = mixtures.view(dimensionality)

    return mixtures, legend


def compute_superposition_loss(hcqt, model, audio, activations, mix_probability=0.5):
    with torch.no_grad():
        # Randomly mix the audio tracks in the batch
        mixtures, legend = get_random_mixtures(audio, mix_probability)

        # Ignore mixing weights
        legend = torch.ceil(legend)

        # Determine how many tracks were included in each mixture
        n_mix = torch.sum(legend, dim=-1).unsqueeze(-1)

        # Obtain normalization coefficients for log-softmax operator
        normalization_coeffs = n_mix * torch.tensor(1).exp()

        # Superimpose thresholded activations for mixture targets
        mixture_activations = torch.sparse.mm(legend, activations.flatten(-2).exp())

        # Normalize the log-softmax mixture activations using coefficients
        mixture_activations = mixture_activations.log() / normalization_coeffs

        # Restore original dimensionality to the mixture activations
        mixture_activations = mixture_activations.view(activations.size())

    # Obtain log-scale features for the mixtures
    mixture_features = rescale_decibels(hcqt(mixtures))

    # Process the audio mixtures with the model
    mixture_embeddings = model(mixture_features).squeeze()

    # Compute superpositions loss as BCE of embeddings computed from mixtures with respect to mixed activations
    superposition_loss = F.binary_cross_entropy_with_logits(mixture_embeddings, mixture_activations, reduction='none')

    # Sum across frequency bins and average across time and batch
    superposition_loss = superposition_loss.sum(-2).mean(-1).mean(-1)

    return superposition_loss
