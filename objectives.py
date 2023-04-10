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


def compute_geometric_loss(model, features, embeddings, max_shift_f=12,
                           max_shift_t=25, min_stretch=0.5, max_stretch=2):
    # Determine the number of samples in the batch
    B = features.size(0)

    # Sample a random frequency and time shift for each sample in the batch
    freq_shifts = torch.randint(low=-max_shift_f, high=max_shift_f + 1, size=(B,)).tolist()
    time_shifts = torch.randint(low=-max_shift_t, high=max_shift_t + 1, size=(B,)).tolist()

    # Sample a random stretch factor for each sample in the batch
    stretch_factors = (torch.rand(size=(B,)) * (max_stretch - min_stretch) + min_stretch).tolist()

    with torch.no_grad():
        # Translate the features by the sampled number of bins and frames
        distorted_features = translate_batch(features, freq_shifts, -2)
        distorted_features = translate_batch(distorted_features, time_shifts)
        # Stretch the features by the sampled stretch factors
        distorted_features = stretch_batch(distorted_features, stretch_factors)

    # Translate the original embeddings by the sampled number of bins and frames
    distorted_embeddings = translate_batch(embeddings, freq_shifts, -2, -torch.inf)
    distorted_embeddings = translate_batch(distorted_embeddings, time_shifts, -1, -torch.inf)
    # Stretch the original embeddings by the sampled stretch factors
    distorted_embeddings = stretch_batch(distorted_embeddings, stretch_factors)

    # Process the distorted features with the model
    distortion_embeddings = model(distorted_features).squeeze()

    # Convert both sets of logits to activations (implicit pitch salience)
    distorted_salience = torch.sigmoid(distorted_embeddings)
    distortion_salience = torch.sigmoid(distortion_embeddings)

    # Compute geometric loss as BCE of embeddings computed from distorted features with respect to distorted activations
    geometric_loss_ds = F.binary_cross_entropy_with_logits(distortion_embeddings, distorted_salience.detach(), reduction='none')

    # Compute geometric loss as BCE of distorted embeddings with respect to activations computed from distorted features
    geometric_loss_og = F.binary_cross_entropy_with_logits(distorted_embeddings, distortion_salience.detach(), reduction='none')

    # Ignore NaNs introduced by computing BCE loss on -∞
    geometric_loss_og[distorted_embeddings.isinf()] = 0

    # Sum across frequency bins and average across time and batch for both variations of geometric loss
    geometric_loss = (geometric_loss_ds.sum(-2).mean(-1).mean(-1) + geometric_loss_og.sum(-2).mean(-1).mean(-1)) / 2

    return geometric_loss


# TODO - can initial logic be simplified at all?
def compute_timbre_loss(model, features, embeddings, fbins_midi, bins_per_octave, points_per_octave=2):
    # Determine the number of samples in the batch
    B, H, K, _ = features.size()

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine the semitone span of the frequency support
    semitone_span = fbins_midi.max() - fbins_midi.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(torch.ceil(n_psuedo_bins / bins_per_octave))

    # Determine the number of cut/boost points to sample
    n_points = 1 + points_per_octave * n_octaves

    # Cover the full octave for proper interpolation
    out_size = n_octaves * bins_per_octave

    # Sample a random stretch factor for each sample in the batch
    equalization_curves = 0.5 + torch.rand(size=(B, 1, n_points), device=features.device)
    # Upsample the equalization curve to the number of frequency bins via linear interpolation
    equalization_curves = F.interpolate(equalization_curves,
                                        size=out_size,
                                        mode='linear',
                                        align_corners=True).squeeze()

    # Determine which bins correspond to which equalization
    nearest_bins = torch.round(bins_per_semitone * (fbins_midi - fbins_midi.min())).long()

    with torch.no_grad():
        # Obtain indices corresponding to equalization for each sample in the batch
        equalization_idcs = torch.meshgrid(torch.arange(B), nearest_bins.flatten())
        # Obtain the equalization for each sample in the batch
        equalization = equalization_curves[equalization_idcs].view(B, H, K, -1)
        # Apply the sampled equalization curves to the batch
        equalization_features = torch.clip(equalization * features, min=0, max=1)

    # Process the equalized features with the model
    equalization_embeddings = model(equalization_features).squeeze()

    # Convert both sets of logits to activations (implicit pitch salience)
    original_salience, equalization_salience = torch.sigmoid(embeddings), torch.sigmoid(equalization_embeddings)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss_eq = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience.detach(), reduction='none')

    # Compute timbre loss as BCE of embeddings computed from original features with respect to equalization activations
    timbre_loss_og = F.binary_cross_entropy_with_logits(embeddings, equalization_salience.detach(), reduction='none')

    # Sum across frequency bins and average across time and batch for both variations of timbre loss
    timbre_loss = (timbre_loss_eq.sum(-2).mean(-1).mean(-1) + timbre_loss_og.sum(-2).mean(-1).mean(-1)) / 2

    return timbre_loss


def compute_scaling_loss(model, features, activations):
    # Determine the number of samples in the batch
    B = features.size(0)

    # Sample a random scaling factor for each sample in the batch
    scaling_factors = torch.rand(size=(B, 1, 1), device=features.device)

    # Apply the scaling factors to the batch
    scaled_features = scaling_factors.unsqueeze(-1) * features
    scaled_activations = scaling_factors * activations

    # Process the scaled features with the model and convert to activations
    #scale_embeddings = model(scaled_features).squeeze()
    scale_activations = torch.sigmoid(model(scaled_features)).squeeze()

    # Compute scaling loss as MSE between embeddings computed from scaled features and scaled activations
    #scaling_loss = F.binary_cross_entropy_with_logits(scale_embeddings, scaled_activations.detach(), reduction='none')
    #scaling_loss = F.mse_loss(scale_activations, scaled_activations.detach(), reduction='none')
    #scaling_loss = F.mse_loss(scaled_activations, scale_activations.detach(), reduction='none')
    scaling_loss = F.mse_loss(scale_activations, scaled_activations, reduction='none')

    # Sum across frequency bins and average across time and batch
    #scaling_loss = (scaling_loss_sc.sum(-2).mean(-1).mean(-1) + scaling_loss_og.sum(-2).mean(-1).mean(-1)) / 2
    scaling_loss = scaling_loss.sum(-2).mean(-1).mean(-1)

    return scaling_loss


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
    #legend = (0.5 + torch.rand((N, N), device=legend.device)) * legend

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
