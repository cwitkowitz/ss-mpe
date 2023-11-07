# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .hcqtmodule import HCQT

# Regular imports
import torch.nn.functional as F
import torch


__all__ = [
    'compute_support_loss',
    'compute_harmonic_loss',
    'compute_sparsity_loss',
    'sample_random_equalization',
    'sample_parabolic_equalization',
    'sample_gaussian_equalization',
    'compute_timbre_loss',
    'compute_geometric_loss',
    #'compute_superposition_loss',
    #'compute_scaling_loss',
    #'compute_power_loss'
]


def compute_support_loss(embeddings, h1_features):
    # Set the weight for positive activations to zero
    pos_weight = torch.tensor(0)

    # Compute support loss as BCE of activations with respect to features (negative activations only)
    support_loss = F.binary_cross_entropy_with_logits(embeddings, h1_features, reduction='none', pos_weight=pos_weight)
    #support_loss = F.mse_loss(torch.sigmoid(embeddings), h1_features, reduction='none')

    # Sum across frequency bins and average across time and batch
    support_loss = support_loss.sum(-2).mean(-1).mean(-1)

    return support_loss


def compute_harmonic_loss(embeddings, salience):
    # Set the weight for negative activations to zero
    neg_weight = torch.tensor(0)

    # Compute harmonic loss as BCE of activations with respect to salience estimate (positive activations only)
    harmonic_loss = F.binary_cross_entropy_with_logits(-embeddings, (1 - salience), reduction='none', pos_weight=neg_weight)
    #harmonic_loss = F.mse_loss(torch.sigmoid(embeddings), salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    harmonic_loss = harmonic_loss.sum(-2).mean(-1).mean(-1)

    return harmonic_loss


def compute_sparsity_loss(activations):
    # Compute sparsity loss as the L1 norm of the activations
    sparsity_loss = torch.norm(activations, 1, dim=-2)

    # Average loss across time and batch
    sparsity_loss = sparsity_loss.mean(-1).mean(-1)

    return sparsity_loss


def sample_random_equalization(n_bins, batch_size=1, n_points=None, std_dev=0.10, device='cpu'):
    """
    Uniformly sample multiplicative equalization factors and upsample to cover whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Final number of frequency bins
    batch_size : int
      Number of curves to sample
    n_points : int or None (optional)
      Number of peaks/troughs to sample
    std_dev : float
      Standard deviation of boost/cut
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    if n_points is None:
        # Default to provided output size
        n_points = n_bins

    # Sample a random equalization curve factor for each sample in batch
    curves = 1 + torch.randn(size=(batch_size, 1, n_points), device=device) * std_dev

    if n_bins != n_points:
        # Upsample equalization curves to number of frequency bins via linear interpolation
        curves = F.interpolate(curves, size=n_bins, mode='linear', align_corners=True)

    # Remove channel dimension
    curves = curves.squeeze(-2)

    return curves


def sample_parabolic_equalization(n_bins, batch_size=1, pointiness=1, device=None):
    """
    Randomly sample parabolic equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    pointiness : float
      Multiplier to shrink parabolic opening
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    # Randomly sample parametric parabolic functions
    alpha, beta = torch.rand(size=(2, batch_size, 1), device=device)
    # Scale parameters to appropriate ranges
    alpha, beta = alpha / (n_bins - 1) ** 2, beta * (n_bins - 1)

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute parabolic equalization curves
    curves = 1 - pointiness * alpha * (idcs - beta) ** 2

    return curves


def sample_gaussian_equalization(n_bins, batch_size=1, max_A=0.25, max_std_dev=None, device=None):
    """
    Randomly sample Gaussian equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    max_std_dev : float or None (optional)
      Maximum standard deviation of sampled Gaussians
    max_A : float
      Maximum amplitude of sampled Gaussians
    device : string
      Device on which to initialize curves

    Returns
    ----------
    curves : Tensor (B x F)
      Sampled equalization curves
    """

    if max_std_dev is None:
        # Default to 10% of frequency bins
        max_std_dev = 0.10 * n_bins

    # Randomly sample parametric Gaussian functions
    A, mu, sigma = torch.rand(size=(3, batch_size, 1), device=device)
    # Scale parameters to appropriate ranges
    A, mu, sigma = max_A * (A * 2 - 1), mu * (n_bins - 1), sigma * max_std_dev

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute Gaussian equalization curves
    curves = 1 + A * torch.exp(-0.5 * (idcs - mu) ** 2 / sigma ** 2)

    return curves


def compute_timbre_loss(model, features, embeddings, eq_fn, **eq_kwargs):
    # Obtain dimensionality of features and appropriate device
    (B, H, K, _), device = features.size(), features.device

    # Extract relevant HCQT parameters
    bins_per_octave = model.hcqt_params['bins_per_octave']

    # Obtain center frequencies (MIDI) associated with each HCQT bin
    midi_freqs = torch.from_numpy(model.hcqt.midi_freqs).to(device)

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine semitone span of frequency support
    semitone_span = midi_freqs.max() - midi_freqs.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(torch.ceil(n_psuedo_bins / bins_per_octave))

    # Perform equalization over full octave
    out_size = n_octaves * bins_per_octave

    # Randomly sample an equalization curve for each sample in batch
    curves = eq_fn(out_size, batch_size=B, device=device, **eq_kwargs)

    # Determine nearest equalization corresponding to each frequency bin
    equalization_bins = bins_per_semitone * (midi_freqs - midi_freqs.min())
    # Round, convert equalization bins to integers, and flatten
    equalization_bins = equalization_bins.round().long().flatten()
    # Obtain indices corresponding to equalization for each sample in the batch
    equalization_idcs = torch.meshgrid(torch.arange(B, device=device), equalization_bins)
    # Obtain the equalization for each sample in the batch
    equalization = curves[equalization_idcs].view(B, H, K, -1)

    # Apply sampled equalization curves to the batch and clamp features
    equalization_features = torch.clip(equalization * features, min=0, max=1)
    # Process equalized features with provided model
    equalization_embeddings = model(equalization_features)[0]

    # Convert both sets of logits to activations (implicit pitch salience)
    # TODO - will be unnecessary if ground-truth is salience
    original_salience, equalization_salience = torch.sigmoid(embeddings), torch.sigmoid(equalization_embeddings)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    #timbre_loss_eq = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience.detach(), reduction='none')
    #timbre_loss_eq = F.mse_loss(equalization_salience, original_salience.detach(), reduction='none')
    #timbre_loss_eq = F.mse_loss(equalization_embeddings, embeddings.detach(), reduction='none')
    #timbre_loss_eq = F.mse_loss(equalization_embeddings, embeddings, reduction='none')
    timbre_loss_eq = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience, reduction='none')

    # Compute timbre loss as BCE of embeddings computed from original features with respect to equalization activations
    #timbre_loss_og = F.binary_cross_entropy_with_logits(embeddings, equalization_salience.detach(), reduction='none')
    #timbre_loss_og = F.mse_loss(original_salience, equalization_salience.detach(), reduction='none')
    #timbre_loss_og = F.mse_loss(embeddings, equalization_embeddings.detach(), reduction='none')
    #timbre_loss_og = F.mse_loss(embeddings, equalization_embeddings, reduction='none')

    # Sum across frequency bins and average across time and batch for both variations of timbre loss
    #timbre_loss = (timbre_loss_eq.sum(-2).mean(-1).mean(-1) + timbre_loss_og.sum(-2).mean(-1).mean(-1)) / 2
    #timbre_loss = (timbre_loss_eq + timbre_loss_og).sum(-2).mean(-1).mean(-1)
    timbre_loss = timbre_loss_eq.sum(-2).mean(-1).mean(-1)

    return timbre_loss


# TODO - can this function be sped up?
def translate_batch(batch, shifts, dim=-1, val=0):
    """
    TODO
    """

    # Determine the dimensionality of the batch
    dimensionality = batch.size()

    # Combine the original tensor with tensor filled with zeros such that no wrapping will occur
    rolled_batch = torch.cat([batch, val * torch.ones(dimensionality, device=batch.device)], dim=dim)

    # Roll each sample in the batch independently and reconstruct the tensor
    rolled_batch = torch.cat([x.unsqueeze(0).roll(i, dim) for x, i in zip(rolled_batch, shifts)])

    # Trim the rolled tensor to its original dimensionality
    translated_batch = rolled_batch.narrow(dim, 0, dimensionality[dim])

    return translated_batch


# TODO - can this function be sped up?
def stretch_batch(batch, stretch_factors):
    """
    TODO
    """

    # Determine height and width of the batch
    H, W = batch.size(-2), batch.size(-1)

    # Inserted stretched values to a copy of the original tensor
    stretched_batch = batch.clone()

    # Loop through each sample and stretch factor in the batch
    for i, (sample, factor) in enumerate(zip(batch, stretch_factors)):
        # Reshape the sample to B x H x W
        original = sample.reshape(-1, H, W)
        # Stretch the sample by the specified amount
        stretched_sample = F.interpolate(original,
                                         scale_factor=factor,
                                         mode='linear',
                                         align_corners=True)

        # Patch upsampled -∞ values that end up being NaNs (a little hacky)
        stretched_sample[stretched_sample.isnan()] = -torch.inf

        if factor < 1:
            # Determine how much padding is necessary
            pad_amount = W - stretched_sample.size(-1)
            # Pad the stretched sample to fit original width
            stretched_sample = F.pad(stretched_sample, (0, pad_amount))

        # Insert the stretched sample back into the batch
        stretched_batch[i] = stretched_sample[..., :W].view(sample.shape)

    return stretched_batch


def compute_geometric_loss(model, features, embeddings, max_seq_idx=250, max_shift_f=12,
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

    # Process the distorted features with the model and randomize start position
    distortion_embeddings = model(distorted_features, max_seq_idx).squeeze()

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
    #geometric_loss = (geometric_loss_ds.sum(-2).mean(-1).mean(-1) + geometric_loss_og.sum(-2).mean(-1).mean(-1)) / 2
    geometric_loss = (geometric_loss_ds + geometric_loss_og).sum(-2).mean(-1).mean(-1)

    return geometric_loss


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

    # Randomly sample mixture weights
    legend = torch.rand((N, N), device=legend.device) * legend

    # Mix the tracks based on the legend
    mixtures = torch.sparse.mm(legend, audio)

    # Restore the original dimensionality
    mixtures = mixtures.view(dimensionality)

    return mixtures, legend


def compute_superposition_loss(hcqt, model, audio, activations, mix_probability=0.5):
    with torch.no_grad():
        # Randomly mix the audio tracks in the batch
        mixtures, legend = get_random_mixtures(audio, mix_probability)

        # Normalize the mixing weights so their influence is relative
        legend /= torch.max(legend, dim=-1)[0].unsqueeze(-1)

        # Superimpose thresholded activations for mixture targets with max operation
        mixture_activations = torch.sparse.mm(legend.cpu().detach().to_sparse_csr(),
                                              activations.float().flatten(-2).cpu().detach(), 'max')

        # Add the mixed activations to the appropriate device and restore dimensionality
        mixture_activations = mixture_activations.to(legend.device).view(activations.size())

    # Obtain log-scale features for the mixtures
    mixture_features = HCQT.rescale_decibels(hcqt(mixtures))

    # Process the audio mixtures with the model
    mixture_embeddings = model(mixture_features).squeeze()

    # Compute superposition loss as BCE of embeddings computed from mixtures with respect to mixed activations
    superposition_loss = F.binary_cross_entropy_with_logits(mixture_embeddings, mixture_activations, reduction='none')

    # Sum across frequency bins and average across time and batch
    superposition_loss = superposition_loss.sum(-2).mean(-1).mean(-1)

    return superposition_loss


def compute_scaling_loss(model, features, activations):
    # Determine the number of samples in the batch
    B = features.size(0)

    # Sample a random scaling factor for each sample in the batch
    scaling_factors = torch.rand(size=(B, 1, 1), device=features.device)

    # Apply the scaling factors to the batch
    scaled_features = scaling_factors.unsqueeze(-1) * features
    scaled_activations = scaling_factors * activations

    # Process the scaled features with the model and convert to activations
    scale_activations = torch.sigmoid(model(scaled_features)).squeeze()

    # Compute scaling loss as MSE between embeddings computed from scaled features and scaled activations
    scaling_loss = F.mse_loss(scale_activations, scaled_activations, reduction='none')

    # Sum across frequency bins and average across time and batch
    scaling_loss = scaling_loss.sum(-2).mean(-1).mean(-1)

    return scaling_loss


def compute_power_loss(activations, h1_features):
    # Sum the feature power across all frequency bins
    power_features = torch.sum(h1_features ** 2, dim=-2)

    # Sum the activation power across all frequency bins
    power_activations = torch.sum(activations ** 2, dim=-2)

    # Compute power loss as difference between sum of powers
    power_loss = torch.abs(power_features - power_activations)

    # Average loss across time and batch
    power_loss = power_loss.mean(-1).mean(-1)

    return power_loss
