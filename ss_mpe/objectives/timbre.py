# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'sample_random_equalization',
    'sample_butterworth_equalization',
    'sample_parabolic_equalization',
    'sample_gaussian_equalization',
    'apply_random_equalizations',
    'compute_timbre_loss'
]


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


def sample_butterworth_equalization(n_bins, batch_size=1, device=None):
    # Randomly sample bins for the indices corresponding to the cutoff frequencies
    cutoff_bins = torch.randint(high=n_bins + 1, size=(batch_size, 1), device=device)
    # Sample binary values to indicate orientation (i.e. lowpass vs. highpass)
    orientation = torch.rand(size=(batch_size, 1), device=device).round()

    # Initialize allpass filter curves for each sample in batch
    curves = torch.ones(n_bins, device=device).repeat((batch_size, 1))

    for i in range(batch_size):
        # Create lowpass filter curves
        curves[i, cutoff_bins[i]:] = 0

        if orientation[i]:
            # Reverse orientation for highpass filter curves
            curves[i] = curves[i].flip(dims=[-1])

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


def sample_gaussian_equalization(n_bins, batch_size=1, max_A=0.25, max_std_dev=None, fixed_shape=False, device=None):
    """
    Randomly sample Gaussian equalization curves covering whole frequency spectrum.

    Parameters
    ----------
    n_bins : int
      Number of frequency bins
    batch_size : int
      Number of curves to sample
    max_A : float
      Maximum amplitude of sampled Gaussians
    max_std_dev : float or None (optional)
      Maximum standard deviation of sampled Gaussians
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

    if fixed_shape:
        # Amplitude and standard deviation go to maximum values
        A, sigma = A.round(), torch.ones_like(sigma, device=device)

    # Scale parameters to appropriate ranges
    A, mu, sigma = max_A * (A * 2 - 1), mu * (n_bins - 1), sigma * max_std_dev

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute Gaussian equalization curves
    curves = 1 + A * torch.exp(-0.5 * (idcs - mu) ** 2 / sigma ** 2)

    return curves


def apply_random_equalizations(features, hcqt_module, **eq_kwargs):
    # Obtain dimensionality of features and appropriate device
    (B, H, K, _), device = features.size(), features.device

    # Determine expected number of bins per octave
    bins_per_octave = hcqt_module.bins_per_octave

    # Obtain center frequencies (MIDI) associated with each HCQT bin
    midi_freqs = torch.from_numpy(hcqt_module.midi_freqs).to(device)

    # Infer the number of bins per semitone
    bins_per_semitone = bins_per_octave / 12

    # Determine semitone span of frequency support
    semitone_span = midi_freqs.max() - midi_freqs.min()

    # Determine how many bins are represented across all harmonics
    n_psuedo_bins = (bins_per_semitone * semitone_span).round()

    # Determine how many octaves have been covered
    n_octaves = int(torch.ceil(n_psuedo_bins / bins_per_octave))

    # Perform equalization over full octave
    n_total_bins = n_octaves * bins_per_octave

    # Extract the equalization function
    eq_fn = eq_kwargs.pop('eq_fn')

    # Randomly sample an equalization curve for each sample in batch
    curves = eq_fn(n_total_bins, batch_size=B, device=device, **eq_kwargs)

    # TODO - apply EQ curves function?

    # Determine nearest equalization corresponding to each frequency bin
    equalization_bins = bins_per_semitone * (midi_freqs - midi_freqs.min())
    # Round, convert equalization bins to integers, and flatten
    equalization_bins = equalization_bins.round().long().flatten()
    # Obtain indices corresponding to equalization for each sample in the batch
    equalization_idcs = torch.meshgrid(torch.arange(B, device=device), equalization_bins, indexing='ij')
    # Obtain the equalization for each sample in the batch
    equalization = curves[equalization_idcs].view(B, H, K, -1)

    # Apply sampled equalization curves to the batch and clamp features
    equalized_features = torch.clip(equalization * features, min=0, max=1)

    # TODO - return EQ curves?

    return equalized_features


def compute_timbre_loss(model, features, targets, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_equalizations(features, model.hcqt, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original targets
    timbre_loss = F.binary_cross_entropy_with_logits(equalization_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss
