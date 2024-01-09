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
    'compute_timbre_loss_og',
    'compute_timbre_loss_og_rvs',
    'compute_timbre_loss_og_mse',
    'compute_timbre_loss_og_con',
    'compute_timbre_loss_2x',
    'compute_timbre_loss_2x_mse',
    'compute_timbre_loss_2x_lat_mse',
    'compute_timbre_loss_2x_con',
    'compute_timbre_loss_2x_lat_con',
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
    # Scale parameters to appropriate ranges
    A, mu, sigma = max_A * (A * 2 - 1), mu * (n_bins - 1), sigma * max_std_dev

    # Create a Tensor of indices for frequency bins of each curve
    idcs = torch.arange(n_bins, device=device).repeat((batch_size, 1))

    # Compute Gaussian equalization curves
    curves = 1 + A * torch.exp(-0.5 * (idcs - mu) ** 2 / sigma ** 2)

    return curves


def apply_random_eq(features, hcqt, eq_fn, **eq_kwargs):
    # Obtain dimensionality of features and appropriate device
    (B, H, K, _), device = features.size(), features.device

    # Extract relevant HCQT parameters
    bins_per_octave = hcqt.bins_per_octave

    # Obtain center frequencies (MIDI) associated with each HCQT bin
    midi_freqs = torch.from_numpy(hcqt.midi_freqs).to(device)

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

    # Randomly sample an equalization curve for each sample in batch
    curves = eq_fn(n_total_bins, batch_size=B, device=device, **eq_kwargs)

    # Determine nearest equalization corresponding to each frequency bin
    equalization_bins = bins_per_semitone * (midi_freqs - midi_freqs.min())
    # Round, convert equalization bins to integers, and flatten
    equalization_bins = equalization_bins.round().long().flatten()
    # Obtain indices corresponding to equalization for each sample in the batch
    equalization_idcs = torch.meshgrid(torch.arange(B, device=device), equalization_bins)
    # Obtain the equalization for each sample in the batch
    equalization = curves[equalization_idcs].view(B, H, K, -1)

    # Apply sampled equalization curves to the batch and clamp features
    equalized_features = torch.clip(equalization * features, min=0, max=1)

    return equalized_features


def compute_timbre_loss_og(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)[0]

    # Convert both sets of logits to activations (implicit pitch salience)
    # TODO - will be unnecessary if ground-truth is salience
    original_salience, equalization_salience = torch.sigmoid(embeddings), torch.sigmoid(equalization_embeddings)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    #timbre_loss_eq = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience.detach(), reduction='none')
    #timbre_loss_eq = F.mse_loss(equalization_salience, original_salience.detach(), reduction='none')
    #timbre_loss_eq = F.mse_loss(equalization_embeddings, embeddings.detach(), reduction='none')
    #timbre_loss_eq = F.mse_loss(equalization_embeddings, embeddings, reduction='none')
    timbre_loss_eq = F.binary_cross_entropy_with_logits(equalization_embeddings, original_salience, reduction='none')
    #timbre_loss_eq = F.binary_cross_entropy_with_logits(embeddings, equalization_salience, reduction='none')

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


def compute_timbre_loss_og_rvs(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)[0]

    # Convert both sets of logits to activations (implicit pitch salience)
    original_salience, equalization_salience = torch.sigmoid(embeddings), torch.sigmoid(equalization_embeddings)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss_eq = F.binary_cross_entropy_with_logits(embeddings, equalization_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss_eq.sum(-2).mean(-1).mean(-1)

    return timbre_loss


def compute_timbre_loss_og_mse(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)[0]

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss_eq = F.mse_loss(equalization_embeddings, embeddings, reduction='none')

    # Sum across frequency bins and average across time and batch
    timbre_loss = timbre_loss_eq.sum(-2).mean(-1).mean(-1)

    return timbre_loss


def compute_timbre_loss_2x(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform first set of random equalizations on batch of features
    equalized_features_1st = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process first set of equalized features with provided model
    equalization_embeddings_1st = model(equalized_features_1st)[0]

    # Perform second set of random equalizations on batch of features
    equalized_features_2nd = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process second set of equalized features with provided model
    equalization_embeddings_2nd = model(equalized_features_2nd)[0]

    # Convert both sets of logits to activations
    salience_1st = torch.sigmoid(equalization_embeddings_1st)
    salience_2nd = torch.sigmoid(equalization_embeddings_2nd)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to alternate activations
    timbre_loss_1st = F.binary_cross_entropy_with_logits(equalization_embeddings_1st, salience_2nd, reduction='none')
    timbre_loss_2nd = F.binary_cross_entropy_with_logits(equalization_embeddings_2nd, salience_1st, reduction='none')

    # Sum across frequency bins and average across time and batch for both variations of timbre loss
    timbre_loss = 0.5 * (timbre_loss_1st + timbre_loss_2nd).sum(-2).mean(-1).mean(-1)

    return timbre_loss


def compute_timbre_loss_2x_mse(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform first set of random equalizations on batch of features
    equalized_features_1st = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process first set of equalized features with provided model
    equalization_embeddings_1st = model(equalized_features_1st)[0]

    # Perform second set of random equalizations on batch of features
    equalized_features_2nd = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process second set of equalized features with provided model
    equalization_embeddings_2nd = model(equalized_features_2nd)[0]

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = F.mse_loss(equalization_embeddings_1st, equalization_embeddings_2nd, reduction='none')

    # Sum across frequency bins and average across time and batch for both variations of timbre loss
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss


def compute_timbre_loss_2x_lat_mse(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform first set of random equalizations on batch of features
    equalized_features_1st = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process first set of equalized features with provided model
    equalization_latents_1st = model(equalized_features_1st)[1]

    # Perform second set of random equalizations on batch of features
    equalized_features_2nd = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process second set of equalized features with provided model
    equalization_latents_2nd = model(equalized_features_2nd)[1]

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = F.mse_loss(equalization_latents_1st, equalization_latents_2nd, reduction='none')

    # Sum across frequency bins and average across time and batch for both variations of timbre loss
    timbre_loss = timbre_loss.sum(-2).mean(-1).mean(-1)

    return timbre_loss


def compute_contrastive_loss(embeddings_1st, embeddings_2nd, temperature=0.07):
    # Switch frame and feature dimension of embeddings
    embeddings_1st = embeddings_1st.transpose(-1, -2)
    embeddings_2nd = embeddings_2nd.transpose(-1, -2)

    # Keep track of original dimensionality and device
    (B, T, E), device = embeddings_1st.shape, embeddings_1st.device

    # Concatenate both sets of embeddings along the batch dimension
    all_embeddings = torch.cat((embeddings_1st, embeddings_2nd), dim=0)

    # Normalize both sets of embeddings
    all_embeddings = F.normalize(all_embeddings, dim=-1)

    # Switch batch and frame dimensions for the embeddings
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
    contrastive_loss = F.cross_entropy(logits.view(2 * B * T, -1), targets.flatten(), reduction='none')

    # Restore the original dimensions to the computed losses
    contrastive_loss = contrastive_loss.view(T, 2 * B).t()

    # Sum the loss across embeddings and then average across frames
    contrastive_loss = contrastive_loss.sum(0).mean(-1)

    return contrastive_loss


def compute_timbre_loss_og_con(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform random equalizations on batch of features
    equalized_features = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process equalized features with provided model
    equalization_embeddings = model(equalized_features)[0]

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = compute_contrastive_loss(equalization_embeddings, embeddings)

    return timbre_loss


def compute_timbre_loss_2x_con(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform first set of random equalizations on batch of features
    equalized_features_1st = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process first set of equalized features with provided model
    equalization_embeddings_1st = model(equalized_features_1st)[0]

    # Project first set of embeddings with projection head
    equalization_embeddings_1st = model.projection(equalization_embeddings_1st)

    # Perform second set of random equalizations on batch of features
    equalized_features_2nd = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process second set of equalized features with provided model
    equalization_embeddings_2nd = model(equalized_features_2nd)[0]

    # Project second set of embeddings with projection head
    equalization_embeddings_2nd = model.projection(equalization_embeddings_2nd)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = compute_contrastive_loss(equalization_embeddings_1st, equalization_embeddings_2nd)

    return timbre_loss


def compute_timbre_loss_2x_lat_con(model, features, embeddings, eq_fn, **eq_kwargs):
    # Perform first set of random equalizations on batch of features
    equalized_features_1st = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process first set of equalized features with provided model
    equalization_latents_1st = model(equalized_features_1st)[1]

    # Project first set of latent features with projection head
    equalization_latents_1st = model.projection(equalization_latents_1st)

    # Perform second set of random equalizations on batch of features
    equalized_features_2nd = apply_random_eq(features, model.hcqt, eq_fn, **eq_kwargs)

    # Process second set of equalized features with provided model
    equalization_latents_2nd = model(equalized_features_2nd)[1]

    # Project first set of latent features with projection head
    equalization_latents_2nd = model.projection(equalization_latents_2nd)

    # Compute timbre loss as BCE of embeddings computed from equalized features with respect to original activations
    timbre_loss = compute_contrastive_loss(equalization_latents_1st, equalization_latents_2nd)

    return timbre_loss


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
        distorted_ = F.interpolate(x, scale_factor=t.item(), mode='linear')#, align_corners=True)

        # Patch upsampled -∞ values that end up being NaNs (a little hacky)
        #stretched_sample[stretched_sample.isnan()] = -torch.inf

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


def compute_geometric_loss(model, features, embeddings, max_shift_v, max_shift_h, max_stretch_factor):
    # Determine batch size
    B = features.size(0)

    # Sample a random vertical / horizontal shift for each sample in the batch
    v_shifts = torch.randint(low=-max_shift_v, high=max_shift_v + 1, size=(B,))
    h_shifts = torch.randint(low=-max_shift_h, high=max_shift_h + 1, size=(B,))

    # Compute inverse of maximum stretch factor
    min_stretch_factor = 1 / max_stretch_factor

    # Sample a random stretch factor for each sample in the batch, starting at minimum
    stretch_factors, stretch_factors_ = min_stretch_factor, torch.rand(size=(B,))
    # Split sampled values into piecewise ranges
    neg_perc = 2 * stretch_factors_.clip(max=0.5)
    pos_perc = 2 * (stretch_factors_ - 0.5).relu()
    # Scale stretch factor evenly across effective range
    stretch_factors += neg_perc * (1 - min_stretch_factor)
    stretch_factors += pos_perc * (max_stretch_factor - 1)

    # Apply vertical and horizontal translations, inserting zero at empties
    transformed_features = apply_translation(features, v_shifts, axis=-2, val=0)
    transformed_features = apply_translation(transformed_features, h_shifts, axis=-1, val=0)
    # Apply time distortion, maintaining original dimensionality and padding with zeros
    transformed_features = apply_distortion(transformed_features, stretch_factors)

    # Process transformed features with provided model
    transformation_embeddings = model(transformed_features)[0]

    # Apply same transformations to embeddings produced for original features
    #transformed_embeddings = apply_translation(embeddings, v_shifts, axis=-2, val=0)
    #transformed_embeddings = apply_translation(transformed_embeddings, h_shifts, axis=-1, val=0)
    #transformed_embeddings = apply_distortion(transformed_embeddings, stretch_factors)

    # Convert logits to activations (implicit pitch salience)
    #transformed_salience = torch.sigmoid(transformed_embeddings)
    salience = torch.sigmoid(embeddings).unsqueeze(-3)

    # Apply same transformations to activations produced for original features
    transformed_salience = apply_translation(salience, v_shifts, axis=-2, val=0)
    transformed_salience = apply_translation(transformed_salience, h_shifts, axis=-1, val=0)
    transformed_salience = apply_distortion(transformed_salience, stretch_factors)

    # Remove temporarily added channel dimension
    transformed_salience = transformed_salience.squeeze(-3)

    # Compute geometric loss as BCE of embeddings computed from transformed features with respect to transformed activations
    geometric_loss = F.binary_cross_entropy_with_logits(transformation_embeddings, transformed_salience, reduction='none')

    # Sum across frequency bins and average across time and batch
    geometric_loss = geometric_loss.sum(-2).mean(-1).mean(-1)

    # Ignore NaNs introduced by computing BCE loss on -∞
    #geometric_loss_og[distorted_embeddings.isinf()] = 0

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
