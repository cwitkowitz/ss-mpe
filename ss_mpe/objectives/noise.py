# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'add_random_noise',
    'compute_noise_loss'
]


def add_random_noise(audio, max_volume=1.0):
    # Determine batch size
    B = audio.size(0)

    # Sample random volumes for percussion audio
    volumes = max_volume * torch.rand((B, 1, 1), device=audio.device)

    # Sample white noise for each track
    white_noise = torch.randn_like(audio)

    # Mix sampled noise with original audio
    mixtures = audio + volumes * white_noise

    return mixtures


def compute_noise_loss(model, audio, targets, **an_kwargs):
    # Superimpose random white noise onto original audio
    noisy_audio = add_random_noise(audio, **an_kwargs)

    # Compute spectral features for noisy audio
    noisy_features = model.hcqt.to_decibels(model.hcqt(noisy_audio))

    # Process noisy features with provided model
    noisy_embeddings = model(noisy_features)

    # Compute noise loss as BCE of embeddings computed from noisy features with respect to original targets
    noise_loss = F.binary_cross_entropy_with_logits(noisy_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    noise_loss = noise_loss.sum(-2).mean(-1).mean(-1)

    return noise_loss
