# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'mix_random_percussion',
    'compute_percussion_loss'
]


def mix_random_percussion(audio, percussive_set_combo, max_volume=1.0):
    # Determine batch size
    B = audio.size(0)

    # Sample random volumes for percussion audio
    volumes = max_volume * torch.rand((B, 1, 1), device=audio.device)

    # Initialize list for sampled percussive audio
    percussive_audio = list()

    # Sample indices for a batch of percussive audio
    idcs = torch.randperm(len(percussive_set_combo))[:B]

    for i in idcs:
        # Keep track of relative index
        local_idx, dataset_idx = i, 0

        while local_idx >= len(percussive_set_combo.datasets[dataset_idx]):
            # Subtract length of current sub-dataset from global index
            local_idx -= len(percussive_set_combo.datasets[dataset_idx])
            # Check next dataset
            dataset_idx += 1

        # Obtain a reference to the dataset of the sampled track
        percussive_dataset = percussive_set_combo.datasets[dataset_idx]

        # Determine which track was sampled from the dataset
        percussive_track = percussive_dataset.tracks[local_idx]

        # Obtain the full-length audio corresponding to the sampled track
        percussive_audio_ = percussive_dataset.get_audio(percussive_track)
        # Slice the sampled audio to match duration of the provided audio
        percussive_audio_ = percussive_dataset.slice_audio(percussive_audio_,
                                                           n_samples=audio.size(-1))[0]
        # Add percussive audio with batch dimension to overall list for the batch
        percussive_audio.append(percussive_audio_.unsqueeze(0).to(audio.device))

    # Combine percussive audio along the batch dimension
    percussive_audio = torch.cat(percussive_audio, dim=0)

    # Mix sampled percussive audio with original audio
    mixtures = audio + volumes * percussive_audio

    return mixtures
    #return mixtures, volumes, percussive_audio


def compute_percussion_loss(model, audio, targets, **pc_kwargs):
    # Superimpose random percussive audio onto original audio
    percussive_audio = mix_random_percussion(audio, **pc_kwargs)

    # Compute spectral features for percussive audio mixture
    percussive_features = model.hcqt.to_decibels(model.hcqt(percussive_audio))

    # Process percussive features with provided model
    percussive_embeddings, _ = model(percussive_features)

    # Compute percussion loss as BCE of embeddings computed from percussive features with respect to original targets
    percussion_loss = F.binary_cross_entropy_with_logits(percussive_embeddings, targets, reduction='none')
    #percussion_loss = F.cross_entropy(percussive_embeddings, targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    percussion_loss = percussion_loss.sum(-2).mean(-1).mean(-1)
    #percussion_loss = percussion_loss.mean(-1).mean(-1)

    return percussion_loss
