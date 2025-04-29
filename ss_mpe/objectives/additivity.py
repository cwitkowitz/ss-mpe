# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch


__all__ = [
    'create_batch_mixtures',
    'mix_random_tracks',
    'compute_additivity_loss'
]


def constructive_maximum(t1, t2, beta=0.5):
    max_vals = torch.maximum(t1, t2)
    min_vals = torch.minimum(t1, t2)

    cmax = max_vals + beta * min_vals * (1 - max_vals)

    # t1 = torch.linspace(0, 1, steps=2001)
    # t1, t2 = torch.meshgrid(t1, t1, indexing='xy')
    # from timbre_trap.utils import *
    # cmax = to_array(constructive_maximum(t1, t2))
    # from ss_mpe.visualization import plot_bce_loss
    # plot_bce_loss(cmax)

    return cmax


def create_batch_mixtures(audio, targets):
    # Mix adjacent audio samples within batch
    mixed_audio = audio + audio.roll(-1, 0)

    # Mix adjacent targets via the max operation
    mixed_targets = torch.maximum(targets, targets.roll(-1, 0))

    return mixed_audio, mixed_targets


def mix_random_tracks(audio, targets, model, additive_set_combo):
    # Determine batch size
    B = audio.size(0)

    # Initialize list for sampled additive audio
    additive_audio = list()

    # Sample indices for a batch of additive audio
    idcs = torch.randperm(len(additive_set_combo))[:B]

    for i in idcs:
        # Keep track of relative index
        local_idx, dataset_idx = i, 0

        while local_idx >= len(additive_set_combo.datasets[dataset_idx]):
            # Subtract length of current sub-dataset from global index
            local_idx -= len(additive_set_combo.datasets[dataset_idx])
            # Check next dataset
            dataset_idx += 1

        # Obtain a reference to the dataset of the sampled track
        additive_dataset = additive_set_combo.datasets[dataset_idx]

        # Determine which track was sampled from the dataset
        additive_track = additive_dataset.tracks[local_idx]

        # Obtain the full-length audio corresponding to the sampled track
        additive_audio_ = additive_dataset.get_audio(additive_track)
        # Slice the sampled audio to match duration of the provided audio
        additive_audio_ = additive_dataset.slice_audio(additive_audio_,
                                                       n_samples=audio.size(-1))[0]
        # Add additive audio with batch dimension to overall list for the batch
        additive_audio.append(additive_audio_.unsqueeze(0).to(audio.device))

    # Combine additive audio along the batch dimension
    additive_audio = torch.cat(additive_audio, dim=0)

    # Mix sampled additive audio with original audio
    mixed_audio = audio + additive_audio

    # Compute spectral features for additive audio
    additive_features = model.hcqt.to_decibels(model.hcqt(additive_audio))

    # Process additive features and convert to activations
    #additive_targets = torch.sigmoid(model(additive_features))
    additive_targets = torch.softmax(model(additive_features), dim=-2)

    # Mix original and additive targets via the max operation
    mixed_targets = torch.maximum(targets, additive_targets)

    return mixed_audio, mixed_targets


def compute_additivity_loss(model, audio, targets, **ad_kwargs):
    # Determine which function to use for mixing
    additive_fn = ad_kwargs.pop('additive_fn')

    if additive_fn == create_batch_mixtures:
        # Superimpose audio from the same batch onto original each other
        mixed_audio, mixed_targets = create_batch_mixtures(audio, targets)
    else:
        # Superimpose audio from a specific dataset onto original audio
        mixed_audio, mixed_targets = mix_random_tracks(audio, targets, model, **ad_kwargs)

    # Compute spectral features for mixed audio
    mixed_features = model.hcqt.to_decibels(model.hcqt(mixed_audio))

    # Process features with provided model
    mixed_embeddings = model(mixed_features)

    # Compute additivity loss as BCE of embeddings computed from mixed features with respect to mixed targets
    #additivity_loss = F.binary_cross_entropy_with_logits(mixed_embeddings, mixed_targets, reduction='none')
    additivity_loss = F.cross_entropy(mixed_embeddings, mixed_targets, reduction='none')

    # Sum across frequency bins and average across time and batch
    #additivity_loss = additivity_loss.sum(-2).mean(-1).mean(-1)
    additivity_loss = additivity_loss.mean(-1).mean(-1)

    return additivity_loss
