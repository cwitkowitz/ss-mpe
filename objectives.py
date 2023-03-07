# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>


import torch.nn.functional as F
import torch


def compute_linearity_loss(audio, model, transforms=None):
    batch_size = audio.size(0)

    isolated_embeddings = model(audio)

    mixtures = audio.unsqueeze(0).repeat(batch_size, 1, 1) + \
               audio.unsqueeze(1).repeat(1, batch_size, 1)

    mixture_idcs_r = torch.arange(batch_size).unsqueeze(0).repeat(batch_size, 1).flatten()
    mixture_idcs_c = torch.arange(batch_size).unsqueeze(1).repeat(1, batch_size).flatten()

    # TODO - try random mixtures (w/ random scaling) instead of pairwise?
    # TODO - if transforms != None, transform audio before training for linearity
    # TODO - make sure this will work for combinations of the same pitch

    mixtures = mixtures.reshape(batch_size ** 2, -1)

    mixtures = mixtures / 2

    mixture_embeddings = model(mixtures)

    pair_weights = 1 - torch.eye(batch_size).flatten().to('cuda:0')

    target_embeddings = isolated_embeddings[mixture_idcs_r] + \
                        isolated_embeddings[mixture_idcs_c]

    pair_losses = torch.nn.functional.mse_loss(mixture_embeddings, target_embeddings, reduction='none')

    linearity_loss = torch.mean(pair_weights * pair_losses.sum(-1).mean(-1)) / 2

    return linearity_loss


def compute_content_loss(audio, model):
    # TODO - might be OK if input audio contains silence

    # TODO - should scale with loudness of audio

    isolated_embeddings = model(audio)

    # compute RMS value on embedding elements
    content_loss = torch.mean(torch.e ** (-1 * (isolated_embeddings ** 2).mean(-1).sqrt()))

    return content_loss


def compute_contrastive_loss(originals, augmentations, temperature=0.07):
    # SimCLR

    # TODO - more than one augmentation?

    assert originals.shape == augmentations.shape

    # Determine which device to use for processing
    device = originals.device

    # Keep track of original dimensionality
    B, T, E = originals.shape

    # Concatenate both sets of embeddings along the batch dimension
    embeddings = torch.cat((originals, augmentations), dim=0)

    # Normalize both sets of embeddings
    embeddings = F.normalize(embeddings, dim=-1)

    # Switch the batch and frame dimensions for the embeddings
    embeddings = embeddings.transpose(0, 1)

    # Compute cosine similarity between every embedding across each frame
    similarities = torch.bmm(embeddings, embeddings.transpose(-1, -2))

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


def compute_timbre_invariance_loss(audio, model, transforms):
    original_embeddings = model(audio)

    transformed_audio = transforms(audio.unsqueeze(1), sample_rate=16000).squeeze(1)

    transformed_embeddings = model(transformed_audio)

    invariance_loss = compute_contrastive_loss(original_embeddings, transformed_embeddings)

    return invariance_loss


def compute_shift_invariance_loss():
    # TODO
    pass
