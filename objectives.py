# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch


def compute_linearity_loss(audio, model):
    batch_size = audio.size(0)

    isolated_embeddings = model(audio)

    mixtures = audio.unsqueeze(0).repeat(batch_size, 1, 1) + \
               audio.unsqueeze(1).repeat(1, batch_size, 1)

    mixture_idcs_r = torch.arange(batch_size).unsqueeze(0).repeat(batch_size, 1).flatten()
    mixture_idcs_c = torch.arange(batch_size).unsqueeze(1).repeat(1, batch_size).flatten()

    # TODO - try random mixtures (w/ random scaling) instead of pairwise?

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

    isolated_embeddings = model(audio)

    # compute RMS value on embedding elements
    content_loss = torch.mean(torch.e ** (-1 * (isolated_embeddings ** 2).mean(-1).sqrt()))

    return content_loss


def compute_similarity_loss(audio, model, transforms):
    pass
