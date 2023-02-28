# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

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


def compute_invariance_loss(audio, model, transforms):
    original_embeddings = model(audio)

    transformed_audio = transforms(audio.unsqueeze(1), sample_rate=16000).squeeze(1)

    transformed_embeddings = model(transformed_audio)

    test = info_nce_loss(torch.cat((original_embeddings, transformed_embeddings), dim=0))

    pair_losses = torch.nn.functional.mse_loss(transformed_embeddings, original_embeddings, reduction='none')

    invariance_loss = torch.mean(pair_losses.sum(-1).mean(-1))

    return invariance_loss


import torch.nn.functional as F


def SimCLR_loss(features, temperature=0.07):
    labels = torch.cat([torch.arange(8) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(0)


    features = features[:, 0, :]

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(0)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(0)

    logits = logits / temperature
    return logits, labels
