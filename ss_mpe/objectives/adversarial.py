# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from torch.autograd import Function

import torch.nn.functional as F
import torch.nn as nn
import torch


__all__ = [
    'DomainClassifier',
    'reverse_gradient',
    'compute_adversarial_loss'
]


class DomainClassifier(nn.Module):
    def __init__(self, n_bins):

        nn.Module.__init__(self)

        #self.rnn = nn.GRU(n_bins, 64, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(128, 1)
        # TODO - layer normalization?
        # TODO - max/avg pooling?

        """
        self.classifier = nn.Sequential(
            nn.Linear(n_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            #nn.Sigmoid()
        )
        """
        #self.simple_classifier = nn.Linear(n_bins, 1)

        self.conv = nn.Conv1d(1, 1, 7, padding='same')
        #self.rnn = nn.GRU(n_bins, 440, batch_first=True, bidirectional=False)
        #self.mp = nn.MaxPool1d(5, padding=0)
        self.mp = nn.MaxPool2d((5, 1), padding=0)
        self.fc = nn.Linear(88, 1)

    def forward(self, x):
        B, E, T = x.size()
        x = F.relu(self.conv(x.transpose(-1, -2).reshape(B * T, 1, E)).reshape(B, T, E).transpose(-1, -2))
        #x = F.relu(self.rnn(x.transpose(-1, -2))[1].transpose(0, 1)).reshape(-1, 440)
        #x = F.dropout(self.mp(x)[..., torch.randperm(88)], 0.5)
        x = F.dropout(self.mp(x)[..., torch.randperm(88), :], 0.5)
        #return self.fc(x).squeeze(-1)
        return self.fc(x.transpose(-1, -2)).squeeze(-1)
        #x = F.dropout(x, 0.5)
        #x = F.relu(self.rnn(x.transpose(-1, -2))[1].transpose(0, 1))
        #return self.fc(F.dropout(x.reshape(-1, 128), 0.0)).squeeze(-1)
        #return self.classifier(x.transpose(-1, -2)).squeeze(-1)
        #return self.simple_classifier(x.transpose(-1, -2)).squeeze(-1)


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lmbda):
        ctx.lambda_ = lmbda
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def reverse_gradient(x, lmbda=1.0):
    return GradientReversalFunction.apply(x, lmbda)


def compute_adversarial_loss(classifier, features, labels, lmbda=1.0, n_frames=None):
    if n_frames is not None and features.size(-1) >= n_frames:
        # Sample a random starting point within the provided frames
        start = torch.randint(low=0, high=features.size(-1) - n_frames + 1, size=(1,))
        # Slice the features before feeding into domain classifier
        features = features[..., start : start + n_frames]

    # Attempt to classify the features as originating from supervised (1) vs. fully self-supervised (0) data
    domains = classifier(reverse_gradient(features, lmbda))

    # Repeat ground-truth labels across time
    labels = labels.unsqueeze(-1).repeat(1, domains.size(-1))

    # Obtain counts for each domain
    n_total = len(labels.flatten())
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()

    # Compute ratios for equal class weighting
    #weight = torch.tensor([n_neg / n_total, n_pos / n_total])
    #weight = labels * (1 - n_pos / n_total)
    #weight[weight == 0] = (1 - n_neg / n_total)
    #weight /= weight.min()

    # Compute adversarial loss as BCE of embeddings for source predictions with respect to true domains
    adversarial_loss = F.binary_cross_entropy_with_logits(domains, labels, reduction='none')

    # Average across time and batch
    adversarial_loss = adversarial_loss.mean(-1).mean(-1)
    #adversarial_loss = adversarial_loss.mean(-1)

    # Determine which predictions were correct
    correct = torch.logical_not(torch.logical_xor(torch.sigmoid(domains).round(), labels))

    # Compute accuracy for the batch (TODO - precision / recall / FN/TN/TP/FP)
    accuracy = correct.sum() / n_total

    # Determine accuracy for each class independently
    accuracy_sp = correct[labels == 1].sum() / n_pos
    accuracy_ss = correct[labels == 0].sum() / n_neg

    #from timbre_trap.utils import plot_latents
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plot_latents(features[..., 100].cpu().detach(), labels.cpu().detach().numpy().tolist(), fig=fig)
    #plt.show()

    return adversarial_loss, (accuracy, accuracy_sp, accuracy_ss)
