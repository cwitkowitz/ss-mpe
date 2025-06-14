# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .utils import gate_and_average_loss

# Regular imports
from torch.autograd import Function

import torch.nn.functional as F
import torch.nn as nn
import torch


__all__ = [
    'DomainClassifier',
    'reverse_gradient',
    'compute_adversarial_loss',
    'compute_confusion_loss'
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

        self.conv = nn.Conv1d(1, 1, 9, padding='same')
        #self.rnn = nn.GRU(n_bins, 440, batch_first=True, bidirectional=False)
        self.mp = nn.MaxPool1d(5, padding=0)
        self.ap = nn.AvgPool1d(5, stride=1)
        #self.ap = nn.AvgPool1d(5, padding=0)
        #self.mp = nn.MaxPool2d((5, 1))#, padding=(2, 0)) # TODO - ideally actually doing padding on conv
        self.fc = nn.Linear(1, 1)

        n_bins_blur_decay = 2.5
        # Compute standard deviation for kernel
        std_dev = (2 * n_bins_blur_decay) / 5
        # Truncate kernel at 4 deviations
        kernel_size = int(8 * std_dev + 1)
        # Initialize indices for the kernel
        idcs = torch.arange(kernel_size) - kernel_size // 2
        # Compute weights for a Gaussian kernel
        kernel = torch.exp(-0.5 * (idcs / std_dev) ** 2)
        # Set weight of convolutional filter to Gaussian
        self.conv.weight = torch.nn.Parameter(kernel.reshape(1, 1, -1), requires_grad=False)

        self.fc.weight = torch.nn.Parameter(torch.tensor([[1.]]))
        self.fc.bias = torch.nn.Parameter(torch.tensor([[0.]]))

    def forward(self, x):
        B, E, T = x.size()
        #x = F.relu(self.conv(x.transpose(-1, -2).reshape(B * T, 1, E)).reshape(B, T, E).transpose(-1, -2))
        #x = self.conv(x.transpose(-1, -2).reshape(B * T, 1, E))
        #x = F.elu(x).sum(-2).reshape(B, T, -1).transpose(-1, -2)
        #x = F.leaky_relu(x, negative_slope=0.3).transpose(-1, -2)
        x = F.leaky_relu(x).transpose(-1, -2)
        #x = F.relu(self.rnn(x.transpose(-1, -2))[1].transpose(0, 1)).reshape(-1, 440)
        #x = F.dropout(self.mp(x)[..., torch.randperm(88)], 0.5)
        #x = F.dropout(self.mp(x)[..., torch.randperm(88), :], 0.25)
        #x = F.dropout(self.mp(x), 0.25)
        #x = x.transpose(-1, -2)
        x = self.mp(x)
        #x = self.ap(x) # TODO - pooling before or after weighting?
        #x = x * x.softmax(dim=-1)
        #return self.fc(x).squeeze(-1)
        #return self.fc(x.transpose(-1, -2)).squeeze(-1)
        #return self.fc(x.transpose(-1, -2).sum(-1, keepdim=True)).squeeze(-1)
        #return self.fc(x.sum(-1, keepdim=True)).squeeze(-1)
        x = x.sum(-1)
        return x
        #return self.fc(x.max(-1, keepdim=True)[0]).squeeze(-1)
        #return self.fc(x.topk(x.size(-1) // 10)[0].sum(-1, keepdim=True)).squeeze(-1)
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


def compute_adversarial_loss(classifier, features, labels, lmbda=1.0, n_frames=None, rms_vals=None, rms_thr=0.01):
    if n_frames is not None and features.size(-1) >= n_frames:
        # Sample a random starting point within the provided frames
        start = torch.randint(low=0, high=features.size(-1) - n_frames + 1, size=(1,))
        # Slice the features before feeding into domain classifier
        features = features[..., start : start + n_frames]

    # Attempt to classify the features as originating from supervised (1) vs. fully self-supervised (0) data
    #domains = classifier(reverse_gradient(features, lmbda))
    domains = classifier(features.detach())

    mean_sum_sup = domains[labels == 1].mean()
    mean_sum_ss = domains[labels == 0].mean()

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

    if rms_vals is not None:
        # Gate based on RMS values and average across time and batch
        adversarial_loss = gate_and_average_loss(adversarial_loss, rms_vals, rms_thr)
    else:
        # Average across batch and time
        adversarial_loss = adversarial_loss.mean(-1).mean(-1)

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

    return adversarial_loss, (accuracy, accuracy_sp, accuracy_ss, mean_sum_sup, mean_sum_ss)

def compute_confusion_loss(classifier, features, labels=None, rms_vals=None, rms_thr=0.01):
    # Attempt to classify the features
    domains = classifier(features)

    """
    # Compute adversarial loss as BCE of embeddings for source predictions with respect to true domains
    confusion_loss = F.binary_cross_entropy_with_logits(domains, torch.ones_like(domains), reduction='none')

    if rms_vals is not None:
        # Gate based on RMS values and average across time and batch
        confusion_loss = gate_and_average_loss(confusion_loss, rms_vals, rms_thr)
    else:
        # Average across batch and time
        confusion_loss = confusion_loss.mean(-1).mean(-1)
    """

    """"""
    #mean_sum_sup = gate_and_average_loss(domains[labels == 1].detach(), rms_vals, rms_thr)
    #mean_sum_ss = gate_and_average_loss(domains[labels == 0], rms_vals, rms_thr)

    mean_sum_sup = torch.cat([c[rms_vals[i] >= rms_thr] for i, c in enumerate(domains[labels == 1].detach())]).mean()
    mean_sum_ss = torch.cat([c[rms_vals[i] >= rms_thr] for i, c in enumerate(domains[labels == 0])]).mean()

    confusion_loss = (mean_sum_sup - mean_sum_ss) ** 2 # TODO - l1?
    """"""

    return confusion_loss
