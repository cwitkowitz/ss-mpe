# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .utils import gate_and_average_loss

# Regular imports
from torch.nn.utils.parametrizations import weight_norm
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
        #self.simple_classifier = nn.Linear(n_bins, 1)

        n_filters = 32

        self.conv = weight_norm(nn.Conv1d(1, n_filters, 7, padding='same'))
        #self.conv = nn.Conv1d(1, n_filters, 9, padding='same')
        #self.conv.requires_grad_(False)
        self.rnn = nn.GRU(n_bins, 440, batch_first=True, bidirectional=False)
        self.mp = nn.MaxPool1d(5, padding=0)
        #self.ap = nn.AvgPool1d(5, stride=1, padding=0)
        #self.ap = nn.AvgPool1d(5, padding=0)
        #self.mp = nn.MaxPool2d((5, 1))#, padding=(2, 0)) # TODO - ideally actually doing padding on conv
        #self.fc = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(n_filters)
        self.fc = nn.Linear(n_filters, 1)
        """

        self.a = 0

        """
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
        """

        #self.fc.weight = torch.nn.Parameter(torch.tensor([[1.]]))
        #self.fc.bias = torch.nn.Parameter(torch.tensor([[0.]]))

        """
        self.m = torch.nn.Parameter(torch.tensor([[1.]]))
        self.b = torch.nn.Parameter(torch.tensor([[0.]]))
        """

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        #self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        #self.conv5 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        """
        B, E, T = x.size()
        #x = F.relu(self.conv(x.transpose(-1, -2).reshape(B * T, 1, E)).reshape(B, T, E).transpose(-1, -2))
        x = self.conv(x.transpose(-1, -2).reshape(B * T, 1, E))
        #x = self.bn(x)
        #x = F.elu(x).sum(-2).reshape(B, T, -1).transpose(-1, -2)
        #x = F.leaky_relu(x, negative_slope=0.3).transpose(-1, -2)
        #x = F.leaky_relu(x, negative_slope=0.1).transpose(-1, -2)
        x = F.leaky_relu(x).transpose(-1, -2)
        #x = F.relu(self.rnn(x.transpose(-1, -2))[1].transpose(0, 1)).reshape(-1, 440)
        #x = F.dropout(self.mp(x)[..., torch.randperm(88)], 0.5)
        #x = F.dropout(self.mp(x)[..., torch.randperm(88), :], 0.25)
        #x = F.dropout(self.mp(x), 0.25)
        x = F.dropout(x, 0.5)
        #x = x.transpose(-1, -2)
        x = self.fc(x).reshape(B, T, -1)
        #x = x.reshape(B, T, -1)
        #x = F.leaky_relu(x)
        #x = F.dropout(x, 0.5)
        x = self.mp(x)
        x = F.leaky_relu(x)
        #x = x.masked_fill(~(torch.rand_like(x) > 0.5), float('-inf'))
        #x = self.ap(x) # TODO - pooling before or after weighting?
        #x = x * x.softmax(dim=-1)
        #return self.fc(x).squeeze(-1)
        #return self.fc(x.transpose(-1, -2)).squeeze(-1)
        #return self.fc(x.transpose(-1, -2).sum(-1, keepdim=True)).squeeze(-1)
        #return self.fc(x.sum(-1, keepdim=True)).squeeze(-1)
        x = x.sum(-1)
        #x = x.max(-1)[0]
        y = self.m * x + self.b
        """
        x = x.unsqueeze(-3)
        x = F.dropout(F.leaky_relu(self.conv1(x), negative_slope=0.2), 0.5)
        x = F.dropout(F.leaky_relu(self.conv2(x), negative_slope=0.2), 0.5)
        x = F.dropout(F.leaky_relu(self.conv3(x), negative_slope=0.2), 0.5)
        #x = F.dropout(F.leaky_relu(self.conv4(x), negative_slope=0.2), 0.5)
        x = self.conv5(x)
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


def compute_adversarial_loss(classifier, features, labels, n_frames=None, rms_vals=None, rms_thr=0.01):
    if n_frames is not None and features.size(-1) >= n_frames:
        # Sample a random starting point within the provided frames
        start = torch.randint(low=0, high=features.size(-1) - n_frames + 1, size=(1,))
        # Slice the features before feeding into domain classifier
        features = features[..., start : start + n_frames]

    n_sup = (labels == 1).sum().item()
    n_ss = (labels == 0).sum().item()

    #activations = torch.sigmoid(features)

    #activations_sup = activations[labels == 1].detach()
    #activations_ss = activations[labels == 0].detach()
    features_sup = features[labels == 1].detach()
    features_ss = features[labels == 0].detach()

    #sup_idcs = torch.randint(high=n_sup, size=(n_ss,))

    if classifier.a > 0:
        lmbdas = torch.distributions.Beta(classifier.a, classifier.a).sample((n_ss,)).reshape(-1, 1, 1).to(features.device)
    else:
        lmbdas = torch.randint(0, 2, (n_ss,), dtype=torch.float).reshape(-1, 1, 1).to(features.device)

    #mixed_features = lmbdas * features_sup[sup_idcs] + (1 - lmbdas) * features_ss
    #mixed_features = lmbdas * features_sup + (1 - lmbdas) * features_ss
    mixed_features1 = lmbdas * features_sup + (1 - lmbdas) * features_ss
    mixed_features2 = (1 - lmbdas) * features_sup + lmbdas * features_ss
    mixed_features = torch.cat((mixed_features1, mixed_features2), dim=0)
    lmbdas = torch.cat((lmbdas, (1 - lmbdas)), dim=0)

    # Attempt to classify the features as originating from supervised (1) vs. fully self-supervised (0) data
    #domains = classifier(reverse_gradient(features, lmbda))
    #domains = classifier(features.detach())
    domains = classifier(mixed_features).mean(-1).mean(-1)

    # Repeat ground-truth labels across time
    #labels = labels.unsqueeze(-1).repeat(1, domains.size(-1))

    # Obtain counts for each domain
    n_total = len(labels.flatten())
    #n_pos = (labels == 1).sum().item()
    #n_neg = (labels == 0).sum().item()

    # Compute ratios for equal class weighting
    #weight = torch.tensor([n_neg / n_total, n_pos / n_total])
    #weight = labels * (1 - n_pos / n_total)
    #weight[weight == 0] = (1 - n_neg / n_total)
    #weight /= weight.min()

    # Compute adversarial loss as BCE of embeddings for source predictions with respect to true domains
    #adversarial_loss = F.binary_cross_entropy_with_logits(domains, labels, reduction='none')
    adversarial_loss = F.binary_cross_entropy_with_logits(domains, lmbdas.squeeze(-1), reduction='none')

    if rms_vals is not None:
        # TODO - may mess up 50/50 balance
        # Gate based on RMS values and average across time and batch
        adversarial_loss = gate_and_average_loss(adversarial_loss, rms_vals, rms_thr)
    else:
        # Average across batch and time
        adversarial_loss = adversarial_loss.mean(-1).mean(-1)

    with torch.no_grad():
        domains_unmixed = classifier(features).mean(-1).mean(-1).squeeze(-1)

        mean_sup = domains_unmixed[labels == 1].mean()
        mean_ss = domains_unmixed[labels == 0].mean()

        # Determine which predictions were correct
        correct = torch.logical_not(torch.logical_xor(torch.sigmoid(domains_unmixed).round(), labels))

        # Compute accuracy for the batch (TODO - precision / recall / FN/TN/TP/FP)
        accuracy = correct.sum() / len(correct)

        # Determine accuracy for each class independently
        accuracy_sp = correct[labels == 1].sum() / n_sup
        accuracy_ss = correct[labels == 0].sum() / n_ss

    #from timbre_trap.utils import plot_latents
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #plot_latents(features[..., 100].cpu().detach(), labels.cpu().detach().numpy().tolist(), fig=fig)
    #plt.show()

    #dc_m, dc_b = classifier.m, classifier.b

    return adversarial_loss, (accuracy, accuracy_sp, accuracy_ss, mean_sup, mean_ss)

def compute_confusion_loss(classifier, features, labels=None, n_frames=None, rms_vals=None, rms_thr=0.01):
    if n_frames is not None and features.size(-1) >= n_frames:
        # Sample a random starting point within the provided frames
        start = torch.randint(low=0, high=features.size(-1) - n_frames + 1, size=(1,))
        # Slice the features before feeding into domain classifier
        features = features[..., start : start + n_frames]

    n_sup = (labels == 1).sum().item()
    n_ss = (labels == 0).sum().item()

    #activations = torch.sigmoid(features)

    #activations_sup = activations[labels == 1]
    #activations_ss = activations[labels == 0]
    features_sup = features[labels == 1]
    features_ss = features[labels == 0]

    #sup_idcs = torch.randint(high=n_sup, size=(n_ss,))

    if classifier.a > 0:
        lmbdas = torch.distributions.Beta(classifier.a, classifier.a).sample((n_ss,)).reshape(-1, 1, 1).to(features.device)
    else:
        lmbdas = torch.randint(0, 2, (n_ss,), dtype=torch.float).reshape(-1, 1, 1).to(features.device)

    #mixed_features = lmbdas * features_sup[sup_idcs] + (1 - lmbdas) * features_ss
    #mixed_features = lmbdas * features_sup + (1 - lmbdas) * features_ss
    #mixed_features1 = lmbdas * features_sup + (1 - lmbdas) * features_ss
    #mixed_features2 = (1 - lmbdas) * features_sup + lmbdas * features_ss
    #mixed_features = torch.cat((mixed_features1, mixed_features2), dim=0)
    #lmbdas = torch.cat((lmbdas, (1 - lmbdas)), dim=0)

    # Attempt to classify the features
    domains = classifier(features).mean(-1).mean(-1)
    #domains = classifier(mixed_features)

    """"""
    # Compute adversarial loss as BCE of embeddings for source predictions with respect to true domains
    confusion_loss = F.binary_cross_entropy_with_logits(domains, torch.ones_like(domains), reduction='none')
    #confusion_loss = F.binary_cross_entropy_with_logits(domains, (1 - lmbdas.squeeze(-1)), reduction='none')
    #confusion_loss = F.binary_cross_entropy_with_logits(domains, torch.maximum(lmbdas.squeeze(-1), (1 - lmbdas.squeeze(-1))), reduction='none')

    if rms_vals is not None:
        # Gate based on RMS values and average across time and batch
        confusion_loss = gate_and_average_loss(confusion_loss, rms_vals, rms_thr)
    else:
        # Average across batch and time
        confusion_loss = confusion_loss.mean(-1).mean(-1)
    """"""

    """
    #mean_sum_sup = gate_and_average_loss(domains[labels == 1].detach(), rms_vals, rms_thr)
    #mean_sum_ss = gate_and_average_loss(domains[labels == 0], rms_vals, rms_thr)

    mean_sum_sup = torch.cat([c[rms_vals[i] >= rms_thr] for i, c in enumerate(domains[labels == 1].detach())]).mean()
    mean_sum_ss = torch.cat([c[rms_vals[i] >= rms_thr] for i, c in enumerate(domains[labels == 0])]).mean()

    confusion_loss = (mean_sum_sup - mean_sum_ss) ** 2 # TODO - l1?
    """

    return confusion_loss
