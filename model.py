# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from torch import nn

import torch


class PlaceHolderNet(nn.Module):
    """
    TODO.
    """

    def __init__(self):
        """
        TODO.
        """

        super().__init__()

        self.conv0 = nn.Conv1d(1, 256, 1024, stride=512)
        self.bn0 = nn.BatchNorm1d(256)

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding="same")
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, (3, 3), padding="same")
        self.relu = nn.ReLU()

    def forward(self, audio):
        """
        TODO.

        Parameters
        ----------
        audio : TODO
          TODO

        Returns
        ----------
        embeddings : TODO
          TODO
        """

        audio = audio.float()
        audio = audio.unsqueeze(1)
        x = self.conv0(audio)
        x = self.bn0(x)
        x = self.relu(x)
        x = x.unsqueeze(1)

        x = torch.transpose(x, 2, 3)

        # input is (batch, channels, time, freq)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # output is (batch, 1, time, freq)

        embeddings = x.squeeze(1)

        return embeddings
