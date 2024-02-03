# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

import torch.nn.functional as F
import torch.nn as nn
import torch


class ProjectionHead(nn.Module):
    """
    Implements a projection head for contrastive loss.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize the projection head.

        Parameters
        ----------
        dim_in : int
          Input feature size
        dim_out : int
          Output feature size
        """

        nn.Module.__init__(self)

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.ELU(inplace=True),
            nn.Linear(dim_in // 2, dim_out)
        )

    def forward(self, x):
        """
        Project features with linear layer.

        Parameters
        ----------
        x : Tensor (B x D_in x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x D_out x W)
          Batch of corresponding output features
        """

        # Swap time / feature axes and project features
        y = self.head(x.transpose(-1, -2)).transpose(-1, -2)

        return y
