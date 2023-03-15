# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>


import torch.nn.functional as F
import torch.nn as nn
import torch


class SAUNet(nn.Module):
    """
    SA U-Net adapted from https://github.com/christofw/multipitch_architectures (simple_u_net_doubleselfattn).
    TODO - better description
    """

    def __init__(self, n_ch_in=6, n_bins_in=216, n_heads=8, model_complexity=1):
        """
        TODO
        """

        nn.Module.__init__(self)

        # Number of channels at each stage
        n_ch_1 = 8 * 2 ** (model_complexity - 1)
        n_ch_2 = 16 * 2 ** (model_complexity - 1)
        n_ch_3 = 32 * 2 ** (model_complexity - 1)
        n_ch_4 = 64 * 2 ** (model_complexity - 1)

        # Embedding sizes for self-attention modules
        d_attention = 8 * 8 * 2 ** (model_complexity - 1)
        d_forward = 1024 * 2 ** (model_complexity - 1)

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_ch_in, n_bins_in])

        self.initial_conv = DoubleConv(in_channels=n_ch_in,
                                       out_channels=n_ch_1,
                                       kernel_size=15,
                                       padding=7)

        self.down_block_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=n_ch_1,
                       out_channels=n_ch_2,
                       kernel_size=15,
                       padding=7)
        )
        self.down_block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=n_ch_2,
                       out_channels=n_ch_3,
                       kernel_size=9,
                       padding=4)
        )
        self.down_block_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=n_ch_3,
                       out_channels=n_ch_4,
                       kernel_size=5,
                       padding=2)
        )
        self.down_block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=n_ch_4,
                       out_channels=n_ch_4,
                       kernel_size=3,
                       padding=1)
        )

        #self.bottleneck = nn.Sequential(
        self.pos = SinusoidalEncodings()#,
        self.sa1 = nn.TransformerEncoderLayer(d_model=d_attention,
                                       nhead=n_heads,
                                       dim_feedforward=d_forward,
                                       batch_first=True)#,
        self.sa2 = nn.TransformerEncoderLayer(d_model=d_attention,
                                       nhead=n_heads,
                                       dim_feedforward=d_forward,
                                       batch_first=True)
            # TODO - reshape
        #)

        self.concat_up = ConcatenativeUpSample2d(factor=2)

        self.up_conv_1 = DoubleConv(in_channels=2 * n_ch_4,
                                    mid_channels=n_ch_4,
                                    out_channels=n_ch_3,
                                    kernel_size=3,
                                    padding=1)
        self.up_conv_2 = DoubleConv(in_channels=2 * n_ch_3,
                                    mid_channels=n_ch_3,
                                    out_channels=n_ch_2,
                                    kernel_size=5,
                                    padding=2)
        self.up_conv_3 = DoubleConv(in_channels=2 * n_ch_2,
                                    mid_channels=n_ch_2,
                                    out_channels=n_ch_1,
                                    kernel_size=9,
                                    padding=4)
        self.up_conv_4 = DoubleConv(in_channels=2 * n_ch_1,
                                    mid_channels=n_ch_1,
                                    out_channels=1,
                                    kernel_size=15,
                                    padding=7)

    def forward(self, hcqt):
        """
        TODO
        """

        # TODO - remove the following line
        hcqt = hcqt[..., :75]

        # Normalize harmonic channels and frequency bins of HCQTs
        hcqt = self.layernorm(hcqt.transpose(-1, -2).transpose(-2, -3)
                              ).transpose(-2, -3).transpose(-1, -2)

        # Obtain an initial set of features
        x1 = self.initial_conv(hcqt)

        # Feed features through all downsampling blocks
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        embeddings = self.down_block_4(x4)

        # Flatten time and frequency dimensions, then switch with channel dimension
        embeddings = embeddings.flatten(-2).transpose(-1, -2)

        #embeddings = self.bottleneck(embeddings)

        xt = self.pos(embeddings)
        xt = self.sa1(xt)
        xt = self.sa2(xt)

        x = self.up_conv_1(self.concat_up(embeddings, x4))
        x = self.up_conv_2(self.concat_up(x, x3))
        x = self.up_conv_3(self.concat_up(x, x2))
        salience = self.up_conv_4(self.concat_up(x, x1))

        return salience


class DoubleConv(nn.Module):
    """
    TODO
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
        """
        TODO
        """

        nn.Module.__init__(self)

        if mid_channels is None:
            mid_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        TODO
        """

        features = self.conv_block(features)

        return features


class SinusoidalEncodings(nn.Module):
    """
    TODO
    """

    def __init__(self, max_freq=10000):
        """
        TODO
        """

        nn.Module.__init__(self)

        self.max_freq = max_freq

    def forward(self, features, interleave=True):
        """
        TODO
        """

        # Determine the dimensionality of the input features
        B, T, E = features.size()

        # Determine the frequencies corresponding to each dimension (pair)
        frequencies = self.max_freq ** (torch.arange(0, E, 2) / E)
        # Multiply every position by every frequency
        angles = torch.outer(torch.arange(0, T), 1 / frequencies)

        # Compute the sine and cosine of the angles
        angles_sin = torch.sin(angles)
        angles_cos = torch.cos(angles)

        if interleave:
            # Add an extra dimension to each vector
            angles_sin = torch.unsqueeze(angles_sin, dim=-1)
            angles_cos = torch.unsqueeze(angles_cos, dim=-1)

        # Interleave the two positional encodings
        sinusoidal_encodings = torch.cat((angles_sin, angles_cos), dim=-1)

        if interleave:
            # Collapse the added dimension to interleave the vectors
            sinusoidal_encodings = sinusoidal_encodings.view(T, E)

        # Repeat encodings for each sample in the batch and add to appropriate device
        features += torch.tile(sinusoidal_encodings.to(features.device), (B, 1, 1))

        return features


class ConcatenativeUpSample2d(nn.Module):
    """
    TODO
    """

    def __init__(self, factor=2):
        """
        TODO
        """

        nn.Module.__init__(self)

        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

    def forward(self, in_features, skip_features):
        """
        TODO
        """

        # Upsample the features using biliniear interpolation
        x = self.upsample(in_features)

        # Determine the target dimensionality of the upsampled features
        target_height, target_width = skip_features.shape[-2:]

        # Determine the actual dimensionality of the upsampled features
        actual_height, actual_width = x.shape[-2:]

        # Compute the number of missing rows and/or columns
        missing_rows = target_height - actual_height
        missing_cols = target_width - actual_width

        # Compute the appropriate padding for the features
        pad_l = missing_cols // 2
        pad_r = missing_cols - pad_l
        pad_t = missing_rows // 2
        pad_b = missing_rows - pad_t

        # Pad the upsampled features to match the dimensionality of the skip features
        x = F.pad(x, [pad_l, pad_r, pad_t, pad_b])

        # Concatenate the features along the channel dimension
        out_features = torch.cat([skip_features, x], dim=-3)

        return out_features
