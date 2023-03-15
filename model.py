# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>


import torch.nn.functional as F
import torch.nn as nn
import torch


class SAUNet(nn.Module):
    """
    SA U-Net adapted from https://github.com/christofw/multipitch_architectures (simple_u_net_doubleselfattn).
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
        d_attention = 8 * (8 * 2 ** (model_complexity - 1))
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
        self.flatten = nn.Flatten(start_dim=-2)#,
        self.pos = SinusoidalEncodings(feature_size=d_attention)#,
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

        self.upsample = ConcatenativeUpSample2d(factor=2)

        self.up_conv_1 = DoubleConv(in_channels=1024//sc,
                                    out_channels=512//(sc*2),
                                    mid_channels=1024//(sc*2),
                                    kernel_size=3,
                                    padding=1)
        self.up_conv_2 = DoubleConv(in_channels=512//sc,
                                    out_channels=256//(sc*2),
                                    mid_channels=512//(sc*2),
                                    kernel_size=5,
                                    padding=2)
        self.up_conv_3 = DoubleConv(in_channels=256//sc,
                                    out_channels=128//(sc*2),
                                    mid_channels=256//(sc*2),
                                    kernel_size=9,
                                    padding=4)
        self.up_conv_4 = DoubleConv(in_channels=128//sc,
                                    out_channels=1,
                                    mid_channels=128//(sc*2),
                                    kernel_size=15,
                                    padding=7)

    def forward(self, hcqt):
        """
        TODO
        """

        # TODO - remove the following line
        #hcqt = hcqt[..., :75]

        # Normalize harmonic channels and frequency bins of HCQTs
        hcqt = self.layernorm(hcqt.transpose(-1, -2).transpose(-2, -3)
                              ).transpose(-2, -3).transpose(-1, -2)

        x1 = self.initial_conv(hcqt)
        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x5 = self.down_block_4(x4)

        #x5 = self.bottleneck(x5)
        #xt = self.flatten(x5)
        #xt = self.pos(xt)
        #xt = self.sa1(xt)
        #xt = self.sa2(xt)

        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        salience = self.upconv4(self.upconcat(x, x1))

        return salience


class DoubleConv(nn.Module):
    """
    TODO
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=None, padding=None):
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

    def __init__(self, feature_size):
        """
        TODO
        """

        nn.Module.__init__(self)

        self.feature_size = feature_size

    def forward(self, features, interleave=True):
        """
        TODO
        """

        # Determine the dimensionality of the input features
        B, T, E = features.size()

        # Determine the frequencies corresponding to each (pair of) dimension
        frequencies = 10000 ** (torch.arange(0, self.feature_size, 2) / self.feature_size)
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
            sinusoidal_encodings = sinusoidal_encodings.view(T, self.feature_size)

        sinusoidal_encodings = torch.tile(sinusoidal_encodings, (B, 1, 1))

        features += pe[:features.shape[1], :]

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

        # Since using bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

    def forward(self, x1, skip_features):
        """
        TODO
        """

        x1 = self.up(x1)
        diffY = skip_features.size()[2] - x1.size()[2]
        diffX = skip_features.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([skip_features, x1], dim=1)
        return x
