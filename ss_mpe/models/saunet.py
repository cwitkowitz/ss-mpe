# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import SS_MPE

# Regular imports
import torch.nn.functional as F
import torch.nn as nn
import torch


class AE_Base(SS_MPE):
    """
    Implements adaptation of base U-Net from https://ieeexplore.ieee.org/document/9865174.
    """

    def __init__(self, hcqt_params, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See SS_MPE class for others...

        model_complexity : int
          Scaling factor for number of filters
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        super().__init__(hcqt_params)

        # Extract HCQT parameters to infer dimensionality of input features
        n_bins, n_harmonics = hcqt_params['n_bins'], len(hcqt_params['harmonics'])
        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_harmonics, n_bins])

        self.encoder = Encoder(in_channels=n_harmonics,
                               model_complexity=model_complexity)
        self.decoder = Decoder(out_channels=1,
                               model_complexity=model_complexity,
                               skip_connections=skip_connections)

        self.skip_connections = skip_connections

    def forward(self, features):
        """
        Process spectral features to obtain pitch salience logits (for training/evaluation).

        Parameters
        ----------
        features : Tensor (B x H x F x T)
          Batch of HCQT spectral features

        Returns
        ----------
        output : Tensor (B x F X T)
          Batch of (implicit) pitch salience logits
        latents : Tensor (B x 8 * H x F_lat, T_lat)
          Batch of latent codes
        losses : dict containing
          ...
        """

        # Normalize the spectral features
        features = self.layernorm(features.permute(0, -1, -3, -2)).permute(0, -2, -1, -3)

        # Process features with the encoder
        latents, padding, embeddings, losses = self.encoder(features)

        if not self.skip_connections:
            # Discard embeddings from encoder
            embeddings = [None for _ in embeddings]

        # Process latents with the decoder
        output = self.decoder(latents, padding, embeddings)

        # Collapse channel dimension
        output = output.squeeze(-3)

        return output, latents, losses


class Encoder(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, in_channels, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (8  * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    32 * 2 ** (model_complexity - 1),
                    64 * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        self.convin = DoubleConv(in_channels, channels[0], kernel_size=15)

        self.block1 = EncoderBlock(channels[0], channels[1], kernel_size=15)
        self.block2 = EncoderBlock(channels[1], channels[2], kernel_size=9)
        self.block3 = EncoderBlock(channels[2], channels[3], kernel_size=5)
        self.block4 = EncoderBlock(channels[3], channels[3], kernel_size=3)

    def forward(self, features):
        """
        Encode a batch of input spectral features.

        Parameters
        ----------
        features : Tensor (B x C_in x F X T)
          Batch of input spectral features

        Returns
        ----------
        latents : Tensor (B x 8 * C_in x F_lat, T_lat)
          Batch of latent codes
        padding : list of tuple (int, int)
          Padding required at each stage
        embeddings : list of [Tensor (B x C x H x W)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Initialize a list to hold features for skip connections
        padding, embeddings = list(), list()

        embeddings.append(self.convin(features))
        padding.append((embeddings[-1].size(-2) % 2,
                        embeddings[-1].size(-1) % 2))

        embeddings.append(self.block1(embeddings[-1]))
        padding.append((embeddings[-1].size(-2) % 2,
                        embeddings[-1].size(-1) % 2))

        embeddings.append(self.block2(embeddings[-1]))
        padding.append((embeddings[-1].size(-2) % 2,
                        embeddings[-1].size(-1) % 2))

        embeddings.append(self.block3(embeddings[-1]))
        padding.append((embeddings[-1].size(-2) % 2,
                        embeddings[-1].size(-1) % 2))

        # Compute latent vectors from embeddings
        latents = self.block4(embeddings[-1])

        # Reverse order of lists
        padding.reverse()
        embeddings.reverse()

        # No encoder losses
        loss = dict()

        return latents, padding, embeddings, loss


class Decoder(nn.Module):
    """
    Implements a 2D convolutional decoder.
    """

    def __init__(self, out_channels=1, model_complexity=1, skip_connections=True):
        """
        Initialize the decoder.

        Parameters
        ----------
        out_channels : int
          Number of output feature channels
        model_complexity : int
          Scaling factor for number of filters
        skip_connections : bool
          Whether to expect skip connections from encoder
        """

        nn.Module.__init__(self)

        channels = (64 * 2 ** (model_complexity - 1),
                    32 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        self.block1 = DecoderBlock(channels[0], channels[0], kernel_size=3, skip_connections=False)
        self.block2 = DecoderBlock(channels[0], channels[1], kernel_size=5, skip_connections=skip_connections)
        self.block3 = DecoderBlock(channels[1], channels[2], kernel_size=9, skip_connections=skip_connections)
        self.block4 = DecoderBlock(channels[2], channels[3], kernel_size=15, skip_connections=skip_connections)

        self.convout = DoubleConv(channels[3] * 2 ** int(skip_connections), out_channels, kernel_size=15, final_layer=True)

    def forward(self, latents, padding, encoder_embeddings=None):
        """
        Decode a batch of input latent codes.

        Parameters
        ----------
        latents : Tensor (B x C_lat x F_lat, T_lat)
          Batch of latent codes
        padding : list of tuple (int, int)
          Padding required at each stage
        encoder_embeddings : list of [Tensor (B x C x H x W)] or None (no skip connections)
          Embeddings produced by encoder at each level

        Returns
        ----------
        output : Tensor (B x 1 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Process latents and apply skip connections
        embeddings = self.block1(latents, padding[0], encoder_embeddings[0])
        embeddings = self.block2(embeddings, padding[1], encoder_embeddings[1])
        embeddings = self.block3(embeddings, padding[2], encoder_embeddings[2])
        embeddings = self.block4(embeddings, padding[3], encoder_embeddings[3])

        # Decode embeddings into spectral logits
        output = self.convout(embeddings)

        return output


class EncoderBlock(nn.Module):
    """
    Implements a dimensionality reduction followed by a double-convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        kernel_size : int
          Kernel size for convolutions
        """

        nn.Module.__init__(self)

        self.redux = nn.MaxPool2d(kernel_size=2)

        self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        """
        Feed features through the encoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H // 2 x W // 2)
          Batch of corresponding output features
        """

        # Reduce dimensionality of features along frequency/time
        y = self.redux(x)

        # Process features
        y = self.conv(y)

        return y


class DecoderBlock(nn.Module):
    """
    Implements upsampling followed by a double-convolution, with support for optional skip connections.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, skip_connections=True):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        kernel_size : int
          Kernel size for convolutions
        skip_connections : bool
          Whether to expect skip connections from encoder
        """

        nn.Module.__init__(self)

        self.upsmp = ConcatenativeUpSample2d(factor=2)

        mid_channels = in_channels

        in_channels *= 2 ** int(skip_connections)

        self.conv = DoubleConv(in_channels, out_channels, mid_channels, kernel_size=kernel_size)

    def forward(self, x, padding=(0, 0), z=None):
        """
        Feed features through the decoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features
        padding : int
          Number of features to pad after up-sampling
        z : Tensor (B x C_in x 2 * H x 2 * W)
          Batch of skip features

        Returns
        ----------
        y : Tensor (B x C_out x 2 * H x 2 * W)
          Batch of corresponding output features
        """

        # Up-sample and incorporate skip connection
        y = self.upsmp(x, padding)

        # Process features
        y = self.conv(y)

        if z is not None:
            # Concatenate the features along channels
            y = torch.cat([y, z], dim=-3)

        return y


class DoubleConv(nn.Module):
    """
    Implements a 2D convolutional block with two convolutions and batch normalization.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, final_layer=False):
        """
        Initialize the convolutional block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        mid_channels : int (Optional)
          Number of intermediary feature channels
        kernel_size : int
          Kernel size for convolutions
        final_layer : bool
          Whether to apply final batch normalization and activation function
        """

        nn.Module.__init__(self)

        if mid_channels is None:
            mid_channels = out_channels

        padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

        if not final_layer:
            self.conv_block.append(nn.BatchNorm2d(out_channels))
            self.conv_block.append(nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Feed features through the convolutional block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        y = self.conv_block(x)

        return y


class ConcatenativeUpSample2d(nn.Module):
    """
    Implements an upsampling block.
    """

    def __init__(self, factor=2):
        """
        Initialize the upsampling block.

        Parameters
        ----------
        factor : int
          Upsampling factor
        """

        nn.Module.__init__(self)

        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)

    def forward(self, x, padding=(0, 0)):
        """
        Feed features through the upsampling block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features
        padding : tuple of (int, int)
          Padding along frequency and time dimension

        Returns
        ----------
        y : Tensor (B x C_in x factor * H x factor * W)
          Batch of corresponding output features
        """

        # Upsample the features using bilinear interpolation
        y = self.upsample(x)

        # Compute the appropriate padding for the features
        pad_t = padding[0] // 2
        pad_b = padding[0] - pad_t
        pad_l = padding[1] // 2
        pad_r = padding[1] - pad_l

        # Pad the upsampled features
        y = F.pad(y, [pad_l, pad_r, pad_t, pad_b])

        return y

# TODO - working through the following


class AE_SA(SS_MPE):
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

        self.positional = SinusoidalEncodings()

        self.bottleneck = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=d_attention,
                                       nhead=n_heads,
                                       dim_feedforward=d_forward,
                                       batch_first=True),
            nn.TransformerEncoderLayer(d_model=d_attention,
                                       nhead=n_heads,
                                       dim_feedforward=d_forward,
                                       batch_first=True)
        )

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
                                    padding=7,
                                    final_layer=True)

    def forward(self, hcqt, max_seq=None):
        """
        TODO
        """

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

        # Keep track of dimensionality before the bottleneck
        dimensionality = embeddings.size()

        # Flatten time and frequency dimensions and switch with channel dimension
        embeddings = embeddings.flatten(-2).transpose(-1, -2)

        # Feed the features through the self-attention bottleneck
        embeddings = self.bottleneck(self.positional(embeddings, max_seq))

        # Restore original dimensions to the embeddings
        embeddings = embeddings.transpose(-1, -2).view(dimensionality)

        # Feed features through all upsampling blocks
        x = self.up_conv_1(self.concat_up(embeddings, x4))
        x = self.up_conv_2(self.concat_up(x, x3))
        x = self.up_conv_3(self.concat_up(x, x2))
        salience = self.up_conv_4(self.concat_up(x, x1))

        return salience


class SinusoidalEncodings(nn.Module):
    """
    Module to add fixed (sinusoidal) positional encoding to embeddings.
    """

    def __init__(self, upper_bound=10000, interleave=True):
        """
        Initialize the module.

        Parameters
        ----------
        upper_bound : int or float
          Upper boundary for geometric progression of periods (in 2π radians)
        interleave : bool
          TODO
        """

        nn.Module.__init__(self)

        self.upper_bound = upper_bound
        self.interleave = interleave

    def forward(self, features, max_seq=None):
        """
        TODO
        """

        # Determine the dimensionality of the input features
        batch_size, seq_length, d_model = features.size()

        # Determine the periods of the sinusoids
        periods = self.upper_bound ** (torch.arange(0, d_model, 2) / d_model)
        # Construct a tensor of position indices for the features
        positions = torch.arange(0, seq_length)

        if self.training and max_seq is not None:
            if max_seq > seq_length + 1:
                # Add a random offset to diversify positions
                positions += torch.randint(high=(max_seq - seq_length + 1), size=(1,))

        # Multiply every position by every period
        angles = torch.outer(positions, 1 / periods)

        # Compute the sine and cosine of the angles
        angles_sin = torch.sin(angles)
        angles_cos = torch.cos(angles)

        if self.interleave:
            # Add an extra dimension to each vector
            angles_sin = torch.unsqueeze(angles_sin, dim=-1)
            angles_cos = torch.unsqueeze(angles_cos, dim=-1)

        # Interleave the two positional encodings
        sinusoidal_encodings = torch.cat((angles_sin, angles_cos), dim=-1)

        if self.interleave:
            # Collapse the added dimension to interleave the vectors
            sinusoidal_encodings = sinusoidal_encodings.view(seq_length, d_model)

        # Add the encodings to the appropriate device
        sinusoidal_encodings = sinusoidal_encodings.to(features.device)

        # Repeat encodings for each sample in the batch and add to appropriate device
        features = features + torch.tile(sinusoidal_encodings, (batch_size, 1, 1))

        return features
