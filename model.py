# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>


import torch.nn.functional as F
import torch.nn as nn
import torch


class double_conv(nn.Module):
    """ Two convolutional layers, each followed by batch normalization and ReLU """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=(3,3), \
                padding=(1,1), convdrop=0, residual=False, alt_order=False):
        super().__init__()
        self.residual = residual
        self.out_channels = out_channels
        if not mid_channels:
            mid_channels = out_channels
        if not alt_order and convdrop==None:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif not alt_order:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=convdrop),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=convdrop)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.ELU(alpha=1.0, inplace=False),
                nn.BatchNorm2d(in_channels),
                nn.Dropout(p=convdrop),
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
                nn.ELU(alpha=1.0, inplace=False),
                nn.BatchNorm2d(mid_channels),
                nn.Dropout(p=convdrop),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
        if residual:
            self.resize = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), padding=(0,0))

    def forward(self, x):
        x_conv = self.double_conv(x)
        if self.residual:
#             x_resized = torch.cat((x,x,x,x), dim=1)[:, :self.out_channels, :, :]
            x_resized = self.resize(x)
            x_out = x_resized + x_conv
        else:
            x_out = x_conv
        return x_out


class unet_up_concat_padding(nn.Module):
    """ 2-dimensional upsampling and concatenation with fixing padding issues """

    def __init__(self, upsamp_fac=(2,2), bilinear=True):
        super().__init__()
        # Since using bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=upsamp_fac, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class transformer_enc_layer(nn.Module):
    """ Transformer encoder layer, with multi-head self-attention and fully connected network (MLP) """

    def __init__(self, embed_dim=32, num_heads=8, mlp_dim=512, p_dropout=0.2, pos_encoding=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoding=pos_encoding
        # Self-Attention mechanism
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        # max_len = 174
        max_len = 600
        if pos_encoding=='sinusoidal':
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
            pe = torch.zeros(max_len, embed_dim, requires_grad=False, device="cuda:0")
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe
            self.dropout_pe = nn.Dropout(p=p_dropout)
        elif pos_encoding=='learnable':
            position = torch.arange(max_len).unsqueeze(1)
            self.pe = nn.Parameter(torch.zeros(max_len, embed_dim, device="cuda:0"), requires_grad=True)
            nn.init.kaiming_uniform_(self.pe)
            self.dropout_pe = nn.Dropout(p=p_dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.o_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        # Fully connected network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        # Dropout and layer norm
        self.dropout1 = nn.Dropout(p=p_dropout)
        self.layernorm1 = nn.LayerNorm(normalized_shape=[embed_dim])
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[embed_dim])

    def forward(self, x):
        unflat_size = x.size()
        x = self.flatten(x).transpose(1, 2)
        if self.pos_encoding!=None:
            x = self.dropout_pe(x + self.pe[:x.shape[1], :])
        x1 = self.attn(self.q_linear(x), self.k_linear(x), self.v_linear(x))[0]
        x1_proj = self.o_linear(x1)
        x1_norm = self.layernorm1(x + self.dropout1(x1_proj))
        x2 = self.mlp(x1_norm)
        x2_norm = self.layernorm2(x1_norm + self.dropout2(x2))
        x2_norm = x2_norm.transpose(1, 2).view(torch.Size([-1, self.embed_dim, unflat_size[-2], unflat_size[-1]]))
        return x2_norm


# U-net as above, with two self-attention layers at bottom
class SAUNet(nn.Module):
    """
    SA U-Net adapted from https://github.com/christofw/multipitch_architectures
    """
    def __init__(self, n_chan_input=6, n_chan_layers=[64,30,20,10], n_bins_in=216, n_bins_out=12, \
                 a_lrelu=0.3, p_dropout=0.2, convdrop=0, residual=False, alt_order=False, scalefac=16, embed_dim=4*8, num_heads=8, mlp_dim=512, pos_encoding=None):
        super(SAUNet, self).__init__()
        n_in = n_chan_input
        n_ch = n_chan_layers
        last_kernel_size = n_bins_in//3 + 1 - n_bins_out

        # Layer normalization over frequency and channels (harmonics of HCQT)
        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])

        # Prefiltering with smaller U-Net ########################################
        # Downsampling part
        sc=scalefac
        self.inc = double_conv(in_channels=n_in, mid_channels=64//sc, out_channels=64//sc, kernel_size=(15,15), padding=(7,7), convdrop=convdrop, alt_order=alt_order)
        self.down1 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=64//sc, out_channels=128//sc, mid_channels=128//sc, kernel_size=(15,15), padding=(7,7), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=128//sc, out_channels=256//sc, mid_channels=256//sc, kernel_size=(9,9), padding=(4,4), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=256//sc, out_channels=512//sc, mid_channels=512//sc, kernel_size=(5,5), padding=(2,2), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            double_conv(in_channels=512//sc, out_channels=1024//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1), convdrop=convdrop, residual=residual, alt_order=alt_order)
        )
        # Self-Attention part (two Transformer encoder layers)
        self.attention1 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, pos_encoding=pos_encoding)
        self.attention2 = transformer_enc_layer(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)
        # Upsampling part
        self.upconcat = unet_up_concat_padding((2,2))
        self.upconv1 = double_conv(in_channels=1024//sc, out_channels=512//(sc*2), mid_channels=1024//(sc*2), kernel_size=(3,3), padding=(1,1), convdrop=convdrop, residual=residual, alt_order=alt_order)
        self.upconv2 = double_conv(in_channels=512//sc, out_channels=256//(sc*2), mid_channels=512//(sc*2), kernel_size=(5,5), padding=(2,2), convdrop=convdrop, residual=residual, alt_order=alt_order)
        self.upconv3 = double_conv(in_channels=256//sc, out_channels=128//(sc*2), mid_channels=256//(sc*2), kernel_size=(9,9), padding=(4,4), convdrop=convdrop, residual=residual, alt_order=alt_order)
        self.upconv4 = double_conv(in_channels=128//sc, out_channels=n_ch[0], mid_channels=128//(sc*2), kernel_size=(15,15), padding=(7,7), convdrop=convdrop, residual=residual, alt_order=alt_order)

        # Binning to MIDI pitches
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[0], out_channels=n_ch[1], kernel_size=(3,3), padding=(1,0), stride=(1,3)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.MaxPool2d(kernel_size=(13,1), stride=(1,1), padding=(6,0)),
            nn.Dropout(p=p_dropout)
        )
        # Time reduction
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[1], out_channels=n_ch[2], kernel_size=(75,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout)
        )
        # Chroma reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=n_ch[2], out_channels=n_ch[3], kernel_size=(1,1), padding=(0,0), stride=(1,1)),
            nn.LeakyReLU(negative_slope=a_lrelu),
            nn.Dropout(p=p_dropout),
            nn.Conv2d(in_channels=n_ch[3], out_channels=1, kernel_size=(1,last_kernel_size), padding=(0,0), stride=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.layernorm(x.transpose(1, 2)).transpose(1, 2)
        x1 = self.inc(x_norm)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.attention1(x5)
        x5 = self.attention2(x5)
        x = self.upconv1(self.upconcat(x5, x4))
        x = self.upconv2(self.upconcat(x, x3))
        x = self.upconv3(self.upconcat(x, x2))
        x = self.upconv4(self.upconcat(x, x1))
        conv2_lrelu = self.conv2(x)
        conv3_lrelu = self.conv3(conv2_lrelu)
        y_pred = self.conv4(conv3_lrelu)
        return y_pred
