""" 
The code is adapted from official implementation of CycleGAN:
https://github.com/AliaksandrSiarohin/cycle-gan/blob/master/models/networks.py#L319
"""

import torch
import torch.nn as nn
import functools


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


# ######################################
# Encoder and Decoder Modules
# ######################################
class ResnetEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels_conv1,
        channel_mult=[1, 2, 4, 8],
        n_blocks=6,
        use_dropout=False,
        norm_layer=nn.BatchNorm2d,
        padding_type="replicate",
    ):
        super(ResnetEncoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # use_bias = False
        # norm_layer = nn.AdaptiveAvgPool2d

        model = []

        model += [
            nn.ReplicationPad2d(3),
            nn.Conv2d(
                in_channels, out_channels_conv1, kernel_size=7, padding=0, bias=use_bias
            ),
            norm_layer(out_channels_conv1),
            nn.ReLU(True),
        ]

        channel_mult = [1] + channel_mult
        block_ch_out = [out_channels_conv1 * m for m in channel_mult]
        for i in range(len(channel_mult) - 1):
            model += [
                nn.Conv2d(
                    in_channels=block_ch_out[i],
                    out_channels=block_ch_out[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(block_ch_out[i + 1]),
                nn.ReLU(True),
            ]

        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    block_ch_out[-1],
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        self.model = nn.Sequential(*model)

    def forward(self, in_feature):
        """Standard forward"""
        return self.model(in_feature)


class ResnetDecoder(nn.Module):

    def __init__(
        self,
        feature_channels,
        out_channels,
        channel_mult=[1, 2, 4, 8],
        norm_layer=nn.BatchNorm2d,
        padding_type="replicate",
    ):
        super(ResnetDecoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = []

        channel_mult = [1] + channel_mult
        block_ch_out = list(reversed([feature_channels * m for m in channel_mult]))
        for i in range(len(channel_mult) - 1):
            model += [
                nn.ConvTranspose2d(
                    in_channels=block_ch_out[i],
                    out_channels=block_ch_out[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(block_ch_out[i + 1]),
                nn.ReLU(True),
            ]

        model += [
            nn.ReplicationPad2d(3),  # padding_type: replicate
            nn.Conv2d(feature_channels, out_channels, kernel_size=7, padding=0),
            # nn.Tanh(),
            nn.Sigmoid(),  # depth reconstruction scale [0, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, in_feature):
        """Standard forward"""
        return self.model(in_feature)


class Resnet18Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resnet18Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),  # depth reconstruction scale [0, 1]
        )

    def forward(self, in_feature):
        return self.model(in_feature)
