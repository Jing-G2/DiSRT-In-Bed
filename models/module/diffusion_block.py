import numpy as np
import torch as th
import torch.nn as nn

from models.module.diffusion_nn import (
    linear,
    conv_nd,
    zero_module,
    normalization,
    checkpoint,
)
from models.module.diffusion_layer import (
    ResidualSequential,
    AdaLN,
    VanillaSelfAttention,
    VanillaCrossAttention,
    TimestepBlock,
    QKVAttention,
    QKVAttentionLegacy,
    Upsample,
    Downsample,
)


# ######################################
# Transformer Blocks
# ######################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        # pe = pe.unsqueeze(0)#.transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None, activation="gelu"):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        if emb is not None:
            x_norm = self.norm(x, emb)
        else:
            x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        use_self_attention,
        use_cross_attention,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.0,
        **kargs,
    ):
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention

        if use_self_attention:
            self.sa_layer = ResidualSequential(
                VanillaSelfAttention(latent_dim, num_heads, dropout)
            )
        if use_cross_attention:
            self.ca_layer = ResidualSequential(
                VanillaCrossAttention(
                    latent_dim, latent_dim, num_heads, dropout, latent_dim
                )
            )

        self.ff_layer = ResidualSequential(
            FFN(
                latent_dim,
                ff_size,
                dropout,
                latent_dim,
                activation=kargs.get("activation", "gelu"),
            )
        )

    def forward(self, x, emb, xf=None):
        if self.use_self_attention:
            x = self.sa_layer(x, emb)
        if self.use_cross_attention:
            x = self.ca_layer(x, xf, emb)
        x = self.ff_layer(x, emb)

        return x


# ######################################
# Resnet / Unet Blocks
# ######################################
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, y=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(TimestepBlock):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        # *self.norm = normalization(channels)
        self.norm = AdaLN(channels, emb_channels)  # !
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, emb, y=None):
        return checkpoint(self._forward, (x, emb), self.parameters(), True)

    def _forward(self, x, emb):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        emb = emb.unsqueeze(1)
        x_norm = self.norm(x.permute(0, 2, 1), emb).permute(0, 2, 1)
        qkv = self.qkv(x_norm)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class CrossAttentionBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        # *self.norm = nn.LayerNorm(channels)
        self.x_norm = AdaLN(channels, emb_channels)  # !
        self.y_norm = AdaLN(channels, emb_channels)
        self.q = conv_nd(1, channels, channels, 1)
        self.kv = conv_nd(1, channels, channels * 2, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, emb, y=None):
        return checkpoint(self._forward, (x, emb, y), self.parameters(), True)

    def _forward(self, x, emb, y=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        y = y.reshape(b, c, -1)
        assert x.shape == y.shape, "x: {}, y: {}".format(x.shape, y.shape)
        emb = emb.unsqueeze(1)
        x_norm = self.x_norm(x.permute(0, 2, 1), emb).permute(0, 2, 1)
        y_norm = self.y_norm(y.permute(0, 2, 1), emb).permute(0, 2, 1)
        kv = self.kv(x_norm)
        q = self.q(y_norm)
        qkv = th.cat([q, kv], dim=1)  # [B, 3C, N]
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class SaCaBlocks(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.sa = AttentionBlock(
            channels=channels,
            emb_channels=emb_channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_checkpoint=use_checkpoint,
            use_new_attention_order=use_new_attention_order,
        )

        self.ca = CrossAttentionBlock(
            channels=channels,
            emb_channels=emb_channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_checkpoint=use_checkpoint,
            use_new_attention_order=use_new_attention_order,
        )

    def forward(self, x, emb, y=None):
        x = self.sa(x, emb)
        x = self.ca(x, emb, y)
        return x
