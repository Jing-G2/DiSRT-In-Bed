import torch as th
import torch.nn as nn
from utils import logger

from models.module.diffusion_nn import (
    conv_nd,
    linear,
    zero_module,
    normalization,
)

from models.module.diffusion_layer import (
    Downsample,
    TimestepEmbedSequential,
    Upsample,
)

from models.module.diffusion_block import (
    PositionalEncoding,
    TransformerBlock,
    ResBlock,
    AttentionBlock,
    SaCaBlocks,
)


class Transformer(nn.Module):

    def __init__(
        self,
        net_type,
        time_embed_dim,
        out_channels,
        dropout=0,
        num_heads=1,
        num_transformer_blocks=2,
    ):
        super(Transformer, self).__init__()

        self.net_type = net_type  # transformer_sa | transformer_ca | transformer_saca
        self.use_self_attention = "sa" in self.net_type
        self.use_cross_attention = "ca" in self.net_type

        self.positional_encoding = PositionalEncoding(time_embed_dim, dropout)

        self.blocks = nn.ModuleList()
        for _ in range(num_transformer_blocks):
            self.blocks.append(
                TransformerBlock(
                    use_self_attention=self.use_self_attention,
                    use_cross_attention=self.use_cross_attention,
                    latent_dim=time_embed_dim,
                    num_heads=num_heads,
                    ff_size=time_embed_dim * 2,
                    dropout=dropout,
                    activation="gelu",
                )
            )

        self.out = nn.Sequential(
            nn.LayerNorm(time_embed_dim),
            zero_module(linear(time_embed_dim, out_channels)),
        )

    def forward(self, x, time_emb, cond):
        """Forward method for Transformer to get the output
        :param x: parameter embedding (same shape as cond, [B, T, D])
        :param time_emb: a 1-D batch of timesteps [B, D]
        :param cond: condtion feature extracted from the depth images [B, T, D]
        """
        assert len(x.shape) == 3, f"x.shape={x.shape} is not (B, T, D)"
        assert (
            x.shape[2] == time_emb.shape[1]
        ), f"x.shape[2]={x.shape[2]} != time_emb.shape[1]={time_emb.shape[1]}"

        emb = time_emb.unsqueeze(1) + cond
        emb = emb.type(th.float32)
        h = self.positional_encoding(x).type(th.float32)
        for block in self.blocks:
            h = block(h, emb, emb)
        h = h.type(x.dtype)
        return self.out(h).view(h.shape[0], -1)


class ResBlocks(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        time_embed_dim,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_cross_attention=False,
    ):
        super(ResBlocks, self).__init__()

        ch = input_ch = int(channel_mult[0] * model_channels)  # model_channels
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            emb_channels=time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.param_proj = nn.Sequential(
                nn.SiLU(),
                zero_module(
                    linear(
                        time_embed_dim,
                        ch * (image_size // 2 ** (len(channel_mult) - 1)) ** 2,
                    )
                ),
            )  # resblocks_ca
            self.middle_block = TimestepEmbedSequential(
                SaCaBlocks(
                    ch,
                    emb_channels=time_embed_dim,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_checkpoint=use_checkpoint,
                    use_new_attention_order=use_new_attention_order,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )  # resblocks_ca
        else:
            middle_layers = [
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ]
            if len(attention_resolutions) != 0: # use attention
                middle_layers.append(
                    AttentionBlock(
                    ch,
                    emb_channels=time_embed_dim,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_checkpoint=use_checkpoint,
                    use_new_attention_order=use_new_attention_order,
                )
                )
            else: # use ResBlock
                middle_layers.append(
                    ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                )
            middle_layers.append(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            self.middle_block = TimestepEmbedSequential(*middle_layers)

        num_out_channels = model_channels * channel_mult[-1]
        self.num_out_hw = (image_size // 2 ** (len(channel_mult) - 1)) ** 2
        
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(num_out_channels * self.num_out_hw),
            nn.SiLU(),
            zero_module(linear(num_out_channels * self.num_out_hw, out_channels)),
        )

    def forward(self, x, time_emb, cond):
        """Forward method for ResBlocks to get the output.
        :param x: parameter embedding (same shape as time_emb, [B, D])
        :param time_emb: a 1-D batch of timesteps
        :param cond: condition depth images
        """
        x = x.type(th.float32)
        h = cond.type(th.float32)
        time_emb = time_emb + x  # resblocks2

        for module in self.input_blocks:
            h = module(h, time_emb)

        if self.use_cross_attention:
            # h.view(B, C, HW) has the same shape as x
            x = self.param_proj(x.unsqueeze(1)).view(*h.shape)  # resblocks_ca
            h = self.middle_block(h, time_emb, x)
        else:
            h = self.middle_block(h, time_emb)

        h = h.type(cond.dtype)
        h = self.out(h)
        return h


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param : works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        time_embed_dim,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        ch = input_ch = int(channel_mult[0] * model_channels)  # model_channels
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            emb_channels=time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                emb_channels=time_embed_dim,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_checkpoint=use_checkpoint,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            emb_channels=time_embed_dim,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, time_emb, cond):
        """Forward method for UNetModel to get the output.
        :param x: parameter embedding (same shape as cond, [B, C, H, W])
        :param time_emb: a 1-D batch of timesteps
        :param cond: condtion feature extracted from the depth images
        """
        assert x.shape == cond.shape, f"x.shape={x.shape} != cond.shape={cond.shape}"
        hs = []

        # concatenate x and cond along the channel dimension
        h = th.cat([x, cond], dim=1).type(th.float32)

        for module in self.input_blocks:
            h = module(h, time_emb)
            hs.append(h)
        h = self.middle_block(h, time_emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)  # h=[8, 384, 32, 12] hs[-1]=[8,256,32,13]
            h = module(h, time_emb)
        h = h.type(x.dtype)
        return self.out(h).view(h.shape[0], -1)
