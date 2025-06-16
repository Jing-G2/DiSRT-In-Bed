import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module import get_diffusion_latent_model, get_encoder
from models.module.diffusion_nn import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from models.module.diffusion_block import FFN

from utils import logger


class TimestepEmbedding(nn.Module):

    def __init__(self, model_channels, time_embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x):
        return self.layers(x)


class ParameterEncoder(nn.Module):

    def __init__(self, num_param, encoder_net, embed_dim, num_out_hw):
        """
        Convert the parameters to specific shape of embedding

        :param num_param: the number of parameters of the output. 88
        :param encoder_net: the encoder network type.
        :param embed_dim: the dimension of the embedding.
        :param num_out_hw: the number of output height \times width from the encoder.
        """

        super().__init__()

        if encoder_net == "none" and embed_dim != 1:
            self.embed_shape = (
                (embed_dim,) if num_out_hw == 1 else (embed_dim, num_out_hw)
            )
        elif encoder_net == "vit":
            self.embed_shape = (num_out_hw, embed_dim)
        else:
            h = int(math.sqrt(num_out_hw))
            self.embed_shape = (embed_dim, h, h)
        self.layers = nn.Sequential(
            linear(num_param, embed_dim),
            nn.SiLU(),
            linear(embed_dim, embed_dim * num_out_hw),
        )

    def forward(self, x):
        out = self.layers(x)
        return out.view(x.shape[0], *self.embed_shape)


class ParameterModel(nn.Module):
    """
    The ResNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param num_param: the number of parameters of the output. 88
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
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param num_transformer_blocks: number of transformer blocks in the regressor part.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_param,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 4),
        conv_resample=True,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        num_transformer_blocks=3,
        encoder_net="",
        diffusion_net="",
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_param = num_param
        self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order

        self.num_transformer_blocks = num_transformer_blocks
        self.encoder_net = encoder_net  # resnet18 | res_encoder | vit | none
        self.diffusion_net = diffusion_net  # resblocks | resblocks_ca | unet | transformer_sa | transformer_saca

        if diffusion_net == "unet":
            assert encoder_net == "res_encoder" or encoder_net == "none"
        logger.log("\nBuilding diffusion model...")

        # timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)

        # condition encoder for extract feature from images
        if encoder_net == "none":
            diffusion_in_channels = 1
            if diffusion_net == "resblocks":
                num_out_channels = 1
                num_out_hw = 1
                param_embed_dim = time_embed_dim
            elif diffusion_net == "resblocks_ca":
                num_out_channels = model_channels * channel_mult[-1]
                num_out_hw = (image_size // 2 ** (len(channel_mult) - 1)) ** 2
                param_embed_dim = num_out_channels
            elif diffusion_net.startswith("resblocks_ca"):
                num_out_channels = 1
                num_out_hw = 1
                param_embed_dim = time_embed_dim
            elif diffusion_net == "unet":
                num_out_channels = 1
                num_out_hw = image_size**2
                param_embed_dim = 1
            else:
                raise ValueError(
                    f"unsupported achitecture for none encoder with diffusion_net: {diffusion_net}"
                )
        else:
            ch = int(channel_mult[0] * model_channels)  # model_channels
            self.encoder_blocks, num_out_channels, num_out_hw = get_encoder(
                image_size=image_size,
                channel=in_channels,
                feature_channels=ch,
                param_size=num_param,
                recon_channel_mult="1,2,2",  # TODO: control the input by passing the parameter
                num_res_blocks=num_res_blocks,
                encoder_net=encoder_net,  # resnet18 | res_encoder | vit
            )
            self.feature_norm = nn.LayerNorm(num_out_channels)
            self.feature_emb = conv_nd(
                dims=1,
                in_channels=num_out_channels,
                out_channels=time_embed_dim,
                kernel_size=1,
            )  # Conv1d
            # linear(num_out_channels, time_embed_dim) # Linear

            diffusion_in_channels = (
                num_out_channels if diffusion_net == "unet" else time_embed_dim
            )
            param_embed_dim = diffusion_in_channels

        logger.log(
            f"encoder network: {encoder_net}, with num_out_channels: {num_out_channels} and num_out_hw: {num_out_hw}"
        )

        # diffuion model
        logger.log(
            f"diffusion network: {diffusion_net} with input num channels: {diffusion_in_channels}, num_hw: {num_out_hw}"
        )
        self.diffusion_model, self.num_out_dim = get_diffusion_latent_model(
            diffusion_net=diffusion_net,
            image_size=image_size,
            in_channels=diffusion_in_channels,
            time_embed_dim=time_embed_dim,
            model_channels=model_channels,
            out_channels=time_embed_dim,
            num_transformer_blocks=num_transformer_blocks,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            num_out_hw=num_out_hw,
        )

        # param embedding
        self.parm_embed = ParameterEncoder(
            num_param=num_param,
            encoder_net=encoder_net,
            embed_dim=param_embed_dim,
            num_out_hw=num_out_hw,
        )

        # regressor
        self.fc = nn.Sequential(
            nn.LayerNorm(self.num_out_dim),
            linear(self.num_out_dim, time_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            linear(time_embed_dim * 2, num_param * 2),
        )

    def forward(self, x, timesteps, cond, is_train=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs. (smpl parameters)
        :param timesteps: a 1-D batch of timesteps.
        :param cond: conditioning images.
        :return: the model output.
        """
        # 1. timestep embedding
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 2. param embedding
        x_emb = self.parm_embed(x)

        # 3.pass condition through the encoder if needed
        cond = cond.type(torch.float32)
        if self.encoder_net != "none":
            cond = self.encoder_blocks(cond)
            if self.diffusion_net != "unet":
                if len(cond.shape) == 4:  # from resnet
                    B, C, H, W = cond.shape
                    cond = cond.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
                    cond = self.feature_norm(cond)  # [B, HW, C]
                else:  # from vit
                    B, HW, C = cond.shape
                    cond = self.feature_norm(cond)  # [B, HW, C]
                cond = self.feature_emb(cond.permute(0, 2, 1)).permute(
                    0, 2, 1
                )  # [B, HW, time_embed_dim] Conv1d
                # cond = self.feature_emb(cond)  # [B, HW, time_embed_dim] linear

                x_emb = x_emb.view(B, time_emb.shape[1], -1).permute(
                    0, 2, 1
                )  # [B, HW, time_embed_dim]

        # 4. pass through the diffusion model
        x_emb = self.diffusion_model(x_emb, time_emb, cond)

        # 5. regressor
        param_out = self.fc(x_emb).view(-1, 2, self.num_param)

        return param_out
