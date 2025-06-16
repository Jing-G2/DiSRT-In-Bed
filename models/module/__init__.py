import math
import torch
import torch.nn as nn
import timm
import torchvision.models as models

from models.module.diffusion_model import Transformer, ResBlocks, UNetModel
from models.module.encoder_decoder import ResnetEncoder, ResnetDecoder, Resnet18Decoder

from models.module.regressor import MLPRegressor
from models.module.MeshEstimator import MeshEstimator


def get_diffusion_latent_model(
    diffusion_net,
    image_size,
    in_channels,
    time_embed_dim,
    model_channels,
    out_channels,
    num_transformer_blocks,
    num_res_blocks,
    attention_resolutions,
    dropout,
    channel_mult,
    use_checkpoint,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    num_out_hw=-1,
):
    """Get the diffusion latent model based on the network type
    Args:
        diffusion_net (str): network type (resblocks | resblocks_ca | unet | transformer_sa | transformer_saca)
        image_size (int): image size
        in_channels (int): number of input channels of the diffusion model, it is for diffusioned parameters or images in resblocks and unet
        time_embed_dim (int): time embedding dimension
        model_channels (int): number of channels in the model
        out_channels (int): number of output channels of the diffusion model
        num_transformer_blocks (int): number of transformer blocks
        num_res_blocks (int): number of residual blocks
        attention_resolutions (str): attention resolutions
        dropout (float): dropout rate
        channel_mult (str): channel multiplier
        use_checkpoint (bool): use checkpoint or not
        num_heads (int): number of heads
        num_head_channels (int): number of head channels
        num_heads_upsample (int): number of heads for upsample
        use_scale_shift_norm (bool): use scale shift norm or not
        resblock_updown (bool): use resblock updown or not
        use_new_attention_order (bool): use new attention order or not
        num_out_hw (int): output height and width of the encoder, if diffusion_net is resblocks_ca, this is the size after the downsample blocks
    Returns:
        model: diffusion model, output 1D tensor [B, num_out_dim]
        num_out_dim: number of output dimensions
    """
    if diffusion_net.startswith("transformer"):
        # num_tokens = num_out_hw
        model = Transformer(
            net_type=diffusion_net,
            time_embed_dim=time_embed_dim,
            out_channels=out_channels,
            dropout=dropout,
            num_heads=num_heads,
            num_transformer_blocks=num_transformer_blocks,
        )
        num_out_dim = num_out_hw * out_channels

    elif diffusion_net.startswith("resblocks"):
        num_out_channels = model_channels * channel_mult[-1] * 2
        model = ResBlocks(
            image_size=image_size,
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            model_channels=model_channels,
            out_channels=num_out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_cross_attention="ca" in diffusion_net,
        )
        num_out_dim = num_out_channels

    elif diffusion_net == "unet":
        h = int(math.sqrt(num_out_hw))
        if in_channels == 1:
            out_channels = in_channels
        model = UNetModel(
            image_size=image_size,
            in_channels=in_channels * 2,  # param + feature
            time_embed_dim=time_embed_dim,
            model_channels=model_channels,
            out_channels=out_channels,
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
        )
        num_out_dim = out_channels * h * h
    else:
        raise ValueError(f"unsupported diffusion_net: {diffusion_net}")

    return model, num_out_dim


def get_encoder(
    image_size,
    channel,
    feature_channels,
    param_size,
    recon_channel_mult="1,2,4,8",
    use_dropout=False,
    norm_layer="batch",
    padding_type="reflect",
    num_res_blocks=2,
    encoder_net="resnet18",
):
    """get the encoder model based on the network type
    Args:
        image_size (int): image size
        channel (int): number of modality channels
        feature_channels (int): number of feature channels
        param_size (int): number of parameters, (output size of the encoder)
        recon_channel_mult (str): channel multiplier
        use_dropout (bool): use dropout or not
        norm_layer (str): normalization layer type
        padding_type (str): padding type
        num_res_blocks (int): number of residual blocks
        encoder_net (str): network type
    Returns:
        encoder: encoder model
        num_out_channels: number of output channels
    """
    if norm_layer == "batch":
        norm_layer = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm_layer = nn.InstanceNorm2d

    recon_channel_mult = list(map(int, recon_channel_mult.split(",")))

    if encoder_net == "resnet18":
        encoder = timm.create_model("resnet18", pretrained=False, num_classes=0)
        encoder.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        num_out_channels = 512
        num_out_hw = 8 * 8
    elif encoder_net.startswith("vit"):
        encoder = timm.create_model(
            "vit_little_patch16_reg1_gap_256.sbb_in12k_ft_in1k",
            # "vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k",
            pretrained=False,
        )
        patch_size = 16
        out_channels = encoder.patch_embed.proj.out_channels
        encoder.patch_embed.proj = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
        )
        encoder = nn.Sequential(*list(encoder.children())[:-4])
        num_out_channels = out_channels
        num_out_hw = (256 // patch_size) ** 2
    else:
        encoder = ResnetEncoder(
            in_channels=channel,
            out_channels_conv1=feature_channels,
            channel_mult=recon_channel_mult,
            n_blocks=num_res_blocks,
            use_dropout=use_dropout,
            norm_layer=norm_layer,
            padding_type=padding_type,
        )
        num_out_channels = feature_channels * recon_channel_mult[-1]
        num_out_hw = (image_size // 2 ** (len(recon_channel_mult))) ** 2
    return encoder, num_out_channels, num_out_hw


def get_decoder(
    channel,
    feature_channels,
    recon_channel_mult="1,2,4,8",
    norm_layer="batch",
    encoder_net="resnet18",
):
    """get the decoder model based on the network type
    Args:
        channel (int): number of modality channels
        feature_channels (int): number of feature channels
        recon_channel_mult (str): channel multiplier
        norm_layer (str): normalization layer type
        encoder_net (str): network type
    Returns:
        decoder: decoder model
    """
    if norm_layer == "batch":
        norm_layer = nn.BatchNorm2d
    elif norm_layer == "instance":
        norm_layer = nn.InstanceNorm2d

    recon_channel_mult = list(map(int, recon_channel_mult.split(",")))

    if encoder_net == "resnet18":
        decoder = Resnet18Decoder(in_channels=512, out_channels=1)
    else:
        decoder = ResnetDecoder(
            feature_channels=feature_channels,
            out_channels=channel,
            channel_mult=recon_channel_mult,
            norm_layer=norm_layer,
        )

    return decoder
