import models.diffusion.gaussian_diffusion as gd
from models.diffusion.respace import SpacedDiffusion, space_timesteps
from models.diffusion.model import ParameterModel

diffusion_keys = [
    "diffusion_steps",
    "learn_sigma",
    "sigma_small",
    "noise_schedule",
    "predict_xstart",
    "rescale_timesteps",
    "rescale_learned_sigmas",
    "timestep_respacing",
]


def get_gaussian_diffusion(
    *,
    diffusion_steps=100,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(
        noise_schedule, diffusion_steps
    )  # linear or cosine

    if rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE  # default
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


model_keys = [
    "diffusion_model_channels",
    "diffuion_model_res_blocks",
    "channel_mult",
    "use_checkpoint",
    "attention_resolutions",
    "num_heads",
    "num_head_channels",
    "num_heads_upsample",
    "use_scale_shift_norm",
    "dropout",
    "num_transformer_blocks",
    "encoder_net",
    "diffusion_net",
]


def get_model(
    image_size,
    in_channels,
    num_param,
    diffusion_model_channels=128,
    diffuion_model_res_blocks=2,
    channel_mult="",
    use_checkpoint=False,
    attention_resolutions="16,8",
    num_heads=4,
    num_head_channels=32,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    num_transformer_blocks=2,
    encoder_net="",  # resnet18 | res_encoder | vit
    diffusion_net="",  # resblocks | resblocks_ca | unet | transformer_sa | transformer_saca
):
    """Get the Unet model for diffusion model."""
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (1, 2, 2, 4, 4, 8)
        elif image_size == 256:
            channel_mult = (1, 2, 4, 4, 8)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 4)
        elif image_size == 64:
            channel_mult = (1, 1, 2, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    elif isinstance(channel_mult, str):
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    else:
        channel_mult = tuple(channel_mult)

    attention_ds = []
    if attention_resolutions == ",":
        attention_ds = []
    else:
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

    return ParameterModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=diffusion_model_channels,
        num_param=num_param,
        num_res_blocks=diffuion_model_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        num_transformer_blocks=num_transformer_blocks,
        encoder_net=encoder_net,
        diffusion_net=diffusion_net,
    )
