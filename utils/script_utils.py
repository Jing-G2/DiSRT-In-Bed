import os
import json
import argparse


from utils.constants import *


def parse_args_dict(args):
    # exp setup
    args["data_path"] = os.path.join(BASE_PATH, "BodyPressure/data_BP")
    args["ds_fd"] = os.path.join(
        BASE_PATH, "data"
    )  # data folder which contains the SLP directory
    args["save_path"] = args.get("save_path", os.path.join(BASE_PATH, "Pose_exps"))
    args["viz_type"] = args.get(
        "viz_type", None
    )  # visualize the inference results, [None, '3D', 'image', 'video']

    args["name"] = args.get("name", "name not set")

    # ------------------------------------------------
    # data loading setup
    args["syn_ratio"] = args.get("syn_ratio", 1.0)
    args["syn_train_file"] = args.get("syn_train_file", "synth_all.txt")
    args["syn_val_file"] = args.get("syn_val_file", None)
    args["real_train_file"] = args.get("real_train_file", "real_1_80.txt")
    args["real_val_file"] = args.get("real_val_file", "real_81_102.txt")
    args["use_real"] = args.get("use_real", True)
    args["use_synth"] = args.get("use_synth", True)
    args["uncover_only"] = args.get("uncover_only", False)
    args["image_size"] = args.get("image_size", 256)

    # data scaling setup
    args["depth_scaling"] = args.get(
        "depth_scaling", "minmax"
    )  # minmax | sigmoid | none | bodymap
    args["pressure_scaling"] = args.get(
        "pressure_scaling", "minmax"
    )  # minmax | sigmoid | none | bodymap
    args["ir_scaling"] = args.get(
        "ir_scaling", "minmax"
    )  # minmax | sigmoid | none | bodymap

    # data augmentation setup
    args["is_affine"] = args.get("is_affine", True)
    args["is_erase"] = args.get("is_erase", True)

    # ------------------------------------------------
    # opt setup
    args["lr"] = args.get("lr", 1e-4)
    args["lr_policy"] = args.get(
        "lr_policy", "none"
    )  # linear | step | plateau | cosine | none
    args["weight_decay"] = args.get("weight_decay", 5e-4)
    args["batch_size"] = args.get("batch_size", 32)

    # ------------------------------------------------
    # loss setup
    args["use_smpl_loss"] = args.get("use_smpl_loss", True)
    args["use_v2v_loss"] = args.get("use_v2v_loss", True)
    args["lambda_smpl_loss"] = args.get("lambda_smpl_loss", 1.0)
    args["lambda_v2v_loss"] = args.get("lambda_v2v_loss", 0.0)
    args["lambda_recon_loss"] = args.get("lambda_recon_loss", 0.0)
    args["lambda_diffusion_loss"] = args.get("lambda_diffusion_loss", 0.0)

    # ------------------------------------------------
    # diffusion trainer setup
    args["lr_anneal_steps"] = args.get("lr_anneal_steps", 0)
    args["lr_policy_diffusion"] = args.get(
        "lr_policy_diffusion", "none"
    )  # linear | step | plateau | cosine | none
    args["weight_decay_diffusion"] = args.get("weight_decay_diffusion", 5e-4)
    args["microbatch"] = args.get("microbatch", -1)  # -1 disables microbatches
    args["ema_rate"] = args.get("ema_rate", "0.9999")
    args["schedule_sampler"] = args.get("schedule_sampler", "uniform")
    args["trans_loss_type"] = args.get("trans_loss_type", "l1")  # l1 | l2
    # loss weights setup
    args["lambda_trans_loss"] = args.get("lambda_trans_loss", None)
    args["lambda_cyc_loss"] = args.get("lambda_cyc_loss", None)

    # ------------------------------------------------
    # epochs settings setup
    args["epochs"] = args.get("epochs", 50)
    args["epochs_metric"] = 50
    args["epochs_save"] = 5

    # ------------------------------------------------
    # model loading setup
    args["encoder_epoch"] = args.get("encoder_epoch", -1)
    args["decoder_epoch"] = args.get("decoder_epoch", -1)
    args["regressor_epoch"] = args.get("regressor_epoch", -1)
    args["diffusion_epoch"] = args.get("diffusion_epoch", -1)

    # ------------------------------------------------
    # model setup
    args["pretrained_model_path"] = args.get("pretrained_model_path", None)
    args["main_model_fn"] = args.get("main_model_fn", "PMM")
    args["modality"] = args.get("modality", ["depth"])
    args["feature_channels"] = args.get("feature_channels", 128)
    args["vertex_size"] = 6890
    args["param_size"] = 88

    # ------------------------------------------------
    # reconstrcution setup (encoder, decoder)
    args["freeze_encoder"] = args.get("freeze_encoder", False)
    args["freeze_decoder"] = args.get("freeze_decoder", False)
    args["freeze_regressor"] = args.get("freeze_regressor", False)
    args["use_recon"] = args.get("use_recon", False)
    args["use_dropout"] = args.get("use_dropout", False)
    args["recon_channel_mult"] = args.get("recon_channel_mult", "")
    args["norm_layer"] = args.get("norm_layer", "batch")
    args["padding_type"] = args.get("padding_type", "reflect")
    args["recon_num_res_blocks"] = args.get("recon_num_res_blocks", 2)

    # ------------------------------------------------
    # diffusion setup
    args["use_diffusion"] = args.get("use_diffusion", False)
    args["diffusion_steps"] = args.get("diffusion_steps", 1000)
    args["learn_sigma"] = args.get("learn_sigma", False)
    args["sigma_small"] = args.get("sigma_small", False)
    args["noise_schedule"] = args.get("noise_schedule", "linear")
    args["use_kl"] = args.get("use_kl", False)
    args["predict_xstart"] = args.get("predict_xstart", False)
    args["rescale_timesteps"] = args.get("rescale_timesteps", False)
    args["rescale_learned_sigmas"] = args.get("rescale_learned_sigmas", False)
    args["timestep_respacing"] = args.get("timestep_respacing", "")

    # ------------------------------------------------
    # diffusion model setup
    args["encoder_net"] = args.get(
        "encoder_net", "resnet18"
    )  # resnet18 | res_encoder | vit
    args["diffusion_net"] = args.get(
        "diffusion_net", "unet"
    )  # resblocks | resblocks_ca | unet | transformer_sa | transformer_saca

    args["diffuion_unet_image_size"] = args.get(
        "diffuion_unet_image_size", 32
    )  # unused
    args["diffuion_model_res_blocks"] = args.get("diffuion_model_res_blocks", 2)
    args["channel_mult"] = args.get("channel_mult", "")
    args["diffusion_cond"] = args.get("diffusion_cond", True)
    args["class_cond"] = args.get("class_cond", False)
    args["use_checkpoint"] = args.get("use_checkpoint", False)
    args["num_classes"] = args.get("num_classes", 2)
    args["num_heads"] = args.get("num_heads", 4)
    args["attention_resolutions"] = args.get("attention_resolutions", "16,8")
    args["num_head_channels"] = args.get("num_head_channels", 32)
    args["num_heads_upsample"] = args.get("num_heads_upsample", -1)
    args["dropout"] = args.get("dropout", 0.0)
    args["use_scale_shift_norm"] = args.get("use_scale_shift_norm", True)
    args["resblock_updown"] = args.get("resblock_updown", True)
    args["use_fp16"] = args.get("use_fp16", False)
    args["fp16_scale_growth"] = args.get("fp16_scale_growth", 1e-3)
    args["use_new_attention_order"] = args.get("use_new_attention_order", False)

    # transformer
    args["num_transformer_blocks"] = args.get("num_transformer_blocks", 2)

    return args


def args_dict_get_train_val_files(args):
    # data file setup
    if args["exp_type"] == "overfit":
        # this setting is to verify if the model is learning anything so only train and test on the same data
        if args["exp_run"] == "sim_only":
            args["syn_train_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/synth/synth_overfit_train.txt"
            )
            args["syn_val_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/synth/synth_overfit_val.txt"
            )
            args["real_train_file"] = None
            args["real_val_file"] = None
            args["real_val_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/real/real_overfit.txt"
            )
        elif args["exp_run"] == "real_only":
            args["syn_train_file"] = None
            args["syn_val_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/synth/synth_overfit_val.txt"
            )
            args["real_train_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/real/real_train.txt"
            )
            args["real_val_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/real/real_overfit.txt"
            )
        else:
            args["syn_train_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/synth/synth_overfit_train.txt"
            )
            args["syn_val_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/synth/synth_overfit_val.txt"
            )
            args["real_train_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/real/real_train.txt"
            )
            args["real_val_file"] = os.path.join(
                BASE_PATH, "0-Pose/data_files/real/real_overfit.txt"
            )

    elif args["exp_type"] == "normal":
        if args["exp_run"] == "sim_only":
            # train on the synthetic training data and validate on the real validation data
            args["syn_train_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/synth/{args['syn_train_file']}"
            )
            args["syn_val_file"] = None
            args["real_train_file"] = None
            args["real_val_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/real/{args['real_val_file']}"
            )
        elif args["exp_run"] == "real_only":
            # train on the real training data and validate on the real validation data
            args["syn_train_file"] = None
            args["syn_val_file"] = None
            args["real_train_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/real/{args['real_train_file']}"
            )
            args["real_val_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/real/{args['real_val_file']}"
            )
            # args["epochs"] = 100 # ~50k steps for 1-80 (default)
        elif args["exp_run"] == "full-train-test":
            # train on the synthetic training data and part of the real training data and validate on the real validation data(same in the every setting)
            args["syn_train_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/synth/{args['syn_train_file']}"
            )
            args["syn_val_file"] = None
            args["real_train_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/real/{args['real_train_file']}"
            )
            args["real_val_file"] = os.path.join(
                BASE_PATH, f"0-Pose/data_files/real/{args['real_val_file']}"
            )
            # args["epochs"] = 30 # ~200k steps (default)
    elif args["exp_type"] == "hospital":
        # test on slp hospital dataset
        args["syn_train_file"] = None
        args["syn_val_file"] = None
        args["real_train_file"] = None
        args["real_val_file"] = os.path.join(
            BASE_PATH, "0-Pose/data_files/hospital/hospital_all.txt"
        )

    else:
        print("ERROR: invalid setting")

    return args


def load_args_dict_from_json(json_file):
    try:
        with open(json_file, "r") as file:
            arguments = json.load(file)
            return arguments
    except FileNotFoundError:
        print(f"File '{json_file}' not found.")
        return None
    except json.JSONDecodeError as exc:
        print(f"Error decoding JSON: {exc}")
        return None


def create_argparser(args_file):
    if os.path.exists(args_file):
        print(f"Model Setting Path: {args_file}")
        args_dict = load_args_dict_from_json(args_file)
    else:
        print(f"Model Setting Path: {args_file} not found. Using default settings.")
        args_dict = {}
    args_dict = parse_args_dict(args_dict)
    parser = argparse.ArgumentParser(description="In-bed Pose Estimation")
    add_dict_to_argparser(parser, args_dict)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "train", "finetune", "test"],
        required=True,
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Directory root path(contains checkpoints/) to load the model from, only used in finetune mode",
    )

    return parser


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    if isinstance(args, dict):
        return {k: args[k] for k in keys}

    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
