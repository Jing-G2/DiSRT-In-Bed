import os
import json
import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils.constants import DEVICE, MISS_MODALITY_FLAG
from utils.data_utils.trans_utils import get_descaling_func
from utils.metric_utils import (
    create_metric_dict,
    get_anatomy_metrics,
    get_gt_data,
    get_errors,
    save_pd_as_npy,
    save_mesh,
    PI,
)
from utils.viz_utils.plot import plot

from utils import logger
import wandb


# ##################################################################
# calculate metrics
# ##################################################################
def Metric(
    diffusion,
    model,
    mesh_estimator,
    test_loader,
    writer=None,
    epoch=-1,
    save_gt=False,
    viz_type="image",
    infer_save_path=None,
    args=None,
):
    """Compute metrics for the model on the test_loader dataset
    Args:
        diffusion: diffusion model
        model: model to predict eps during the diffusion process
        mesh_estimator: mesh estimator
        test_loader: dataloader for the test dataset
        writer: tensorboard or wandb writer
        epoch: current epoch
        save_gt: save ground truth
        viz_type: type of visualization "3D, "image" or "video"
        infer_save_path: path to save the inference
        args: arguments
    """
    if test_loader is None:
        return None

    dict_map = {
        "uncover": create_metric_dict(epoch, PI),
        "cover1": create_metric_dict(epoch, PI),
        "cover2": create_metric_dict(epoch, PI),
        "real": create_metric_dict(epoch, PI),
        "synth": create_metric_dict(epoch, PI),
        "f": create_metric_dict(epoch, PI),
        "m": create_metric_dict(epoch, PI),
        "overall": create_metric_dict(epoch, PI),
    }

    model.eval()
    model.to(DEVICE)

    viz_count = 0  # count the number of visualizations

    with torch.no_grad():
        for (
            batch_original_pressure,  # [B, 2, 1, H, W]
            batch_pressure_images,  # [B, 2, 1, H, W]
            batch_original_ir,  # [B, 2, 1, H, W]
            batch_ir_images,  # [B, 2, 1, H, W]
            batch_original_depth,  # [B, 2, 1, H, W]
            batch_depth_images,  # [B, 2, 1, H, W]
            batch_labels,  # [B, 162]
            batch_gt_pmap,  # [B, 6890]
            batch_gt_verts,  # [B, 6890, 3]
            batch_names,  # [B,]
        ) in tqdm(
            iter(test_loader),
            desc="metric",
            bar_format="{l_bar}{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            batch_depth_images = (
                batch_depth_images.to(DEVICE)
                if batch_depth_images is not None
                else None
            )
            batch_pressure_images = (
                batch_pressure_images.to(DEVICE)
                if batch_pressure_images is not None
                else None
            )
            batch_ir_images = (
                batch_ir_images.to(DEVICE) if batch_ir_images is not None else None
            )
            batch_labels_copy = batch_labels.clone().to(DEVICE)
            smpl_param_gt = torch.cat(
                (
                    batch_labels[:, 72:82],  # 10
                    batch_labels[:, 154:157],  # 3
                    torch.cos(batch_labels[:, 82:85]),  # 3
                    torch.sin(batch_labels[:, 82:85]),  # 3
                    batch_labels[:, 85:154],  # 69
                ),  # [B, 88]
                axis=1,
            ).detach()
            # -------------------------------------
            # run diffusion model to get predictions
            batch_samples = None
            for i in range(0, batch_depth_images.shape[0], args["microbatch"]):
                model_kwargs = {
                    "cond": batch_depth_images[i : i + args["microbatch"], 1].to(
                        DEVICE
                    ),  # [b, C, H, W]  # condition on images
                    "is_train": False,
                }
                batch_micro = torch.randn(
                    model_kwargs["cond"].shape[0], args["param_size"]
                ).to(DEVICE)

                sample = diffusion.ddim_sample_loop(
                    model=model,
                    shape=batch_micro.shape,
                    model_kwargs=model_kwargs,
                )

                batch_samples = (
                    sample
                    if batch_samples is None
                    else torch.cat((batch_samples, sample), 0)
                )

            batch_mesh_pred = mesh_estimator(
                batch_gender=batch_labels[:, 157:159],
                x=batch_samples,
            )
            batch_mesh_pred["out_joint_pos"] = batch_mesh_pred["out_joint_pos"].reshape(
                -1, 24, 3
            )

            # -------------------------------------
            """ permute the pressure and depth images to BSHWC"""
            if (batch_original_pressure != MISS_MODALITY_FLAG).any():
                batch_original_pressure = batch_original_pressure.permute(0, 1, 3, 4, 2)
            if (batch_original_depth != MISS_MODALITY_FLAG).any():
                batch_original_depth = batch_original_depth.permute(0, 1, 3, 4, 2)
            if (batch_original_ir != MISS_MODALITY_FLAG).any():
                batch_original_ir = batch_original_ir.permute(0, 1, 3, 4, 2)

            # -------------------------------------
            # iterate over batch to calculate metrics
            for i, name in enumerate(batch_names):
                # get ground truth data
                label = batch_labels_copy[i]
                # assert len(batch_gt_verts[0]) != 0, "gt verts is empty"
                gt = get_gt_data(
                    label=label,
                    verts=batch_gt_verts[i].to("cpu"),
                )

                # prepare data for metric calculation
                pd_betas = batch_mesh_pred["out_betas"][i]
                pd_jtr = batch_mesh_pred["out_joint_pos"][i]
                pred_out_verts = batch_mesh_pred["out_verts"][i]

                height, chest, waist, hips = get_anatomy_metrics(pd_betas, label[157])

                pred = {
                    "jtr": pd_jtr,
                    "height": height,
                    "chest": chest,
                    "waist": waist,
                    "hips": hips,
                    "verts": pred_out_verts.to("cpu"),
                    "betas": pd_betas,
                }

                # -------------------------------------
                # get the inference path
                name_split = name.split("_")
                cover_type = name_split[-1]

                # set the save path and save the inference
                if infer_save_path is not None:  # and viz_count < 45 * 3:
                    viz_count += 1  # for faster testing without visualizing all

                    if "syn" in name:
                        # synthetic dataset, name: e.g. synth_1to40_lay_f_5_cover1
                        p_idx = int(name_split[-2])
                        pose_idx = -1
                        cur_save_path = os.path.join(
                            infer_save_path, name.replace("_", "/")
                        )  # e.g. {infer_path}/synth/1to40/lay/f/5/cover1
                    elif "slp" in name:
                        # SLP dataset, name: e.g. slp_071_m_1_cover1
                        dataset = name_split[0]
                        p_idx = int(name_split[1])
                        pose_idx = int(name_split[-2])
                        cur_save_path = os.path.join(
                            infer_save_path,
                            f"{dataset}",
                            f"{p_idx:05d}",
                            f"{cover_type}",
                            f"{pose_idx:06d}",
                        )  # e.g. {infer_path}/slp/00071/cover1/000001

                    if epoch != -1:
                        cur_save_path = os.path.join(cur_save_path, f"epoch_{epoch}")

                    if not os.path.exists(cur_save_path):
                        os.makedirs(cur_save_path)

                    # save predicted mesh parameters
                    save_pd_as_npy(
                        cur_save_path=cur_save_path,
                        pd_betas=pd_betas,
                        pd_jtr=pd_jtr,
                        pd_verts=pred_out_verts,
                    )

                    # save mesh to the directory of the first cover state
                    try:
                        if save_gt:
                            save_mesh(
                                cur_save_path=cur_save_path,
                                verts=batch_gt_verts[i],
                                is_gt=True,
                                writer=writer,
                            )

                        save_mesh(
                            cur_save_path=cur_save_path,
                            verts=pred_out_verts,
                            is_gt=False,
                            writer=writer,
                        )
                    except:
                        pass

                    # -------------------------------------
                    # render the mesh and save the image
                    plot(
                        save_path=cur_save_path,
                        pred=pred,
                        depth_pair=batch_original_depth[i].to("cpu").numpy(),
                        gender=label[157],
                        recon_depth=None,
                        pressure=batch_original_pressure[i, 1].to("cpu").numpy(),
                        ir=batch_original_ir[i, 1].to("cpu").numpy(),
                        gt=gt if save_gt else None,
                        p_idx=p_idx,
                        pose_idx=pose_idx,
                        cover_type=cover_type,
                        viz_type=viz_type,
                    )

                # -------------------------------------
                # calculate metrics
                error = get_errors(pred=pred, gt=gt)

                try:
                    metric_dicts = [
                        dict_map[cover_type],  # uncover, cover1, cover2
                        dict_map[name_split[-3]],  # f, m
                        dict_map["overall"],
                    ]
                    if "syn" in name:
                        metric_dicts.append(dict_map["synth"])
                    else:
                        metric_dicts.append(dict_map["real"])
                except:
                    metric_dicts = [dict_map["overall"]]

                for metric_dict in metric_dicts:
                    metric_dict["count"] += 1
                    metric_dict["3D MPJPE"] += error["MPJPE"].sum()
                    metric_dict["PVE"] += error["PVE"].sum()
                    metric_dict["height"] += error["height"].sum()
                    metric_dict["chest"] += error["chest"].sum()
                    metric_dict["waist"] += error["waist"].sum()
                    metric_dict["hips"] += error["hips"].sum()

        for mdict in dict_map.values():
            mdict["3D MPJPE"] = round(
                ((mdict["3D MPJPE"] / (mdict["count"]))).item(), 6
            )
            mdict["PVE"] = round(((mdict["PVE"] / (mdict["count"]))).item(), 6)
            mdict["height"] = round((mdict["height"] / mdict["count"]).item(), 6)
            mdict["chest"] = round((mdict["chest"] / mdict["count"]).item(), 6)
            mdict["waist"] = round((mdict["waist"] / mdict["count"]).item(), 6)
            mdict["hips"] = round((mdict["hips"] / mdict["count"]).item(), 6)

    # -------------------------------------
    if writer is not None:  # use tensorboard or wandb
        try:
            wandb.log(dict_map)
        except:
            for mkey, mdict in dict_map.items():
                mdict_str = json.dumps(mdict)
                writer.add_text(f"{mkey}", mdict_str)
    else:
        logger.log(dict_map)

    return dict_map
