import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np
import imageio
from PIL import Image
import utils.data_utils.data_utils as data_utils
from utils.constants import *
from utils.viz_utils.lib_render2D import *
from utils.viz_utils.lib_render3D import *


def plot(
    save_path,
    pred,
    depth_pair,
    gender,
    recon_depth=None,
    pressure=MISS_MODALITY_FLAG.numpy(),
    ir=MISS_MODALITY_FLAG.numpy(),
    gt=None,
    p_idx=-1,
    pose_idx=-1,
    cover_type="uncover",
    viz_type="image",
):
    """Plot visualization of the given data
        from left to right: RGB(uncover), RGB(cover), IR, Pressure, Depth, GT mesh,  Pred mesh
    Args:
        save_path (str): directory to save the visualization
        pred (dict): predicted data
        depth_pair (np.array): depth image [uncover, cover]
        recon_depth (np.array): reconstructed image
        pressure (np.array): pressure image
        ir (np.array): -1 denotes that ir is not available
        gt (dict): ground truth data
        p_idx (int): participant index
        pose_idx (int): pose index
        cover_type (str): cover type
        viz_type (str): visualization type
    """
    # -------------------------------------
    if viz_type == "3D":
        plot3D(
            pred=pred,
            depth=depth_pair[1],
            gender=gender,
            gt=gt,
            p_idx=p_idx,
            cover=cover_type,
            pose_idx=pose_idx,
        )
    else:
        plot2D(
            save_path=save_path,
            pred=pred,
            depth_pair=depth_pair,
            gender=gender,
            recon_depth=recon_depth,
            pressure=pressure,
            ir=ir,
            gt=gt,
            p_idx=p_idx,
            pose_idx=pose_idx,
            cover_type=cover_type,
            viz_type=viz_type,
        )


# ##################################################################
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def plot3D(
    pred,
    depth,
    gender,
    gt=None,
    p_idx=-1,
    cover="uncover",
    pose_idx=-1,
    filter_pc=True,
    downsample=True,
):
    # get reference image plane mesh
    depth_image = get_depth_image(
        depth, cmap_func=lambda x: BONE(normalize(x))
    )  # (128, 54, 1)->(440, 185, 3)
    try:
        ref = data_utils.get_array_A2B(
            p_id=p_idx, cover="uncover", pose_id=pose_idx, modA="RGB", modB="PM"
        )  # (192, 84, 3)
    except:
        ref = depth_image

    # get pc from depth
    try:
        transform_1 = np.identity(4)
        transform_1[1, 3] = 1.0  # move things over
        transform_2 = np.identity(4)
        transform_2[1, 3] = 2.0
        pc_mesh_list = [
            get_point_cloud_mesh(
                p_idx,
                cover,
                pose_idx,
                color=cv2.resize(depth_image, depth.shape[1::-1]),
                filter_pc=filter_pc,
                transform=transform_1,
                downsample=downsample,
            ),
            get_point_cloud_mesh(
                p_idx,
                cover,
                pose_idx,
                color=cv2.resize(depth_image, depth.shape[1::-1]),
                filter_pc=filter_pc,
                transform=transform_2,
                downsample=downsample,
            ),
        ]
    except:
        pc_mesh_list = None

    if gt is not None:
        gt_trimesh = get_smpl_trimesh(gt["verts"].to("cpu").numpy(), gender)
    pred_trimesh = get_smpl_trimesh(pred["verts"].to("cpu").numpy(), gender)
    view_3D_scene(
        pred_smpl_trimesh=pred_trimesh,
        gt_smpl_trimesh=gt_trimesh,
        ref=ref,
        pc_mesh_list=pc_mesh_list,
    )


# ##################################################################
def plot2D(
    save_path,
    pred,
    depth_pair,
    gender,
    recon_depth=None,
    pressure=MISS_MODALITY_FLAG.numpy(),
    ir=MISS_MODALITY_FLAG.numpy(),
    gt=None,
    p_idx=-1,
    pose_idx=-1,
    cover_type="uncover",
    viz_type="image",
):
    """Plot visualization of the given data
        from left to right: RGB(uncover), RGB(cover), IR, Pressure, Depth, GT mesh,  Pred mesh
    Args:
        save_path (str): directory to save the visualization
        pred (dict): predicted data
        depth_pair (np.array): depth image (2, 128, 54, 1)
        gender (int): 1-female, 0-male
        recon_depth (np.array): reconstructed image
        pressure (np.array): pressure image
        ir (np.array): -1 denotes that ir is not available
        gt (dict): ground truth data
        p_idx (int): participant index
        pose_idx (int): pose index
        cover_type (str): cover type
        viz_type (str): visualization type
    """
    # -------------------------------------
    ref_offset = 10
    # get RGB refs
    if "Hos" in save_path:
        dsFd = os.path.join(SLP_PATH, "simLab")
    else:
        dsFd = os.path.join(SLP_PATH, "danaLab")
    try:
        rgb_uncover, rgb = get_RGB_refs(
            p_idx, cover_type, pose_idx, dsFd
        )  # (440, 191, 3)
        ref_uncover = np.ones((520, 250, 3)).astype(int) * 255
        ref_uncover[40:480, ref_offset : ref_offset + 191, :] = rgb_uncover[:, :, :3]
        ref = np.ones((520, 211, 3)).astype(int) * 255
        ref[40:480, 10:201, :] = rgb[:, :, :3]
        ref_uncover = ref_uncover.astype(np.uint8)
        ref = ref.astype(np.uint8)
    except:
        ref, ref_uncover = None, None  # (np.ones((520, 211, 3)) * 255).astype(np.uint8)

    # -------------------------------------
    # generate ir, pressure, depth
    num_cols = 0
    if (ir != MISS_MODALITY_FLAG.numpy()).any():
        # -1 denotes that ir is not available
        num_cols += 1
        ir = get_ir_image(ir)
    if (pressure != MISS_MODALITY_FLAG.numpy()).any():
        num_cols += 1
        pressure = get_pressure_image(pressure)
    if recon_depth is not None:
        num_cols += 1
        assert (
            recon_depth.shape == depth_pair[1].shape
        ), f"recon_depth.shape: {recon_depth.shape}, depth.shape: {depth.shape}"
        recon_depth = get_depth_image(recon_depth)
    if depth_pair is not None:
        num_cols += 1
        depth = depth_pair[1]
        depth = get_depth_image(depth)  # (128, 54, 1)->(440, 185, 3)

    # get image array
    image_arr = np.ones((520, 195 * num_cols, 3)).astype(int) * 255
    start_idx = 10
    if (ir != MISS_MODALITY_FLAG.numpy()).any():
        image_arr[40:480, start_idx : start_idx + 186, :] = ir
        start_idx += 195
    if (pressure != MISS_MODALITY_FLAG.numpy()).any():
        image_arr[40:480, start_idx : start_idx + 185, :] = pressure
        start_idx += 195
    image_arr[40:480, start_idx : start_idx + 185, :] = depth
    if recon_depth is not None:
        start_idx += 195
        image_arr[40:480, start_idx : start_idx + 185, :] = recon_depth
    image_arr = image_arr.astype(np.uint8)

    if ref_uncover is None:
        # use depth as reference
        depth_ref = depth_pair[0]
        depth_ref = get_depth_image(depth_ref)
        ref_uncover = np.ones((520, 250, 3)).astype(int) * 255
        ref_uncover[40:480, ref_offset : ref_offset + 185, :] = depth_ref

    # -------------------------------------
    # generate mesh render
    if gt is not None:
        gt_mesh = get_smpl_trimesh(gt["verts"].to("cpu").numpy(), gender)
        if viz_type == "image":
            gt_render = render_scene_image(gt_mesh, is_gt=True)
            gt_render = np.array(gt_render)[20:540, 0:250, :]  # shift the smpl mesh
            if ref_uncover is not None:
                replace_index = np.argwhere((gt_render == [255, 255, 255]).all(2))
                gt_render[replace_index[:, 0], replace_index[:, 1], :] = ref_uncover[
                    replace_index[:, 0], replace_index[:, 1], :
                ]
        elif viz_type == "video":
            gt_renders = render_scene_video(gt_mesh)
            gt_renders = [np.array(render)[20:540, 0:250, :] for render in gt_renders]
        else:
            raise ValueError("Invalid viz_type")

    pred_mesh = get_smpl_trimesh(pred["verts"].to("cpu").numpy(), gender)
    if viz_type == "image":
        pred_render = render_scene_image(pred_mesh)
        pred_render = np.array(pred_render)[20:540, 0:250, :]  # shift the smpl mesh
        if ref_uncover is not None:
            replace_index = np.argwhere((pred_render == [255, 255, 255]).all(2))
            pred_render[replace_index[:, 0], replace_index[:, 1], :] = ref_uncover[
                replace_index[:, 0], replace_index[:, 1], :
            ]

    elif viz_type == "video":
        pred_renders = render_scene_video(pred_mesh)
        pred_renders = [np.array(render)[20:540, 0:250, :] for render in pred_renders]
    else:
        raise ValueError("Invalid viz_type")

    # -------------------------------------
    # visualize
    if viz_type == "image":
        # -------------------------------------
        # concatenate images and save
        final_render = [] if ref is None else [ref]
        final_render.append(image_arr)
        if gt is not None:
            final_render.append(gt_render)
        final_render.append(pred_render)
        final_render = np.concatenate(final_render, axis=1)

        start_final_index = 40
        if ref is not None:
            final_render = cv2.putText(
                final_render,
                "Reference",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            start_final_index += 200

        if (ir != MISS_MODALITY_FLAG.numpy()).any():
            final_render = cv2.putText(
                final_render,
                "    IR  ",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            start_final_index += 200

        if (pressure != MISS_MODALITY_FLAG.numpy()).any():
            final_render = cv2.putText(
                final_render,
                "Pressure",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            start_final_index += 200

        final_render = cv2.putText(
            final_render,
            "  Depth  ",
            (start_final_index, 30),
            0,
            0.85,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        start_final_index += 200

        if recon_depth is not None:
            final_render = cv2.putText(
                final_render,
                "Recon Depth",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            start_final_index += 200

        if gt is not None:
            final_render = cv2.putText(
                final_render,
                "  GT Mesh  ",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            start_final_index += 200

        final_render = cv2.putText(
            final_render,
            "   Pred Mesh  ",
            (start_final_index, 30),
            0,
            0.85,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        image = Image.fromarray(final_render.astype(np.uint8))
        image.save(os.path.join(save_path, f"viz_{p_idx}_{pose_idx}.png"))

    elif viz_type == "video":
        # -------------------------------------
        # save video
        final_renders = []
        for i in range(len(pred_renders)):
            final_render = [] if ref is None else [ref]
            final_render.append(image_arr)
            if gt is not None:
                final_render.append(gt_renders[i])
            final_render.append(pred_renders[i])
            final_render = np.concatenate(final_render, axis=1)

            start_final_index = 40
            if ref is not None:
                final_render = cv2.putText(
                    final_render,
                    "Reference",
                    (start_final_index, 30),
                    0,
                    0.85,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                start_final_index += 200

            if ir != MISS_MODALITY_FLAG:
                final_render = cv2.putText(
                    final_render,
                    "    IR  ",
                    (start_final_index, 30),
                    0,
                    0.85,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                start_final_index += 200

            if pressure != MISS_MODALITY_FLAG:
                final_render = cv2.putText(
                    final_render,
                    "Pressure",
                    (start_final_index, 30),
                    0,
                    0.85,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                start_final_index += 200

            final_render = cv2.putText(
                final_render,
                "  Depth  ",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            start_final_index += 200

            if recon_depth is not None:
                final_render = cv2.putText(
                    final_render,
                    "Recon Depth",
                    (start_final_index, 30),
                    0,
                    0.85,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                start_final_index += 200

            if gt is not None:
                final_render = cv2.putText(
                    final_render,
                    "  GT Mesh  ",
                    (start_final_index, 30),
                    0,
                    0.85,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                start_final_index += 200

            final_render = cv2.putText(
                final_render,
                "   Pred Mesh  ",
                (start_final_index, 30),
                0,
                0.85,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            final_renders.append(Image.fromarray(final_render.astype(np.uint8)))
        imageio.mimsave(
            os.path.join(save_path, f"viz_{p_idx}_{pose_idx}.gif"),
            final_renders,
            fps=10,
            loop=0,
        )
