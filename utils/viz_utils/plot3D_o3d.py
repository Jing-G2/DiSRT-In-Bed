import sys
import numpy as np
import os
import open3d as o3d

from utils.data_utils.data_utils import get_array_A2B
from utils.constants import *
from utils.viz_utils.lib_o3d import *

base_path = os.path.join(BASE_PATH, "0-Pose")

sys.path.insert(0, base_path)


def o3d_plot(
    save_path,
    pred,
    depth,
    gt=None,
    p_idx=-1,
    pose_idx=-1,
    cover_type="uncover",
    visible=False,
):
    """Plot visualization of the given data using open3d
    Args:
        save_path (str): directory to save the visualization
        pred (dict): predicted data
        depth (np.array): depth image
        gt (dict): ground truth data
        p_idx (int): participant index
        pose_idx (int): pose index
        cover_type (str): cover type
        visible (bool): whether to show the visualization
    """
    # -------------------------------------
    # get refs
    # try:
    ref = get_array_A2B(
        p_id=p_idx, cover=cover_type, pose_id=pose_idx, modA="RGB", modB="depthRaw"
    )
    # except:
    # depth = get_array_A2B(
    #     p_id=p_idx,
    #     cover=cover_type,
    #     pose_id=pose_idx,
    #     modA="depthRaw",
    #     modB="depthRaw",
    # )
    # ref = get_depth_image(depth)

    # -------------------------------------
    # generate mesh render
    if gt is not None:
        plot3D(
            verts=gt["verts"].to("cpu").numpy(),
            p_id=p_idx,
            cover=cover_type,
            pose_id=pose_idx,
            ref=ref,
            file_path=os.path.join(save_path, f"gt_{p_idx}_{pose_idx}_o3d.png"),
            visible=visible,
        )

    plot3D(
        verts=pred["verts"].to("cpu").numpy(),
        p_id=p_idx,
        cover=cover_type,
        pose_id=pose_idx,
        ref=ref,
        file_path=os.path.join(save_path, f"pred_{p_idx}_{pose_idx}_o3d.png"),
        visible=visible,
    )


def plot3D(verts, p_id, cover, pose_id, ref=None, file_path="test.png", visible=False):
    """Plot 3D visualization of the SMPL mesh and point cloud
    Args:
        verts (np.array): SMPL mesh vertices
        p_id (int): person ID
        cover (str): cover name
        pose_id (int): pose ID
        ref: reference image (rgb or depth)
        file_path (str): file path to save the visualization
        visible (bool): whether to show the visualization
    """
    smpl_o3d, smpl_o3d_offset = o3d_get_smpl_mesh(verts=verts)
    point_could = o3d_get_point_cloud(p_id=p_id, cover=cover, pose_id=pose_id)

    ref_pcd, intrinsic = o3d_get_ref_image(ref=ref)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=True)  # Create an window
    if ref_pcd is not None:
        vis.add_geometry(ref_pcd)

    # ----------------------------------------------
    # Render the scene with only point cloud
    vis.add_geometry(point_could)
    o3d_set_camera_extrinsic(vis, transform=np.eye(4))
    vis.update_geometry(point_could)
    vis.poll_events()
    vis.update_renderer()
    # depth_pcd_image = vis.capture_depth_image(
    #     file_path.replace(".png", "_depth_pc.png"), do_render=True
    # )
    depth_pcd_buffer = vis.capture_depth_float_buffer(do_render=True)
    depth_pcd_np = np.asarray(depth_pcd_buffer)

    # ----------------------------------------------
    # Render the scene with only smpl mesh
    vis.add_geometry(smpl_o3d)
    vis.add_geometry(smpl_o3d_offset)

    cameraLines = o3d.geometry.LineSet.create_camera_visualization(
        intrinsic=intrinsic,
        extrinsic=np.eye(4),
    )
    vis.add_geometry(cameraLines)

    o3d_set_camera_extrinsic(vis, transform=np.eye(4))
    vis.poll_events()
    vis.update_renderer()
    # depth_smpl_image = vis.capture_depth_image(
    #     file_path.replace(".png", "_depth_smpl.png"), do_render=True
    # )
    depth_smpl_buffer = vis.capture_depth_float_buffer(do_render=True)
    depth_smpl_np = np.asarray(depth_smpl_buffer)

    # ----------------------------------------------
    img = vis.capture_screen_image(file_path)
    vis.run()
    vis.destroy_window()


    return depth_pcd_np, depth_smpl_np
