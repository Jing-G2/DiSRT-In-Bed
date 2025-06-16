import sys
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import torch
import numpy as np
import imageio
import pyrender
import trimesh
import open3d as o3d
import matplotlib
from scipy.ndimage.interpolation import zoom

import utils.data_utils.data_utils as data_utils
from utils.constants import *
from utils.viz_utils.lib_basic import *
from utils.viz_utils.camera import get_slp_3D_camera_pose

artags_3d = BoundryMarkers(x_bump=0, y_bump=Y_BUMP)


# ##################################################################
# ----------------- point cloud functions -----------------
# ##################################################################
def apply_homography(points, h, yx=True):
    # Apply 3x3 homography matrix to points
    # Note that the homography matrix is parameterized as XY,
    # but all image coordinates are YX

    if yx:
        points = np.flip(points, 1)

    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    tform_h = np.matmul(h, points_h.T).T
    tform_h = tform_h / tform_h[:, 2][:, np.newaxis]

    points = tform_h[:, :2]

    if yx:
        points = np.flip(points, 1)

    return points


def get_modified_depth_to_pressure_homography(p_idx):
    """
    Magic function to get the homography matrix to warp points in depth-cloud space into pressure-mat space.
    However, this modifies the scaling of the homography matrix to keep the same scale, so that the points can be
    projected into 3D metric space.
    :param slp_dataset: Dataset object to extract homography from
    :param idx: SLP dataset sample index
    :return:
    """
    WARPING_MAGIC_SCALE_FACTOR = (
        192.0 / 345.0
    )  # Scale matrix to align to PM. 192 is height of pressure mat, 345 is height of bed in depth pixels

    depth_Tr = data_utils.get_PTr_A2B(
        p_id=p_idx, modA="depthRaw", modB="PM"  # modB="depthRaw"
    )  # Get SLP homography matrix
    depth_Tr /= depth_Tr[2, 2]  # Make more readable matrix

    depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3] / WARPING_MAGIC_SCALE_FACTOR
    return depth_Tr


def project_depth_with_warping(p_idx, cover, pose_idx):
    """
    Project a 2D depth image into 3D space. Additionally, this warps the 2D points of the input image
    using a homography matrix. Importantly, there is no interpolation in the warping step.
    :param slp_dataset: Dataset object to extract image from
    :param depth_arr: The input depth image to use
    :param idx: SLP dataset sample index
    :return: A [N, 3] numpy array of the 3D pointcloud
    """
    depth_arr = data_utils.get_array_A2B(
        p_idx, cover, pose_idx, modA="depthRaw", modB="depthRaw"
    )

    # The other methods of using cv2.warpPerspective apply interpolation to the image, which is bad. This doesn't
    # Input image is YX
    depth_homography = get_modified_depth_to_pressure_homography(p_idx)
    orig_x, orig_y = np.meshgrid(
        np.arange(0, depth_arr.shape[1]), np.arange(0, depth_arr.shape[0])
    )

    image_space_coordinates = np.stack((orig_x.flatten(), orig_y.flatten()), 0).T
    image_space_coordinates = image_space_coordinates * 64 * 0.0286 / 1.92
    warped_image_space_coordinates = apply_homography(
        image_space_coordinates, depth_homography, yx=False
    )

    cd_modified = np.matmul(
        depth_homography,
        np.array([SLP_CAM_PARAMS["c_d"][0], SLP_CAM_PARAMS["c_d"][1], 1.0]).T,
    )  # Multiply the center of the depth image by homography
    cd_modified = cd_modified / cd_modified[2]  # Re-normalize

    depth_arr[0, 0] = 2101
    projection_input = np.concatenate(
        (warped_image_space_coordinates, depth_arr.flatten()[..., np.newaxis]), axis=1
    )
    point_cloud = (
        data_utils.pixel2cam(projection_input, SLP_CAM_PARAMS["f_d"], cd_modified[0:2])
        / 1000.0
    )

    return point_cloud


def get_points_from_slp_depth(p_idx, cover, pose_idx, filter_pc=False):
    """Get point cloud from original depth array"""
    point_cloud = project_depth_with_warping(p_idx, cover, pose_idx)
    ptc_first_point = np.array(point_cloud[0])

    if filter_pc:
        valid_x = np.logical_and(
            point_cloud[:, 0] > -0.25, point_cloud[:, 0] < 0.55
        )  # Cut X
        valid_y = np.logical_and(
            point_cloud[:, 1] > -1.0, point_cloud[:, 1] < 1.0
        )  # Cut Y

        # Cut out any outliers above the bed
        # valid_z = np.logical_and(point_cloud[:, 2] > 1.55, point_cloud[:, 2] < 2.1)
        valid_all = np.logical_and.reduce((valid_x, valid_y))  # , valid_z))
        point_cloud = point_cloud[valid_all, :]

        rot_angle_fixed = np.deg2rad(2.5)
        point_cloud[:, 0] = (point_cloud[:, 0]) * np.cos(rot_angle_fixed) - (
            point_cloud[:, 2]
        ) * np.sin(rot_angle_fixed)
        point_cloud[:, 2] = (point_cloud[:, 0]) * np.sin(rot_angle_fixed) + (
            point_cloud[:, 2]
        ) * np.cos(rot_angle_fixed)

        point_cloud = point_cloud[point_cloud[:, 2] < 2.08]  # *2.1

        point_cloud[:, 0] = (point_cloud[:, 0]) * np.cos(-rot_angle_fixed) - (
            point_cloud[:, 2]
        ) * np.sin(-rot_angle_fixed)
        point_cloud[:, 2] = (point_cloud[:, 0]) * np.sin(-rot_angle_fixed) + (
            point_cloud[:, 2]
        ) * np.cos(-rot_angle_fixed)

    # this is because we set the first point in the depth image to 2101.
    point_cloud[:, 2] -= ptc_first_point[2]

    # manually adjust the point cloud to align with the bed
    point_cloud[:, 0] -= 27 * 0.0286 # + X_BUMP
    point_cloud[:, 1] -= 64 * 0.0286 / 2 - 5 * 0.0286 # * vertical shift hard to align all

    point_cloud = np.concatenate(
        (-point_cloud[:, 1:2], point_cloud[:, 0:1], point_cloud[:, 2:3]), axis=1
    )

    cmap_norm = matplotlib.colors.Normalize(
        vmin=-point_cloud.max(axis=0)[-1], vmax=-point_cloud.min(axis=0)[-1]
    )
    colors = JET(cmap_norm(-point_cloud[:, -1]))[:, :3]

    return point_cloud, colors


def get_point_cloud_mesh(
    p_idx,
    cover,
    pose_idx,
    color=None,
    filter_pc=False,
    transform=np.eye(4),
    downsample=True,
):
    """Get point cloud mesh for rendering"""
    pc, color = get_points_from_slp_depth(p_idx, cover, pose_idx, filter_pc)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)

    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd_points = np.asarray(pcd.points)

    pcd_points_homo = np.swapaxes(
        np.concatenate([pcd_points, np.ones((pcd_points.shape[0], 1))], axis=1), 0, 1
    )
    pcd_trans = np.swapaxes(np.matmul(transform, pcd_points_homo), 0, 1)[:, :3]

    if color is None:
        color = np.zeros((pcd_trans.shape[0], 3))
    else:
        color = np.asarray(pcd.colors)

    pc_mesh = pyrender.Mesh.from_points(pcd_trans, colors=color)

    return pc_mesh


# ##################################################################
# ----------------- other mesh functions -----------------
# ##################################################################
def get_joints_trimesh(joints, transform=np.eye(4)):
    """Get joints markers for rendering"""
    joints = joints.reshape(-1, 3)
    joints_homo = np.concatenate([joints, np.ones((joints.shape[0], 1))], axis=1)
    joints_homo = np.swapaxes(joints_homo, 0, 1)
    joints_trans = np.swapaxes(np.matmul(transform, joints_homo), 0, 1)[:, :3]

    sm = trimesh.creation.uv_sphere(radius=0.03)
    sm.visual.vertex_colors = [0.15, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (len(joints_trans), 1, 1))
    tfs[:, :3, 3] = joints_trans
    joints_trimesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    return joints_trimesh


def get_img_plane_trimesh(height, width, color=None):
    translucency = 0.9
    if color is None:
        # the last dimension is translucency
        color = np.ones((height, width, 4)) * [0.8, 0.8, 0.8, translucency]
    else:
        assert (
            height == color.shape[0] and width == color.shape[1]
        ), f"color.shape: {color.shape}, height and width should match the size of color image"
        if (color > 1.0 + 1e-5).any():
            color = color.astype(np.float32) / 255.0
        color = np.concatenate(
            [color, np.ones([color.shape[0], color.shape[1], 1]) * translucency], axis=2
        )  # the last dimension is translucency

    cali_size = [64, 27]
    plane_xyz = np.zeros((height + 1, width + 1, 3))
    plane_faces = []
    plane_faces_color = []

    for i in range(height + 1):
        for j in range(width + 1):
            plane_xyz[i, j, 1] = j * 0.0286 * (cali_size[1] / width)  # * + X_BUMP
            plane_xyz[i, j, 0] = (
                (height - i) * 0.0286 * (cali_size[0] / height)
            ) * 1.04 + Y_BUMP
            plane_xyz[i, j, 2] = 0.075  # *0.01

            if i < height and j < width:
                coord1 = i * (width + 1) + j
                coord2 = coord1 + 1
                coord3 = (i + 1) * (width + 1) + j
                coord4 = coord3 + 1

                plane_faces.append([coord1, coord2, coord3])  # bottom surface
                plane_faces.append([coord1, coord3, coord2])  # top surface
                plane_faces.append([coord4, coord3, coord2])  # bottom surface
                plane_faces.append([coord2, coord3, coord4])  # top surface
                plane_faces_color.append(color[i, j, :])
                plane_faces_color.append(color[i, j, :])
                plane_faces_color.append(color[i, j, :])
                plane_faces_color.append(color[i, j, :])

    plane_faces = np.array(plane_faces)
    plane_faces_color = np.array(plane_faces_color)

    plane_verts = plane_xyz.reshape(-1, 3)

    plane_trimesh = trimesh.base.Trimesh(
        vertices=plane_verts, faces=plane_faces, face_colors=plane_faces_color
    )

    return plane_trimesh


# ##################################################################
# ----------------- view functions -----------------
# ##################################################################
def view_3D_scene(pred_smpl_trimesh, gt_smpl_trimesh=None, ref=None, pc_mesh_list=None):
    """View 3D scene with SMPL mesh, image plane and point cloud"""
    scene = pyrender.Scene()

    # add pred SMPL mesh
    pred_smpl_mesh = pyrender.Mesh.from_trimesh(
        pred_smpl_trimesh, material=HUMAN_MATERIAL.get("human_mat_3d"), smooth=True
    )
    # scene.add(pred_smpl_mesh)

    if gt_smpl_trimesh is not None:
        # add gt SMPL mesh
        # gt_smpl_trimesh.apply_translation([0.0, -1.0, 0.0])
        gt_smpl_mesh = pyrender.Mesh.from_trimesh(
            gt_smpl_trimesh, material=HUMAN_MATERIAL.get("human_mat_3d"), smooth=True
        )
        scene.add(gt_smpl_mesh)

    # add camera
    camera_pose = get_slp_3D_camera_pose()
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=camera_pose)

    # add reference image plane
    ref_trimesh = get_img_plane_trimesh(ref.shape[0], ref.shape[1], ref)
    ref_trimesh.apply_translation([0.0, 0.0, 0.0])
    ref_mesh = pyrender.Mesh.from_trimesh(ref_trimesh, smooth=False)
    scene.add(ref_mesh)
    ref_trimesh.apply_translation([0.0, -1.0, 0.0])
    ref_mesh = pyrender.Mesh.from_trimesh(ref_trimesh, smooth=False)
    scene.add(ref_mesh)

    if pc_mesh_list is not None:
        # add point cloud mesh
        for pc_mesh in pc_mesh_list:
            scene.add(pc_mesh)
        scene.add(pc_mesh)

    if artags_3d is not None:
        # add AR tags
        artags_trimesh = artags_3d.get_artags_trimesh(viz_type="3D")
        artags_trimesh_transformed = []
        for artag in artags_trimesh:
            artag.apply_translation([0.0, 1.0, 0.0])
            artags_trimesh_transformed.append(artag)
        artags_mesh = pyrender.Mesh.from_trimesh(
            artags_trimesh_transformed, smooth=False
        )
        scene.add(artags_mesh)

    viewer = pyrender.Viewer(
        scene,
        use_raymond_lighting=True,
        lighting_intensity=5.0,  # *20.0
        point_size=5.0,
        viewport_size=(1000, 700),
    )

    input("Press Enter to continue...")
