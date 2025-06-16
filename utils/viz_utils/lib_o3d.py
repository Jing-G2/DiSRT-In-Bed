import os
import sys
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import utils.data_utils.data_utils as data_utils
from utils.constants import (
    SLP_PATH,
    SLP_CAM_PARAMS,
    FACES,
)

MIDDLE_FILLER = 100
PERC_TOTAL = 1.0
FACES_NP = FACES.squeeze(0).to("cpu").numpy()


# #################################################
# o3d get functions
# #################################################
def o3d_get_smpl_mesh(verts, offset=np.array([1.3, 0, 0])):
    """Get SMPL mesh in Open3D format
    Args:
        verts (np.array): SMPL mesh vertices
        offset (np.array): offset for visualization
    Returns:
        smpl_o3d: Open3D mesh of SMPL
        smpl_o3d_offset: Open3D mesh of SMPL offset for visualization
    """
    smpl_o3d = o3d.geometry.TriangleMesh()
    smpl_o3d.triangles = o3d.utility.Vector3iVector(FACES_NP)
    smpl_o3d.vertices = o3d.utility.Vector3dVector(verts)
    smpl_o3d.compute_vertex_normals()

    smpl_o3d_offset = o3d.geometry.TriangleMesh()
    smpl_o3d_offset.triangles = o3d.utility.Vector3iVector(FACES_NP)
    smpl_o3d_offset.vertices = o3d.utility.Vector3dVector(verts + offset)
    smpl_o3d_offset.compute_vertex_normals()
    smpl_o3d_offset.paint_uniform_color([0.85, 0.3, 0.3])

    return smpl_o3d, smpl_o3d_offset


def o3d_get_smpl_joints(joints, cor_list=None, camera=None, radius=0.02):
    """Get SMPL joints in Open3D format
    Args:
        joints (np.array): SMPL joints
        cor_list (list): list of joint indices
        radius (float): radius of the joints
        camera (dict): camera parameters, if not None then it's gt joints
    Returns:
        all_joint_markers: list of Open3D sphere joints
    """
    all_joint_markers = []
    for i in range(len(cor_list)):
        color = cm.jet((i / 25.0 * 3) % 1)[:3]

        if cor_list is None:
            assert camera is not None
            z_depth = 2
            pos = inverse_camera_perspective_transform(
                points=joints[cor_list[i], :],
                z_dist=z_depth,
                camera_rotation=camera["camera_rotation"],
                camera_translation=camera["camera_translation"],
                camera_center=camera["camera_center"],
                camera_focal_length_x=camera["camera_focal_length_x"],
                camera_focal_length_y=camera["camera_focal_length_y"],
            )
        else:
            pos = joints[cor_list[i], :]

        joint_marker = o3d_get_sphere(color=color, pos=pos, radius=radius)
        all_joint_markers.append(joint_marker)

    return all_joint_markers


def o3d_get_point_cloud(
    p_id, cover="uncover", pose_id=1, dsFd=os.path.join(SLP_PATH, "danaLab")
):
    """Get Open3D point cloud from depth array"""
    pointcloud = project_depth_with_warping(
        p_id=p_id, cover=cover, pose_id=pose_id, dsFd=dsFd
    )

    valid_z = np.logical_and(
        pointcloud[:, 2] > 1.55,
        pointcloud[:, 2] < 2.08,  # * modified to get human part only
    )  # Cut out any outliers above the bed
    valid_x = np.logical_and(pointcloud[:, 0] > -0.3, pointcloud[:, 0] < 0.8)  # Cut X
    valid_y = np.logical_and(pointcloud[:, 1] > -1.1, pointcloud[:, 1] < 1.0)  # Cut Y
    valid_all = np.logical_and.reduce((valid_x, valid_y, valid_z))
    pointcloud = pointcloud[valid_all, :]

    ptc_depth = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))

    # cmap_norm = matplotlib.colors.Normalize(vmin=pointcloud.min(axis=0)[-1], vmax=pointcloud.max(axis=0)[-1])
    # cmap_norm = matplotlib.colors.Normalize(vmin=1.55, vmax=2.15)
    cmap_norm = matplotlib.colors.Normalize(
        vmin=-pointcloud.max(axis=0)[-1], vmax=-pointcloud.min(axis=0)[-1]
    )
    point_colors = plt.get_cmap("jet")(cmap_norm(-pointcloud[:, -1]))[:, 0:3]
    ptc_depth.colors = o3d.utility.Vector3dVector(point_colors)

    return ptc_depth


def o3d_get_ref_image(ref):
    """covert ref image to Open3D format"""
    height, width = ref.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=SLP_CAM_PARAMS["f_d"][0],
        fy=SLP_CAM_PARAMS["f_d"][1],
        cx=SLP_CAM_PARAMS["c_d"][0],
        cy=SLP_CAM_PARAMS["c_d"][1],
    )

    ref = ref.astype(np.float32)
    depth_raw = np.ones((height, width), dtype=np.float32) * 2.15
    rows, cols = np.where(ref[:, :, 0] == 0)
    depth_raw[rows, cols] = np.inf

    ref_image = o3d.geometry.Image(ref)
    depth_image = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        ref_image, depth_image, depth_scale=1
    )

    ref_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    rows, cols = np.where(ref[:, :, 0] != 0)
    pcd_colors = np.array(ref[rows, cols, :], dtype=np.float32) / 255.0
    ref_pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    ref_pcd = ref_pcd.translate((-1.5, 0, 0))

    return ref_pcd, intrinsic


def o3d_set_camera_extrinsic(vis, transform=np.eye(4)):
    """
    Sets the Open3D camera position and orientation
    :param vis: Open3D visualizer object
    :param transform: 4x4 numpy defining a rigid transform where the camera should go
    """
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = transform
    ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)


# #################################################
# helper functions
# #################################################
def o3d_get_sphere(color=[0.3, 1.0, 0.3], pos=[0, 0, 0], radius=0.06):
    """
    Generates an Open3D mesh object as a sphere
    :param color: RGB color 3-list, 0-1
    :param pos: Center of sphere, 3-list
    :param radius: Radius of sphere
    :return: Open3D TriangleMesh sphere
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color(color)
    mean = np.asarray(mesh_sphere.vertices).mean(axis=0)
    diff = np.asarray(pos) - mean
    mesh_sphere.translate(diff)
    return mesh_sphere


def camera_perspective_transform(
    points,
    camera_rotation,
    camera_translation,
    camera_center,
    camera_focal_length_x,
    camera_focal_length_y,
):
    """Apply the camera perspective transform to points in the world coordinate.
    Args:
        points: 3D world points
        camera_rotation: camera rotation matrix
        camera_translation: camera translation vector
        camera_center: camera center
        camera_focal_length_x: camera focal length x
        camera_focal_length_y: camera focal length y
    Returns:
        pixel_points: points in the image coordinate
    """
    camera_mat = np.ones((3, 3))  # Make a 3x3 matrix with focal length on diagonal
    camera_mat[0, 0] = camera_focal_length_x
    camera_mat[1, 1] = camera_focal_length_y
    camera_mat[0, 2] = camera_center[0]
    camera_mat[1, 2] = camera_center[1]

    camera_transform = np.eye(4)  # Make 4x4 rigid transform matrix
    camera_transform[:3, :3] = camera_rotation
    camera_transform[:3, 3] = camera_translation
    camera_transform = camera_transform[:3, :]  # Remove last row

    points_homo = np.ones((points.shape[0], 4))
    points_homo[:, :3] = points

    pixel_points = np.matmul(camera_transform, points.T)  # 3xn
    pixel_points = np.matmul(camera_mat, pixel_points).T  # nx3

    return pixel_points[:, :2]


def inverse_camera_perspective_transform(
    points,
    z_dist,
    camera_rotation,
    camera_translation,
    camera_center,
    camera_focal_length_x,
    camera_focal_length_y,
):
    """Apply the inverse camera perspective transform to points in the image space.
    Args:
        points: 2D image points
        z_dist: distance from camera
        camera_rotation: camera rotation matrix
        camera_translation: camera translation vector
        camera_center: camera center
        camera_focal_length_x: camera focal length x
        camera_focal_length_y: camera focal length y
    Returns:
        non_homo_points: points in the world coordinate
    """
    camera_mat = np.zeros((2, 2))  # Make a 2x2 matrix with focal length on diagonal
    camera_mat[0, 0] = camera_focal_length_x
    camera_mat[1, 1] = camera_focal_length_y
    camera_mat_inv = np.linalg.inv(camera_mat)

    camera_transform = np.eye(4)  # Make 4x4 rigid transform matrix
    camera_transform[:3, :3] = camera_rotation
    camera_transform[:3, 3] = camera_translation
    camera_transform_inv = np.linalg.inv(camera_transform)
    pixel_points = np.matmul(
        camera_mat_inv, points.T - camera_center[..., np.newaxis]
    ).T
    img_points = np.ones((points.shape[0], 4))
    img_points[:, :2] = pixel_points * z_dist
    img_points[:, 2] = z_dist

    projected_points = np.matmul(
        camera_transform_inv, img_points.T
    ).T  # Apply rigid transform
    non_homo_points = projected_points[:, :3]

    return non_homo_points


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


def get_modified_depth_to_pressure_homography(
    p_id, dsFd=os.path.join(SLP_PATH, "danaLab")
):
    """
    Magic function to get the homography matrix to warp points in depth-cloud space into pressure-mat space.
    However, this modifies the scaling of the homography matrix to keep the same scale, so that the points can be
    projected into 3D metric space.

    Args:
        p_id: person id, base 1
        cover: cover str "uncover", "cover1", "cover2"
        pose_id: pose id, base 1
    Return:
        depth to pressure homography matrix
    """
    WARPING_MAGIC_SCALE_FACTOR = (
        192.0 / 345.0
    )  # Scale matrix to align to PM. 192 is height of pressure mat, 345 is height of bed in depth pixels
    depth_Tr = data_utils.get_PTr_A2B(
        p_id=p_id, modA="depthRaw", modB="PM", dsFd=dsFd
    )  # Get SLP homography matrix
    depth_Tr /= depth_Tr[2, 2]  # Make more readable matrix

    depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3] / WARPING_MAGIC_SCALE_FACTOR
    return depth_Tr


def pixel2cam(pixel_coord, f, c):
    pixel_coord = pixel_coord.astype(float)
    jt_cam = np.zeros_like(pixel_coord)
    jt_cam[..., 0] = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    jt_cam[..., 1] = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    jt_cam[..., 2] = pixel_coord[..., 2]

    return jt_cam


def project_depth_with_warping(
    p_id, cover="uncover", pose_id=1, dsFd=os.path.join(SLP_PATH, "danaLab")
):
    """
    Project a 2D depth image into 3D space. Additionally, this warps the 2D points of the input image
    using a homography matrix. Importantly, there is no interpolation in the warping step.
    :param p_id: SLP dataset sample index
    :return: A [N, 3] numpy array of the 3D pointcloud
    """
    # The other methods of using cv2.warpPerspective apply interpolation to the image, which is bad. This doesn't
    # Input image is YX
    depth_arr = data_utils.get_array_A2B(
        p_id=p_id,
        cover=cover,
        pose_id=pose_id,
        modA="depthRaw",
        modB="depthRaw",
        dsFd=dsFd,
    )
    depth_homography = get_modified_depth_to_pressure_homography(p_id=p_id, dsFd=dsFd)
    orig_x, orig_y = np.meshgrid(
        np.arange(0, depth_arr.shape[1]), np.arange(0, depth_arr.shape[0])
    )

    image_space_coordinates = np.stack((orig_x.flatten(), orig_y.flatten()), 0).T
    warped_image_space_coordinates = apply_homography(
        image_space_coordinates, depth_homography, yx=False
    )

    cd_modified = np.matmul(
        depth_homography,
        np.array([SLP_CAM_PARAMS["c_d"][0], SLP_CAM_PARAMS["c_d"][1], 1.0]).T,
    )  # Multiply the center of the depth image by homography
    cd_modified = cd_modified / cd_modified[2]  # Re-normalize

    projection_input = np.concatenate(
        (warped_image_space_coordinates, depth_arr.flatten()[..., np.newaxis]), 1
    )
    ptc = pixel2cam(projection_input, SLP_CAM_PARAMS["f_d"], cd_modified[0:2]) / 1000.0
    return ptc


smplx_alphapose_corrs = np.asarray(
    [
        [0, 19],
        [1, 11],
        [2, 12],
        [4, 13],
        [5, 14],
        [7, 15],
        [8, 16],
        [10, 20],
        [11, 21],
        [12, 18],
        [15, 0],
        [16, 5],
        [17, 6],
        [18, 7],
        [19, 8],
        [20, 9],
        [21, 10],
    ],
    dtype=np.int32,
)

smplx_openpose_corrs = np.asarray(
    [
        [0, 24],
        [1, 12],
        [2, 17],
        [3, 19],
        [4, 21],
        [5, 16],
        [6, 18],
        [7, 20],
        [8, 0],
        [9, 2],
        [10, 5],
        [11, 8],
        [12, 1],
        [13, 4],
        [14, 7],
        [15, 25],
        [16, 26],
        [17, 27],
        [18, 28],
        [19, 29],
        [20, 30],
        [21, 31],
        [22, 32],
        [23, 33],
        [24, 34],
    ],
    dtype=np.int32,
)


def smpl_to_openpose():
    """
    :return: Returns the indices of the permutation that maps OpenPose joints to SMPL joints
    """

    return np.array(
        [
            [0, 24],
            [1, 12],
            [2, 17],
            [3, 19],
            [4, 21],
            [5, 16],
            [6, 18],
            [7, 20],
            [8, 0],
            [9, 2],
            [10, 5],
            [11, 8],
            [12, 1],
            [13, 4],
            [14, 7],
            [15, 25],
            [16, 26],
            [17, 27],
            [18, 28],
            [19, 29],
            [20, 30],
            [21, 31],
            [22, 32],
            [23, 33],
            [24, 34],
        ],
        dtype=np.int32,
    )
