import os
import numpy as np
import trimesh


def get_slp_2D_camera_pose(camera_angle=0):
    camera_pose = np.eye(4)
    camera_pose[0, 0] = np.cos(np.pi / 2)
    camera_pose[0, 1] = np.sin(np.pi / 2)
    camera_pose[1, 0] = -np.sin(np.pi / 2)
    camera_pose[1, 1] = np.cos(np.pi / 2)

    rot_udpim = np.eye(4)
    rot_y = 180 * np.pi / 180.0
    rot_udpim[1, 1] = np.cos(rot_y)
    rot_udpim[2, 2] = np.cos(rot_y)
    rot_udpim[1, 2] = np.sin(rot_y)
    rot_udpim[2, 1] = -np.sin(rot_y)
    camera_pose = np.matmul(rot_udpim, camera_pose)

    camera_pose[0, 3] = 0  # (middle_horiz_filler/880)*64*0.0286 #64*0.0286/2  # -1.0
    camera_pose[1, 3] = 1.2
    camera_pose[2, 3] = -1.0

    # rotate around the smpl human mesh
    radias_angle = np.radians(camera_angle)
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=radias_angle, direction=[1, 0, 0], point=[0, 0.4, 0]
    )
    camera_pose = np.dot(rotation_matrix, camera_pose)

    return camera_pose


def get_slp_3D_camera_pose(camera_angle=0):
    camera_pose = np.eye(4)
    camera_pose[0, 0] = np.cos(np.pi / 2)
    camera_pose[0, 1] = np.sin(np.pi / 2)
    camera_pose[1, 0] = -np.sin(np.pi / 2)
    camera_pose[1, 1] = np.cos(np.pi / 2)

    rot_udpim = np.eye(4)
    rot_y = 180 * np.pi / 180.0
    rot_udpim[1, 1] = np.cos(rot_y)
    rot_udpim[2, 2] = np.cos(rot_y)
    rot_udpim[1, 2] = np.sin(rot_y)
    rot_udpim[2, 1] = -np.sin(rot_y)
    camera_pose = np.matmul(rot_udpim, camera_pose)

    camera_pose[0, 3] = 1.0
    camera_pose[1, 3] = 0.5
    camera_pose[2, 3] = -2.103

    return camera_pose
