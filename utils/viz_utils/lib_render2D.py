import sys
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import torch
import numpy as np
import imageio
import pyrender
import trimesh
import matplotlib.cm as cm

import utils.data_utils.data_utils as data_utils

from scipy.ndimage.interpolation import zoom

from utils.constants import *
from utils.viz_utils.lib_basic import *
from utils.viz_utils.camera import get_slp_2D_camera_pose


# ##################################################################
# ----------------- get functions -----------------
# ##################################################################
def get_RGB_refs(p_idx, cover_type, pose_idx, dsFd):
    if p_idx < 1 or pose_idx < 1:
        raise ValueError("p_idx and pose_idx should be greater than 0")

    rgb_uncover = data_utils.get_array_A2B(
        p_id=p_idx, cover="uncover", pose_id=pose_idx, modA="RGB", modB="PM", dsFd=dsFd
    )
    rgb_uncover = data_utils.zoom_color_image(
        color_arr=rgb_uncover, image_zoom=4.585 / 2
    )

    rgb = data_utils.get_array_A2B(
        p_id=p_idx, cover=cover_type, pose_id=pose_idx, modA="RGB", modB="PM", dsFd=dsFd
    )
    rgb = data_utils.zoom_color_image(color_arr=rgb, image_zoom=4.585 / 2)

    return rgb_uncover, rgb


def get_depth_image(depth, cmap_func=BONE):
    if len(depth.shape) == 3:
        # pyrender
        depth = depth.squeeze(-1)
        depth = zoom(depth, 3.435, order=1)

    depth -= 1700
    depth = depth.astype(float) / 500.0
    depth = cmap_func(np.clip(depth, a_min=0, a_max=1))[:, :, 0:3] * 255.0
    depth = depth.astype(np.uint8)
    return depth


def get_pressure_image(pressure):
    pressure = pressure.squeeze(-1)

    pressure = zoom(pressure, (3.435 * 2, 3.435 * 2), order=0)
    pressure = (
        np.clip((JET(pressure / 40)[:, :, 0:3] + 0.1), a_min=0, a_max=1) * 255
    ).astype(np.uint8)
    return pressure


def get_ir_image(ir):
    ir = ir.squeeze(-1)

    ir = zoom(ir, (3.435 * 2, 3.435 * 2.01), order=0)
    ir -= 29000  # max 30878.859375
    ir = ir.astype(float) / 2000.0
    ir = HOT(np.clip(ir, a_min=0, a_max=1))[:, :, 0:3] * 255.0
    ir = ir.astype(np.uint8)
    return ir


def get_pmap_colors(pmap):
    verts_color_jet = np.clip(
        cm.jet(pmap.to("cpu").numpy() / 30.0)[:, 0:3], a_min=0.0, a_max=1.0
    )
    return torch.tensor(verts_color_jet).float().to(DEVICE)


# ##################################################################
# ----------------- render functions -----------------
# ##################################################################
def render_scene_image(mesh, camera_angle=0, is_gt=False):
    lval = 50
    scene = pyrender.Scene(ambient_light=(lval, lval, lval))
    human_material_name = "human_mat_gt" if is_gt else "human_mat"
    mesh = pyrender.Mesh.from_trimesh(
        mesh, material=HUMAN_MATERIAL.get(human_material_name), smooth=True
    )
    scene.add(mesh)

    camera_pose = get_slp_2D_camera_pose(camera_angle)

    magnify = (1 + MIDDLE_FILLER / 880) * (64 * 0.0286) / PERC_TOTAL

    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=10.0,
        innerConeAngle=np.pi / 10.0,
        outerConeAngle=np.pi / 2.0,
    )
    light_pose = np.copy(camera_pose)
    light_pose[0, 3] = 0.8
    light_pose[1, 3] = -0.5
    light_pose[2, 3] = -2.5

    light_pose2 = np.copy(camera_pose)
    light_pose2[0, 3] = 2.5
    light_pose2[1, 3] = 1.0
    light_pose2[2, 3] = -5.0

    light_pose3 = np.copy(camera_pose)
    light_pose3[0, 3] = 1.0
    light_pose3[1, 3] = 5.0
    light_pose3[2, 3] = -4.0

    scene.add(light, pose=light_pose)
    scene.add(light, pose=light_pose2)
    scene.add(light, pose=light_pose3)

    camera = pyrender.OrthographicCamera(xmag=magnify, ymag=magnify)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
    # light = pyrender.PointLight(color=np.ones(3), intensity=50.0)
    light_pose = np.copy(camera_pose)
    light_pose[1, 3] = 2.5
    light_pose[2, 3] = -3.0

    for i in range(0, 361, 30):
        light_rot1 = trimesh.transformations.rotation_matrix(
            angle=np.radians(i), direction=[1, 0, 0], point=[0.4, 0, 0]
        )
        light_rot2 = trimesh.transformations.rotation_matrix(
            angle=np.radians(i + 15), direction=[1, 0, 0], point=[0.1, 0, 0]
        )
        light_pose[0, 3] = 1.0
        scene.add(light, pose=np.dot(light_rot1, light_pose))
        light_pose[0, 3] = 4.0
        scene.add(light, pose=np.dot(light_rot2, light_pose))

    r = pyrender.OffscreenRenderer(600, 880 + MIDDLE_FILLER)
    render, depth = r.render(scene)

    return render


def render_scene_video(mesh):
    renders = []
    for rot in range(0, 360 * 2 + 5, 10):
        renders.append(render_scene_image(mesh, rot))
    return renders
