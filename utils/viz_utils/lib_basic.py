import colorsys
import os
import sys
import numpy as np
import trimesh
import pyrender

import matplotlib.cm as cm

from utils.constants import (
    BASE_PATH,
    X_BUMP,
    Y_BUMP,
)

base_path = os.path.join(BASE_PATH, "0-Pose")

sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "smpl"))

from smpl.smpl_webuser.serialization import load_model

smpl_female = load_model(
    os.path.join(base_path, "smpl_models", "smpl", "SMPL_FEMALE.pkl")
)
smpl_male = load_model(os.path.join(base_path, "smpl_models", "smpl", "SMPL_MALE.pkl"))

CIVIDIS = cm.get_cmap("cividis", 255)  # ugly
INFERNO = cm.get_cmap("inferno", 255)
VIRIDIS = cm.get_cmap("viridis", 255)
BONE = cm.get_cmap("bone", 255)  # Depth colormap
JET = cm.get_cmap("jet", 255)  # PM colormap
HOT = cm.get_cmap("hot", 255)  # IR colormap
MIDDLE_FILLER = 100
PERC_TOTAL = 1.0

CTRL_PNL = {
    "batch_size": 1,
    "verbose": False,
    "mesh_recon_map_labels": False,  # can only be true if we have 100% synthetic data for training
    "mesh_recon_map_labels_test": False,  # can only be true is we have 100% synth for testing
    "recon_map_input_est": False,  # do this if we're working in a two-part regression
    "x_y_offset_synth": [12, -35],  # [-7, -45]#[200, 200]#
    "clean_slp_depth": False,
}


def get_smpl_trimesh(smpl_verts, gender, vertex_colors=None):
    """Get the SMPL mesh"""
    smpl_verts[:, 0] += 0.0286
    smpl_verts[:, 1] += 0.0143
    smpl_verts_A = np.copy(smpl_verts)

    smpl_verts_A *= 64 * 0.0286 / 1.92
    smpl_verts_A[:, 0] += (1.92 - 64 * 0.0286) / 2
    smpl_verts_A[:, 1] += (0.84 - 27 * 0.0286) / 2

    if gender == 1:
        smpl_faces = np.array(smpl_female.f)
    else:
        smpl_faces = np.array(smpl_male.f)

    mesh = trimesh.base.Trimesh(
        vertices=np.array(smpl_verts_A), faces=smpl_faces, vertex_colors=vertex_colors
    )
    return mesh


# #################################################
# pyrender materials
# #################################################
class HumanMaterial:
    def __init__(self):

        self.mat_name_list = [
            "human_mat",
            "human_mat_3d",
            "human_mat_gt",
            "human_mat_GT",
            "human_arm_mat",
            "human_mat_for_study",
            "human_bed_for_study",
            "human_mat_D",
            "mesh_parts_mat_list",
            "artag_mat",
            "artag_mat_other",
        ]
        # color: (1.0, 1.0, 0.9), (0.412,0.663,1.0), (0.8, 0.3, 0.3, 1.0), colorsys.hsv_to_rgb(1/n, 0.5, 1.0)
        self.human_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.05, 0.5, 1.0),
            metallicFactor=0.2,
            alphaMode="OPAQUE",
        )  # [0.0, 0.0, 0.0, 0.0]
        self.human_mat_gt = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.8, 0.3, 0.3, 1.0),
            metallicFactor=0.2,
            alphaMode="OPAQUE",
        )
        self.human_mat_3d = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.05, 0.05, 0.8, 0.0],
            metallicFactor=0.0,
            roughnessFactor=0.5,
        )  # [0.0, 0.0, 0.0, 0.0]

        self.human_mat_GT = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.0, 0.3, 0.0, 0.0]
        )
        self.human_arm_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.1, 0.1, 0.8, 1.0]
        )
        self.human_mat_for_study = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.3, 0.3, 0.3, 0.5]
        )
        self.human_bed_for_study = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.2, 0.5]
        )
        self.human_mat_D = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.1, 0.1, 0.1, 1.0], alphaMode="BLEND"
        )

        mesh_color_mult = 0.25

        self.mesh_parts_mat_list = [
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 166.0 / 255.0,
                    mesh_color_mult * 206.0 / 255.0,
                    mesh_color_mult * 227.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 31.0 / 255.0,
                    mesh_color_mult * 120.0 / 255.0,
                    mesh_color_mult * 180.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 251.0 / 255.0,
                    mesh_color_mult * 154.0 / 255.0,
                    mesh_color_mult * 153.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 227.0 / 255.0,
                    mesh_color_mult * 26.0 / 255.0,
                    mesh_color_mult * 28.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 178.0 / 255.0,
                    mesh_color_mult * 223.0 / 255.0,
                    mesh_color_mult * 138.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 51.0 / 255.0,
                    mesh_color_mult * 160.0 / 255.0,
                    mesh_color_mult * 44.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 253.0 / 255.0,
                    mesh_color_mult * 191.0 / 255.0,
                    mesh_color_mult * 111.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 255.0 / 255.0,
                    mesh_color_mult * 127.0 / 255.0,
                    mesh_color_mult * 0.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 202.0 / 255.0,
                    mesh_color_mult * 178.0 / 255.0,
                    mesh_color_mult * 214.0 / 255.0,
                    0.0,
                ]
            ),
            pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[
                    mesh_color_mult * 106.0 / 255.0,
                    mesh_color_mult * 61.0 / 255.0,
                    mesh_color_mult * 154.0 / 255.0,
                    0.0,
                ]
            ),
        ]

        self.artag_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.3, 1.0, 0.3, 0.5]
        )
        self.artag_mat_other = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.1, 0.1, 0.1, 0.0]
        )

    def get(self, name: str):
        """Get the material by name"""
        if name in self.mat_name_list:
            return getattr(self, name)
        else:
            raise ValueError(f"Material {name} not found")


HUMAN_MATERIAL = HumanMaterial()


# #################################################
# markers: AR tags
# #################################################
class BoundryMarkers:
    def __init__(self, x_bump=0, y_bump=0):
        self.markers = {
            "2D": [
                [0.0, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "3D": [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        }

        self.artag_mat = HumanMaterial().get("artag_mat")
        self.artag_mat_other = HumanMaterial().get("artag_mat_other")

        self.x_bump = x_bump
        self.y_bump = y_bump

        # self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
        self.artag_r = np.array(
            [
                [0.0 + self.y_bump, 0.0 + self.x_bump, 0.075],
                [0.0286 * 64 * 1.04 + self.y_bump, 0.0 + self.x_bump, 0.075],
                [0.0 + self.y_bump, 0.01 + self.x_bump, 0.075],
                [0.0286 * 64 * 1.04 + self.y_bump, 0.01 + self.x_bump, 0.075],
                [0.0 + self.y_bump, 0.0 + self.x_bump, 0.075],
                [0.0 + self.y_bump, 0.0286 * 27 + self.x_bump, 0.075],
                [0.01 + self.y_bump, 0.0 + self.x_bump, 0.075],
                [0.01 + self.y_bump, 0.0286 * 27 + self.x_bump, 0.075],
                [0.0 + self.y_bump, 0.0286 * 27 + self.x_bump, 0.075],
                [
                    0.0286 * 64 * 1.04 + self.y_bump,
                    0.0286 * 27 + self.x_bump,
                    0.075,
                ],
                [0.0 + self.y_bump, 0.0286 * 27 + 0.01 + self.x_bump, 0.075],
                [
                    0.0286 * 64 * 1.04 + self.y_bump,
                    0.0286 * 27 + 0.01 + self.x_bump,
                    0.075,
                ],
                [0.0286 * 64 * 1.04 + self.y_bump, 0.0 + self.x_bump, 0.075],
                [
                    0.0286 * 64 * 1.04 + self.y_bump,
                    0.0286 * 27 + self.x_bump,
                    0.075,
                ],
                [0.0286 * 64 * 1.04 - 0.01 + self.y_bump, 0.0 + self.x_bump, 0.075],
                [
                    0.0286 * 64 * 1.04 - 0.01 + self.y_bump,
                    0.0286 * 27 + self.x_bump,
                    0.075,
                ],
            ]
        )
        # self.artag_f = np.array([[0, 1, 3], [3, 1, 0], [0, 2, 3], [3, 2, 0], [1, 3, 2]])
        self.artag_f = np.array(
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 2, 3],
                [1, 3, 2],
                [4, 5, 6],
                [4, 6, 5],
                [5, 6, 7],
                [5, 7, 6],
                [8, 9, 10],
                [8, 10, 9],
                [9, 10, 11],
                [9, 11, 10],
                [12, 13, 14],
                [12, 14, 13],
                [13, 14, 15],
                [13, 15, 14],
            ]
        )
        # self.artag_facecolors_root = np.array([[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
        self.artag_facecolors_root = np.array(
            [
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
                [0.0, 0.8, 0.8],
            ]
        )
        self.artag_facecolors_root_gt = (
            np.array(
                [
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1],
                ]
            )
            * 0.5
        )
        # self.artag_facecolors = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],])
        self.artag_facecolors = np.copy(self.artag_facecolors_root)
        self.artag_facecolors_gt = np.copy(self.artag_facecolors_root_gt)

    def get_artags_trimesh(self, viz_type="2D", shift_both_amount=0.0):
        """create mini meshes for AR tags"""
        artag_trimeshes = []
        markers = self.markers[viz_type]
        if viz_type == "3D":
            shift_both_amount = -markers[2][1]

        for marker in markers:
            if markers[2] is None:
                artag_trimeshes.append(None)
            elif marker is None:
                artag_trimeshes.append(None)
            else:
                if marker is markers[2] and viz_type == "3D":
                    artag_trimesh = trimesh.base.Trimesh(
                        vertices=self.artag_r,
                        faces=self.artag_f,
                        face_colors=self.artag_facecolors_root,
                    )
                    artag_trimeshes.append(artag_trimesh)
                else:
                    artag_trimesh = trimesh.base.Trimesh(
                        vertices=self.artag_r + [0.0, shift_both_amount, 0.0],
                        faces=self.artag_f,
                        face_colors=self.artag_facecolors,
                    )
                    artag_trimeshes.append(artag_trimesh)

        return artag_trimeshes
