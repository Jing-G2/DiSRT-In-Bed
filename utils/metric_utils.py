import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull

import utils.data_utils.angle_utils as angle_utils
from mesh_mesh_intersection import MeshMeshIntersection
from utils.viz_utils.lib_basic import smpl_male, smpl_female, get_smpl_trimesh
from utils.constants import DEVICE, SMPL_FEMALE, SMPL_MALE, POSE, FACES, BASE_PATH

TRANS_PREV_GT = np.identity(4)
TRANS_PREV_GT[0, 3] = -2.0
TRANS_NEXT_GT = np.identity(4)
TRANS_NEXT_GT[0, 3] = -2.0
TRANS_NEXT_GT[1, 3] = 1.0

TRANS_PREV_PD = np.identity(4)
TRANS_NEXT_PD = np.identity(4)
TRANS_NEXT_PD[1, 3] = 1.0

# ##################################################################
# ----------- Constants for Pressure Metrics ------------
# ##################################################################
PI = np.load(
    os.path.join(
        BASE_PATH, "BodyPressure", "data_BP", "parsed", "segmented_mesh_idx_faces.npy"
    ),
    allow_pickle=True,
).item()
EA1 = np.load(
    os.path.join(BASE_PATH, "BodyPressure", "data_BP", "parsed", "EA1.npy"),
    allow_pickle=True,
)
EA2 = np.load(
    os.path.join(BASE_PATH, "BodyPressure", "data_BP", "parsed", "EA2.npy"),
    allow_pickle=True,
)
FACES = FACES.squeeze(0).long()

head_top_face_idx = 435
head_top_bc = torch.tensor([0.0, 1.0, 0.0]).float().to(DEVICE)
left_heel_face_idx = 5975
left_heel_bc = torch.tensor([0.0, 0.0, 1.0]).float().to(DEVICE)
chest_face_index = 11885
chest_bcs = torch.tensor([0.0, 0.0, 1.0]).float().to(DEVICE)
belly_face_index = 6833
belly_bcs = torch.tensor([0.0, 0.0, 1.0]).float().to(DEVICE)
hips_face_index = 1341
hips_bcs = torch.tensor([0.0, 1.0, 0.0]).float().to(DEVICE)

isect_module = MeshMeshIntersection(max_collisions=256)


# ##################################################################
# metrics functions
# ##################################################################
def create_metric_dict(epoch=-1, PI=PI):
    """
    Create all metrics for evaluation
    3D Pose:
        3D 3MPJPE
        PVE
    3D Shape
        Height Error
        Chest Error
        Waist Error
        Hips Error
    """
    return {
        "epoch": epoch,
        "count": 0,
        "3D MPJPE": torch.tensor(0).float(),
        "PVE": torch.tensor(0).float(),
        "height": torch.tensor(0).float(),
        "chest": torch.tensor(0).float(),
        "waist": torch.tensor(0).float(),
        "hips": torch.tensor(0).float(),
    }


def get_verts_from_label(data_label):
    data_label = data_label.to("cpu").numpy()

    betas_gt = data_label[72:82]
    angles_gt = data_label[82:154]
    root_shift_est_gt = data_label[154:157]
    gender = data_label[157]
    root_shift_est_gt[1] *= -1
    root_shift_est_gt[2] *= -1

    R_root = angle_utils.matrix_from_dir_cos_angles(angles_gt[0:3])
    flip_root_euler = np.pi
    flip_root_R = angle_utils.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])
    angles_gt[0:3] = angle_utils.dir_cos_angles_from_matrix(
        np.matmul(flip_root_R, R_root)
    )

    if gender == 1:
        smpl_model = smpl_female
    else:
        smpl_model = smpl_male

    for beta in range(betas_gt.shape[0]):
        smpl_model.betas[beta] = betas_gt[beta]
    for angle in range(angles_gt.shape[0]):
        smpl_model.pose[angle] = angles_gt[angle]

    smpl_verts_gt = np.array(smpl_model.r)
    for s in range(root_shift_est_gt.shape[0]):
        smpl_verts_gt[:, s] += root_shift_est_gt[s] - float(
            smpl_model.J_transformed[0, s]
        )

    smpl_verts_gt[:, 1:2] *= -1

    return torch.tensor(smpl_verts_gt)


def get_verts_from_betas(betas, gender):
    if gender == 1:  # FEMALE
        smpl_model = SMPL_FEMALE
    else:  # MALE
        smpl_model = SMPL_MALE

    smpl_verts, _ = smpl_model(POSE, th_betas=betas.reshape(1, -1))
    return smpl_verts[0]


def get_anatomy_metrics(betas, gender):
    """Get the anatomy metrics (height, chest, waist, hips) from the body shape(betas) and geder
    Args:
        betas: torch.Tensor, the body shape parameters
        gender: 1-female, 0-male
    Returns:
        height: torch.Tensor, the height of the person
        chest: torch.Tensor, the chest perimeter
        waist: torch.Tensor, the waist perimeter
        hips: torch.Tensor, the hips perimeter
    """
    smpl_verts = get_verts_from_betas(betas, gender).unsqueeze(0)
    height, chest, waist, hips = compute_anatomy(smpl_verts)
    return height, chest, waist, hips


def get_gt_data(label, verts):
    """Get the ground truth or precicted joints from the label and the path
    Args:
        label: torch.Tensor, the label tensor
        verts: torch.Tensor, the predicted vertices
    Returns:
        a dictionary containing
            jtr: torch.Tensor, the joint positions
            height: torch.Tensor, the height of the person
            chest: torch.Tensor, the chest perimeter
            waist: torch.Tensor, the waist perimeter
            hips: torch.Tensor, the hips perimeter
            verts: torch.Tensor, the predicted vertices
            pmap: torch.Tensor, the predicted pressure map
            betas: torch.Tensor, the predicted betas
    """
    betas = label[72:82]
    gender = label[157]
    height, chest, waist, hips = get_anatomy_metrics(betas, gender)

    if len(verts) == 0:
        verts = torch.zeros(6890, 3)

    return {
        "jtr": label[:72].reshape(-1, 3),
        "height": height,
        "chest": chest,
        "waist": waist,
        "hips": hips,
        "verts": verts,
        "betas": betas,
    }


# ##################################################################
# calculate error
# ##################################################################
def get_errors(pred, gt):
    """calculate mesh metrics
    Args:
        pred: predicted values
        gt: ground truth values
    Returns:
        error: a dictionary containing the error values
    """
    error_jtr = (
        torch.norm(gt["jtr"] - pred["jtr"] * 1000.0, dim=1).mean(dim=0).to("cpu")
    )  # error in mm
    error_verts = (
        torch.norm(gt["verts"] - pred["verts"], dim=1).mean(dim=0).to("cpu") * 1000.0 # mm
    )
    error_height = F.l1_loss(gt["height"], pred["height"], reduction="none").to("cpu")
    error_chest = F.l1_loss(gt["chest"], pred["chest"], reduction="none").to("cpu")
    error_waist = F.l1_loss(gt["waist"], pred["waist"], reduction="none").to("cpu")
    error_hips = F.l1_loss(gt["hips"], pred["hips"], reduction="none").to("cpu")

    error = {
        "MPJPE": error_jtr,
        "PVE": error_verts,  # pose vertex error
        "height": error_height,
        "chest": error_chest,
        "waist": error_waist,
        "hips": error_hips,
    }

    return error


# ##################################################################
# save inference
# ##################################################################\
def save_pd_as_npy(cur_save_path, pd_betas, pd_jtr, pd_verts):
    """save the mesh parameters as .npy files
    Args:
        cur_save_path: path to save the inference
        pd_betas: predicted betas
        pd_jtr: predicted joint positions
        pd_verts: predicted vertices
    """
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)

    np.save(
        os.path.join(cur_save_path, f"pd_betas.npy"),
        pd_betas.to("cpu").numpy(),
    )
    np.save(
        os.path.join(cur_save_path, f"pd_vertices.npy"),
        pd_verts.to("cpu").numpy(),
    )
    np.save(
        os.path.join(cur_save_path, f"pd_jtr.npy"),
        pd_jtr.to("cpu").numpy() * 1000.0,  # convert to mm
    )


def save_mesh(cur_save_path, verts, is_gt=False, writer=None):
    """save the mesh as .obj file
    Args:
        cur_save_path: path to save the mesh
        verts: gt or predicted vertices
        is_gt: whether the mesh is ground truth
    """
    if not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path)

    render_pose = get_smpl_trimesh(verts.to("cpu").numpy(), 1)  # green - gt

    if writer is not None:  # use tensorboard
        name = cur_save_path.split("/")[-3:].join("_")
        if is_gt:
            name = f"{name}/gt"
        writer.add_mesh(
            name,
            vertices=torch.as_tensor(render_pose.vertices, dtype=torch.float).unsqueeze(
                0
            ),
            colors=torch.as_tensor(
                render_pose.visual.vertex_colors[:, :3], dtype=torch.int
            ).unsqueeze(0),
            faces=torch.as_tensor(render_pose.faces, dtype=torch.int).unsqueeze(0),
        )
    else:  # use wandb
        # save the mesh as .obj file
        name = "gt" if is_gt else "pred"
        render_pose.export(f"{cur_save_path}/{name}.obj", include_color=True)


# ##################################################################
# Code borrowed from SHAPY (https://github.com/muelea/shapy/blob/master/mesh-mesh-intersection/body_measurements/body_measurements.py)
# ##################################################################
def compute_height(triangles):
    head_top_tri = triangles[:, head_top_face_idx]
    head_top = (
        head_top_tri[:, 0, :] * head_top_bc[0]
        + head_top_tri[:, 1, :] * head_top_bc[1]
        + head_top_tri[:, 2, :] * head_top_bc[2]
    )
    head_top = (head_top_tri * head_top_bc.reshape(1, 3, 1)).sum(dim=1)
    left_heel_tri = triangles[:, left_heel_face_idx]
    left_heel = (left_heel_tri * left_heel_bc.reshape(1, 3, 1)).sum(dim=1)
    return torch.abs(head_top[:, 1] - left_heel[:, 1])


def get_plane_at_heights(height):
    device = height.device
    batch_size = height.shape[0]
    verts = (
        torch.tensor([[-1.0, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], device=device)
        .unsqueeze(dim=0)
        .expand(batch_size, -1, -1)
        .clone()
    )
    verts[:, :, 1] = height.reshape(batch_size, -1)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device, dtype=torch.long)
    return verts, faces, verts[:, faces]


def compute_peripheries(triangles):
    batch_size, num_triangles = triangles.shape[:2]
    device = triangles.device
    batch_indices = (
        torch.arange(batch_size, dtype=torch.long, device=device).reshape(-1, 1)
        * num_triangles
    )
    meas_data = {}
    meas_data["chest"] = (chest_face_index, chest_bcs)
    meas_data["waist"] = (belly_face_index, belly_bcs)
    meas_data["hips"] = (hips_face_index, hips_bcs)
    output = {}
    for name, (face_index, bcs) in meas_data.items():
        vertex = (triangles[:, face_index] * bcs.reshape(1, 3, 1)).sum(axis=1)
        _, _, plane_tris = get_plane_at_heights(vertex[:, 1])
        with torch.no_grad():
            collision_faces, collision_bcs = isect_module(plane_tris, triangles)
        selected_triangles = triangles.view(-1, 3, 3)[
            (collision_faces + batch_indices).view(-1)
        ].reshape(batch_size, -1, 3, 3)
        points = (
            (selected_triangles[:, :, None] * collision_bcs[:, :, :, :, None])
            .sum(axis=-2)
            .reshape(batch_size, -1, 2, 3)
        )
        np_points = points.detach().cpu().numpy()
        collision_faces = collision_faces.detach().cpu().numpy()
        collision_bcs = collision_bcs.detach().cpu().numpy()
        output[name] = {
            "points": [],
            "valid_points": [],
            "value": [],
            "plane_height": vertex[:, 1],
        }
        for ii in range(batch_size):
            valid_face_idxs = np.where(collision_faces[ii] > 0)[0]
            points_in_plane = np_points[
                ii,
                valid_face_idxs,
                :,
            ][
                :, :, [0, 2]
            ].reshape(-1, 2)
            hull = ConvexHull(points_in_plane)
            point_indices = hull.simplices.reshape(-1)
            hull_points = (
                points[ii][valid_face_idxs].view(-1, 3)[point_indices].reshape(-1, 2, 3)
            )
            meas_value = (
                (hull_points[:, 1] - hull_points[:, 0]).pow(2).sum(dim=-1).sqrt().sum()
            )
            output[name]["value"].append(meas_value)
        output[name]["tensor"] = torch.stack(output[name]["value"])
    return output


def compute_anatomy(verts):
    triangles = verts[:, FACES]
    height = (compute_height(triangles) * 100).to("cpu")
    peri = compute_peripheries(triangles)
    chest = (peri["chest"]["tensor"] * 100).to("cpu")
    waist = (peri["waist"]["tensor"] * 100).to("cpu")
    hips = (peri["hips"]["tensor"] * 100).to("cpu")
    return height, chest, waist, hips
