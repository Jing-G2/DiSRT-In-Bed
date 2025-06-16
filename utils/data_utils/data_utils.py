import os
import cv2
import numpy as np
import pickle as pkl
from scipy.ndimage.interpolation import zoom

from utils.constants import SLP_PATH, SLP_RAW_IMG_SIZE


# #########################################################
# data loading helper functions
# #########################################################
def load_pickle(filename):
    try:
        with open(filename, "rb") as f:
            return pkl.load(f, encoding="latin1")
    except:
        print(f"Error in {filename}")
        return None


def load_data_lines(data_lines_file):
    with open(data_lines_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def concatenate(trg, src):
    if trg is not None and src is not None:
        return np.concatenate((trg, src))
    elif trg is not None:
        return trg
    elif src is not None:
        return src
    else:
        return None


# #########################################################
# SLP data helper functions
# #########################################################
def uni_mod(mod):
    """
    unify the mod name, so depth and depth raw both share depth related geometry parameters, such as resolution and homography
    :param mod:
    :return:
    """
    if "depth" in mod:
        mod = "depth"
    if "IR" in mod:
        mod = "IR"
    if "PM" in mod:
        mod = "PM"
    return mod


def genPTr_dict(subj_li, mod_li, dsFd=os.path.join(SLP_PATH, "danaLab")):
    """
    loop idx_li, loop mod_li then generate dictionary {mod[0]:PTr_li[...], mod[1]:PTr_li[...]}
    history: 6/3/20: add 'PM' as eye matrix for simplicity
    :param subj_li: base 1 person id list
    :param mod_li: modality list
    :return: PTr_dct_li_src: a dict of list of PTr
    """
    PTr_dct_li_src = {}  # a dict
    for modNm in mod_li:  # initialize the dict_li
        PTr_dct_li_src[modNm] = []  # make empty list  {md:[], md2:[]...}
    for i in subj_li:
        for mod in mod_li:  # add mod PTr
            mod = uni_mod(mod)  # clean
            if "PM" not in mod:
                pth_PTr = os.path.join(
                    dsFd, "{:05d}".format(i), "align_PTr_{}.npy".format(mod)
                )
                PTr = np.load(pth_PTr)
            else:
                PTr = np.eye(3)  # fill PM with identical matrix
            PTr_dct_li_src[mod].append(PTr)
    return PTr_dct_li_src


def get_PTr_A2B(
    p_id=0, modA="IR", modB="depthRaw", dsFd=os.path.join(SLP_PATH, "danaLab")
):
    """
    get PTr from A2B
    :param p_id: base 1 person id
    :param modA: source mod
    :param modB: target mod
    :param dsFd: dataset folder path to "danaLab" or "simLab"
    :return:
    """
    assert p_id > 0, "p_id(p_id) should be base 1"
    modA = uni_mod(modA)
    modB = uni_mod(modB)
    PTrA = genPTr_dict([p_id], [modA], dsFd)[modA][0]
    PTrB = genPTr_dict([p_id], [modB], dsFd)[modB][0]

    PTr_A2B = np.dot(np.linalg.inv(PTrB), PTrA)
    PTr_A2B = PTr_A2B / np.linalg.norm(PTr_A2B)  # normalize

    return PTr_A2B


def get_array_A2B(
    p_id,
    cover,
    pose_id,
    modA="IR",
    modB="depthRaw",
    dsFd=os.path.join(SLP_PATH, "danaLab"),
):
    """
    Get array A after project to B space.
    :param p_id: base 1 person id
    :param cover: cover name
    :param pose_id: pose id base 1
    :param modA: source mod
    :param modB: target mod
    :param dsFd: dataset folder path to "danaLab" or "simLab"
    :return:
    """
    if modA == "RGB":
        arr = cv2.imread(
            os.path.join(
                dsFd,
                "{:05d}".format(p_id),
                modA,
                cover,
                f"image_{(pose_id):06d}.png",
            ),
        )
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        arr = np.load(
            file=os.path.join(
                dsFd,
                "{:05d}".format(p_id),
                modA,
                cover,
                f"{(pose_id):06d}.npy",
            )
        )

    PTr_A2B = get_PTr_A2B(p_id=p_id, modA=modA, modB=modB, dsFd=dsFd)
    modB = uni_mod(modB)  # the unified name
    dst = cv2.warpPerspective(arr, PTr_A2B, SLP_RAW_IMG_SIZE[modB])
    return dst


def zoom_color_image(color_arr, image_zoom=4.585 / 2):
    color_arr_r = zoom(color_arr[:, :, 0], image_zoom, order=1)
    color_arr_g = zoom(color_arr[:, :, 1], image_zoom, order=1)
    color_arr_b = zoom(color_arr[:, :, 2], image_zoom, order=1)
    color_arr = np.stack((color_arr_r, color_arr_g, color_arr_b), axis=2)
    color_arr = color_arr[:, 1:-1]
    return color_arr


def pixel2cam(pixel_coord, f, c):
    """Convert pixel coordinate to camera coordinate."""
    pixel_coord = pixel_coord.astype(float)
    jt_cam = np.zeros_like(pixel_coord)
    jt_cam[..., 0] = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    jt_cam[..., 1] = (pixel_coord[..., 1] - c[1]) / (f[1]) * pixel_coord[..., 2]
    jt_cam[..., 2] = pixel_coord[..., 2]

    return jt_cam
