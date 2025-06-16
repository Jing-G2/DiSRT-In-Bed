import os
import numpy as np
import torch

# Set your base path which has the file structure as shown in the readme
BASE_PATH = "../"
SLP_PATH = os.path.join(BASE_PATH, "BodyPressure/data_BP/SLP")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPUS_PER_NODE = 1

# data sigmoid transformation parameters
K_DEPTH = 0.005
K_PRESSURE = 0.1
K_IR = 0.005

# BP dataset
MAX_PRESSURE_SYNTH = 100.0  # 170.0 # synth
MAX_DEPTH_SYNTH = 2215.0  # synth
MAX_PMAP_SYNTH = 100.0
MIN_PRESSURE_SYNTH = 0.0
MIN_DEPTH_SYNTH = 1500.0
MIN_PMAP_SYNTH = 0.0
BP_BED_HEIGHT = MAX_DEPTH_SYNTH  # accurate 2121

# SLP dataset
SLP_SIMLAB_BED = 2264.0
SLP_DANALAB_BED = 2101.0
SLP_RAW_IMG_SIZE = {
    "RGB": [576, 1024],
    "depth": [424, 512],
    "PM": [84, 192],
    "IR": [120, 160],
}
SLP_IMG_DATA_SIZE = {
    "depth": [128, 54],
    "PM": [64, 27],
}
SLP_CAM_PARAMS = {
    "camera_center": [38.207363, 179.873],  # from pkl file
    "camera_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "camera_translation": [0.0, 0.0, 0.0],
    "f_d": [367.8, 367.8],
    "c_d": [208.1, 259.7],  # z/f = x_m/x_p so m or mm doesn't matter
}

MAX_DEPTH_REAL = 2575.0  # real
MAX_PRESSURE_REAL = 100.0  # 620.0 # real
MAX_IR_REAL = 30797.0  # real
MIN_IR_REAL = 29500.0  # lower bound
MAX_PMAP_REAL = 100.0

# inference parameters
USE_TILL = 1024
USE_FOR_INFER = 5

# data loader parameters
MISS_MODALITY_FLAG = torch.tensor(-1).int()

# visualization parameters
X_BUMP = -0.0143 * 2
Y_BUMP = -(0.0286 * 64 * 1.04 - 0.0286 * 64) / 2 - 0.0143 * 2


from smplpytorch.pytorch.smpl_layer import SMPL_Layer

SMPL_FEMALE = SMPL_Layer(
    center_idx=0,
    gender="female",
    model_root="./smpl_models/smpl",
).to(DEVICE)

SMPL_MALE = SMPL_Layer(
    center_idx=0,
    gender="male",
    model_root="./smpl_models/smpl",
).to(DEVICE)

POSE = torch.tensor(SMPL_FEMALE.smpl_data["pose"].r).unsqueeze(0).float().to(DEVICE)

FACES = SMPL_FEMALE.th_faces.unsqueeze(0).int().to(DEVICE)