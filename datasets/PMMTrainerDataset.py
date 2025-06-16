import numpy as np
import os

import torch
from torch.utils.data import Dataset

from utils import logger
from utils.constants import *
import utils.data_utils.data_utils as data_utils
import utils.data_utils.trans_utils as trans_utils
from datasets.BPDataset import BPDataset
from datasets.SLPDataset import SLPDataset
from datasets.HospitalDataset import SLPHospitalDataset


class PMMTrainerDataset(Dataset):

    def __init__(
        self,
        data_path,
        data_files,
        syn_ratio, # 1.0
        transforms,
        exp_type,  # "overfit" | "normal" | "hospital"
        is_train,
        modality,
        pressure_scaling,  # "minmax" | "sigmoid" | "none"
        ir_scaling,  # "minmax" | "sigmoid" | "none"
        depth_scaling,  # "minmax" | "sigmoid" | "none"
        is_affine,
        uncover_only,
    ):
        super(PMMTrainerDataset, self).__init__()
        self.data_path = data_path
        self.syn_ratio = syn_ratio
        self.transforms = transforms
        self.exp_type = exp_type
        self.is_train = is_train
        self.modality = modality

        self.pressure_scaling_func = trans_utils.get_scaling_func(
            scaling=pressure_scaling, modality="pressure"
        )
        self.pressure_scaling = pressure_scaling
        self.ir_scaling_func = trans_utils.get_scaling_func(
            scaling=ir_scaling, modality="ir"
        )
        self.ir_scaling = ir_scaling
        self.depth_scaling_func = trans_utils.get_scaling_func(
            scaling=depth_scaling, modality="depth"
        )
        self.depth_scaling = depth_scaling
        if self.depth_scaling == "bodymap":
            logger.log("Scaling: using Bodymap scaling for depth")

        self.is_affine = is_affine
        self.load_verts = not is_train
        self.uncover_only = uncover_only

        self._prepare_dataset(data_files)

    def _concat_data_returns(self, data_returns):

        self.data_pressure_x = data_utils.concatenate(
            self.data_pressure_x, data_returns["data_pressure_x"]
        )
        self.data_ir_x = data_utils.concatenate(
            self.data_ir_x, data_returns["data_ir_x"]
        )
        self.data_depth_x = data_utils.concatenate(
            self.data_depth_x, data_returns["data_depth_x"]
        )
        self.data_label_y = data_utils.concatenate(
            self.data_label_y, data_returns["data_label_y"]
        )
        self.data_pmap_y = data_utils.concatenate(
            self.data_pmap_y, data_returns["data_pmap_y"]
        )
        self.data_verts_y = data_utils.concatenate(
            self.data_verts_y, data_returns["data_verts_y"]
        )
        self.data_names_y = data_utils.concatenate(
            self.data_names_y, data_returns["data_names_y"]
        )

    def _purge_for_overfitting(self, use_till):
        self.data_pressure_x = (
            self.data_pressure_x[:use_till, ...]
            if self.data_pressure_x is not None
            else None
        )
        self.data_ir_x = (
            self.data_ir_x[:use_till, ...] if self.data_ir_x is not None else None
        )
        self.data_depth_x = self.data_depth_x[:use_till, ...]
        self.data_label_y = self.data_label_y[:use_till, ...]
        self.data_pmap_y = (
            self.data_pmap_y[:use_till, ...] if self.data_pmap_y is not None else None
        )
        self.data_verts_y = self.data_verts_y[:use_till, ...]
        self.data_names_y = self.data_names_y[:use_till, ...]

    def _prepare_syn_data(self, data_file):
        cover_str_list = ["uncover"] if self.uncover_only else ["uncover", "cover1"]
        data_returns = []
        data_lines = data_utils.load_data_lines(data_file)
        data_returns = BPDataset(self.data_path).prepare_dataset(
            data_lines=data_lines,
            cover_str_list=cover_str_list,
            syn_ratio=self.syn_ratio,
            load_verts=self.load_verts,
            load_pressure=False,
            for_infer=(self.exp_type == "overfit"),
        )

        self._concat_data_returns(data_returns)

    def _prepare_slp_data(self, data_file):
        cover_str_list = (
            ["uncover"] if self.uncover_only else ["uncover", "cover1", "cover2"]
        )
        data_returns = []
        data_lines = data_utils.load_data_lines(data_file)
        if self.exp_type == "hospital":
            self.bed_height = SLP_SIMLAB_BED
            data_returns = SLPHospitalDataset().prepare_dataset(
                data_lines=data_lines,
                mod=self.modality,
                cover_str_list=cover_str_list,
            )
        else:
            self.bed_height = SLP_DANALAB_BED
            data_returns = SLPDataset(self.data_path).prepare_dataset(
                data_lines=data_lines,
                mod=self.modality,
                cover_str_list=cover_str_list,
                load_verts=self.load_verts,
                for_infer=(self.exp_type == "overfit"),
            )
        self._concat_data_returns(data_returns)

    def _prepare_dataset(self, data_files):
        real_file, synth_file = data_files
        self.data_pressure_x = None
        self.data_ir_x = None
        self.data_depth_x = None
        self.data_label_y = None
        self.data_pmap_y = None
        self.data_verts_y = None
        self.data_names_y = None

        if real_file is not None:
            self._prepare_slp_data(real_file)
        else:
            logger.log("No real data used")

        if synth_file is not None:
            self._prepare_syn_data(synth_file)
        else:
            logger.log("No synth data used")

        if self.exp_type == "overfit":
            self._purge_for_overfitting(USE_TILL)

        """ permute the data from NSHWC to NSCHW , S: Covering State """
        if self.data_pressure_x is not None:
            self.data_pressure_x = (
                torch.tensor(self.data_pressure_x).float().permute(0, 1, 4, 2, 3)
            )
            self.data_pmap_y = torch.tensor(self.data_pmap_y).float()

        if self.data_ir_x is not None:
            self.data_ir_x = torch.tensor(self.data_ir_x).float().permute(0, 1, 4, 2, 3)

        self.data_depth_x = (
            torch.tensor(self.data_depth_x).float().permute(0, 1, 4, 2, 3)
        )
        self.data_label_y = torch.tensor(self.data_label_y).float()
        self.data_verts_y = torch.tensor(self.data_verts_y).float()

        logger.log(
            "Mixed Training Data:",
            "data_pressure_x",
            self.data_pressure_x.shape if self.data_pressure_x is not None else None,
            "data_ir_x",
            self.data_ir_x.shape if self.data_ir_x is not None else None,
            "data_depth_x",
            self.data_depth_x.shape,
            "data_label_y",
            self.data_label_y.shape,
            "data_pmap_y",
            self.data_pmap_y.shape if self.data_pmap_y is not None else None,
            "data_verts_y",
            self.data_verts_y.shape,
            "data_names_y",
            self.data_names_y.shape,
        )

    def _apply_transforms(self, label, images, ori_images):
        if self.is_train and self.is_affine:
            # rotate the image and pose randomly to augment the data
            images.extend(ori_images)
            label[82:154], images = trans_utils.RandomRotate(label[82:154], images)
            ori_images = images[6:]
            images = images[:6]

        if images is not None:
            for transform_fn in self.transforms:
                images = transform_fn(images)

        images_trans = images + ori_images
        images_trans = [
            (image.float() if image is not None else MISS_MODALITY_FLAG)
            for image in images_trans
        ]  # -1 denotes that the image is None

        return label.clone().detach().float(), images_trans

    def __len__(self):
        return self.data_depth_x.shape[0]

    def __getitem__(self, index):
        """
        Label contents
        0-72    3D Marker Positions
        72-82   Body Shape Params
        82-154  Joint Angles
        154-157 Root XYZ Shift
        157-159 Gender
        159     [1]
        160     Body Mass
        161     Body Height
        """
        """
        Returns --- 
        pressure_image
        depth_image 
        label 
        pmap 
        verts
        file_name
        """
        label = self.data_label_y[index]
        file_name = self.data_names_y[index]
        images = []
        ori_images = []

        if self.data_depth_x is not None:
            ori_depth_image = self.data_depth_x[index].clone()
            if self.depth_scaling == "bodymap":
                depth_image = ori_depth_image.clone()
                if file_name.startswith("synth"):
                    depth_image /= MAX_DEPTH_SYNTH
                else:
                    if not self.exp_type == "hospital":
                        depth_image /= MAX_DEPTH_REAL
                    else:
                        depth_image /= MAX_DEPTH_REAL + SLP_SIMLAB_BED - SLP_DANALAB_BED
            else:
                # if not file_name.startswith("synth"):
                # ori_depth_image += MAX_DEPTH_SYNTH - ori_depth_image.max() # 1
                # ori_depth_image += MAX_DEPTH_SYNTH - SLP_DANALAB_BED  # 2
                # ori_depth_image += MAX_DEPTH_SYNTH - MAX_DEPTH_REAL # 3
                # ori_depth_image += MAX_DEPTH_SYNTH - ori_depth_image.max() # 4

                # # 5
                # if not file_name.startswith("synth"):
                #     ori_depth_image += MAX_DEPTH_SYNTH - SLP_DANALAB_BED
                # ori_depth_image += MAX_DEPTH_SYNTH - ori_depth_image.max()

                # 6
                if not file_name.startswith("synth"):  # real data
                    ori_depth_image += MAX_DEPTH_SYNTH - self.bed_height
                else:
                    ori_depth_image += MAX_DEPTH_SYNTH - ori_depth_image.max()

                ori_depth_image = ori_depth_image.clamp(min=0)
                depth_image = ori_depth_image.clone()

                depth_image = self.depth_scaling_func(depth_image)  # Scaling to [0-1]

            images.append(depth_image[0])
            images.append(depth_image[1])
            ori_images.append(ori_depth_image[0])
            ori_images.append(ori_depth_image[1])
        else:
            images.extend([None, None])
            ori_images.extend([None, None])

        if self.data_pressure_x is not None:
            ori_pressure_image = self.data_pressure_x[index].clone()
            pressure_image = self.data_pressure_x[index].clone()
            if self.pressure_scaling == "bodymap":
                if file_name.startswith("synth"):
                    pressure_image /= MAX_PRESSURE_SYNTH
                else:
                    pressure_image /= MAX_PRESSURE_REAL
            else:
                pressure_image = self.pressure_scaling_func(
                    pressure_image
                )  # Scaling to 0-1
            images.append(pressure_image[0])
            images.append(pressure_image[1])
            ori_images.append(ori_pressure_image[0])
            ori_images.append(ori_pressure_image[1])

            pmap = self.data_pmap_y[index]
        else:
            images.extend([None, None])
            ori_images.extend([None, None])
            pmap = np.array([])

        if self.data_ir_x is not None:  # only SLP
            ori_ir_image = self.data_ir_x[index].clone()
            ir_image = self.data_ir_x[index].clone()
            if self.ir_scaling == "bodymap":
                ir_image = ori_ir_image / MAX_IR_REAL
            else:
                ir_image = self.ir_scaling_func(ir_image)  # Scaling to 0-1
            images.append(ir_image[0])
            images.append(ir_image[1])
            ori_images.append(ori_ir_image[0])
            ori_images.append(ori_ir_image[1])
        else:
            images.extend([None, None])
            ori_images.extend([None, None])

        label, images_trans = self._apply_transforms(
            label=label,
            images=images,
            ori_images=ori_images,
        )

        depth_image = torch.stack(images_trans[0:2])
        pressure_image = torch.stack(images_trans[2:4])
        ir_image = torch.stack(images_trans[4:6])

        ori_depth_image = torch.stack(images_trans[6:8])
        ori_pressure_image = torch.stack(images_trans[8:10])
        ori_ir_image = torch.stack(images_trans[10:12])

        try:
            verts = self.data_verts_y[index] if self.load_verts else torch.tensor([])
        except:
            verts = torch.tensor([])

        return (
            ori_pressure_image,  # SCHW
            pressure_image,  # SCHW
            ori_ir_image,  # SCHW
            ir_image,  # SCHW
            ori_depth_image,  # SCHW
            depth_image,  # SCHW
            label,
            pmap,
            verts,
            file_name,
        )
