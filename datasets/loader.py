import cv2
import numpy as np
import os

import torch
from torch.utils.data import DataLoader

from utils.constants import *
import utils.data_utils.trans_utils as trans_utils
from datasets.PMMTrainerDataset import PMMTrainerDataset


def prepare_transforms(image_size, is_affine, is_erase, is_train=False):
    transforms = [lambda x: trans_utils.Resize(x, image_size)]
    if is_train:
        if is_affine:
            transforms.append(trans_utils.RandomAffine)
        if is_erase:
            # transforms.append(trans_utils.RandomBlur)
            transforms.append(trans_utils.RandomErase)
            transforms.append(trans_utils.RandomNoise)
    return transforms


def prepare_loader(
    data_path,
    data_files,
    syn_ratio,
    batch_size,
    image_size,
    exp_type,
    is_train,
    modality,
    pressure_scaling,
    ir_scaling,
    depth_scaling,
    is_affine,
    is_erase,
    uncover_only,
):
    if all([file is None for file in data_files]):
        return None, None

    data_transforms = prepare_transforms(
        image_size=image_size,
        is_affine=is_affine,
        is_erase=is_erase,
        is_train=is_train,
    )

    dataset = PMMTrainerDataset(
        data_path=data_path,
        data_files=data_files,
        syn_ratio=syn_ratio,
        transforms=data_transforms,
        exp_type=exp_type,
        is_train=is_train,
        modality=modality,
        pressure_scaling=pressure_scaling,
        ir_scaling=ir_scaling,
        depth_scaling=depth_scaling,
        is_affine=is_affine,
        uncover_only=uncover_only,
    )

    # shuffle, drop_last should be True for training case
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=4,
    )
    return loader, dataset


def prepare_dataloaders(
    data_path,
    train_files,
    val_files,
    syn_ratio,
    batch_size,
    image_size,
    exp_type,
    is_train,
    modality,
    pressure_scaling,
    ir_scaling,
    depth_scaling,
    is_affine,
    is_erase,
    uncover_only,
):
    print("Starting train dataset prepare")
    train_loader, train_dataset = prepare_loader(
        data_path=data_path,
        data_files=train_files,
        syn_ratio=syn_ratio,
        batch_size=batch_size,
        image_size=image_size,
        exp_type=exp_type,
        is_train=is_train,
        modality=modality,
        pressure_scaling=pressure_scaling,
        ir_scaling=ir_scaling,
        depth_scaling=depth_scaling,
        is_affine=is_affine,
        is_erase=is_erase,
        uncover_only=uncover_only,
    )
    print("Prepared train dataset")
    print()

    print("Starting val dataset prepare")
    val_loader, val_dataset = prepare_loader(
        data_path=data_path,
        data_files=val_files,
        syn_ratio=1.0,
        batch_size=batch_size,
        image_size=image_size,
        exp_type=exp_type,
        is_train=False,
        modality=modality,
        pressure_scaling=pressure_scaling,
        ir_scaling=ir_scaling,
        depth_scaling=depth_scaling,
        is_affine=False,
        is_erase=False,
        uncover_only=uncover_only,
    )
    print("Prepared val dataset")
    print()

    return train_loader, val_loader, train_dataset, val_dataset
