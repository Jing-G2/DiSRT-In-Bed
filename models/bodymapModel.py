import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger
from utils.constants import DEVICE
from utils.data_utils.trans_utils import get_descaling_func
from utils.train_utils.file_utils import get_model_path_in_directory

from models.module import get_encoder, MLPRegressor, MeshEstimator


class PMMModel(nn.Module):

    def __init__(
        self,
        image_size,
        feature_channels,
        param_size,
        vertex_size,
        batch_size,
        modality,
    ):
        super(PMMModel, self).__init__()
        # get the encoder
        self.channel = len(modality)
        self.encoder, self.encoder_out_channels, self.encoder_out_hw = get_encoder(
            image_size=image_size,
            channel=self.channel,
            feature_channels=feature_channels,
            param_size=param_size,
        )

        # get the regressor
        self.regressor = MLPRegressor(
            in_channels=self.encoder_out_channels,
            out_channels=param_size,
        )

        # get the mesh model
        self.mesh_model = MeshEstimator(batch_size).to(DEVICE)

    def forward(
        self,
        batch_gender,
        img,
    ):
        """Forward pass of the model, the regressor will redict the smpl parameters from input depth image
            Estimated params - shape - [B, 88]
            0-10    Body Shape (10)
            10-13   root xyz (3)
            13-19   root angle (atan2 3)
            19-88   joint angles (69)
        Args:
            batch_gender (tensor): [B, 2]
            img (tensor): input image [B, 2, C, H, W]
        Returns:
            mesh_pred (tensor): mesh predictions dict of [B, N, 3]
            recon_raw_depth (tensor): reconstructed depth image [B, 1, H, W]
            diffusion_result (dict): diffusion results
        """
        # get the feature from the encoder
        feature = self.encoder(img[:, 1])

        # get the smpl parameters from the regressor
        assert feature.shape[1] == 512
        smpl_param_pred = self.regressor(feature)  # [B, 88]

        mesh_pred = self.mesh_model(batch_gender, smpl_param_pred)

        return mesh_pred

    def mesh_infer_gt(self, batch_gender, smpl_gt):
        with torch.no_grad():
            smpl_gt = smpl_gt.to(DEVICE)
            mesh_pred = self.mesh_model(batch_gender, smpl_gt, is_gt=True)  # [B, N, 3]

        return mesh_pred

    def load_model(self, save_dir, encoder_epoch=-1, regressor_epoch=-1):
        """Load all models inside from the given path
        Args:
            save_dir (str): directory to save the models
                -checkpoints (save_dir)
                    |-encoder
                    |-regressor
                    |-diffusion
            encoder_epoch (int): epoch number of the encoder
            decoder_epoch (int): epoch number of the decoder
            regressor_epoch (int): epoch number of the regressor
        """
        encoder_path, self.encoder_epoch = get_model_path_in_directory(
            os.path.join(save_dir, "encoder"), encoder_epoch
        )
        regressor_path, self.regressor_epoch = get_model_path_in_directory(
            os.path.join(save_dir, "regressor"), regressor_epoch
        )

        logger.log(f"Load models from {save_dir}")
        logger.log(f"Load encoder from {encoder_path}")
        logger.log(f"Load regressor from {regressor_path}")

        self.encoder.load_state_dict(torch.load(encoder_path))
        self.regressor.load_state_dict(torch.load(regressor_path))

        return self.regressor_epoch

    def save_model(self, save_dir, epoch):
        """Save all models inside to the given path separately
        Args:
            save_dir (str): directory to save the models
                -checkpoints (save_dir)
                    |-encoder
                    |-decoder
                    |-regressor
                    |-diffusion
            epoch (int): epoch number
        """
        if self.encoder.training:
            torch.save(
                self.encoder.state_dict(),
                os.path.join(save_dir, "encoder", f"model_{epoch}.pt"),
            )
        if self.regressor.training:
            torch.save(
                self.regressor.state_dict(),
                os.path.join(save_dir, "regressor", f"model_{epoch}.pt"),
            )
