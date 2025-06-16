import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logger
from utils.constants import DEVICE
from utils.data_utils.trans_utils import get_descaling_func
from utils.train_utils.file_utils import get_model_path_in_directory

from models.module import get_encoder, get_decoder, ConvRegressor, MeshEstimator


class PMMModel(nn.Module):

    def __init__(
        self,
        image_size,
        feature_channels,
        param_size,
        vertex_size,
        batch_size,
        modality,
        recon_channel_mult="1,2,4,8",
        use_dropout=False,
        norm_layer="batch",
        padding_type="reflect",
        encoder_net="resnet18",
        recon_num_res_blocks=2,
        use_recon=False,
    ):
        super(PMMModel, self).__init__()
        # get the encoder
        self.channel = len(modality)
        self.encoder, self.encoder_out_channels, self.encoder_out_hw = get_encoder(
            image_size=image_size,
            channel=self.channel,
            feature_channels=feature_channels,
            param_size=param_size,
            recon_channel_mult=recon_channel_mult,
            use_dropout=use_dropout,
            norm_layer=norm_layer,
            padding_type=padding_type,
            num_res_blocks=recon_num_res_blocks,
            encoder_net=encoder_net,
        )

        # get the decoder
        self.decoder = None
        if use_recon:
            self.decoder = get_decoder(
                channel=self.channel,
                feature_channels=feature_channels,
                recon_channel_mult=recon_channel_mult,
                norm_layer=norm_layer,
                encoder_net=encoder_net,
            )

        # get the regressor
        self.regressor = ConvRegressor(
            in_channels=self.encoder_out_channels,
            image_size=self.encoder_out_hw,
            hidden_channels=feature_channels,
            out_channels=param_size,
        )

        # get the mesh model
        self.mesh_model = MeshEstimator(batch_size).to(DEVICE)

    def forward(
        self,
        mode,
        batch_gender,
        img,
        ori_img,
        batch_names=None,
        depth_scaling="minmax",
        use_recon=False,
        use_diffusion=False,
        diffusion_trainer=None,
    ):
        """Forward pass of the model, the regressor will redict the smpl parameters from input depth image
            Estimated params - shape - [B, 88]
            0-10    Body Shape (10)
            10-13   root xyz (3)
            13-19   root angle (atan2 6)
            19-88   joint angles (69)
        Args:
            mode (str): train or test
            batch_gender (tensor): [B, 2]
            img (tensor): input image [B, 2, C, H, W]
            ori_img (tensor): original image [B, 2, C, H, W]
            batch_names (str): image names [B,]
            use_recon (bool): use image reconstruction or not
            use_diffusion (bool): use diffusion model or not
            diffusion_trainer (obj): diffusion trainer (optional)
        Returns:
            mesh_pred (tensor): mesh predictions dict of [B, N, 3]
            recon_raw_depth (tensor): reconstructed depth image [B, 1, H, W]
            diffusion_result (dict): diffusion results
        """
        # get the feature from the encoder
        feature_pair_list = [
            self.encoder(img[:, i]) for i in [0, 1]
        ]  # list of [B, C, H, W]

        diffusion_result = None
        if use_diffusion and diffusion_trainer is not None:
            if mode == "train":
                """ the results of the forward pass.
                    t: the timestep.
                    x_t: the results from q_sample.
                    vb: the variational bound.
                    model_output: the output from the model.
                    model_target: the target of the model.
                    pred_ori_xstart: the original x_start.
                    pred_cyc_xstart: the cycle x_start.
                    pred_trans_xstart: the translated x_start. """
                feature_pair = torch.stack(feature_pair_list, dim=1)  # [B, 2, C, H, W]
                diffusion_result = diffusion_trainer.forward_train_latent(feature_pair, batch_names)
                feature_pair_list[1] = diffusion_result["pred_trans_xstart"]
            elif mode == "test":
                feature_pair_list[1] = diffusion_trainer.forward_test_latent(
                    feature_pair_list[1]
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

        # get the reconstructed image from the decoder
        recon_raw_depth = None
        if use_recon and self.decoder is not None:
            recon_img = self.decoder(feature_pair_list[1])
            descaling_func = get_descaling_func(scaling=depth_scaling, modality="depth")
            recon_raw_depth = descaling_func(recon_img)
            recon_raw_depth = F.interpolate(
                recon_raw_depth, size=ori_img.shape[-2:], mode="bilinear"
            )  # recon_raw_depth: [B, 1, H, W]
            assert (
                recon_raw_depth.shape[1:] == ori_img.shape[-3:]
            ), f"Recon shape {recon_raw_depth.shape} != Ori shape {ori_img[:,0].shape}"

        # get the smpl parameters from the regressor
        smpl_param_pred = self.regressor(feature_pair_list[1])  # [B, 88]

        mesh_pred = self.mesh_model(batch_gender, smpl_param_pred)

        return mesh_pred, recon_raw_depth, diffusion_result

    def mesh_infer_gt(self, batch_gender, smpl_gt):
        with torch.no_grad():
            smpl_gt = smpl_gt.to(DEVICE)
            mesh_pred = self.mesh_model(batch_gender, smpl_gt, is_gt=True)  # [B, N, 3]

        return mesh_pred

    def reconstruct_clean_feature(self, x):
        """Reconstruct input image using the encoder and decoder
        this is only for pretraining the encoder and decoder and uses clean depth map
        """
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out

    def load_model(
        self, save_dir, encoder_epoch=-1, decoder_epoch=-1, regressor_epoch=-1
    ):
        """Load all models inside from the given path
        Args:
            save_dir (str): directory to save the models
                -checkpoints (save_dir)
                    |-encoder
                    |-decoder
                    |-regressor
                    |-diffusion
            encoder_epoch (int): epoch number of the encoder
            decoder_epoch (int): epoch number of the decoder
            regressor_epoch (int): epoch number of the regressor
        """
        encoder_path, self.encoder_epoch = get_model_path_in_directory(
            os.path.join(save_dir, "encoder"), encoder_epoch
        )
        decoder_path, self.decoder_epoch = get_model_path_in_directory(
            os.path.join(save_dir, "decoder"), decoder_epoch
        )
        regressor_path, self.regressor_epoch = get_model_path_in_directory(
            os.path.join(save_dir, "regressor"), regressor_epoch
        )

        logger.log(f"Load models from {save_dir}")
        logger.log(f"Load encoder from {encoder_path}")
        logger.log(f"Load decoder from {decoder_path}")
        logger.log(f"Load regressor from {regressor_path}")

        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
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
        if self.decoder is not None and self.decoder.training:
            torch.save(
                self.decoder.state_dict(),
                os.path.join(save_dir, "decoder", f"model_{epoch}.pt"),
            )
        if self.regressor.training:
            torch.save(
                self.regressor.state_dict(),
                os.path.join(save_dir, "regressor", f"model_{epoch}.pt"),
            )
