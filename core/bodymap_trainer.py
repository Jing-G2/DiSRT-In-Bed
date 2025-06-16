from collections import defaultdict
from datetime import datetime
import json
import numpy as np
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from utils import logger
from torch.utils.tensorboard import SummaryWriter

from core.bodymap_metric import Metric
from datasets.loader import prepare_dataloaders
from utils.constants import *
from utils.train_utils.scheduler import get_scheduler
from models.bodymapModel import PMMModel as PMM

MODEL_FN_DICT = {
    "PMM": PMM,
}


class Trainer:
    def __init__(self, args):
        self.args = args
        if self.args["mode"] == "finetune":
            self.args["load_model_path"] = os.path.join(
                self.args["save_path"], self.args["load_model_path"]
            )
        print(f'Starting PM Trainer Object for {self.args["name"]} model')
        self.epoch = -1

        self.criterion_l1_mean = nn.L1Loss(reduction="mean")
        self.criterion_mse_mean = nn.MSELoss(reduction="mean")

        if self.args["trans_loss_type"] == "l2":
            self.criterion_recon = nn.MSELoss()
        else:  # default to l1
            self.criterion_recon = nn.L1Loss()

        self.args_file_name = "exp.json"

    # ######################################
    # helper functions
    # ######################################
    def _setup(self):
        json_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args_file_name = "exp_" + self.args["mode"] + f"_{json_time}.json"

        self.args["save_path"] = os.path.join(self.args["save_path"], self.args["name"])
        self.output_path = self.args["save_path"]

        logger.configure(
            dir=os.path.join(
                self.output_path, "logs", time.strftime("%Y-%m-%d_%H-%M-%S")
            )
        )

        # --------------------------------------
        if self.args["mode"] != "test":

            for dirname in [
                "encoder",
                "decoder",
                "regressor",
            ]:
                # save each modules weights separately
                os.makedirs(
                    os.path.join(self.output_path, "checkpoints", dirname),
                    exist_ok=True,
                )

            try:
                wandb.init(
                    project="BodyMAP",
                    name=self.args["name"],
                    dir=self.output_path,
                    resume=True,
                )
            except Exception as e:
                logger.log(f"Error in wandb init: {e}")
                exit(-1)

        self._load_data()
        self._save_args()

    def _load_data(self):
        start_time = time.time()
        if (
            self.args["mode"] != "test"
            and self.args["real_train_file"] is None
            and self.args["syn_train_file"] is None
        ):
            logger.log("ERROR: No dataset being used")
            exit(-1)

        if self.args["mode"] == "test":
            train_files = (None, None)
        else:
            train_files = (
                self.args["real_train_file"] if self.args["use_real"] else None,
                self.args["syn_train_file"] if self.args["use_synth"] else None,
            )

        (
            self.train_loader,
            self.val_loader,
            train_dataset,
            val_dataset,
        ) = prepare_dataloaders(
            data_path=self.args["data_path"],
            train_files=train_files,
            val_files=(
                self.args["real_val_file"] if self.args["use_real"] else None,
                self.args["syn_val_file"] if self.args["use_synth"] else None,
            ),
            batch_size=self.args["batch_size"],
            image_size=self.args["image_size"],
            exp_type=self.args["exp_type"],
            is_train=self.args["mode"] != "test",
            modality=self.args["modality"],
            pressure_scaling=self.args["pressure_scaling"],
            ir_scaling=self.args["ir_scaling"],
            depth_scaling=self.args["depth_scaling"],
            is_affine=self.args["is_affine"],
            is_erase=self.args["is_erase"],
            uncover_only=self.args["uncover_only"],
        )

        self.metric_loader = self.val_loader
        logger.log("Using val dataset for metric testing as well")

        end_time = time.time()
        self.args["dataset_setup_time"] = end_time - start_time
        logger.log(f'Dataset Setup Time = {self.args["dataset_setup_time"]: .0f} s')
        self.args["train_len"] = len(train_dataset) if train_dataset is not None else 0
        self.args["val_len"] = len(val_dataset)

    def _save_args(self):
        args_str = json.dumps(self.args)
        with open(os.path.join(self.output_path, self.args_file_name), "w") as f:
            f.write(args_str)

    # ######################################
    # model functions
    # ######################################
    def _setup_model(self):
        main_model_fn = MODEL_FN_DICT.get(self.args["main_model_fn"], None)
        if main_model_fn is None:
            logger.log("ERROR: invalid main_model_fn")
            exit(-1)

        # get PMM model (encoder, decoder, regressor, mesh estimator)
        model = main_model_fn(
            image_size=self.args["image_size"],
            feature_channels=self.args["feature_channels"],
            param_size=self.args["param_size"],
            vertex_size=self.args["vertex_size"],
            batch_size=self.args["batch_size"],  # required for mesh estimator
            modality=self.args["modality"],
        )

        # calculate the number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.log(f"Total trainable parameters = {total_params}")

        for param in model.parameters():
            param.requires_grad = True
        for param in model.mesh_model.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )

        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            lr_policy=self.args["lr_policy"],
            total_epochs=self.args["epochs"],
        )

        return model

    def _load_model(self, model, save_dir):
        model_path = os.path.join(save_dir, "model.pt")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])

            self.epoch = checkpoint["epoch"]
            self.starting_epoch = self.epoch + 1

            self.optimizer.load_state_dict(checkpoint["opt_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
        else:
            self.starting_epoch = 1 + model.load_model(
                save_dir=os.path.join(save_dir, "checkpoints"),
                encoder_epoch=self.args["encoder_epoch"],
                regressor_epoch=self.args["regressor_epoch"],
            )

        return model

    def _save_model(self, model, epoch):
        self.epoch = epoch
        model = model.to("cpu")
        if epoch == self.args["epochs"] - 1 or epoch == -1:
            model.save_model(os.path.join(self.output_path, "checkpoints"), epoch)

        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.output_path, "model.pt"),
        )

    # ######################################
    # train and test functions
    # ######################################
    def _get_losses(self, mesh_pred, mesh_gt, batch_labels):
        """Compute the losses for the model
        Args:
            mesh_pred (dict): predicted mesh params [B, N, 3]
            mesh_gt (dict): ground truth mesh params [B, N, 3]
            batch_labels (tensor): batch labels [B, 162]
        Returns:
            losses (dict): dictionary of losses
        """
        betas_loss = root_angle_loss = joint_angles_loss = joint_pos_loss = (
            smpl_loss
        ) = (torch.tensor(0).float().to(DEVICE))
        if self.args["use_smpl_loss"]:
            betas_loss = self.criterion_l1_mean(
                mesh_pred["out_betas"], batch_labels[:, 72:82]
            )
            root_angle_loss = self.criterion_l1_mean(
                mesh_pred["out_root_angles"][:, :3], torch.cos(batch_labels[:, 82:85])
            ) + self.criterion_l1_mean(
                mesh_pred["out_root_angles"][:, 3:], torch.sin(batch_labels[:, 82:85])
            )
            joint_angles_loss = self.criterion_mse_mean(
                mesh_pred["out_joint_angles"][:, 3:], batch_labels[:, 85:154]
            )  # skip root rotation from loss
            joint_pos_loss = (
                (((mesh_pred["out_joint_pos"] - mesh_gt["out_joint_pos"]) + 1e-7) ** 2)
                .reshape(-1, 24, 3)
                .sum(dim=-1)
                .sqrt()
                .mean()
            )

            # weight the smpl parameters by the std
            smpl_loss = (
                betas_loss * (1 / 1.728158146914805)
                + joint_angles_loss * (1 / 0.29641429463719227)
                + root_angle_loss * (1 / 0.3684988513298487)
                + joint_pos_loss * (1 / 0.1752780723422608)
            )

        v2v_loss = torch.tensor(0).float().to(DEVICE)
        if self.args["use_v2v_loss"] and mesh_gt is not None:
            # weight by the std
            v2v_loss = (
                ((mesh_gt["out_verts"] - mesh_pred["out_verts"]) ** 2)
                .sum(dim=-1)
                .sqrt()
                .mean()
            ) * (1 / 0.1752780723422608)

        # total loss
        loss = (
            self.args["lambda_smpl_loss"] * smpl_loss
            + self.args["lambda_v2v_loss"] * v2v_loss
        )

        return {
            "total_loss": loss,
            "betas_loss": betas_loss,
            "joint_angles_loss": joint_angles_loss,
            "root_angle_loss": root_angle_loss,
            "joint_pos_loss": joint_pos_loss,
            "smpl_loss": smpl_loss,
            "v2v_loss": v2v_loss,
        }

    def _train_epoch(self, model):
        model = model.to(DEVICE)
        model.train()
        running_losses = defaultdict(float)
        with torch.autograd.set_detect_anomaly(True):

            for (
                batch_original_pressure,  # [B, S, 1, H, W]
                batch_pressure_images,  # [B, S, 1, H, W]
                batch_original_ir,  # [B, S, 1, H, W]
                batch_ir_images,  # [B, S, 1, H, W]
                batch_original_depth,  # [B, S, 1, H, W]
                batch_depth_images,  # [B, S, 1, H, W]
                batch_labels,  # [B, 162]
                batch_gt_pmap,  # [B, 6890]
                batch_gt_verts,  # [B, 6890, 3]
                batch_names,  # [B,]
            ) in tqdm(
                iter(self.train_loader),
                desc="Training epoch",
                bar_format="{l_bar}{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                total=self.args["train_len"] // self.args["batch_size"],
                leave=False,
            ):

                self.optimizer.zero_grad()

                batch_pressure_images = (
                    batch_pressure_images.to(DEVICE)
                    if (batch_pressure_images != MISS_MODALITY_FLAG).all()
                    else None
                )
                batch_ir_images = (
                    batch_ir_images.to(DEVICE)
                    if (batch_ir_images != MISS_MODALITY_FLAG).all()
                    else None
                )
                if (batch_depth_images != MISS_MODALITY_FLAG).all():
                    batch_original_depth = batch_original_depth.to(DEVICE)
                    batch_depth_images = batch_depth_images.to(DEVICE)
                else:
                    batch_original_depth = None
                    batch_depth_images = None

                batch_labels = batch_labels.to(DEVICE)

                batch_mesh_pred = model(
                    batch_gender=batch_labels[:, 157:159],  # gender
                    img=batch_depth_images,
                )  # mesh_pred: [B, N, 3]

                # Due to data augmentation (random rotation), the gt smpl params need to be recomputed
                smpl_param_gt = torch.cat(
                    (
                        batch_labels[:, 72:82],
                        batch_labels[:, 154:157],
                        torch.cos(batch_labels[:, 82:85]),
                        torch.sin(batch_labels[:, 82:85]),
                        batch_labels[:, 85:154],
                    ),
                    axis=1,
                )
                mesh_gt = model.mesh_infer_gt(
                    batch_labels[:, 157:159],  # gender
                    smpl_param_gt,
                )  # [B, N, 3]

                losses = self._get_losses(
                    mesh_pred=batch_mesh_pred,
                    mesh_gt=mesh_gt,
                    batch_labels=batch_labels,
                )

                losses["total_loss"].backward()
                self.optimizer.step()

                wandb.log(losses)
                for k in losses:
                    running_losses[k] += losses[k].item()

            self.scheduler.step()

        for k in running_losses:
            running_losses[k] /= self.args["train_len"]
        return running_losses

    def _train_model(self, model):
        logger.log(f"Starting training for experiment - {self.args['name']}")
        logger.log(
            f"Starting model training for {self.args['epochs'] - self.starting_epoch} epochs starting from {self.starting_epoch}"
        )

        for epoch in tqdm(
            range(self.starting_epoch, self.args["epochs"], 1),
            desc="Training",
            bar_format="{l_bar}{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):

            train_losses = self._train_epoch(model)

            # wandb logging
            # for k in train_losses:
            #     wandb.log(
            #         {f"Train/{k}": train_losses[k]},
            #         step=epoch,
            #     )
            # wandb.log(
            #     {
            #         "Learning_rate": self.optimizer.param_groups[0]["lr"],
            #         "Epoch": epoch,
            #     },
            #     step=epoch,
            # )

            # save the model
            if (epoch + 1) % self.args["epochs_save"] == 0 or (epoch + 1) % self.args[
                "epochs_metric"
            ] == 0:
                model.train()
                self._save_model(model, epoch)

            # run metric
            if (epoch + 1) % self.args["epochs_metric"] == 0 and (
                epoch + 1
            ) != self.args["epochs"]:
                Metric(
                    model=model,
                    test_loader=self.metric_loader,
                    epoch=epoch,
                    save_gt=self.args["viz_type"] is not None,
                    infer_save_path=(
                        os.path.join(self.output_path, "infer", self.args["mode"])
                        if self.args["viz_type"] is not None
                        else None
                    ),
                )

                self._save_args()

        # ------------------------------------------------
        # save the model at the end of training
        self._save_model(model, self.args["epochs"] - 1)

        metric = Metric(
            model=model,
            test_loader=self.metric_loader,
            epoch=self.args["epochs"] - 1,
            save_gt=True,
            viz_type=(
                self.args["viz_type"] if self.args["viz_type"] is not None else "image"
            ),
            infer_save_path=(
                os.path.join(self.output_path, "infer", self.args["mode"])
            ),
        )
        self.args["metric"] = metric
        logger.log("Final Metric")
        logger.log(f"{metric}")

        return model

    def train_model(self):
        """Train the model with the given arguments."""
        if self.args["mode"] == "finetune" and (
            self.args["load_model_path"] is None
            or not os.path.exists(self.args["load_model_path"])
        ):
            print("Finetune mode requires a valid `load_model_path`")
            exit(-1)

        # Setup and Load Data
        start_time = time.time()
        self._setup()

        # --------------------------------------
        # Load Model
        model = self._setup_model()
        try:
            model = self._load_model(model, self.output_path)
            logger.log(
                f"Model loaded: epoch = {self.epoch}. Starting epoch = {self.starting_epoch}"
            )
        except Exception as e:
            if self.args["mode"] == "finetune":
                model = self._load_model(model, self.args["load_model_path"])
                self.starting_epoch = 0
                logger.log(
                    f"Finetune epoch start from {self.starting_epoch}, finetuning model from {self.args['load_model_path']}"
                )
            else:
                self.starting_epoch = 0
                logger.log(
                    f"No pretrained model loaded: {e}, starting from epoch {self.starting_epoch}"
                )

        # Train Model
        model = self._train_model(model)
        end_time = time.time()
        self.args["training_time"] = end_time - start_time
        logger.log(
            f'model trained in time = {self.args["training_time"]: .0f} s = {self.args["training_time"]/60: .2f} min = {self.args["training_time"]/3600: .2f} hr'
        )
        self._save_args()

    def test_model(
        self,
    ):
        """Evaluate the model overall."""

        # Setup and Load Data
        start_time = time.time()
        self._setup()
        assert self.val_loader is not None, "[Error]: val loader is None"

        # --------------------------------------
        # Load Model
        model = self._setup_model()
        model = self._load_model(model, self.output_path)
        model = model.to(DEVICE)

        # --------------------------------------
        # Infer and Test
        real_val_results = Metric(
            model,
            self.val_loader,
            epoch=-1,
            save_gt=self.args["viz_type"] is not None,
            infer_save_path=(
                os.path.join(self.output_path, "infer", self.args["mode"])
                if self.args["viz_type"] is not None
                else None
            ),
        )

        self.args["real_val_metric"] = real_val_results
        self._save_args()
        logger.log("\nReal Val Results")
        logger.log(f"{real_val_results}")
        logger.log("Test run complete")
        logger.log(f"Total time = {time.time() - start_time: .0f} s")
