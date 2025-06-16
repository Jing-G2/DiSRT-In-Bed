from collections import defaultdict
from datetime import datetime
import json
import numpy as np
import os
import time
import copy
import blobfile as bf
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from utils import logger
from torch.utils.tensorboard import SummaryWriter

from core.diffusion_metric import Metric
from datasets.loader import prepare_dataloaders
from utils.constants import *

from utils.script_utils import args_to_dict
from models.module import MeshEstimator
from models.diffusion import (
    dist_util,
    diffusion_keys,
    model_keys,
    get_gaussian_diffusion,
    get_model,
)
from models.module.diffusion_nn import update_ema
from models.diffusion.resample import (
    create_named_schedule_sampler,
)

from utils.train_utils.scheduler import get_scheduler
from utils.train_utils.file_utils import get_model_path_in_directory


class Trainer:
    def __init__(self, args):
        self.args = args
        if self.args["mode"] == "finetune":
            self.args["load_model_path"] = os.path.join(
                self.args["save_path"], self.args["load_model_path"]
            )
        print(f'Starting PM Trainer Object for model: {self.args["name"]}')
        self.epoch = -1

        self.criterion_l1_mean = nn.L1Loss(reduction="mean")
        self.criterion_mse_mean = nn.MSELoss(reduction="mean")

        self.args_file_name = "exp.json"

    # ######################################
    # helper functions
    # ######################################
    def _setup(self):
        json_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.args_file_name = "exp_" + self.args["mode"] + f"_{json_time}.json"

        self.output_path = os.path.join(self.args["save_path"], self.args["name"])

        logger.configure(
            dir=os.path.join(
                self.output_path, "logs", time.strftime("%Y-%m-%d_%H-%M-%S")
            )
        )

        # --------------------------------------
        if self.args["mode"] != "test":
            os.makedirs(
                os.path.join(self.output_path, "checkpoints", "diffusion"),
                exist_ok=True,
            )

            try:
                wandb.init(
                    project="BodyMAP",
                    name=self.args["name"],
                    dir=self.output_path,
                    resume=True,
                )
                self.writer = None
            except Exception as e:
                logger.log(f"Error in wandb init: {e}")
                self.writer = SummaryWriter(
                    os.path.join(self.output_path, "runs", self.args["name"])
                )
        else:
            self.writer = None

        if self.args["microbatch"] == -1:
            self.args["microbatch"] = self.args["batch_size"]
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
            syn_ratio=self.args["syn_ratio"],
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
            uncover_only=self.args["uncover_only"] or self.args["mode"] == "pretrain",
            # pretrain only uses uncover images
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
        if self.writer is not None:
            self.writer.add_text("Args", args_str)

    # ######################################
    # model functions
    # ######################################
    def _setup_model(self):
        # setup the environment
        dist_util.setup_dist()

        # get the diffusion model and unet
        diffusion_dict = args_to_dict(self.args, diffusion_keys)
        diffusion_unet_dict = args_to_dict(self.args, model_keys)
        self.diffusion = get_gaussian_diffusion(**diffusion_dict)
        self.model = get_model(
            image_size=self.args["image_size"],
            in_channels=1,
            num_param=self.args["param_size"],
            **diffusion_unet_dict,
        )
        self.model = self.model.to(DEVICE)

        # calculate the number of parameters
        num_params = sum(
            p.numel() if p.requires_grad else 0 for p in self.model.parameters()
        )
        logger.log(f"-- Number of trainable parameters: {num_params} --")

        # get the mesh estimator
        self.mesh_estimator = MeshEstimator(self.args["batch_size"]).to(DEVICE)

        self.loss_type = self.diffusion.loss_type  # LossType.MSE
        self.model_var_type = (
            self.diffusion.model_var_type
        )  # ModelVarType.LEARNED_RANGE

        self.ema_rate = (
            [self.args["ema_rate"]]
            if isinstance(self.args["ema_rate"], float)
            else [float(x) for x in self.args["ema_rate"].split(",")]
        )
        self.schedule_sampler = create_named_schedule_sampler(
            name=self.args["schedule_sampler"],
            diffusion=self.diffusion,
        )

        """ 
            use AdamW optimizer for training
            AdamW = Adam + L2 regularization + weight decay
        """
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay_diffusion"],
        )
        if self.args["mode"] == "finetune":
            self.args["lr_anneal_steps"] = (
                self.args["epochs"] * self.args["train_len"] // self.args["batch_size"]
            )
        self.scheduler = get_scheduler(
            self.opt,
            self.args["lr_policy_diffusion"],
            self.args["lr_anneal_steps"],
        )

    def _load_and_sync_parameters(self, save_dir, epoch=-1):
        try:
            diffusion_model_path, resume_epoch = get_model_path_in_directory(
                save_dir, epoch
            )
        except:
            logger.log("Haven't found resume diffusion checkpoint")
            return -1

        logger.log(
            f"loading diffusion model from checkpoint path: {diffusion_model_path}..."
        )
        self.model.load_state_dict(
            dist_util.load_state_dict(diffusion_model_path, map_location=DEVICE)
        )

        dist_util.sync_params(self.model.parameters())

        return resume_epoch

    def _load_optimizer_state(self, save_dir, epoch):
        scheduler_checkpoint = bf.join(save_dir, "opt", f"scheduler_{epoch}.pt")
        opt_checkpoint = bf.join(save_dir, "opt", f"opt_{epoch}.pt")
        opt_checkpoint_old = bf.join(save_dir, "opt", f"opt.pt")

        if bf.exists(scheduler_checkpoint):
            logger.log(
                f"loading scheduler state from checkpoint: {scheduler_checkpoint}"
            )
            state_dict = dist_util.load_state_dict(
                scheduler_checkpoint, map_location=DEVICE
            )
            self.scheduler.load_state_dict(state_dict)
        else:
            logger.log("No scheduler checkpoint found, using default scheduler")

        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=DEVICE)
            self.opt.load_state_dict(state_dict)
        elif bf.exists(opt_checkpoint_old):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint_old}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint_old, map_location=DEVICE
            )
            self.opt.load_state_dict(state_dict)
        else:
            logger.log("No optimizer checkpoint found, using default optimizer")
            raise Exception("No optimizer checkpoint found")

    def _load_ema_parameters(self, save_dir, epoch, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))
        ema_checkpoint = bf.join(save_dir, f"ema/ema.pt")
        ema_checkpoint_old = bf.join(save_dir, f"ema/ema_{rate}_{epoch}.pt")

        if bf.exists(ema_checkpoint):
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=DEVICE)
            ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]
        elif bf.exists(ema_checkpoint_old):
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint_old}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint_old, map_location=DEVICE
            )
            ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]

        dist_util.sync_params(ema_params)
        return ema_params

    def load_model(self, save_dir, epoch=-1):
        """Load model from the given path and resume training"""
        self.epoch = self._load_and_sync_parameters(save_dir, epoch)
        self.starting_epoch = self.epoch + 1

        self._load_optimizer_state(save_dir, self.epoch)
        # Model was resumed, either due to a restart or a checkpoint
        # being specified at the command line.
        self.ema_params = [
            self._load_ema_parameters(save_dir, self.epoch, rate)
            for rate in self.ema_rate
        ]

    def save_model(self, save_dir, epoch):
        """Save model to the given path"""

        def save_checkpoint(rate, params):
            state_dict = self.model.state_dict()
            for i, (name, _) in enumerate(self.model.named_parameters()):
                state_dict[name] = params[i]
            if not rate:
                filename = f"model_{epoch}.pt"
            else:
                filename = f"ema/ema_{rate}_{epoch}.pt"
                # filename = f"ema/ema.pt"
            with bf.BlobFile(bf.join(save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        # save the model
        save_checkpoint(0, list(self.model.parameters()))
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # save the scheduler state
        with bf.BlobFile(
            bf.join(save_dir, f"opt/scheduler_{epoch}.pt"),
            "wb",
        ) as f:
            torch.save(self.scheduler.state_dict(), f)

        # save the optimizer state
        with bf.BlobFile(
            # bf.join(save_dir, f"opt/opt.pt"),
            bf.join(save_dir, f"opt/opt_{epoch}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)

    # ######################################
    # train and test functions
    # ######################################
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(list(self.model.parameters()), params, rate)

    def _get_losses(self, diffusion_terms, param_gt, batch_gender):
        """Compute the losses for the model
        Args:
            diffusion_terms (dict): dictionary of diffusion terms`
            param_gt (torch.Tensor): ground truth parameters [B, 88]
            batch_gender (torch.Tensor)
        Returns:
            losses (dict): dictionary of losses
        """
        # get the mesh prediction and ground truth from the mesh estimator
        mesh_pred = self.mesh_estimator(
            batch_gender=batch_gender, x=diffusion_terms["pred_xstart"].clone()
        )
        with torch.no_grad():
            mesh_gt = self.mesh_estimator(
                batch_gender=batch_gender, x=param_gt, is_gt=True
            )  # [B, N, 3]

        betas_loss = root_angle_loss = joint_angles_loss = joint_pos_loss = (
            smpl_loss
        ) = (torch.tensor(0).float().to(DEVICE))

        if self.args["use_smpl_loss"]:
            betas_loss = self.criterion_l1_mean(
                mesh_pred["out_betas"], mesh_gt["out_betas"]
            )

            root_angle_loss = self.criterion_l1_mean(
                mesh_pred["out_root_angles"], mesh_gt["out_root_angles"]
            )

            joint_angles_loss = self.criterion_mse_mean(
                mesh_pred["out_joint_angles"], mesh_gt["out_joint_angles"]
            )  # include root rotation

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
                + root_angle_loss * (1 / 0.3684988513298487)
                + joint_angles_loss * (1 / 0.29641429463719227)
                + joint_pos_loss * (1 / 0.1752780723422608)
            )

        v2v_loss = torch.tensor(0).float().to(DEVICE)
        if self.args["use_v2v_loss"]:
            # weight by the std
            v2v_loss = (
                ((mesh_pred["out_verts"] - mesh_gt["out_verts"]) ** 2)
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
            "smpl_loss": smpl_loss,
            "v2v_loss": v2v_loss,
            "betas_loss": betas_loss,
            "root_angle_loss": root_angle_loss,
            "joint_angles_loss": joint_angles_loss,
            "joint_pos_loss": joint_pos_loss,
        }

    def _train_epoch(self):
        self.model.train()
        if self.args["mode"] == "finetune":
            for param in self.model.parm_embed.parameters():
                param.requires_grad = False
            for param in self.model.time_embed.parameters():
                param.requires_grad = False

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

                self.opt.zero_grad()

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

                # ------------------------------------------------
                # Due to data augmentation (random rotation), the gt smpl params need to be recomputed
                param_gt = torch.cat(
                    (
                        batch_labels[:, 72:82],  # 10
                        batch_labels[:, 154:157],  # 3
                        torch.cos(batch_labels[:, 82:85]),  # 3
                        torch.sin(batch_labels[:, 82:85]),  # 3
                        batch_labels[:, 85:154],  # 69
                    ),  # [B, 88]
                    axis=1,
                )

                # ------------------------------------------------
                # Run the model
                res = None
                for i in range(0, batch_depth_images.shape[0], self.args["microbatch"]):
                    batch_micro = param_gt[i : i + self.args["microbatch"]].to(
                        DEVICE
                    )  # [b, 88]
                    t, sampler_weights = self.schedule_sampler.sample(
                        batch_size=batch_micro.shape[0], device=DEVICE
                    )  # t: [b], weights: [b]

                    model_kwargs = {
                        "cond": batch_depth_images[
                            i : i + self.args["microbatch"], 1
                        ].to(
                            DEVICE
                        ),  # [b, C, H, W]  # condition on images
                        "is_train": True,
                    }

                    terms = self.diffusion.run_training_step(
                        model=self.model,
                        x_start=batch_micro,
                        t=t,
                        sampler_weights=sampler_weights,
                        model_kwargs=model_kwargs,
                        noise=None,
                    )

                    if res is None:
                        res = terms
                    else:
                        for k, v in terms.items():
                            res[k] = torch.cat((res[k], v), dim=0)

                losses = self._get_losses(
                    diffusion_terms=res,
                    param_gt=param_gt,
                    batch_gender=batch_labels[:, 157:159],
                )
                losses["total_loss"].backward()
                self._update_ema()
                self.opt.step()
                self.scheduler.step()

                log_dict = losses.copy()
                log_dict["Learning_rate"] = self.opt.param_groups[0]["lr"]
                wandb.log(log_dict)

                for k in losses:
                    running_losses[k] += losses[k].item()

        for k in running_losses:
            running_losses[k] /= self.args["train_len"]
        return running_losses

    def _train_model(self):
        logger.log(f"Starting training for experiment - {self.args['name']}")
        logger.log(
            f"Starting model training for {self.args['epochs'] - self.starting_epoch} epochs starting from {self.starting_epoch}"
        )

        for epoch in tqdm(
            range(self.starting_epoch, self.args["epochs"], 1),
            desc="Training",
            bar_format="{l_bar}{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):

            train_losses = self._train_epoch()

            # wandb logging
            # for k in train_losses:
            #     wandb.log(
            #         {f"Train/{k}": train_losses[k]},
            #         step=epoch,
            #     )

            # save the model
            if (epoch + 1) % self.args["epochs_save"] == 0 or (epoch + 1) % self.args[
                "epochs_metric"
            ] == 0:
                self.save_model(
                    save_dir=os.path.join(self.output_path, "checkpoints", "diffusion"),
                    epoch=epoch,
                )

            # run metric
            if (epoch + 1) % self.args["epochs_metric"] == 0 and (
                epoch + 1
            ) != self.args["epochs"]:
                Metric(
                    diffusion=self.diffusion,
                    model=self.model,
                    mesh_estimator=self.mesh_estimator,
                    test_loader=self.metric_loader,
                    writer=self.writer,
                    epoch=epoch,
                    save_gt=self.args["viz_type"] is not None,
                    viz_type=self.args["viz_type"],
                    infer_save_path=(
                        os.path.join(self.output_path, "infer", self.args["mode"])
                        if self.args["viz_type"] is not None
                        else None
                    ),
                    args=self.args,
                )

                self._save_args()
                if self.writer is not None:
                    self.writer.flush()

        # ------------------------------------------------
        # save the model at the end of training
        self.save_model(
            save_dir=os.path.join(self.output_path, "checkpoints", "diffusion"),
            epoch=self.args["epochs"] - 1,
        )

        metric = Metric(
            diffusion=self.diffusion,
            model=self.model,
            mesh_estimator=self.mesh_estimator,
            test_loader=self.metric_loader,
            writer=self.writer,
            epoch=self.args["epochs"] - 1,
            save_gt=True,
            viz_type=(
                self.args["viz_type"] if self.args["viz_type"] is not None else "image"
            ),
            infer_save_path=(
                os.path.join(self.output_path, "infer", self.args["mode"])
            ),
            args=self.args,
        )
        self.args["metric"] = metric
        logger.log("Final Metric")
        logger.log(f"{metric}")

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
        self._setup_model()
        try:
            self.load_model(
                save_dir=os.path.join(self.output_path, "checkpoints", "diffusion"),
                epoch=-1,
            )
        except Exception as e:
            if self.args["mode"] == "finetune":
                logger.log("Error in loading model: ", e)
                logger.log(
                    f"The model has not been finetuned yet, so starting to load the model from `load_model_path`: {self.args['load_model_path']}"
                )
                self.load_model(
                    save_dir=os.path.join(
                        self.args["load_model_path"], "checkpoints", "diffusion"
                    ),
                    epoch=-1,
                )
                self.starting_epoch = 0

                self.save_model(
                    save_dir=os.path.join(self.output_path, "checkpoints", "diffusion"),
                    epoch=-1,
                )

                self.scheduler = get_scheduler(
                    self.opt,
                    self.args["lr_policy_diffusion"],
                    self.args["lr_anneal_steps"],
                )

                # test and metrics of the initial model
                # init_results = Metric(
                #     diffusion=self.diffusion,
                #     model=self.model,
                #     mesh_estimator=self.mesh_estimator,
                #     test_loader=self.val_loader,
                #     writer=self.writer,
                #     epoch=0,
                #     save_gt=None,
                #     viz_type=self.args["viz_type"],
                #     infer_save_path=(
                #         os.path.join(self.output_path, "infer", self.args["mode"])
                #         if self.args["viz_type"] is not None
                #         else None
                #     ),
                #     args=self.args,
                # )
                # init_results_json = json.dumps(init_results)
                # with open(
                #     os.path.join(self.output_path, "exp_init_results.json"), "w"
                # ) as f:
                #     f.write(init_results_json)
                # logger.log("\nInitial Results")
                # logger.log(f"{init_results}")
            else:
                self.ema_params = [
                    copy.deepcopy(list(self.model.parameters()))
                    for _ in range(len(self.ema_rate))
                ]

        # Train Model
        self._train_model()
        end_time = time.time()
        self.args["training_time"] = end_time - start_time
        logger.log(
            f'model trained in time = {self.args["training_time"]: .0f} s = {self.args["training_time"]/60: .2f} min = {self.args["training_time"]/3600: .2f} hr'
        )
        self._save_args()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

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
        self._setup_model()
        self.load_model(
            save_dir=os.path.join(self.output_path, "checkpoints", "diffusion"),
            epoch=self.args["epochs"] - 1,
        )

        # --------------------------------------
        # Infer and Test
        real_val_results = Metric(
            diffusion=self.diffusion,
            model=self.model,
            mesh_estimator=self.mesh_estimator,
            test_loader=self.val_loader,
            writer=self.writer,
            epoch=self.args["epochs"] - 1,
            save_gt=self.args["viz_type"] is not None,
            viz_type=self.args["viz_type"],
            infer_save_path=(
                os.path.join(self.output_path, "infer", self.args["mode"])
                if self.args["viz_type"] is not None
                else None
            ),
            args=self.args,
        )

        self.args["real_val_metric"] = real_val_results
        self._save_args()
        logger.log("\nReal Val Results")
        logger.log(f"{real_val_results}")
        logger.log("Test run complete")
        logger.log(f"Total time = {time.time() - start_time: .0f} s")
