from functools import singledispatchmethod
import math
import os
import random
from dataclasses import dataclass
from typing import *

import hydra
import lightning as L
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchmetrics.functional import kl_divergence
from transformers.activations import ACT2FN

from data.diffusion import KnownLatentDiffusionDataset
from lightning_modules.metalearn import MetaLearningTask
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from models.utils import exists, right_pad_dims_to
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

@dataclass
class DiffusionTaskConfig:
    diffusion: DiffusionEncoderConfig
    dataset: dict 
    pretrained_id: str
    batch_size: int
    val_split: float
    loss: str
    lr: float
    lr_scheduler: bool = True


class DiffusionPriorTask(L.LightningModule):
    """Takes a base model and trains a variation encoder for its decoder
    you need to specify : diffusion config, dataset, pretrained model, some training info
    """

    @singledispatchmethod
    def __init__(self, cfg):
        pass

    @__init__.register(DiffusionTaskConfig)
    def _from_cfg(self, cfg: DiffusionTaskConfig) -> None:
        super().__init__()

        # Load pre-trained meta-learning task (and freeze it)
        self.base_task = MetaLearningTask(cfg.pretrained_id)
        for param in self.base_task.parameters():
            param.requires_grad = False
        
        # Init diffusion model
        self.diffusion_prior = DiffusionEncoder(cfg.diffusion)
        
        self.cfg = cfg        
        self.wandb_dict = dict({})
        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    @__init__.register(str)
    def _from_id(self, id: str):
        # Parse the checkpoint directory
        dir = os.path.join(os.environ["SCRATCH"], "diffusion_train_log/checkpoints/", id)
        ckpts = []
        for f in os.listdir(dir):
            if f != "last.ckpt":
                ckpts.append(f)

        # Load the last checkpoint
        ckpt = torch.load(os.path.join(dir, ckpts[-1]), weights_only=False)
        cfg = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.create(DiffusionTaskConfig),
                OmegaConf.create(ckpt[DiffusionPriorTask.CHECKPOINT_HYPER_PARAMS_KEY]),
            )
        )
        self._from_cfg(cfg)
        
        # LOAD VARIABLES
        self.train_data = ckpt["train_data"]
        self.val_data = ckpt["val_data"]
        self.wandb_dict.update(
            {
                "id": id,
                "ckpts_dir": dir,
                "default_ckpt": "last.ckpt",
                "ckpts_names": ckpts,
            }
        )
        self.set_to_checkpoint(-1)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["train_data"] = self.train_data
        checkpoint["val_data"] = self.val_data

    def set_to_checkpoint(
        self, ckpt_id: Optional[int] = None, step: Optional[int] = None
    ):
        assert len(self.wandb_dict) > 0
        assert sum([ckpt_id is None, step is None]) == 1

        if step is not None:
            steps = [
                int(filename.split(".ckpt")[0].split("step=")[1])
                for filename in self.wandb_dict["ckpts_names"]
            ]
            ckpt_id = np.abs((np.array(steps) / step) - 1).argmin()

        if ckpt_id == -1:
            ckpt_f = self.wandb_dict["default_ckpt"]
        else:
            ckpt_f = self.wandb_dict["ckpts_names"][ckpt_id]

        self.load_state_dict(
            torch.load(
                os.path.join(
                    self.wandb_dict["ckpts_dir"],
                    ckpt_f,
                ),
                weights_only=False,
            )["state_dict"]
        )
        print(f"Loaded checkpoing : {ckpt_f}")

    @property
    def loss_fn(self):
        if self.cfg.loss == "l1":
            return F.l1_loss
        elif self.cfg.loss == "l2":
            return F.mse_loss
        elif self.cfg.loss == "smooth_l1":
            return F.smooth_l1_loss
        else:
            raise ValueError(f"invalid loss type {self.cfg.loss}")


    def setup(self, stage):
        # Init dataset (which also makes sure everything is compatible)
        with torch.no_grad():
            dataset_cfg = hydra.utils.instantiate(self.cfg.dataset)
            if 'KnownLatentDiffusionDatasetConfig' in self.cfg.dataset['_target_']:
                dataset = KnownLatentDiffusionDataset(dataset_cfg, self.base_task, self.diffusion_prior)

            self.train_data, self.val_data = random_split(dataset, [1 - self.cfg.val_split, self.cfg.val_split])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.diffusion_prior.parameters(), lr=self.cfg.lr)
        if self.cfg.lr_scheduler:
            # this is probably fake af but we put it for good luck
            scheduler = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=100, max_epochs=10000)
            return [opt], [{"scheduler": scheduler, "interval": 'step'}]
        else:
            return opt

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
        )

    def compute_diffusion_loss(
        self, latent, class_id=None, cond_tokens=None, cond_ignore_mask=None
    ):
        # NOTE: Important to flip the <ignore_mask> to a <don't_ignore_mask>
        cond_mask = torch.logical_not(cond_ignore_mask)

        bs, l, d = (*latent.shape,)
        device = latent.device

        times = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
        noise = torch.randn_like(latent)

        alpha = self.diffusion_prior.train_schedule(times)
        alpha = right_pad_dims_to(latent, alpha)

        z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise

        self_cond = None

        if self.diffusion_prior.cfg.self_condition and (
            random.random() < self.diffusion_prior.cfg.train_prob_self_cond
        ):
            with torch.no_grad(): 
                model_output = self.diffusion_prior.diffusion_model_predictions(
                    z_t,
                    times,
                    class_id=class_id,
                    cond_tokens=cond_tokens,
                    cond_mask=cond_mask,
                )
                self_cond = model_output.pred_x_start.detach()
                if self.diffusion_prior.cfg.l2_normalize:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(
                        self_cond.shape[-1]
                    )

        # predict and take gradient step

        predictions = self.diffusion_prior.diffusion_model_predictions(
            z_t,
            times,
            x_self_cond=self_cond,
            class_id=class_id,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )

        if self.diffusion_prior.cfg.objective == "pred_x0":
            target = latent
            pred = predictions.pred_x_start
        elif self.diffusion_prior.cfg.objective == "pred_noise":
            target = noise
            pred = predictions.pred_noise
        elif self.diffusion_prior.cfg.objective == "pred_v":
            target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
            assert exists(predictions.pred_v)
            pred = predictions.pred_v

        loss = self.loss_fn(pred, target, reduction="none")
        loss = rearrange(
            [
                reduce(loss[i], "l d -> 1", "mean")
                for i in range(latent.shape[0])
            ],
            "b 1 -> b 1",
        )

        return loss.mean()

    def training_step(self, batch, batch_idx=None):
        latent = batch["latent"]
        cond_tokens, cond_ignore_mask = None, None
        
        if self.cfg.diffusion.seq_conditional:
            cond_tokens, cond_ignore_mask = batch["cond_tokens"], batch["cond_ignore_mask"]

        if self.diffusion_prior.cfg.normalize_latent:
            latent_ = rearrange(latent, 'b s d -> (b s) d')
            self.diffusion_prior.latent_mean = torch.mean(latent_, dim=0)
            self.diffusion_prior.latent_scale = torch.std(
                latent_ - self.diffusion_prior.latent_mean, unbiased=False
            )
            latent = self.diffusion_prior.normalize_latent(latent)

        loss = self.compute_diffusion_loss(latent, cond_tokens=cond_tokens, cond_ignore_mask=cond_ignore_mask)
        self.log("train/loss", loss.detach().cpu().numpy().item(), prog_bar=True, add_dataloader_idx=False, batch_size=latent.shape[0])

        return loss
    
    def validation_step(self, batch):
        latent, cond, cond_ignore_mask, cond_input_ids = batch.get("latent"), batch.get("cond"), batch.get("cond_ignore_mask"), batch.get("cond_input_ids")
        # for know_transformer, we can compare the empirical distribution of the diffusion model to p(theta | cond_input_ids)
        pass        
