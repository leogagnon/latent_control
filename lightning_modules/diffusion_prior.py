import math
import random
from dataclasses import dataclass
from typing import *

import hydra
import lightning as L
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.functional import kl_divergence
from transformers.activations import ACT2FN

from data.diffusion import KnownLatentDiffusionDataset
from lightning_modules.metalearn import MetaLearningTask
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from models.utils import exists, right_pad_dims_to


@dataclass
class DiffusionTaskConfig:
    diffusion: DiffusionEncoderConfig
    dataset: dict 
    pretrained_id: str
    batch_size: int
    val_split: float
    loss: str
    lr: float


class DiffusionPriorTask(L.LightningModule):
    """Takes a base model and trains a variation encoder for its decoder
    you need to specify : diffusion config, dataset, pretrained model, some training info
    """

    def __init__(self, cfg: DiffusionTaskConfig):
        super().__init__()
        # Load pre-trained meta-learning task (and freeze it)
        self.base_task = MetaLearningTask(cfg.pretrained_id)
        for param in self.base_task.parameters():
            param.requires_grad = False
        
        # Init diffusion model
        self.diffusion_prior = DiffusionEncoder(cfg.diffusion)
        
        self.cfg = cfg

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
        return torch.optim.AdamW(self.diffusion_prior.parameters(), lr=self.cfg.lr)

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
        self, latent, class_id=None, cond=None, cond_mask=None
    ):

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
                    seq2seq_cond=cond,
                    seq2seq_mask=cond_mask,
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
            seq2seq_cond=cond,
            seq2seq_mask=cond_mask,
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
        latent, cond, cond_mask = batch.get("latent"), batch.get("cond"), batch.get("cond_mask")

        if self.diffusion_prior.cfg.normalize_latent:
            latent = torch.cat(
                [latent[i][: torch.sum(cond_mask[i])] for i in range(latent.shape[0])],
                dim=0,
            )
            self.diffusion_prior.latent_mean = torch.mean(latent, dim=0)
            self.diffusion_prior.latent_scale = torch.std(
                latent - self.diffusion_prior.latent_mean, unbiased=False
            )
            latent = self.diffusion_prior.normalize_latent(latent)

        loss = self.compute_diffusion_loss(latent, cond=cond, cond_mask=cond_mask)
        self.log("train/loss", loss.detach().cpu().numpy().item(), prog_bar=True, add_dataloader_idx=False, batch_size=latent.shape[0])

        return loss
    
    def validation_step(self, batch):
        latent, cond, cond_mask, cond_input_ids = batch.get("latent"), batch.get("cond"), batch.get("cond_mask"), batch.get("cond_input_ids")
        # for know_transformer, we can compare the empirical distribution of the diffusion model to p(theta | cond_input_ids)
        pass        
