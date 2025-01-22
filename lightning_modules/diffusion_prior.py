import math

import random
from dataclasses import dataclass
from typing import *


import lightning as L
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN
from lightning_modules.metalearn import MetaLearningTask
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from models.utils import exists, right_pad_dims_to
from models.diffusion import DiffusionTransformerConfig


@dataclass
class DiffusionPriorTaskConfig:
    name: str
    batch_size: int
    diffusion_config: DiffusionEncoderConfig
    train_size: int
    val_size: int
    loss_type: str
    normalize_latent: bool
    pretrained_id: Optional[str] = None
    lr: Optional[float] = 1e-3


class DiffusionPriorTask(L.LightningModule):
    """Takes a base model and trains a variation encoder for its decoder"""

    def __init__(self, cfg: DiffusionPriorTaskConfig):
        # Make sure pretrained_id, data_type and enc_cfg are compatible
        assert cfg.name in ["known_transformer", "implicit_rnn"]

        self.base_task = MetaLearningTask(cfg.pretrained_id)
        self.diffusion_prior = DiffusionEncoder(cfg.diffusion_config)

        self.cfg = cfg

    def setup(self):
        if self.cfg.name == "known_transformer":
            self.train_data = KnownLatentDiffusionDataset(
                self.base_task, size=self.cfg.train_size
            )
            self.val_data = KnownLatentDiffusionDataset(
                self.base_task, size=self.cfg.val_size
            )
        else:
            raise NotImplementedError

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

    @property
    def loss_fn(self):
        if self.cfg.loss_type == "l1":
            return F.l1_loss
        elif self.cfg.loss_type == "l2":
            return F.mse_loss
        elif self.cfg.loss_type == "smooth_l1":
            return F.smooth_l1_loss
        else:
            raise ValueError(f"invalid loss type {self.cfg.loss_type}")

    def compute_diffusion_loss(
        self, latent, mask=None, class_id=None, cond=None, cond_mask=None
    ):

        bs, l, d = (*latent.shape,)
        device = latent.device

        times = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
        noise = torch.randn_like(latent)

        alpha = self.diffusion_prior.train_schedule(times)
        alpha = right_pad_dims_to(latent, alpha)

        z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise

        self_cond = None

        if self.diffusion_prior.model.cfg.self_condition and (
            random.random() < self.diffusion_prior.cfg.train_prob_self_cond
        ):
            with torch.no_grad():
                model_output = self.diffusion_prior.diffusion_model_predictions(
                    z_t,
                    times,
                    mask,
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
            mask,
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
                reduce(loss[i][: torch.sum(mask[i])], "l d -> 1", "mean")
                for i in range(latent.shape[0])
            ],
            "b 1 -> b 1",
        )

        return loss.mean()

    def training_step(self, batch, batch_idx=None):
        latent, cond, cond_mask = batch["latent"], batch["cond"], batch["cond_mask"]

        if self.cfg.normalize_latent:
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
        wandb.log({"train/loss": loss.detach().numpy()})

        return loss


class KnownLatentDiffusionDataset(Dataset):
    def __init__(
        self, task: MetaLearningTask, size: int, context_length: Tuple[int]
    ) -> None:
        super().__init__()
        assert "KnownEncoder" in str(task.model.encoder.__class__)
        assert "TransformerDecoder" in str(task.model.decoder.__class__)

        # Sample some environemnts
        env_indices = torch.randint(
            low=0, high=len(task.full_data), size=size, device="cpu"
        )

        # Compute task-latent embedding from known latent encoder
        env_latents = j2t(task.full_data.index_to_latent)[env_indices]
        self.env_latents = task.model.encoder(env_latents)

        # Generate sequences
        cond = []
        mask = []
        for batch in torch.chunk(env_indices, 1024):
            out = task.full_data.__getitems__(batch, length=context_length)
            cond.append(out["input_ids"])
            mask.append(out["ignore_mask"])
        self.cond = torch.stack(cond, dim=0)
        self.mask = torch.stack(mask, dim=0)

    def __getitem__(self, idx):
        return self.__getitem__([idx])

    def __getitems__(self, indices):
        indices = torch.LongTensor(indices)

        return {
            "latent": self.env_latents[indices],
            "cond": self.cond[indices],
            "cond_mask": self.mask[indices],
        }
