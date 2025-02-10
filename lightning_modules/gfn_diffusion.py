from functools import singledispatchmethod
import math
import os
import random
from dataclasses import dataclass
from typing import *
from models.utils import *

import hydra
import lightning as L
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce, repeat
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchmetrics.functional import kl_divergence
from transformers.activations import ACT2FN

from data.diffusion import *
from lightning_modules.metalearn import MetaLearningTask
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from models.utils import exists, right_pad_dims_to
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import jax.numpy as jnp
import jax
from jax.scipy.special import rel_entr
from models.diffusion import DiT, DiTConfig

logtwopi = math.log(2 * math.pi)


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


@dataclass
class GFNDiffusionConfig:
    model: DiTConfig
    dataset: dict
    pretrained_id: str
    batch_size: int
    loss: str
    lr: float
    max_steps: int

    log_var_range: float
    t_scale: float
    learned_variance: bool
    trajectory_length: int
    energy: str = None
    train_direction: str
    exploratory: bool
    exploration_factor: float
    exploration_wd: bool


class GFNDiffusion(L.LightningModule):
    """
    Trains a diffusion model with a Trajectory Balance (TB) objective (Sendera et al. 2025; https://arxiv.org/pdf/2402.05098), i.e. with a GFlowNet.
    NOTE: The code doesn't support SubTB and learnable backward policy for simplicity and because Sendera et al. found it didn't help. Could be added.
    NOTE: Local search as in Sendera et al. is also not supported, but very well could be
    NOTE: Replay buffer could also be implemented
    NOTE: Could support fine-tuning an unconditional diffusion model trained with DSM (i.e. with RelativeTB) without too many modifications (like in https://arxiv.org/pdf/2405.20971)
    """

    def __init__(self, cfg: GFNDiffusionConfig):

        assert cfg.train_direction in ["both_ways", "fwd", "bwd"]

        self.pf_std_per_traj = np.sqrt(cfg.t_scale)
        self.dt = 1.0 / cfg.trajectory_length

        self.base_task = MetaLearningTask(cfg.pretrained_id)
        for param in self.base_task.parameters():
            param.requires_grad = False

        self.model = DiT(cfg.model)

        self.cfg = cfg

    def log_reward(self, x, cond):
        # TODO: compute log-likelihood of sequence given hidden state
        pass

    def get_exploration_std(self, iter):
        if self.cfg.exploratory is False:
            return None
        if self.cfg.exploration_wd:
            exploration_std = self.cfg.exploration_factor * max(
                0, 1.0 - (iter / (self.cfg.max_steps / 2))
            )
        else:
            exploration_std = self.cfg.exploration_factor
        expl = lambda x: exploration_std
        return expl

    def setup(self, stage):
        with torch.no_grad():
            dataset_cfg = hydra.utils.instantiate(self.cfg.dataset)
            if "GRU" in self.cfg.dataset["_target_"]:
                dataset_cls = GRUDiffusionDataset
            elif "Mamba" in self.cfg.dataset["_target_"]:
                dataset_cls = MambaDiffusionDataset
            elif "KnownEncoder" in self.cfg.dataset["_target_"]:
                dataset_cls = KnownEncoderDiffusionDataset
            else:
                assert False

            dataset = dataset_cls(dataset_cfg, self.base_task, self.model)
            self.full_data = dataset

            self.train_data = self.full_data

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def split_params(self, tensor):
        mean, logvar = gaussian_params(tensor)
        if not self.cfg.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.cfg.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.0

    def get_trajectory_fwd(self, s, exploration_std, log_r, condition=None):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.cfg.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.cfg.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.cfg.trajectory_length + 1), device=self.device)
        states = torch.zeros(
            (bsz, self.cfg.trajectory_length + 1, *self.model.cfg.latent_shape),
            device=self.device,
        )

        for i in range(self.cfg.trajectory_length):
            pfs, flow = self.model(s, i * self.dt, log_r, condition)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, i] = flow

            if exploration_std is None:
                pflogvars_sample = pflogvars.detach()
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars.detach()
                else:
                    add_log_var = torch.full_like(
                        pflogvars, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2
                    )
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var).detach()

            s_ = (
                s
                + self.dt * pf_mean.detach()
                + np.sqrt(self.dt)
                * (pflogvars_sample / 2).exp()
                * torch.randn_like(s, device=self.device)
            )

            noise = ((s_ - s) - self.dt * pf_mean) / (
                np.sqrt(self.dt) * (pflogvars / 2).exp()
            )
            logpf[:, i] = -0.5 * (
                noise**2 + logtwopi + np.log(self.dt) + pflogvars
            ).sum(1)

            back_mean_correction, back_var_correction = torch.ones_like(
                s_
            ), torch.ones_like(s_)

            if i > 0:
                back_mean = (
                    s_ - self.dt * s_ / ((i + 1) * self.dt) * back_mean_correction
                )
                back_var = (
                    (self.pf_std_per_traj**2)
                    * self.dt
                    * i
                    / (i + 1)
                    * back_var_correction
                )
                noise_backward = (s - back_mean) / back_var.sqrt()
                logpb[:, i] = -0.5 * (
                    noise_backward**2 + logtwopi + back_var.log()
                ).sum(1)

            s = s_
            states[:, i + 1] = s

        return states, logpf, logpb, logf

    def get_trajectory_bwd(self, s, exploration_std, log_r, condition=None):
        bsz = s.shape[0]
        logpf = torch.zeros((bsz, self.cfg.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.cfg.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.cfg.trajectory_length + 1), device=self.device)
        states = torch.zeros(
            (bsz, self.cfg.trajectory_length + 1, *self.model.cfg.latent_shape),
            device=self.device,
        )
        states[:, -1] = s

        for i in range(self.cfg.trajectory_length):
            if i < self.cfg.trajectory_length - 1:
                back_mean_correction, back_var_correction = torch.ones_like(
                    s
                ), torch.ones_like(s)

                mean = s - self.dt * s / (1.0 - i * self.dt) * back_mean_correction
                var = (
                    ((self.pf_std_per_traj**2) * self.dt * (1.0 - (i + 1) * self.dt))
                    / (1 - i * self.dt)
                    * back_var_correction
                )
                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(
                    s, device=self.device
                )
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, self.cfg.trajectory_length - i - 1] = -0.5 * (
                    noise_backward**2 + logtwopi + var.log()
                ).sum(1)
            else:
                s_ = torch.zeros_like(s)

            pfs, flow = self.model(s_, (1.0 - (i + 1) * self.dt), log_r, condition)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, self.cfg.trajectory_length - i - 1] = flow

            noise = ((s - s_) - self.dt * pf_mean) / (
                np.sqrt(self.dt) * (pflogvars / 2).exp()
            )
            logpf[:, self.cfg.trajectory_length - i - 1] = -0.5 * (
                noise**2 + logtwopi + np.log(self.dt) + pflogvars
            ).sum(1)

            s = s_
            states[:, self.cfg.trajectory_length - i - 1] = s

        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r, condition=None):
        s = torch.zeros(batch_size, *self.model.cfg.latent_shape).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r, condition)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std, condition=None):
        s = torch.zeros(batch_size, *self.model.cfg.latent_shape).to(self.device)
        return self.get_trajectory_fwd(
            s, exploration_std, log_r=None, condition=condition
        )[0][:, -1]

    def fwd_tb_loss(
        self, initial_state=None, exploration_std=None, return_exp=False, condition=None
    ):
        if initial_state == None:
            initial_state = torch.zeros(
                self.cfg.batch_size, *self.model.cfg.latent_shape
            ).to(self.device)

        states, log_pfs, log_pbs, log_fs = self.get_trajectory_fwd(
            initial_state, exploration_std, condition
        )

        with torch.no_grad():
            log_r = self.log_reward(states[:, -1], condition).detach()

        loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)
        if return_exp:

            return loss.mean(), states, log_pfs, log_pbs, log_r
        else:

            return loss.mean()

    def bwd_tb_loss(self, samples=None, exploration_std=None, condition=None):

        if samples == None:
            samples = self.sleep_phase_sample(
                self.cfg.batch_size, exploration_std, condition=condition
            ).to(self.device)

        states, log_pfs, log_pbs, log_fs = self.get_trajectory_bwd(
            samples, exploration_std, log_r=self.log_reward, condition=condition
        )

        with torch.no_grad():
            log_r = self.log_reward(states[:, -1], condition).detach()

        loss = 0.5 * ((log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r) ** 2)

        return loss.mean()

    def training_step(self, batch, batch_idx):
        latent = batch["latent"]

        cond = (
            batch["cond_input_ids"]
            if batch["cond_tokens"] is None
            else batch["cond_tokens"]
        )
        exploration_std = self.get_exploration_std(self.global_step)

        if self.cfg.train_direction == "both_ways":
            if self.global_step % 2 == 0:
                loss = self.fwd_tb_loss(exploration_std=exploration_std, condition=cond)
            else:
                loss = self.bwd_tb_loss(exploration_std=exploration_std, condition=cond)
        elif self.cfg.train_direction == "fwd":
            loss = self.fwd_tb_loss(exploration_std=exploration_std, condition=cond)
        else:
            loss = self.bwd_tb_loss(exploration_std=exploration_std, condition=cond)

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
        )

        return loss
