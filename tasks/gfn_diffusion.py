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
from einops import rearrange, reduce, repeat
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchmetrics.functional import kl_divergence
from transformers.activations import ACT2FN

from data.diffusion import *
from tasks.metalearn import MetaLearningTask
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import jax.numpy as jnp
import jax
from jax.scipy.special import rel_entr
from models.diffusion import DiT, DiTConfig
from tasks.utils import CustomCheckpointing

logtwopi = math.log(2 * math.pi)


@dataclass
class GFNDiffusionConfig:
    model: DiTConfig
    dataset: dict
    pretrained_id: str
    batch_size: int
    lr: float
    max_steps: int

    log_var_range: float = 4.0
    t_scale: float = 5.0
    trajectory_length: int = 100
    train_direction: str = "fwd"
    exploratory: bool = False
    exploration_factor: float = 0.1
    exploration_wd: bool = False
    vargrad_repeats: int = 10


class GFNDiffusion(L.LightningModule, CustomCheckpointing):
    """
    Trains a diffusion model from an energy function (Sendera et al. 2025; https://arxiv.org/pdf/2402.05098), i.e. with a GFlowNet.
    NOTE: For simplicity or because previous work has shown they don't work well in conditional settings, a lot of possible features are not implemented yet.
        Notably left-out features (which could easily be re-added) are :
        1) SubTB
        2) Classical TB loss with LogZ estimation (we use VarGrad). Would need to add a flow model to the DiT.
        3) Learnable backward policy (we use constant; brownian bridge)
        4) Local search
        5) Replay buffer
        6) Relative TB (https://arxiv.org/pdf/2405.20971)
        7) Unconditional diffusion (probably a few minor things to change, e.g. the loss)
    """

    def __init__(self, cfg: GFNDiffusionConfig):
        super().__init__()

        assert cfg.train_direction in ["both_ways", "fwd", "bwd"]

        self.pf_std_per_traj = np.sqrt(cfg.t_scale)
        self.dt = 1.0 / cfg.trajectory_length

        self.base_task = MetaLearningTask.from_id(cfg.pretrained_id)
        for param in self.base_task.parameters():
            param.requires_grad = False

        self.model = DiT(cfg.model)

        self.cfg = cfg

    def log_reward(self, x, cond_input_ids, cond_mask):
        """Returns log_p(sequence | latent) from the decoder. This is the log-reward for the GFlowNet"""
        logits = self.base_task.model.decoder(input_ids=cond_input_ids, context_enc=x)
        log_prob = torch.log_softmax(logits, dim=-1)[:, :-1, :-1]

        loglike = log_prob[
            torch.arange(log_prob.shape[0], device=log_prob.device)[:, None],
            torch.arange(log_prob.shape[1], device=log_prob.device).repeat(
                log_prob.shape[0], 1
            ),
            cond_input_ids[:, 1:],
        ]

        log_reward = loglike[cond_mask[:,1:]].sum()

        return log_reward

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

            dataset = dataset_cls(dataset_cfg, self)
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
        mean, logvar = torch.chunk(tensor, 2, dim=-1)
        if not self.model.cfg.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.cfg.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.0

    def get_trajectory_fwd(self, s, exploration_std=None, cond=None, cond_input_ids=None, cond_mask=None):
        bsz = s.shape[0]

        logpf = torch.empty((bsz, self.cfg.trajectory_length), device=self.device).fill_(0.0)
        logpb = torch.empty((bsz, self.cfg.trajectory_length), device=self.device).fill_(0.0)
        states = torch.empty(
            (bsz, self.cfg.trajectory_length + 1, *self.model.cfg.latent_shape),
            device=self.device,
        ).fill_(0.0)

        for i in range(self.cfg.trajectory_length):
            time = torch.full(size=(bsz,), fill_value=i * self.dt, device=self.device)
            pfs = self.model.forward(
                x=s,
                time=time,
                log_r_fn=self.log_reward,
                cond=cond,
                cond_input_ids=cond_input_ids,
                cond_mask=cond_mask,
            )
            pf_mean, pflogvars = self.split_params(pfs)

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
            ).view(bsz,-1).sum(1)

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
                ).view(bsz,-1).sum(1)

            s = s_
            states[:, i + 1] = s

        return states, logpf, logpb

    def get_trajectory_bwd(
        self, s, exploration_std=None, cond=None, cond_input_ids=None, cond_mask=None
    ):
        bsz = s.shape[0]
        logpf = torch.empty((bsz, self.cfg.trajectory_length), device=self.device).fill_(0.0)
        logpb = torch.empty((bsz, self.cfg.trajectory_length), device=self.device).fill_(0.0)
        states = torch.empty(
            (bsz, self.cfg.trajectory_length + 1, *self.model.cfg.latent_shape),
            device=self.device,
        ).fill_(0.0)

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
                ).view(bsz,-1).sum(1)
            else:
                s_ = torch.zeros_like(s)
            
            time = torch.full(size=(bsz,), fill_value=(1.0 - (i + 1) * self.dt), device=self.device)
            pfs = self.model.forward(
                x=s_,
                time=time,
                log_r_fn=self.log_reward,
                cond=cond,
                cond_input_ids=cond_input_ids,
                cond_mask=cond_mask,
            )

            pf_mean, pflogvars = self.split_params(pfs)

            noise = ((s - s_) - self.dt * pf_mean) / (
                np.sqrt(self.dt) * (pflogvars / 2).exp()
            )
            logpf[:, self.cfg.trajectory_length - i - 1] = -0.5 * (
                noise**2 + logtwopi + np.log(self.dt) + pflogvars
            ).view(bsz,-1).sum(1)

            s = s_
            states[:, self.cfg.trajectory_length - i - 1] = s

        return states, logpf, logpb

        

    # Called <fwd_tb_avg_cond> in original code [https://github.com/GFNOrg/gfn-diffusion/blob/15a0d78d6d2fd6cfc620ce0102e67f25f042fa94/vae/gflownet_losses.py#L56]
    def fwd_vargrad_loss(
        self,
        initial_state=None,
        exploration_std=None,
        return_exp=False,
        cond=None,
        cond_input_ids=None,
        cond_mask=None,
        repeats=10,
    ):

        if initial_state == None:
            initial_state = torch.zeros(
                self.cfg.batch_size, *self.model.cfg.latent_shape
            ).to(self.device)

        cond = repeat(cond, "b l d -> (r b) l d", r=repeats)
        cond_mask = repeat(cond_mask, "b l -> (r b) l", r=repeats)
        cond_input_ids = repeat(cond_input_ids, "b l -> (r b) l", r=repeats)
        initial_state = repeat(initial_state, "b L D -> (r b) L D", r=repeats)

        states, log_pfs, log_pbs = self.get_trajectory_fwd(
            s=initial_state,
            exploration_std=exploration_std,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
        )
        with torch.no_grad():
            log_r = self.log_reward(x=states[:, -1], cond_input_ids=cond_input_ids, cond_mask=cond_mask).detach()

        log_Z = repeat(log_r + log_pbs.sum(-1) - log_pfs.sum(-1), '(r b) -> r b', r=repeats).mean(0, keepdim=True)   
        loss = log_Z + repeat(log_pfs.sum(-1) - log_r - log_pbs.sum(-1), '(r b) -> r b', r=repeats)     
        

        if return_exp:
            return 0.5 * (loss**2).mean(), states, log_pfs, log_pbs, log_r
        else:
            return 0.5 * (loss**2).mean()

    def bwd_vargrad_loss(
        self,
        samples=None,
        exploration_std=None,
        cond=None,
        cond_input_ids=None,
        cond_mask=None,
        repeats=10,
    ):

        if samples == None:
            # Gather samples by rolling the policy forward
            s = torch.zeros(self.cfg.batch_size, *self.model.cfg.latent_shape).to(self.device)
            samples = self.get_trajectory_fwd(s, exploration_std=None, cond=cond, cond_input_ids=cond_input_ids, cond_mask=cond_mask)[0][:, -1]

        cond = repeat(cond, "b l d -> (r b) l d", r=repeats)
        cond_mask = repeat(cond_mask, "b l -> (r b) l", r=repeats)
        cond_input_ids = repeat(cond_input_ids, "b l -> (r b) l", r=repeats)
        samples = repeat(samples, "b L D -> (r b) L D", r=repeats)

        states, log_pfs, log_pbs = self.get_trajectory_bwd(
            s=samples,
            exploration_std=exploration_std,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
        )

        with torch.no_grad():
            log_r = self.log_reward(x=states[:, -1], cond_input_ids=cond_input_ids, cond_mask=cond_mask).detach()

        log_Z = repeat(log_r + log_pbs.sum(-1) - log_pfs.sum(-1), '(r b) -> r b', r=repeats).mean(0, keepdim=True)   
        loss = log_Z + repeat(log_pfs.sum(-1) - log_r - log_pbs.sum(-1), '(r b) -> r b', r=repeats)     

        return 0.5 * (loss**2).mean()

    def training_step(self, batch, batch_idx):
        latent = batch["latent"]

        cond_mask = torch.logical_not(batch["cond_ignore_mask"])
        exploration_std = self.get_exploration_std(self.global_step)

        if self.cfg.train_direction == "both_ways":
            if self.global_step % 2 == 0:
                loss = self.fwd_vargrad_loss(
                    exploration_std=exploration_std,
                    cond=batch["cond_tokens"],
                    cond_input_ids=batch["cond_input_ids"],
                    cond_mask=cond_mask,
                    repeats=self.cfg.vargrad_repeats,
                )
            else:
                loss = self.bwd_vargrad_loss(
                    exploration_std=exploration_std,
                    cond=batch["cond_tokens"],
                    cond_input_ids=batch["cond_input_ids"],
                    cond_mask=cond_mask,
                    repeats=self.cfg.vargrad_repeats,
                )
        elif self.cfg.train_direction == "fwd":
            loss = self.fwd_vargrad_loss(
                exploration_std=exploration_std,
                cond=batch["cond_tokens"],
                cond_input_ids=batch["cond_input_ids"],
                cond_mask=cond_mask,
                repeats=self.cfg.vargrad_repeats,
            )
        else:
            loss = self.bwd_vargrad_loss(
                exploration_std=exploration_std,
                cond=batch["cond_tokens"],
                cond_input_ids=batch["cond_input_ids"],
                cond_mask=cond_mask,
                repeats=self.cfg.vargrad_repeats,
            )

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
        )

        return loss
