import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import *

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch2jax import j2t, t2j
from torch.nn import init
from torch.utils.data import (
    DataLoader,
    ConcatDataset,
)
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN
from models.decoder import TransformerDecoder
from tasks.metalearn import MetaLearningTask
from data.active import HMMEnv, PPODataset, PPOBatch
import math 

@dataclass
class ActiveLearningConfig:
    pretrained_id: str
    batch_size: int
    traj_len: int
    traj_per_epoch: int
    traj_repeat: int
    seed: int
    reward_type: str ='target'
    clip_coef: float = 0.2
    clip_vloss: bool = False
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    pred_coef: float = 0.5
    normalize_advantages: bool = False
    normalize_rewards: bool = True
    gae_lambda: float = 0.95
    discount_factor: float = 0.0
    rollout_batch_size: int = 1024
    lr: Optional[float] = 1e-4


class ActiveLearning(L.LightningModule):
    """
    Train a pretrained implicit model perform actions in order to minimize prediction error
    """

    def __init__(self, cfg: Optional[ActiveLearningConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(ActiveLearningConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg

        base_task = MetaLearningTask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"], cfg.pretrained_id, "last.ckpt"
            ),
            strict=False,
        )

        # Setup model
        self.model = base_task.model.decoder
        self.model: TransformerDecoder
        assert base_task.model.encoder is None

        # Setup dataset
        self.full_data = base_task.full_data
        assert self.full_data.cfg.has_actions

        # Add actor and critic heads
        self.actor_head = nn.Linear(
            in_features=self.model.cfg.n_embd,
            out_features=len(self.full_data.ACTIONS),
        )
        
        self.critic_head = nn.Linear(in_features=self.model.cfg.n_embd, out_features=1)

        # Init the HMM gym env
        self.hmm_env = HMMEnv(self.full_data, seed=cfg.seed)

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

    def train_dataloader(self) -> DataLoader:
        # Generate rollouts
        sequences = self.rollout_trajs(self.cfg.traj_per_epoch)

        # Log some of them
        table = wandb.Table(columns=['Rollouts'])
        for i in range(5):
            table.add_data(str(sequences[i].tolist()))
        wandb.log({"train/rollouts": table})

        # Process rollouts to make dataset
        dataset = self.make_ppo_dataset(sequences)

        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=lambda x: x,
        )

    @torch.no_grad()
    def rollout_trajs(self, n: int):
        """Rollout some HMM trajectories with actions from the model

        Args:
            n (int): Number of trajectories
        """
        remaining = n
        done = False
        sequences = []

        while done == False:
            # Figure out the number of HMMs
            size = min(self.cfg.rollout_batch_size, remaining)

            # Run the HMMs for <traj_len> steps
            self.hmm_env.reset(size)
            for i in range(self.cfg.traj_len):
                action_probs = torch.softmax(
                    self.actor_head(
                        self.model(self.hmm_env.history, return_embeddings=True)
                    )[:, -1],
                    dim=-1,
                )
                self.hmm_env.step(action_probs)

            sequences += [self.hmm_env.history]

            remaining -= size
            if remaining <= 0:
                assert remaining == 0
                done = True

        # Put all sequences together
        sequences = torch.concatenate(sequences, dim=0)

        return sequences

    @torch.no_grad()
    def make_ppo_dataset(self, sequences: torch.Tensor):
        """
        Computes all relevant quantities from the rollouts (reward, returns, advantages, ...)
        and puts them into a PPODataset.

        Args:
            sequences (torch.Tensor): Sequences [o_1, a_1, ..., o_n, a_n]
        """
        bs = sequences.shape[0]
        seqlen = sequences.shape[1] // 2
        device = sequences.device

        # Run model on sequence
        obs_logprob, act_logprob, act_entropy, values = self.forward(sequences)

        # Compute rewards
        if self.cfg.reward_type == 'sum':
            action_mask = (
                sequences[:, 1::2] != self.full_data.ACTIONS.NOOP.value
            )
            obs_logprob[action_mask] = -torch.inf
            rewards = torch.exp(torch.logsumexp(obs_logprob, dim=-1))
        elif self.cfg.reward_type == 'target':
            actions = sequences[:, 1::2]
            # Before timestep 10, probability of actions other than no-op should be 0.1. After it should be 1e-4
            act_target_logprob_0 = torch.where(actions[:, :10] == self.full_data.ACTIONS.NOOP.value, 0.9, 0.1/5).log().to(device=sequences.device)
            act_target_logprob_1 = torch.where(actions[:, 10:] == self.full_data.ACTIONS.NOOP.value, 1.0 - 1e-4, (1e-4)/5).log().to(device=sequences.device)
            act_target_logprob = torch.concatenate([act_target_logprob_0, act_target_logprob_1], dim=-1)
            rewards = torch.sum(obs_logprob, dim=-1) + torch.sum(act_target_logprob, dim=-1)

        if self.cfg.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        wandb.log({
            "train/avg_reward" : rewards.mean().item()}
        )

        # Compute advantages
        advantages = torch.zeros(size=(bs, seqlen), dtype=torch.float, device=device)
        last_advantage = torch.zeros(size=(bs,), dtype=torch.float, device=device)
        rewards_ = torch.zeros(size=(bs, seqlen), dtype=torch.float, device=device)
        rewards_[:, -1] = rewards

        for t in reversed(range(seqlen)):
            if t == seqlen - 1:
                # We assume that
                next_value = torch.zeros(size=(bs,), dtype=torch.float, device=device)
            else:
                next_value = values[:, t + 1]

            # Compute the TD error (delta)
            delta = rewards_[:, t] + self.cfg.discount_factor * next_value - values[:, t]

            # Compute the advantage using GAE formula
            advantages[:, t] = (
                delta + self.cfg.discount_factor * self.cfg.gae_lambda * last_advantage
            )

            # Update the running advantage for the next step backward
            last_advantage = advantages[:, t]

        returns = advantages + values

        dataset = PPODataset(
            sequences, advantages, returns, values, act_logprob, act_entropy, repeats=self.cfg.traj_repeat
        )

        return dataset

    def forward(self, sequences: torch.Tensor):

        # Embeddings and observation head
        obs_logits, embeds = self.model(sequences, return_logits_and_embeddings=True)
        obs_logits = obs_logits[:, 1::2]

        # Action and values heads
        act_logits = self.actor_head(embeds[:, 0::2])[:, :-1]
        values = self.critic_head(embeds[:, 0::2])[:, :-1, 0]

        # Logprob of observed action
        shift_actions = sequences[:, 1::2]
        act_logprob = torch.log_softmax(act_logits, dim=-1)[
            torch.arange(shift_actions.shape[0], device=shift_actions.device)[:, None],
            torch.arange(shift_actions.shape[1], device=shift_actions.device).repeat(
                shift_actions.shape[0], 1
            ),
            shift_actions,
        ]

        # Action entropy
        act_entropy = -torch.sum(torch.exp(act_logprob) * act_logprob, dim=-1)

        # Logprob of observed observations
        shift_obs = sequences[:, 2::2]
        obs_logprob = torch.log_softmax(obs_logits, dim=-1)[
            torch.arange(shift_obs.shape[0], device=shift_obs.device)[:, None],
            torch.arange(shift_obs.shape[1], device=shift_obs.device).repeat(
                shift_obs.shape[0], 1
            ),
            shift_obs,
        ]

        return obs_logprob, act_logprob, act_entropy, values

    def policy_loss(
        self, advantages: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
        )
        return torch.max(pg_loss1, pg_loss2).mean()

    def value_loss(
        self, new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        if not self.cfg.clip_vloss:
            values_pred = new_values
        else:
            values_pred = old_values + torch.clamp(
                new_values - old_values, -self.cfg.clip_coef, self.cfg.clip_coef
            )
        return self.cfg.vf_coef * F.mse_loss(values_pred, returns)

    def entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        return -entropy.mean() * self.cfg.ent_coef

    def training_step(self, batch: PPOBatch, batch_idx=None):
        
        bs = batch.sequences.shape[0]
        obs_logprob, act_logprob, act_entropy, values = self.forward(batch.sequences)

        ratio = torch.exp(act_logprob - batch.logprobs)

        advantages = batch.advantages
        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = self.policy_loss(advantages, ratio)
        v_loss = self.value_loss(values, batch.values, batch.returns)
        ent_loss = self.entropy_loss(act_entropy)

        ppo_loss = pg_loss + v_loss + ent_loss

        pred_loss = self.cfg.pred_coef * -obs_logprob.mean()

        self.log(
            "train/ppo_loss",
            ppo_loss,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=bs
        )
        self.log(
            "train/pred_loss",
            pred_loss,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=bs
        )

        return ppo_loss + pred_loss
