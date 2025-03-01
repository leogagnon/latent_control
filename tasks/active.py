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
from torch.utils.data import DataLoader, StackDataset, Subset, TensorDataset, ConcatDataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN

from data.hmm import (
    CompositionalHMMDataset,
    CompositionalHMMDatasetConfig,
    PrecomputedDataset,
    SubsetIntervened,
)
from models.base import MetaLearner, MetaLearnerConfig
from models.decoder import TransformerDecoder
from tasks.metalearn import MetaLearningTask
from data.active import HMMEnv


@dataclass
class ActiveLearningConfig:
    pretrained_id: str
    batch_size: int
    traj_len: int
    traj_per_epoch: int
    traj_repeat: int
    lr: Optional[float] = 1e-4


class ActiveLearning(L.LightningModule):

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
            in_features=self.model.decoder.cfg.n_embd,
            out_features=len(self.full_data.ACTIONS),
        )
        self.critic_head = nn.Linear(
            in_features=self.model.decoder.cfg.n_embd, out_features=1
        )

        # Init the HMM gym env
        self.hmm_env = HMMEnv(self.full_data, seed=0)

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def train_dataloader(self) -> DataLoader:
        # generate rollouts
        sequences = self.rollout_trajs(self.cfg.traj_per_epoch)
        dataset = self.make_ppo_dataset(sequences)

        # Potentially repeat the dataset
        if self.cfg.traj_repeat != 1:
            dataset = ConcatDataset([dataset]* self.cfg.traj_repeat)

        return DataLoader(dataset, batch_size=self.cfg.batch_size)

    def rollout_trajs(self, n: int):
        """Rollout some HMM trajectories with actions from the model

        Args:
            n (int): _description_
        """
        remaining = n
        done = False
        sequences = []

        while done == False:
            size = min(self.cfg.batch_size, remaining)
            seq = [self.hmm_env.reset(size)]

            for i in range(self.cfg.traj_len):
                action_probs = torch.nn.functional.softmax(
                    self.actor_head(
                        self.model(torch.stack(seq), return_embeddings=True)
                    )
                )
                actions, obs = self.hmm_env.step(action_probs)
                seq += [actions, obs]

            sequences += [jnp.stack(seq)]

            remaining -= size
            if remaining <= 0:
                assert remaining == 0
                done = True

        return torch.stack(sequences)

    def make_ppo_dataset(
        self, sequences: torch.Tensor, gamma: float = 0.0, lam: float = 0.95
    ):
        """Computes all relevant quantities from the rollouts (reward, returns, advantages, ...)

        Args:
            sequences (torch.Tensor): Sequences [o_1, a_1, ..., o_n, a_n]
            gamma (float, optional): Discount factor. Defaults to 0.0.
            lam (float, optional): GAE lambda. Defaults to 0.95.
        """
        bs = sequences.shape[0]
        seqlen = sequences.shape[1] // 2
        device = sequences.device

        obs_logits, embeds = self.model(sequences, return_logits_and_embeddings=True)
        obs_logits = obs_logits[:, 1::2]
        action_logits = self.actor_head(embeds[0::2])
        values = self.critic_head(embeds[0::2])

        obs = sequences[0::2]
        actions = sequences[1::2]

        # Action logprobs
        act_log_prob = torch.log_softmax(action_logits)[..., :-1][
            torch.arange(action_logits.shape[0], device=action_logits.device)[:, None],
            torch.arange(action_logits.shape[1], device=action_logits.device).repeat(
                action_logits.shape[0], 1
            ),
            actions[..., 1:],
        ]

        # Compute reward : log-likelihood of observations following noop actions
        obs_log_prob = torch.log_softmax(obs_logits)[..., :-1][
            torch.arange(obs_logits.shape[0], device=obs_logits.device)[:, None],
            torch.arange(obs_logits.shape[1], device=obs_logits.device).repeat(
                obs_logits.shape[0], 1
            ),
            obs[..., 1:],
        ]
        noop_mask = (
            sequences[:, 1::2] == self.full_data.ACTION_IDS[self.full_data.ACTIONS.NOOP]
        )
        obs_log_prob[noop_mask] = 0.0
        rewards = obs_log_prob.sum(-1)

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
                next_value = values[t + 1]

            # Compute the TD error (delta)
            delta = rewards[t] + gamma * next_value - values[t]

            # Compute the advantage using GAE formula
            advantages[t] = delta + gamma * lam * last_advantage

            # Update the running advantage for the next step backward
            last_advantage = advantages[t]

        returns = advantages + values

        data = {
            "sequences": sequences,
            "logprobs": act_log_prob,
            "advantages": advantages,
            "returns": returns,
            "values": values,
        }

        return data

    def training_step(self, batch, batch_idx=None):

        pass
        # rollout some trajectories
        # train on the batch of the replay buffer :
        #   - compute the reward/value of the sequence (train the policy)
        #   - compute the observation prediction with the model (train the critic)
