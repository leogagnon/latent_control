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
from torch.utils.data import DataLoader, StackDataset, Subset, TensorDataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN

from data.hmm import (CompositionalHMMDataset, CompositionalHMMDatasetConfig,
                      PrecomputedDataset, SubsetIntervened)
from models.base import MetaLearner, MetaLearnerConfig
from tasks.metalearn import MetaLearningTask
from data.active import HMMEnv, ReplayBufferDataset


@dataclass
class ActiveLearningConfig:
    pretrained_id: str
    batch_size: int
    rollout_len: int
    buffer_capacity: int
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

        # Load MetaLearningTask
        base_task = MetaLearningTask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"], cfg.pretrained_id, "last.ckpt"
            ),
            strict=False,
        )
        self.model = base_task.model
        self.full_data = base_task.full_data      

        self.hmm_env = HMMEnv(self.full_data, seed=0)  
        self.buffer = ReplayBufferDataset(self.full_data, self.cfg.batch_size, self.cfg.buffer_capacity)

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.buffer, batch_size=1)
    
    def rollout_trajs(self, n: int):
        remaining = n
        done = False
        sequences = []
    
        while done == False:
            size = min(self.cfg.batch_size, remaining)
            seq = [self.hmm_env.reset(size)]
            
            for i in range(self.cfg.rollout_len):
                actions = self.model(seq[-1])
                obs = self.hmm_env.step(actions)
                seq += [actions, obs]
            
            sequences += [jnp.stack(seq)]
            
            remaining -= size
            if remaining <= 0:
                assert remaining == 0
                done = True

        return torch.stack(sequences)

    def training_step(self, batch, batch_idx=None):
        pass
        # rollout some trajectories
        # train on the batch of the replay buffer :
        #   - compute the reward/value of the sequence (train the policy)
        #   - compute the observation prediction with the model (train the critic)