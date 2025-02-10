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
from models.diffusion_architectures import DiT, DiTConfig


@dataclass
class GFNTaskConfig:
    model: DiTConfig 
    pretrained_id: str
    batch_size: int
    val_split: float
    loss: str
    lr: float
    lr_scheduler: bool
    sampling_timesteps: int    
    train_schedule: str
    objective: str
    sampling_schedule: Optional[str]
    scale: float
    sampler: str
    normalize_latent: bool