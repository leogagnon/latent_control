import copy
import math
import os
from abc import ABC
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from typing import *

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import utils

from models.base import EncoderModel
from models.diffusion import (DiffusionTransformer, DiffusionTransformerConfig,
                              GaussianDiffusion, GaussianDiffusionConfig)
from models.x_transformer import (AbsolutePositionalEmbedding, Encoder,
                                  LearnedSinusoidalPosEmb, SinusoidalPosEmb,
                                  TransformerWrapper,
                                  VariationalFourierFeatures, exists,
                                  groupby_prefix_and_trim, init_zero_)


@dataclass
class KnownEncoderConfig:
    n_embd: int
    latents_shape: List[int]


class KnownEncoder(EncoderModel):
    def __init__(self, cfg: Optional[KnownEncoderConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = KnownEncoderConfig(**kwargs)
        self.latent_embedding = nn.ModuleList(
            [nn.Embedding(n, cfg.n_embd) for n in cfg.latents_shape]
        )

    def forward(self, true_latents, tokens=None):
        out = torch.stack(
            [self.latent_embedding[i](l) for i, l in enumerate(true_latents.T)]
        ).sum(0)
        return out[:, None]


@dataclass
class TransformerEncoderConfig:
    max_seq_len: int
    num_tokens: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True
    positional_encodings: bool = True
    tag: Optional[str] = None


class TransformerEncoder(TransformerWrapper, EncoderModel):
    """Basic encoder-only transformer"""

    def __init__(self, cfg: Optional[TransformerEncoderConfig] = None, **kwargs):
        if cfg is None:
            cfg = TransformerEncoderConfig(**kwargs)
        super().__init__(
            num_tokens=cfg.num_tokens,
            max_seq_len=cfg.max_seq_len,
            attn_layers=Encoder(dim=cfg.n_embd, depth=cfg.n_layers, heads=cfg.n_head),
        )

    def forward(self, input_ids, true_latents=None, attn_mask=None):
        out = super().forward(x=input_ids, mask=attn_mask)
        out = out[:, -1, :]

        return out


@dataclass
class DiffusionEncoderConfig:
    model_cfg: DiffusionTransformerConfig
    diffusion_cfg: GaussianDiffusionConfig

class DiffusionEncoder(GaussianDiffusion, EncoderModel):
    def __init__(self, cfg: DiffusionEncoderConfig):
        model = hydra.utils.instantiate()
        super().__init__(model, cfg.diffusion_cfg)

    def forward(input_ids, true_latents=None):
        # We don't care about this for now
        # This will be when we use the DiffusionEncoder in a meta-learning setting
        pass