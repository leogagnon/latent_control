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

from models.base import EncoderModel
from models.diffusion import DiT, DiTConfig
from x_transformers import Encoder, TransformerWrapper, Decoder


@dataclass
class KnownEncoderConfig:
    n_embd: int
    latents_shape: Optional[List[int]] = None
    orth_init: bool = True
    sequential: bool = False


class KnownEncoder(EncoderModel):
    """Encodes the ground-truth latents into a continuous space by a superposition of orthogonal vectors"""

    def __init__(self, cfg: Optional[KnownEncoderConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = KnownEncoderConfig(**kwargs)

        assert cfg.latents_shape != None

        self.latent_embedding = nn.ModuleList(
            [nn.Embedding(n, cfg.n_embd) for n in cfg.latents_shape]
        )
        if cfg.orth_init:
            # Initiallize all embeddings to be orthogonal to each other
            directions = nn.init.orthogonal_(
                torch.zeros((cfg.n_embd, cfg.n_embd)), gain=3.0
            )
            i = 0
            for e in self.latent_embedding:
                weight_ = torch.zeros_like(e.weight)
                for j in range(e.weight.shape[0]):
                    weight_[j] = directions[i]
                    i += 1
                e.weight = nn.Parameter(weight_)
        self.cfg = cfg

    def forward(self, tokens=None, true_latents=None):
        out = torch.stack(
            [self.latent_embedding[i](l) for i, l in enumerate(true_latents.T)], dim=1
        )
        if not self.cfg.sequential:
            out = out.sum(1, keepdim=True)

        return out


@dataclass
class TransformerEncoderConfig:
    max_seq_len: int
    num_tokens: int
    n_layer: int
    n_head: int
    n_embd: int
    positional_encodings: bool = True
    causal_mask: bool = False
    dropout: float = 0.0
    bias: bool = True
    tag: Optional[str] = None


class TransformerEncoder(TransformerWrapper, EncoderModel):
    """Basic encoder-only transformer"""

    def __init__(self, cfg: Optional[TransformerEncoderConfig] = None, **kwargs):
        if cfg is None:
            cfg = TransformerEncoderConfig(**kwargs)
        attn_layer_cls = Decoder if cfg.causal_mask else Encoder
        super().__init__(
            num_tokens=cfg.num_tokens,
            max_seq_len=cfg.max_seq_len,
            attn_layers=attn_layer_cls(
                dim=cfg.n_embd, depth=cfg.n_layer, heads=cfg.n_head
            ),
            scaled_sinu_pos_emb=cfg.positional_encodings,
            use_abs_pos_emb=cfg.positional_encodings,
        )  # we use a Decoder to use a causal mask
        self.cfg = cfg

    def forward(self, input_ids, true_latents=None, attn_mask=None):
        out = super().forward(x=input_ids, mask=attn_mask, return_embeddings=True)

        return out


@dataclass
class GRUEncoderConfig:
    num_tokens: int
    n_layer: int
    n_embd: int
    tag: Optional[str] = None


class GRUEncoder(EncoderModel):
    def __init__(self, cfg: Optional[GRUEncoderConfig] = None, **kwargs) -> None:
        if cfg is None:
            cfg = GRUEncoderConfig(**kwargs)
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=cfg.num_tokens, embedding_dim=cfg.n_embd
        )
        self.backbone = nn.GRU(
            input_size=cfg.n_embd,
            hidden_size=cfg.n_embd,
            num_layers=cfg.n_layer,
            batch_first=True,
        )
        self.cfg = cfg

    def forward(self, input_ids, true_latents=None, attn_mask=None):
        """
        context_enc : Initial hidden state. Shape (n_layer, batch, n_embd). Defaults to None.
        """
        x = self.embedding(input_ids)
       
        x, hiddens = self.backbone(x)

        return x


# We don't care about this for now
# This will be when we use the DiffusionEncoder in a meta-learning setting
# For now this is only a wrapper over DiffusionTransformer
@dataclass
class DiffusionEncoderConfig(DiTConfig):
    pass


class DiffusionEncoder(DiT):
    pass
