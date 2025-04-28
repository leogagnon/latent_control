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
import jax
import jax.random as jr
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch2jax import j2t, t2j
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from x_transformers import Decoder, Encoder, TransformerWrapper

from data.hmm import MetaHMM
from models.base import EncoderModel
from models.diffusion import DiT, DiTConfig
from tasks.metalearn import MetaLearningTask


@dataclass
class KnownEncoderConfig:
    out_dim: int
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
            [nn.Embedding(n, cfg.out_dim) for n in cfg.latents_shape]
        )
        if cfg.orth_init:
            # Initiallize all embeddings to be orthogonal to each other
            directions = nn.init.orthogonal_(
                torch.zeros((cfg.out_dim, cfg.out_dim)), gain=3.0
            )
            i = 0
            for e in self.latent_embedding:
                weight_ = torch.zeros_like(e.weight)
                for j in range(e.weight.shape[0]):
                    weight_[j] = directions[i]
                    i += 1
                e.weight = nn.Parameter(weight_)
        self.cfg = cfg

    @property
    def out_dim(self):
        "Dimension of the output (e.g. logits)"
        return self.cfg.out_dim

    @property
    def enc_len(self):
        "Lenght of the encoding"
        return len(self.cfg.latents_shape) if self.cfg.sequential else 1

    def forward(self, true_latents=None, **kwargs):
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
    sin_posemb: bool
    causal_mask: bool
    bottleneck: bool
    out_dim: Optional[int] = None
    tag: Optional[str] = None


class TransformerEncoder(TransformerWrapper, EncoderModel):
    """Basic encoder-only transformer"""

    def __init__(self, cfg: Optional[TransformerEncoderConfig] = None, **kwargs):
        if cfg is None:
            cfg = TransformerEncoderConfig(**kwargs)
        attn_layer_cls = Decoder if cfg.causal_mask else Encoder
        if cfg.out_dim == None:
            cfg.out_dim = cfg.num_tokens
        super().__init__(
            num_tokens=cfg.num_tokens,
            max_seq_len=cfg.max_seq_len,
            attn_layers=attn_layer_cls(
                dim=cfg.n_embd, depth=cfg.n_layer, heads=cfg.n_head
            ),
            scaled_sinu_pos_emb=cfg.sin_posemb,
            use_abs_pos_emb=cfg.sin_posemb,
            use_cls_token=cfg.bottleneck,
            num_cls_tokens=1,
            logits_dim=cfg.out_dim,
        )

        self.cfg = cfg

    def out_proj(self, hidden):
        return self.to_logits(hidden)

    @property
    def out_dim(self):
        "Dimension of the output (e.g. logits)"
        return self.cfg.out_dim

    @property
    def hidden_dim(self):
        "Dimension of the embeddings"
        return self.cfg.n_embd

    @property
    def enc_len(self):
        "Lenght of the encoding"
        return 1 if self.cfg.bottleneck else None

    def forward(
        self, input_ids: torch.Tensor, return_embeddings: bool = False, **kwargs
    ):
        # Run <input_ids> through the backbone Transformer
        out = super().forward(x=input_ids, return_embeddings=return_embeddings)
        if self.cfg.bottleneck:
            out = out[:, None]

        return out


@dataclass
class GRUEncoderConfig:
    num_tokens: int
    n_layer: int
    n_embd: int
    return_last: Optional[bool] = False
    out_dim: Optional[int] = None
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

        if cfg.out_dim == None:
            cfg.out_dim = cfg.num_tokens

        self.out_proj = nn.Linear(cfg.n_embd, cfg.out_dim)

        self.cfg = cfg

    @property
    def out_dim(self):
        "Dimension of the output (e.g. logits)"
        return self.cfg.out_dim

    @property
    def hidden_dim(self):
        "Dimension of the embeddings"
        return self.cfg.n_embd

    @property
    def enc_len(self):
        "Lenght of the encoding"
        return None

    def forward(
        self, input_ids: torch.Tensor, return_embeddings: bool = False, **kwargs
    ):
        """
        context_enc : Initial hidden state. Shape (n_layer, batch, n_embd). Defaults to None.
        """
        x = self.embedding(input_ids)
        x, hiddens = self.backbone(x)

        if self.cfg.return_last:
            x = x[:,[-1]]

        if not return_embeddings:
            x = self.out_proj(x)

        return x


@dataclass
class ContextEncoderConfig:
    trainable: bool
    normalize: bool
    context_is_prefix: Optional[bool] = None
    context_length: Optional[int] = None
    pool_last_n: Optional[int] = None
    pretrained_id: Optional[str] = None
    backbone: Optional[dict] = None
    out_dim: Optional[int] = None

class ContextEncoder(EncoderModel):

    def __init__(self, cfg: Optional[ContextEncoderConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = ContextEncoderConfig(**kwargs)

        if cfg.pretrained_id != None:
            assert cfg.backbone == None
            task = MetaLearningTask.load_from_checkpoint(
                os.path.join(
                    os.environ["LATENT_CONTROL_CKPT_DIR"],
                    cfg.pretrained_id,
                    "last.ckpt",
                ),
                strict=False,
            )
            if "ContextEncoder" in str(task.model.encoder.__class__):
                self.backbone = task.model.encoder.backbone
            else:
                self.backbone = task.model.encoder
        else:
            assert cfg.backbone != None
            self.backbone = hydra.utils.instantiate(cfg.backbone)
        self.backbone: EncoderModel

        # Potentially replace the output projection the backbone
        if cfg.out_dim != None:
            self.out_proj = nn.Linear(self.backbone.hidden_dim, cfg.out_dim)
        else:
            self.out_proj = nn.Identity()

        if not cfg.trainable:
            self.requires_grad_(False)
        self.out_proj.requires_grad_(True)

        self.is_known = "KnownEncoder" in str(self.backbone.__class__)
        if not self.is_known:
            assert (cfg.context_is_prefix != None) & (cfg.context_length != None)

        self.cfg = cfg
        
    @property
    def out_dim(self):
        "Dimension of the output (e.g. logits)"
        return self.backbone.out_dim

    @property
    def enc_len(self):
        "Lenght of the encoding"
        return 1 if self.cfg.pool_last_n != None else self.backbone.enc_len

    def forward(
        self,
        input_ids: torch.Tensor = None,
        dataset: Optional[MetaHMM] = None,
        states: Optional[torch.Tensor] = None,
        true_envs: Optional[torch.Tensor] = None,
        use_input_ids: bool = False,
        **kwargs
    ):

        if self.is_known:
            true_latents = j2t(dataset.index_to_latent[t2j(true_envs.cpu())]).long().cuda()
            out = self.backbone(true_latents=true_latents)
            return out

        # Override <input_ids> with newly generated context
        if not use_input_ids:
            seed = dataset.generator.integers(0, 1e10)
            if self.cfg.context_is_prefix:
                # Extend the <input_ids> to the past
                initial_states = t2j(states)[:, 0]
                input_ids, _ = jax.vmap(dataset.sample, (0, None, 0, 0, None))(
                    t2j(true_envs),
                    self.cfg.context_length + 1,
                    jr.split(jr.PRNGKey(seed), len(true_envs)),
                    initial_states,  # end on the start of the sequence
                    True,  # reverse
                )
            else:
                # Generate a new independent sequence
                input_ids, _ = jax.vmap(dataset.sample, (0, None, 0, None, None))(
                    t2j(true_envs),
                    self.cfg.context_length + 1,
                    jr.split(jr.PRNGKey(seed), len(true_envs)),
                    None,  # Random initial state
                    False,  # reverse
                )

            input_ids = j2t(input_ids)[:, :-1]

        out = self.backbone(input_ids, return_embeddings=self.cfg.out_dim != None)
        out = self.out_proj(out)

        # Pool across last n tokens
        if self.cfg.pool_last_n != None:
            out = out[:, -self.cfg.pool_last_n :]
            out = out.mean(1, keepdim=True)

        # Normalize latent
        if self.cfg.normalize:
            out = out / out.norm(dim=2, keepdim=True)

        return out
