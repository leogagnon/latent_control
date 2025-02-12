import argparse
import copy
import csv
import json
import math
import os
import random
import timeit
from abc import ABC, abstractmethod, abstractproperty
from collections import Counter, defaultdict, namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from torch import einsum, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from models.x_transformer import (
    AbsolutePositionalEmbedding,
    Encoder,
    ScaledSinusoidalEmbedding,
    SinusoidalPosEmb,
    init_zero_,
)

class LangevinScalingModel(nn.Module):
    def __init__(self, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 3,
                 zero_init: bool = False):
        super(LangevinScalingModel, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.register_buffer('pe', pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t):
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)


@dataclass
class DiTConfig:
    latent_shape: Tuple[int]
    n_layers: int
    n_heads: int
    dropout: float
    scale_shift: bool
    num_dense_connections: int
    cond_encoder_kwargs: Optional[dict]
    seq_conditional: Optional[bool] = False
    seq_conditional_dim: Optional[int] = None
    class_conditional: Optional[bool] = False
    num_classes: Optional[int] = 0

    # DDPM features
    seq_unconditional_prob: Optional[float] = 0.1
    class_unconditional_prob: Optional[float] = 0.1
    self_condition: Optional[bool] = False
    train_prob_self_cond: Optional[float] = 0.5

    # GFlowNet features
    langevin: Optional[bool] = False
    lgv_clip: Optional[float] = None
    gfn_clip: Optional[float] = None
    learned_variance: Optional[bool] = False


class DiT(nn.Module):
    """Diffusion transformer (https://arxiv.org/pdf/2212.09748) with many tricks and add-ons (self-conditionning, sequence-conditioning, class-conditionning, langevin model)"""

    def __init__(self, cfg: DiTConfig):
        super().__init__(cfg)

        assert isinstance(cfg.latent_shape, Iterable) and (len(cfg.latent_shape) == 2)
        self.n_embd = cfg.latent_shape[1]

        # Init model
        sinu_pos_emb = SinusoidalPosEmb(cfg.latent_shape[1])
        fourier_dim = self.n_embd

        time_emb_dim = self.n_embd * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_pos_embed_mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_emb_dim, self.n_embd)
        )

        self.pos_emb = AbsolutePositionalEmbedding(self.n_embd, cfg.latent_shape[0])

        self.cross = cfg.seq_conditional

        self.latent_encoder = Encoder(
            dim=self.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.dropout,  # dropout post-attention
            ff_dropout=cfg.dropout,  # feedforward dropout
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=self.cross,
            time_emb_dim=self.n_embd * 4 if cfg.scale_shift else None,
            num_dense_connections=cfg.num_dense_connections,
        )

        if cfg.class_conditional:
            assert False, 'Careful, never tested the class conditional setting for real.'
            assert cfg.num_classes > 0
            self.class_embedding = nn.Sequential(
                nn.Embedding(cfg.num_classes + 1, self.n_embd),
                nn.Linear(self.n_embd, time_emb_dim),
            )
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(
                probs=cfg.class_unconditional_prob
            )
        if cfg.seq_conditional:
            self.null_embedding_cond = nn.Embedding(1, self.n_embd)
            self.cond_proj = nn.Linear(cfg.seq_conditional_dim, self.n_embd)

        if cfg.self_condition:
            self.input_proj = nn.Linear(cfg.latent_shape[1] * 2, self.n_embd)
            self.init_self_cond = nn.Parameter(torch.randn(1, cfg.latent_shape[1]))
            nn.init.normal_(self.init_self_cond, std=0.02)
        else:
            self.input_proj = nn.Linear(cfg.latent_shape[1], self.n_embd)
        self.norm = nn.LayerNorm(self.n_embd)
        self.output_proj = nn.Linear(
            self.n_embd,
            cfg.latent_shape[1] * 2 if cfg.learned_variance else cfg.latent_shape[1],
        )

        if cfg.langevin:
            self.langevin_scaling_model = LangevinScalingModel(
                t_dim=self.n_embd, hidden_dim=64, out_dim=1
            )

        if cfg.cond_encoder_kwargs != None:
            if cfg.cond_encoder_kwargs["vocab_size"] != None:
                self.cond_embedding = nn.Embedding(
                    num_embeddings=cfg.cond_encoder_kwargs["vocab_size"],
                    embedding_dim=cfg.seq_conditional_dim,
                )
                self.cond_posemb = ScaledSinusoidalEmbedding(cfg.seq_conditional_dim)

            self.cond_encoder = Encoder(
                dim=cfg.cond_encoder_kwargs["n_embd"],
                depth=cfg.cond_encoder_kwargs["n_layers"],
                heads=cfg.cond_encoder_kwargs["n_heads"],
            )


        self.cfg = cfg

        init_zero_(self.output_proj)

    def cond_embed(self, x):
        if hasattr(self, "cond_embedding"):
            x = self.cond_embedding(x)
        if hasattr(self, "cond_posemb"):
            x = x + self.cond_posemb(x)
        return x

    def forward(
        self,
        x : torch.Tensor,
        time,
        x_self_cond=None,
        class_id=None,
        cond=None,
        cond_mask=None,
        log_r=None
    ):  
        
        if self.cfg.langevin:
            x.requires_grad_(True)
            with torch.enable_grad():
                if cond is not None:
                    grad_log_r = torch.autograd.grad(log_r(x, cond).sum(), x)[0].detach()
                else:
                    grad_log_r = torch.autograd.grad(log_r(x).sum(), x)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.cfg.lgv_clip != None:
                    grad_log_r = torch.clip(grad_log_r, -self.cfg.lgv_clip, self.cfg.lgv_clip)

        time_emb = self.time_mlp(time * 1000)

        time_emb = rearrange(time_emb, "b d -> b 1 d")

        if self.cfg.class_conditional:
            assert class_id != None
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, "b d -> b 1 d")
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(x)

        if self.cfg.self_condition:
            if x_self_cond != None:
                x = torch.cat((x, x_self_cond), dim=-1)
            else:
                repeated_x_self_cond = repeat(
                    self.init_self_cond, "1 d -> b l d", b=x.shape[0], l=x.shape[1]
                )
                x = torch.cat((x, repeated_x_self_cond), dim=-1)

        x_input = self.input_proj(x)
        tx_input = x_input + pos_emb + self.time_pos_embed_mlp(time_emb)

        if self.cross:
            context, context_mask = [], []
            if self.cfg.seq_conditional:
                if cond is None:
                    null_context = repeat(
                        self.null_embedding_cond.weight, "1 d -> b 1 d", b=x.shape[0]
                    )
                    context.append(null_context)
                    context_mask.append(
                        torch.tensor(
                            [[True] for _ in range(x.shape[0])],
                            dtype=bool,
                            device=x.device,
                        )
                    )
                else:
                    if self.cfg.cond_encoder_kwargs != None:
                        cond = self.cond_embed(cond)
                        cond = self.cond_encoder(cond)
                    context.append(self.cond_proj(cond))
                    context_mask.append(cond_mask)
            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)

            x = self.latent_encoder(
                tx_input,
                context=context,
                context_mask=context_mask,
                time_emb=time_emb,
            )
        else:
            x = self.latent_encoder(tx_input, time_emb=time_emb)

        if self.cfg.langevin:
            scale = self.langevin_scaling_model(time)
            x += scale * grad_log_r

        x = self.norm(x)
        x = self.output_proj(x)

        if self.cfg.gfn_clip != None:
            x = torch.clip(x, -self.cfg.gfn_clip, self.cfg.gfn_clip)

        return x
