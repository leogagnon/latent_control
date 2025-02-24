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
from typing import Callable, Iterable, Optional, Tuple, Union

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

from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    Encoder,
    ScaledSinusoidalEmbedding,
    init_zero_,
)
from omegaconf import MISSING


@dataclass
class DiTConfig:
    n_layers: int
    n_heads: int
    dropout: float
    scale_shift: bool
    cond_encoder_kwargs: Optional[dict]
    latent_shape: Optional[Tuple[int]] = None
    n_embd: Optional[int] = None
    seq_conditional: Optional[bool] = False
    seq_conditional_dim: Optional[int] = None
    class_conditional: Optional[bool] = False
    num_classes: Optional[int] = 0
    cond_modulation: Optional[bool] = False
    adalnzero_cond: Optional[bool] = False # only for backward compat, doesn't do anything

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
    """
    Diffusion transformer (DiT, https://arxiv.org/pdf/2212.09748) with adaptive layer norm zero (adaLN-Zero) conditionning.
    Super-charged with other tricks and add-ons (self-conditionning, sequence-conditioning, class-conditionning, langevin model)
    Can be the backbone of a DSM or GFN diffusion model.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()

        self.cfg = cfg

        assert isinstance(cfg.latent_shape, Iterable) and (len(cfg.latent_shape) == 2)
        if self.cfg.n_embd == None:
            self.cfg.n_embd = cfg.latent_shape[1]

        # Init model
        sinu_pos_emb = ScaledSinusoidalEmbedding(self.cfg.n_embd)
        fourier_dim = self.cfg.n_embd

        time_emb_dim = self.cfg.n_embd * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_pos_embed_mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_emb_dim, self.cfg.n_embd)
        )

        self.pos_emb = AbsolutePositionalEmbedding(self.cfg.n_embd, self.cfg.n_embd)

        self.latent_encoder = Encoder(
            dim=self.cfg.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.dropout,
            ff_dropout=cfg.dropout,
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=cfg.seq_conditional,
            # DiT scale-shift stuff
            use_adaptive_layernorm=cfg.scale_shift,
            use_adaptive_layerscale=cfg.scale_shift,
            dim_condition=time_emb_dim,
            adaptive_condition_mlp=cfg.scale_shift,
        )

        if cfg.class_conditional:
            assert (
                False
            ), "Careful, never tested the class conditional setting for real."
            assert cfg.num_classes > 0
            self.class_embedding = nn.Sequential(
                nn.Embedding(cfg.num_classes + 1, self.cfg.n_embd),
                nn.Linear(self.cfg.n_embd, time_emb_dim),
            )
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(
                probs=cfg.class_unconditional_prob
            )
        if cfg.seq_conditional:
            assert cfg.seq_conditional_dim != None
            self.null_embedding_cond = nn.Embedding(1, self.cfg.n_embd)
            self.cond_proj = nn.Linear(cfg.seq_conditional_dim, self.cfg.n_embd)

        if cfg.self_condition:
            self.input_proj = nn.Linear(cfg.latent_shape[1] * 2, self.cfg.n_embd)
            self.init_self_cond = nn.Parameter(torch.randn(1, cfg.latent_shape[1]))
            nn.init.normal_(self.init_self_cond, std=0.02)
        else:
            self.input_proj = nn.Linear(cfg.latent_shape[1], self.cfg.n_embd)

        self.norm = nn.LayerNorm(self.cfg.n_embd)
        self.output_proj = nn.Linear(
            self.cfg.n_embd,
            cfg.latent_shape[1] * 2 if cfg.learned_variance else cfg.latent_shape[1],
        )

        if cfg.langevin:
            self.langevin_scaling_model = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, 1),
            )

        if cfg.cond_encoder_kwargs != None:
            if cfg.cond_encoder_kwargs["vocab_size"] != None:
                self.cond_embedding = nn.Embedding(
                    num_embeddings=cfg.cond_encoder_kwargs["vocab_size"],
                    embedding_dim=cfg.seq_conditional_dim,
                )
                self.cond_posemb = ScaledSinusoidalEmbedding(cfg.seq_conditional_dim)

            self.cond_encoder = Encoder(
                dim=cfg.seq_conditional_dim,
                depth=cfg.cond_encoder_kwargs["n_layers"],
                heads=cfg.cond_encoder_kwargs["n_heads"],
            )

        if cfg.cond_modulation:
            assert cfg.seq_conditional
            self.adalnzero_cond_proj = nn.Sequential(
                nn.Linear(cfg.seq_conditional_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            self.adalnzero_null_embedding = nn.Embedding(1, time_emb_dim)

        init_zero_(self.output_proj)

    def forward(
        self,
        x: torch.Tensor,
        time,
        x_self_cond=None,
        class_id=None,
        cond=None,
        cond_input_ids=None,
        cond_mask=None,
        log_r_fn=None,
    ):

        if self.cfg.langevin:
            x.requires_grad_(True)
            with torch.enable_grad():
                if self.cfg.seq_conditional:
                    assert cond_input_ids != None
                    grad_log_r = torch.autograd.grad(
                        log_r_fn(
                            x=x, cond_input_ids=cond_input_ids, cond_mask=cond_mask
                        ),
                        x,
                    )[0].detach()
                else:
                    assert False, "Didn't test this yet (unconditional langevin stuff)"
                    grad_log_r = torch.autograd.grad(log_r_fn(x).sum(), x)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.cfg.lgv_clip != None:
                    grad_log_r = torch.clip(
                        grad_log_r, -self.cfg.lgv_clip, self.cfg.lgv_clip
                    )

        time_emb = self.time_mlp(time[None] * 1000)

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

        if self.cfg.seq_conditional:
            context, context_mask = [], []
            if (cond is None) & (cond_input_ids is None):
                # If the model is conditional but no conditionning is passed, give <null_embedding_cond>
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

                if self.cfg.cond_modulation:
                    condition = time_emb + repeat(
                        self.adalnzero_null_embedding.weight, "1 d -> b 1 d", b=x.shape[0]
                    )
                else:
                    condition = time_emb
                
            else:
                if self.cfg.cond_encoder_kwargs != None:
                    if self.cfg.cond_encoder_kwargs["vocab_size"] != None:
                        assert cond_input_ids != None
                        cond = self.cond_embedding(cond_input_ids)
                        cond = cond + self.cond_posemb(cond)
                    else:
                        assert cond != None
                context.append(self.cond_proj(self.cond_encoder(cond)))
                context_mask.append(cond_mask)

                if self.cfg.cond_modulation:
                    condition = time_emb + self.adalnzero_cond_proj(
                        torch.where(
                            repeat(cond_mask, "b l -> b l d", d=cond.shape[-1]),
                            cond,
                            0,
                        ).mean(1, keepdim=True)
                    )
                else:
                    condition = time_emb

            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)

            x = self.latent_encoder(
                tx_input,
                context=context,
                context_mask=context_mask,
                condition=condition,
            )
        else:
            x = self.latent_encoder(tx_input, condition=time_emb)

        x = self.norm(x)
        x = self.output_proj(x)

        if self.cfg.langevin:
            scale = self.langevin_scaling_model(time_emb)
            if self.cfg.learned_variance:
                x[..., : self.cfg.latent_shape[1]] += scale * grad_log_r
            else:
                x += scale * grad_log_r

        if self.cfg.gfn_clip != None:
            x = torch.clip(x, -self.cfg.gfn_clip, self.cfg.gfn_clip)

        return x
