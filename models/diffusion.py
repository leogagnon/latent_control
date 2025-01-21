import argparse
import copy
import csv
import json
import math
import os
import random
import timeit
from collections import Counter, defaultdict, namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import (Accelerator, DistributedDataParallelKwargs,
                        InitProcessGroupKwargs)
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
from torch import einsum, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, MT5ForConditionalGeneration,
                          PreTrainedTokenizerBase, T5ForConditionalGeneration,
                          get_scheduler)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from models.utils import *
from models.x_transformer import (AbsolutePositionalEmbedding, Encoder,
                                  SinusoidalPosEmb, init_zero_)

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_v"]
)


@dataclass
class DiffusionTransformerConfig:
    n_embd: int
    n_layers: int
    n_heads: int
    latent_dim: int
    max_seq_len: int
    self_condition: bool = False
    dropout: float = 0.0
    scale_shift: bool = False
    seq2seq: bool = False
    seq2seq_context_dim: int = 0
    dual_output: bool = False
    num_dense_connections: int = 0
    dense_output_connection: bool = False
    class_conditional: bool = False
    num_classes: int = 0
    class_unconditional_prob: int = 0

@dataclass
class GaussianDiffusionConfig:
    max_seq_len: int
    sampling_schedule: str = None
    sampling_timesteps: int = 250
    loss_type: str = "l1"
    train_schedule = "cosine"
    objective: str = "pred_noise"
    scale: float = 1.0
    sampler: str = "ddpm"
    seq2seq_unconditional_prob: str = 0.1
    l2_normalize: bool = False
    train_prob_self_cond: float = 0.5,


class DiffusionTransformer(nn.Module):
    """Encoder-only transformer with bells and whistle to do diffusion"""

    def __init__(self, cfg: DiffusionTransformerConfig):
        super().__init__()

        # time embeddings

        sinu_pos_emb = SinusoidalPosEmb(cfg.n_embd)
        fourier_dim = cfg.n_embd

        time_emb_dim = cfg.n_embd * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.time_pos_embed_mlp = nn.Sequential(
            nn.GELU(), nn.Linear(time_emb_dim, cfg.n_embd)
        )

        self.pos_emb = SinusoidalPosEmb(cfg.n_embd)

        self.cross = cfg.seq2seq

        self.encoder = Encoder(
            dim=cfg.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.dropout,  # dropout post-attention
            ff_dropout=cfg.dropout,  # feedforward dropout
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=self.cross,
            time_emb_dim=cfg.n_embd * 4 if cfg.scale_shift else None,
            num_dense_connections=cfg.num_dense_connections,
        )

        if cfg.class_conditional:
            assert cfg.num_classes > 0
            self.class_embedding = nn.Sequential(
                nn.Embedding(cfg.num_classes + 1, cfg.n_embd),
                nn.Linear(cfg.n_embd, time_emb_dim),
            )
        if cfg.seq2seq:
            self.null_embedding_seq2seq = nn.Embedding(1, cfg.n_embd)
            self.seq2seq_proj = nn.Linear(cfg.seq2seq_context_dim, cfg.n_embd)

        if cfg.self_condition:
            self.input_proj = nn.Linear(cfg.latent_dim * 2, cfg.n_embd)
            self.init_self_cond = nn.Parameter(torch.randn(1, cfg.latent_dim))
            nn.init.normal_(self.init_self_cond, std=0.02)
        else:
            self.input_proj = nn.Linear(cfg.latent_dim, cfg.n_embd)
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.output_proj = nn.Linear(
            cfg.n_embd * 2 if cfg.dense_output_connection else cfg.n_embd,
            cfg.latent_dim * 2 if cfg.dual_output else cfg.latent_dim,
        )

        self.using_latent_model: bool

        self.cfg = cfg

        init_zero_(self.output_proj)

    def forward(
        self,
        x,
        time,
        mask=None,
        x_self_cond=None,
        class_id=None,
        seq2seq_cond=None,
        seq2seq_mask=None,
    ):
        """
        x: input, [batch, length, latent_dim]
        mask: bool tensor where False indicates masked positions, [batch, length]
        time: timestep, [batch]
        """

        time_emb = self.time_mlp(time * 1000)

        time_emb = rearrange(time_emb, "b d -> b 1 d")

        if self.cfg.class_conditional:
            assert exists(class_id)
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, 'b d -> b 1 d')
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(x)

        if self.cfg.self_condition:
            if exists(x_self_cond):
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
            if self.cfg.seq2seq:
                if seq2seq_cond is None:
                    null_context = repeat(
                        self.null_embedding_seq2seq.weight, "1 d -> b 1 d", b=x.shape[0]
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
                    context.append(self.seq2seq_proj(seq2seq_cond))
                    context_mask.append(seq2seq_mask)
            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)

            x = self.encoder(
                tx_input,
                mask=mask,
                context=context,
                context_mask=context_mask,
                time_emb=time_emb,
            )
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)

        x = self.norm(x)

        return self.output_proj(x)

class GaussianDiffusion(nn.Module):
    def __init__(self, model: DiffusionTransformer, cfg: GaussianDiffusionConfig):
        super().__init__()
        assert cfg.sampler in {
            "ddim",
            "ddpm",
            "dpmpp",
        }, "sampler must be one of ddim, ddpm, dpmpp"

        self.model = model
            
        assert cfg.objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_v_dual'}, 'objective must be one of pred_noise, pred_x0, pred_v, pred_v_dual'

        if self.model.cfg.class_conditional:
            if self.model.cfg.class_unconditional_prob > 0:
                self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.model.cfg.class_unconditional_prob)

        if cfg.train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif cfg.train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif cfg.train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif cfg.train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid noise schedule {cfg.train_schedule}")

        self.train_schedule = partial(
            time_to_alpha, alpha_schedule=alpha_schedule, scale=cfg.scale
        )

        # Sampling schedule
        if cfg.sampling_schedule is None:
            sampling_alpha_schedule = None
        elif cfg.sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif cfg.sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif cfg.sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif cfg.sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid sampling schedule {cfg.sampling_schedule}")

        if exists(sampling_alpha_schedule):
            self.sampling_schedule = partial(
                time_to_alpha, alpha_schedule=sampling_alpha_schedule, scale=cfg.scale
            )
        else:
            self.sampling_schedule = self.train_schedule

        # Buffers for latent mean and scale values
        self.register_buffer(
            "latent_mean", torch.tensor([0] * self.model.cfg.latent_dim).to(torch.float32)
        )
        self.latent_mean: torch.FloatTensor
        self.register_buffer("latent_scale", torch.tensor(1).to(torch.float32))
        self.latent_scale: torch.FloatTensor

        self.cfg = cfg

    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.cfg.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1 - alpha).sqrt() * noise) / alpha.sqrt().clamp(min=1e-8)

    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1 - alpha).sqrt() * v

        return x

    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1 - alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps

    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x * (1 - alpha).sqrt()

        return v

    def normalize_latent(self, x_start):
        eps = 1e-5

        return (x_start - self.latent_mean) / (self.latent_scale).clamp(min=eps)

    def unnormalize_latent(self, x_start):
        eps = 1e-5

        return x_start * (self.latent_scale.clamp(min=eps)) + self.latent_mean

    def diffusion_model_predictions(
        self,
        z_t,
        t,
        mask=None,
        x_self_cond=None,
        class_id=None,
        seq2seq_cond=None,
        seq2seq_mask=None,
        sampling=False,
        cls_free_guidance=1.0,
        l2_normalize=False,
    ):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output = self.model(
            z_t,
            time_cond,
            mask,
            x_self_cond,
            class_id=class_id,
            seq2seq_cond=seq2seq_cond,
            seq2seq_mask=seq2seq_mask,
        )
        if cls_free_guidance != 1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(
                    class_id, fill_value=self.model.cfg.num_classes
                )
            else:
                unc_class_id = None
            unc_model_output = self.model(
                z_t,
                time_cond,
                mask,
                x_self_cond,
                class_id=unc_class_id,
                seq2seq_cond=None,
                seq2seq_mask=None,
            )
            model_output = model_output * cls_free_guidance + unc_model_output * (
                1 - cls_free_guidance
            )

        pred_v = None
        if self.cfg.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(
                z_t, t, pred_noise, sampling=sampling
            )
        elif self.cfg.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(
                z_t, t, x_start, sampling=sampling
            )
            pred_v = self.predict_v_from_start_and_eps(
                z_t, t, x_start, pred_noise, sampling=sampling
            )
        elif self.cfg.objective == "pred_v":
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f"invalid objective {self.cfg.objective}")
        
        if l2_normalize:
            assert sampling
            x_start = F.normalize(x_start, dim=-1) * math.sqrt(x_start.shape[-1])
            pred_noise = self.predict_noise_from_start(
                z_t, t, x_start, sampling=sampling
            )
            pred_v = self.predict_v_from_start_and_eps(
                z_t, t, x_start, pred_noise, sampling=sampling
            )

        return ModelPrediction(pred_noise, x_start, pred_v)

    def get_sampling_timesteps(self, batch, *, device, invert=False):
        times = torch.linspace(1.0, 0.0, self.cfg.sampling_timesteps + 1, device=device)
        if invert:
            times = times.flip(dims=(0,))
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddim_sample(
        self,
        shape,
        lengths,
        class_id,
        seq2seq_cond,
        seq2seq_mask,
        cls_free_guidance=1.0,
        l2_normalize=False,
        invert=False,
        z_t=None,
    ):
        print("DDIM sampling")
        batch, device = shape[0], next(self.model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device, invert=invert)
        if invert:
            assert exists(z_t)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent = None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:
            mask = [
                [True] * length + [False] * (self.cfg.max_seq_len - length)
                for length in lengths
            ]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=self.cfg.sampling_timesteps
        ):
            # get predicted x0

            model_output = self.diffusion_model_predictions(
                z_t,
                time,
                mask,
                class_id=class_id,
                x_self_cond=x_start,
                seq2seq_cond=seq2seq_cond,
                seq2seq_mask=seq2seq_mask,
                sampling=True,
                cls_free_guidance=cls_free_guidance,
                l2_normalize=l2_normalize,
            )
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(
                partial(right_pad_dims_to, z_t), (alpha, alpha_next)
            )

            # # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise

            if (not invert) and time_next[0] <= 0:
                z_t = x_start
                continue
            if invert and time_next[0] >= 1:
                z_t = eps
                continue

            # get noise

            z_t = x_start * alpha_next.sqrt() + eps * (1 - alpha_next).sqrt()
        return (z_t, mask)

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape,
        lengths,
        class_id,
        seq2seq_cond,
        seq2seq_mask,
        cls_free_guidance=1.0,
        l2_normalize=False,
        invert=False,
        z_t=None,
    ):
        batch, device = shape[0], next(self.model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent = None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:
            mask = [
                [True] * length + [False] * (self.cfg.max_seq_len - length)
                for length in lengths
            ]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=self.cfg.sampling_timesteps
        ):
            # get predicted x0

            model_output = self.diffusion_model_predictions(
                z_t,
                time,
                mask,
                class_id=class_id,
                x_self_cond=x_start,
                seq2seq_cond=seq2seq_cond,
                seq2seq_mask=seq2seq_mask,
                sampling=True,
                cls_free_guidance=cls_free_guidance,
                l2_normalize=l2_normalize,
            )
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(
                partial(right_pad_dims_to, z_t), (alpha, alpha_next)
            )

            alpha_now = alpha / alpha_next

            # # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise

            if time_next[0] <= 0:
                z_t = x_start
                continue

            # get noise

            noise = torch.randn_like(z_t)

            z_t = (
                1
                / alpha_now.sqrt()
                * (z_t - (1 - alpha_now) / (1 - alpha).sqrt() * eps)
                + torch.sqrt(1 - alpha_now) * noise
            )
        return (z_t, mask)

    @torch.no_grad()
    def dpmpp_sample(
        self,
        shape,
        lengths,
        class_id,
        seq2seq_cond,
        seq2seq_mask,
        cls_free_guidance=1.0,
        l2_normalize=False,
        invert=False,
        z_t=None,
    ):
        batch, device = shape[0], next(self.model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent = None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:
            mask = [
                [True] * length + [False] * (self.cfg.max_seq_len - length)
                for length in lengths
            ]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        old_pred_x = []
        old_hs = []

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=self.sampling_timesteps
        ):
            # get predicted x0

            model_output = self.diffusion_model_predictions(
                z_t,
                time,
                mask,
                class_id=class_id,
                x_self_cond=x_start,
                seq2seq_cond=seq2seq_cond,
                seq2seq_mask=seq2seq_mask,
                sampling=True,
                cls_free_guidance=cls_free_guidance,
                l2_normalize=l2_normalize,
            )
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(
                partial(right_pad_dims_to, z_t), (alpha, alpha_next)
            )
            sigma, sigma_next = 1 - alpha, 1 - alpha_next

            alpha_now = alpha / alpha_next

            lambda_now = (log(alpha) - log(1 - alpha)) / 2
            lambda_next = (log(alpha_next) - log(1 - alpha_next)) / 2
            h = lambda_next - lambda_now

            # calculate x0 and noise
            if time_next[0] <= 0:
                z_t = x_start
                continue

            x_start = model_output.pred_x_start

            phi_1 = torch.expm1(-h)
            if len(old_pred_x) < 2:
                denoised_x = x_start
            else:
                h = lambda_next - lambda_now
                h_0 = old_hs[-1]
                r0 = h_0 / h
                gamma = -1 / (2 * r0)
                denoised_x = (1 - gamma) * x_start + gamma * old_pred_x[-1]

            z_t = (
                sigma_next.sqrt() / sigma.sqrt()
            ) * z_t - alpha_next.sqrt() * phi_1 * denoised_x
        return (z_t, mask)

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        length,
        class_id=None,
        seq2seq_cond=None,
        seq2seq_mask=None,
        cls_free_guidance=1.0,
        l2_normalize=False,
    ):
        max_seq_len, latent_dim = (
            self.cfg.max_seq_len,
            self.model.cfg.latent_dim,
        )

        if self.cfg.sampler == "ddim":
            sample_fn = self.ddim_sample
        elif self.cfg.sampler == "ddpm":
            sample_fn = self.ddpm_sample
        elif self.cfg.sampler == "dpmpp":
            sample_fn = self.dpmpp_sample
        else:
            raise ValueError(f"invalid sampler {self.cfg.sampler}")
        return sample_fn(
            (batch_size, max_seq_len, latent_dim),
            length,
            class_id,
            seq2seq_cond,
            seq2seq_mask,
            cls_free_guidance,
            l2_normalize,
        )
        
