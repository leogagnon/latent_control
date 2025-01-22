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

# helpers functions

def exists(val):
    return val is not None

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = weight_decay, betas=betas, eps=eps)

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def l2norm(t):
    return F.normalize(t, dim=-1)


def log(t, eps=1e-12):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# normalize variance of noised latent, if scale is not 1


def normalize_z_t_variance(z_t, mask, eps=1e-5):
    std = rearrange(
        [
            reduce(
                z_t[i][: torch.sum(mask[i])],
                "l d -> 1 1",
                partial(torch.std, unbiased=False),
            )
            for i in range(z_t.shape[0])
        ],
        "b 1 1 -> b 1 1",
    )
    return z_t / std.clamp(min=eps)


# noise schedules


def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def beta_linear_schedule(t, clip_min=1e-9):
    return torch.exp(-1e-4 - 10 * (t**2)).clamp(min=clip_min, max=1.0)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.0)


# converting gamma to alpha, sigma or logsnr


def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha


def alpha_to_shifted_log_snr(alpha, scale=1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2 * np.log(scale).item()


def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale=scale)
    return log_snr_to_alpha(shifted_log_snr)


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)