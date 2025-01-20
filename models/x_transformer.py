#####################################################################################################
# Modified implementation of the x_transformers library from                                        #
# https://github.com/justinlovelace/latent-diffusion-for-language/blob/main/model/x_transformer.py
# Upgraded with things specific to diffusion models                                                 #
#####################################################################################################

from __future__ import annotations

import math
from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, wraps
from random import random, randrange
from typing import Callable

import einx
import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from loguru import logger
from packaging import version
from torch import Tensor, einsum, nn
from torch.amp import autocast
from torch.nn import Module, ModuleDict, ModuleList
from torch.utils._pytree import tree_flatten, tree_unflatten

# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])

LinearNoBias = partial(nn.Linear, bias = False)

# utils

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def first(it, default = None):
    return it[0] if len(it) > 0 else default

def is_empty(x):
    return len(x) == 0

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def softclamp(t, value):
    return (t / value).tanh() * value

def masked_mean(t, mask = None, dim = 1):
    if not exists(mask):
        return t.mean(dim = dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim).clamp(min = 1.)
    return num / den

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# entropy

def calc_entropy(
    t: Tensor,
    is_prob = False
):
    prob = t.softmax(dim = -1) if not is_prob else t
    return -(prob * log(prob)).sum(dim = -1)

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: list[Tensor],
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = tuple(d.pop(key) for key in  keys)
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    prefix_len = len(prefix)
    kwargs_without_prefix = {key[prefix_len:]: value for key, value in kwargs_with_prefix.items()}
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding

class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.emb.weight)


# positional embeddings

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class VariationalFourierFeatures(nn.Module):
    """ following https://arxiv.org/abs/2107.00630 """

    def __init__(self, n_min=0, n_max=8):
        super().__init__()
        assert n_min <= n_max
        self.n_min = n_min
        self.n_max = n_max

    def forward(self, x):
        fourier_features = []
        for n in range(self.n_min, self.n_max+1):
            freqs = x * (2**n) * math.pi
            fourier_features.extend([freqs.sin(), freqs.cos()])
        fouriered = rearrange(fourier_features, 'n b l d -> b l d n')
        return fouriered

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = einx.subtract('j, i -> i j', k_pos, q_pos)
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class CoPE(Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """
    def __init__ (
        self,
        dim,
        heads,
        max_pos,
        soft_onehot = False,
        talking_heads = False,
        soft_onehot_temp = 5e-2
    ):
        super () . __init__ ()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer('positions', torch.arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(causal_mask, -torch.finfo(attn_logits.dtype).max)

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim = -1).flip(-1)
        pos = pos.clamp(max = self.max_pos - 1)

        logits_int = einsum('b h n d, p d -> b h n p', query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract('i, j -> i j', pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim = -1)
            cope_pos_emb = einsum('b h i j p, b h i p -> b h i j', soft_onehot_pos, logits_int)
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb

class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = ModuleList([])

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = einx.subtract('i, j -> i j', seq_arange, context_arange)
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class AlibiPositionalBias(Module):
    def __init__(
        self,
        heads,
        total_heads = None,
        slopes: list[int] | None = None,
        **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, 'h -> h 1 1')

        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward_custom_pos(
        self,
        pos_i: Tensor,
        pos_j: Tensor | None = None
    ):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract('... j, ... i -> ... i j', pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, 'b i j -> b 1 i j')

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        return bias

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        seq_arange = torch.arange(j - i, j, device = device)
        context_arange = torch.arange(j, device = device)
        bias = -einx.subtract('j, i -> 1 i j', context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        self.register_buffer('bias', bias, persistent = False)
        return self.bias
    

class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads):
        super().__init__(heads)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        def get_slopes(param):
            return F.pad(param.exp(), (0, 0, 0, 0, 0, h - param.shape[0]))

        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :i, :j]
        else:
            bias = self.get_bias(i, j, device)
            self.register_buffer('bias', bias, persistent=False)

        slopes = get_slopes(self.learned_logslopes)
        bias = bias * slopes

        return qk_dots + bias

class DataDependentAlibi(Module):
    """ https://openreview.net/forum?id=q2Lnyegkr8 """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        bias_init = 5.,
        post_log_scale = 1.,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n h -> b h n'),
            nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim = -1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim = 1)

        forget_gates = einx.subtract('b h i, b h j -> b h i j', forget_gates, forget_gates)

        if bidirectional:
            forget_gates_reversed = einx.subtract('b h j, b h i -> b h i j', forget_gates_reversed, forget_gates_reversed)
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates

class PerRowDataDependentAlibi(Module):
    """ same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        dim_head = 8,
        post_log_scale = 1.
    ):
        super().__init__()
        assert causal, 'bidirectional not supported yet'

        self.scale = dim_head ** -0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias = False)

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n (qk h d) -> qk b h n d', qk = 2, d = dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype = torch.bool, device = x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims = (-1,))
        forget_gates = forget_gates.cumsum(dim = -1)
        forget_gates = forget_gates.flip(dims = (-1,))

        return forget_gates

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = torch.stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)

# norms

class Scale(Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.unit_offset = unit_offset

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)
    
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, pos = None, seq_dim = 1, offset = 0):
        if not exists(pos):
            pos = torch.arange(x.shape[seq_dim], device = x.device)

        pos = pos.type_as(self.inv_freq) + offset
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb

class ScaleNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        self.scale = dim ** 0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = F.normalize(x, dim = -1)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.)

class SimpleRMSNorm(Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

# residual and residual gates

class Residual(Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1., **kwargs):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class GRUGating(Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)

# hyper connections

class HyperConnection(Module):
    def __init__(
        self,
        dim,
        *,
        layer_index,
        num_residual_streams,
        tanh = True,
        **kwargs
    ):
        """
        https://arxiv.org/abs/2409.19606
        Appendix J - Algorithm 2, Dynamic only
        """
        super().__init__()

        self.act = nn.Tanh() if tanh else nn.Identity()

        self.norm = nn.LayerNorm(dim, bias = False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, 1))
        init_alpha0[layer_index % num_residual_streams, 0] = 1.

        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, num_residual_streams + 1))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

    def prepare(self, residuals):

        residuals = rearrange(residuals, '(b s) n d -> b n s d', s = self.num_residual_streams)

        normed = self.norm(residuals)

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = self.act(normed @ self.dynamic_beta_fn)
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        # width connection

        mix_h = einsum('... s t, ... s d -> ... t d', alpha, residuals)

        branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]

        return branch_input, residuals, dict(beta = beta)

    def forward(self, x, residuals, *, beta):
        residuals = einsum('b n d, b n s -> b n s d', x, beta) + residuals
        return rearrange(residuals, 'b n s d -> (b s) n d')

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

class TimeConditionedResidual(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.scale_shift = ScaleShift(time_emb_dim, dim_out)

    def forward(self, x, residual, time_emb):
        return self.scale_shift(x, time_emb) + residual

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class ShiftTokens(Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = [shift(*args, mask = mask) for args in zip(segments_to_shift, shifts)]
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

class FoldAxially(Module):
    def __init__(
        self,
        axial_dim,
        fn: Module
    ):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim # will fold the sequence as rearrange("b (n axial_dim) ... -> (b axial_dim) n ...")

    def forward(
        self,
        x,
        **kwargs
    ):
        if self.axial_dim == 1:
            return self.fn(x, **kwargs)

        seq_len, axial_dim = x.shape[1], self.axial_dim

        next_multiple = math.ceil(seq_len / axial_dim) * axial_dim
        x = pad_at_dim(x, (0, next_multiple - seq_len), dim = 1)

        x = rearrange(x, 'b (n axial_dim) ... -> (b axial_dim) n ...', axial_dim = axial_dim)

        out = self.fn(x, **kwargs)

        (out, *rest_out), tree_spec = tree_flatten(out)

        out = rearrange(out, '(b axial_dim) n ... -> b (n axial_dim) ...', axial_dim = axial_dim)

        out = out[:, :seq_len]
        out = tree_unflatten((out, *rest_out), tree_spec)

        return out

# post branch operator

class LayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        init_value = 0.,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset

        self.fn = fn
        self.gamma = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gamma, init_value - float(unit_offset))

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        gamma = self.gamma + float(self.unit_offset)

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

class AdaptiveLayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        self.fn = fn

        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition, **kwargs):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        out = self.fn(x, **kwargs)
        gamma = self.to_gamma(condition).sigmoid()

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

# skip connection combining

class ConcatCombine(Module):
    def __init__(self, dim, prev_layer_ind):
        super().__init__()
        self.prev_layer_ind = prev_layer_ind
        self.combine = LinearNoBias(dim * 2, dim)

    def forward(self, x, prev_layers: list[Tensor]):
        skip = prev_layers[self.prev_layer_ind]
        concatted_skip = torch.cat((skip, x), dim = -1)
        return self.combine(concatted_skip)

# feedforward

class GLU(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate) * self.mult_bias

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

# attention.

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        heads = 8,
        causal = False,
        talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_values = False,
        zero_init_output = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        one_kv_head = False,
        shared_kv = False,
        value_dim_head = None,
        tensor_product = False   # https://arxiv.org/abs/2208.06061
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        value_dim_head = default(value_dim_head, dim_head)
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = value_dim_head * heads

        self.one_kv_head = one_kv_head
        if one_kv_head:
            k_dim = dim_head
            v_dim = value_dim_head
            out_dim = v_dim * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)

        # shared key / values, for further memory savings during inference
        assert not (shared_kv and value_dim_head != dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.to_v = nn.Linear(dim, v_dim, bias = False) if not shared_kv else None

        # relations projection from tp-attention
        self.to_r = nn.Linear(dim, v_dim, bias = False) if tensor_product else None

        # dropout
        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # cosine sim attention
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        assert (not qk_norm) or (dim_head % qk_norm_groups) == 0, 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # attention softmax function
        self.attn_fn = partial(F.softmax, dtype = torch.float32) if not qk_norm else F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(out_dim, dim * 2, bias = False), nn.GLU()) if on_attn else nn.Linear(out_dim, dim, bias = False)

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        sinusoidal_emb = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ):
        b, n, _, h, talking_heads, head_scale, scale, device, has_context = *x.shape, self.heads, self.talking_heads, self.head_scale, self.scale, x.device, exists(context)
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim = -2)
            v_input = torch.cat((mem, v_input), dim = -2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset = offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k
        r = self.to_r(r_input) if exists(self.to_r) else None

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        if not self.one_kv_head:
            k, v, r = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = h), (k, v, r))

        if exists(rotary_pos_emb) and not has_context:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
            q, k, v = map(lambda t: torch.cat(t, dim = -1), ((ql, qr), (kl, kr), (vl, vr)))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
            k_mask = rearrange(k_mask, 'b j -> b 1 1 j')
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value = True)

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

        kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            dots.masked_fill_(~attn_mask, mask_value)

        if exists(self.max_attend_past):
            i, j = dots.shape[-2:]
            range_q = torch.arange(j - i, j, device = device)
            range_k = torch.arange(j, device = device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            mask = dist > self.max_attend_past
            dots.masked_fill_(mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            range_i = torch.arange(i, device = device)
            mask = rearrange(range_i, 'i -> 1 1 i 1') < rearrange(range_i, 'j -> 1 1 1 j')
            mask = F.pad(mask, (j - i, 0), value = False)
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim = -1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        dtype = dots.dtype

        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        if exists(r):
            # https://arxiv.org/abs/2208.06061 proposes to add a residual for better gradients
            out = out * r + out

        if head_scale:
            out = out * self.head_scale_params

        out = rearrange(out, 'b h n d -> b n (h d)')

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return self.to_out(out), intermediates

class ScaleShift(nn.Module):
    def __init__(self, time_emb_dim, dim_out):
        super().__init__()
        self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )
        init_zero_(self.time_mlp[-1])
        

    def forward(self, x, time_emb):
        scale, shift = self.time_mlp(time_emb).chunk(2, dim = 2)

        x = x * (scale + 1) + shift

        return x

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        alibi_learned = False,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        position_infused_attn = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        deepnorm = False,
        shift_tokens = 0,
        sandwich_norm = False,
        zero_init_branch_output = False,
        time_emb_dim = None,
        num_dense_connections = 0,
        **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.num_dense_connections = num_dense_connections

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        self.rel_pos = None
        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif dynamic_pos_bias:
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedAlibiPositionalBias if alibi_learned else AlibiPositionalBias
            self.rel_pos = alibi_pos_klass(heads = alibi_num_heads)

        # determine deepnorm and residual scale

        if deepnorm:
            assert scale_residual_constant == 1, 'scale residual constant is being overridden by deep norm settings'
            pre_norm = sandwich_norm = False
            scale_residual = True
            scale_residual_constant = (2 * depth) ** 0.25

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        self.scale_shift = exists(time_emb_dim)


        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if self.scale_shift and layer_type in ['f']:
                residual = TimeConditionedResidual(time_emb_dim, dim)
            else:
                residual_fn = GRUGating if gate_residual else Residual
                residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm or (self.scale_shift and layer_type in ['f']) else None
            post_main_norm = norm_fn() if not pre_norm and not is_last_layer else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))
        
        self.dense_projections = nn.ModuleList([nn.Linear(dim*2, dim) for _ in range(num_dense_connections)])

        if deepnorm:
            init_gain = (8 * depth) ** -0.25
            deepnorm_init(self, init_gain)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        mems = None,
        time_emb = None,
        return_hiddens = False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        dense_hiddens = []
        attn_idx = 0
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                dense_idx = attn_idx - (self.num_attn_layers - self.num_dense_connections)
                if dense_idx >= 0:
                    assert len(dense_hiddens) > 0, 'dense connections must be in order'
                    x = self.dense_projections[dense_idx](torch.cat([x, dense_hiddens.pop()], dim=-1))
                attn_idx += 1
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_branch_norm):
                x = pre_branch_norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, attn_mask = attn_mask, sinusoidal_emb = self.pia_pos_emb, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, mem = layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            if self.scale_shift and layer_type in ['f']:
                x = residual_fn(out, residual, time_emb)
            else:
                x = residual_fn(out, residual)


            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)
            
            if layer_type == 'f' and len(dense_hiddens) < self.num_dense_connections:
                dense_hiddens.append(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens = hiddens,
                attn_intermediates = intermediates
            )

            return x, intermediates

        return x
    
def deepnorm_init(
    transformer,
    beta,
    module_name_match_list = ['.ff.', '.to_v', '.to_out']
):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain = gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = True, **kwargs)

class TransformerWrapper(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: dict[str, int] = dict(),
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        tie_embedding = False,
        logits_dim = None,
        return_only_embed = False,
        num_output_heads = 1,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        recycling = False,            # from Jumper et al. - Alphafold2
        train_max_recycle_steps = 4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient = 1.,       # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
        average_pool_embed = False,
        use_cls_token = False,
        num_cls_tokens = 1,
        squeeze_out_last_dim = False,
        token_emb: TokenEmbedding | None = None,
        mixture_of_softmax = False,
        mixture_of_softmax_k = 4,
        sigsoftmax_logits = False,
        to_logits: Module | None = None,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_cls_tokens = num_cls_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed

        if not exists(token_emb):
            token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        self.token_emb = token_emb

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        # additional embeddings - say type embedding from BERT

        self.embeds = None

        if len(embed_num_tokens) > 0:
            self.embeds = ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        assert num_output_heads > 0

        assert at_most_one_of(average_pool_embed, use_cls_token)

        # maybe recycling

        self.recycling = recycling
        self.recycled_proj = LinearNoBias(dim, dim) if recycling else None

        self.train_max_recycle_steps = train_max_recycle_steps

        # classic cls token from the bert days

        self.cls_token = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, dim))
            nn.init.normal_(self.cls_token, std = 0.02)

        # whether to average pool the embed (`global average pool`)

        self.average_pool_embed = average_pool_embed

        # output type

        self.output_is_log_prob = mixture_of_softmax

        self.to_mixture = None
        self.combine_mixture = None

        if mixture_of_softmax:
            assert num_output_heads == 1

            self.to_mixture = Sequential(
                LinearNoBias(dim, dim * mixture_of_softmax_k),
                Rearrange('... (k d) -> ... k d', k = mixture_of_softmax_k)
            )

            self.combine_mixture = LinearNoBias(dim, mixture_of_softmax_k)

        # sig softmax

        self.sigsoftmax_logits = sigsoftmax_logits

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, num_tokens)

        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif tie_embedding:
            assert isinstance(token_emb, TokenEmbedding), 'can only tie embedding if using `TokenEmbedding`'
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.to_logits = ModuleList([LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)])
        else:
            self.to_logits = LinearNoBias(dim, logits_dim) if not exists(to_logits) else to_logits

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # squeeze out last dimension if possible

        self.squeeze_out_last_dim = squeeze_out_last_dim

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0 and not recycling and self.attn_layers.can_cache_kv
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if hasattr(self.token_emb, 'init_'):
            self.token_emb.init_()

        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)

    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        return_logit_entropies = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: LayerIntermediates | None = None,
        token_emb_kwargs = dict(),
        to_logits_kwargs = dict(),
        **kwargs,
    ):
        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient, orig_mask = x.shape[0], x.shape[1], x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient, mask

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss
        return_embeddings = return_embeddings | (not exists(self.to_logits))

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        # add additional embeddings

        assert not (exists(self.embeds) ^ (len(embed_ids) > 0)), '`embed_num_tokens` must be defined on `TransformerWrapper`'

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, '... -> b ...', b = b)
            x, cls_packed_shape = pack([cls_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value = True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim = -2, value = 0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        # attention layers

        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, 'you did not train with recycling'

            # regular

            attended, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        else:
            # recycling

            recycle_steps = default(recycle_steps, (randrange(self.train_max_recycle_steps) + 1) if self.training else None)
            assert exists(recycle_steps) and recycle_steps > 0, '`recycle_steps` must be provided on forward if recycling is turned on and not training'

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = self.recycled_proj(attended.detach()) if not first_step else 0.

                    attended, intermediates = self.attn_layers(x + maybe_recycled, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        x = attended

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            x = x[:, :mem_seq]

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        if exists(self.cls_token):
            x, _ = unpack(x, cls_packed_shape, 'b * d')
            x = x.squeeze(1)  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # handle expansion to mixture if needed (for mixture of softmax)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim = -1)
            x = self.to_mixture(x)

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        # maybe sig softmax

        if self.sigsoftmax_logits:
            logits = logits + logits.sigmoid().log()

        # handle maybe combine mixture

        if exists(combine_mixture):
            with autocast('cuda', enabled = False):
                prob = logits.softmax(dim = -1)
                mos = einsum('... k d, ... k -> ... d', prob, combine_mixture)
                logits = log(mos)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple((rearrange(t, '... 1 -> ...') if t.shape[-1] == 1 else t) for t in cast_tuple(logits))

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings:
            out = x
        else:
            out = logits

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in  intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = [torch.cat(pair, dim = -2) for pair in zip(mems, hiddens)] if exists(mems) else hiddens
            new_mems = [t[..., -self.max_mem_len:, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out