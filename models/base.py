from abc import ABC
import math
import inspect
from dataclasses import dataclass, field
from typing import List, Optional
from data.hmm import CompositionalHMMDataset

import hydra
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList
import einops
from models.gpt import Block, LayerNorm, GPT
from models.mamba import MambaLMHeadModel, layer_norm_fn, RMSNorm

@dataclass
class MetaLearnerConfig:
    dec_cfg: dict
    tag: Optional[str] = None
    enc_cfg: Optional[dict] = None

class MetaLearner(nn.Module):
    def __init__(self, cfg: Optional[MetaLearnerConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = MetaLearnerConfig(**kwargs)
        self.cfg = cfg
        self.dec = hydra.utils.instantiate(cfg.dec_cfg)
        self.enc = None if cfg.enc_cfg is None else hydra.utils.instantiate(cfg.enc_cfg)

    def forward(self, input_ids, true_latents, only_last_logits=False):
        context_enc = None
        if self.enc != None:
            context_enc = self.enc(self.dec.wte(input_ids), true_latents)
        logits = self.dec(input_ids, context_enc, only_last_logits=only_last_logits)
        return logits