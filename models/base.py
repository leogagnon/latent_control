from abc import ABC
from dataclasses import dataclass
from typing import Optional

import hydra
import torch
import torch.nn as nn


@dataclass
class MetaLearnerConfig:
    decoder: dict
    tag: Optional[str] = None
    encoder: Optional[dict] = None

# NOTE: Right now this only supports batched training with implicit models or know-explicit model (i.e. not DiffusionEncoder or other more spicy setups)
# Can subclass this or restructure a bit when we want more spicy meta-learners
class MetaLearner(nn.Module):
    """Basic skeleton for a meta-learner. If <self.enc> is None this means this is an implicit model."""
    def __init__(self, cfg: Optional[MetaLearnerConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = MetaLearnerConfig(**kwargs)
        self.cfg = cfg
        self.decoder = hydra.utils.instantiate(cfg.decoder)
        self.encoder = None if cfg.encoder is None else hydra.utils.instantiate(cfg.encoder)

    def forward(self, input_ids=None, true_latents=None, only_last_logits=False):
        context_enc = None
        if self.encoder != None:
            context_enc = self.encoder(input_ids, true_latents)
        logits = self.decoder(input_ids, context_enc, only_last_logits=only_last_logits)
        return logits    
    
class EncoderModel(ABC, nn.Module):
    def forward(self, input_ids, true_latents, **kwargs):
        pass

class DecoderModel(ABC, nn.Module):
    def forward(self, input_ids, context_enc):
        pass