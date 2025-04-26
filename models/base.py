from abc import ABC
from dataclasses import dataclass
from typing import Optional

import hydra
import torch
import torch.nn as nn


@dataclass
class MetaLearnerConfig:
    decoder: Optional[dict] = None
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
        self.decoder = (
            None if cfg.decoder is None else hydra.utils.instantiate(cfg.decoder, _recursive_=False)
        )
        self.decoder: DecoderModel
        self.encoder = (
            None if cfg.encoder is None else hydra.utils.instantiate(cfg.encoder, _recursive_=False)
        )
        self.encoder: EncoderModel

    def forward(
        self, input_ids=None, **kwargs
    ):
        context_enc = None
        if self.encoder != None:
            context_enc = self.encoder(
                input_ids=input_ids,
                **kwargs
            )
        if self.decoder != None:
            logits = self.decoder(
                input_ids=input_ids,
                context_enc=context_enc,
            )
        else:
            logits = context_enc

        return logits
    
class EncoderModel(ABC, nn.Module):
    out_proj: callable

    def forward(self, input_ids, **kwargs):
        pass

    @property
    def out_dim(self):
        "Dimension of the output (e.g. logits)"
        raise NotImplementedError

    @property
    def hidden_dim(self):
        "Dimension of the embeddings"
        raise NotImplementedError

    @property
    def enc_len(self):
        "Lenght of the encoding"
        raise NotImplementedError
       
class DecoderModel(ABC, nn.Module):
    def forward(self, input_ids, context_enc, **kwargs):
        pass