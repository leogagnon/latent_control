from abc import ABC
from dataclasses import dataclass
from typing import *

import einx
import torch
import torch.nn as nn
from x_transformers.x_transformers import Decoder, Encoder, TransformerWrapper

from models.base import DecoderModel


@dataclass
class TransformerDecoderConfig:
    max_seq_len: int
    num_tokens: int
    n_layer: int
    n_head: int
    n_embd: int
    soft_token_enc: bool 
    causal_mask: bool 
    sin_posemb: bool 
    enc_dim: Optional[int] = None 
    tag: Optional[str] = None


# NOTE: the way this model takes a context encoding is by appending it to its context
class TransformerDecoder(TransformerWrapper, DecoderModel):
    def __init__(self, cfg: Optional[TransformerDecoderConfig] = None, **kwargs):
        if cfg is None:
            cfg = TransformerDecoderConfig(**kwargs)
        attn_layer_cls = Decoder if cfg.causal_mask else Encoder
        super().__init__(
            num_tokens=cfg.num_tokens,
            max_seq_len=cfg.max_seq_len,
            attn_layers=attn_layer_cls(
                dim=cfg.n_embd, depth=cfg.n_layer, heads=cfg.n_head, cross_attend=not cfg.soft_token_enc
            ),
            scaled_sinu_pos_emb=cfg.sin_posemb,
            use_abs_pos_emb=cfg.sin_posemb,
        )
        if cfg.enc_dim != None:
            self.enc_proj = nn.Linear(cfg.enc_dim, cfg.n_embd)
        else:
            self.enc_proj = nn.Identity()

        self.cfg = cfg

    def forward(
        self,
        input_ids,
        context_enc,
        **kwargs
    ):   
        context_enc = self.enc_proj(context_enc)
        
        bs, l = input_ids.shape
        if self.cfg.soft_token_enc:
            out = super().forward(
                x=input_ids,
                prepend_embeds=context_enc,
            )
            if context_enc != None:
                # Remove the prepended encodings
                out = out[:, context_enc.shape[1] :]
        else:
            out = super().forward(
                x=input_ids,
                context=context_enc
                )

        return out