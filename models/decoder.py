from abc import ABC
from dataclasses import dataclass
from typing import *

import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaConfig

from models.base import DecoderModel
from models.x_transformer import Decoder, TransformerWrapper


@dataclass
class TransformerDecoderConfig:
    max_seq_len: int
    num_tokens: int
    n_layer: int
    n_head: int
    n_embd: int
    positional_encodings: bool = True
    tag: Optional[str] = None

# NOTE: the way this model takes a context encoding is by appending it to its context
class TransformerDecoder(TransformerWrapper, DecoderModel):
    def __init__(self, cfg: Optional[TransformerDecoderConfig] = None, **kwargs):
        if cfg is None:
            cfg = TransformerDecoderConfig(**kwargs)
        super().__init__(
            num_tokens=cfg.num_tokens,
            max_seq_len=cfg.max_seq_len,
            attn_layers=Decoder(dim=cfg.n_embd, depth=cfg.n_layer, heads=cfg.n_head, use_sin_pos_emb=cfg.positional_encodings),
        )

    def forward(
        self, input_ids, context_enc=None, attn_mask=None, only_last_logits=False, return_embeddings=False
    ):
        out = super().forward(x=input_ids, mask=attn_mask, prepend_embeds=context_enc, return_embeddings=return_embeddings)
        if context_enc != None:
            # Remove the prepended encodings
            out = out[:, context_enc.shape[1] :]
        out = out[:, [-1], :] if only_last_logits else out

        return out

class MambaDecoder(MambaLMHeadModel, DecoderModel):
    def __init__(self, cfg: Optional[MambaConfig] = None, **kwargs):
        if cfg is None:
            cfg = MambaConfig(**kwargs)
        super().__init__(cfg)

    def forward(
        self,
        input_ids,
        context_enc=None,
        only_last_logits=False,
        **mixer_kwargs,
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        
        if context_enc is not None:
            # TODO : give hidden state as context encoding
            # Would have to express this forward function with more low-level function
            # (e.g.) what is in MixerModel.forward()
            raise NotImplementedError
        
        out = super().forward(input_ids, num_last_tokens=1 if only_last_logits else 0)
        return out.logits
