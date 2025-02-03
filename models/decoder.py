from abc import ABC
from dataclasses import dataclass
from typing import *

import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaConfig

from models.base import DecoderModel
from models.x_transformer import Decoder, TransformerWrapper

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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

    @property
    def device(self):
        return self.lm_head.weight.device

    def forward(
        self,
        input_ids,
        context_enc=None,
        only_last_logits=False,
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
        
        hidden_states = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.backbone.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )
        if not self.backbone.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.backbone.norm_f(residual.to(dtype=self.backbone.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.backbone.norm_f.weight,
                self.backbone.norm_f.bias,
                eps=self.backbone.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.backbone.residual_in_fp32,
                is_rms_norm=isinstance(self.backbone.norm_f, RMSNorm)
            )
        if only_last_logits:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
