from abc import ABC
from dataclasses import dataclass
from typing import *

import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaConfig

from models.base import DecoderModel
from x_transformers.x_transformers import Decoder, TransformerWrapper, Encoder

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
import einx


@dataclass
class TransformerDecoderConfig:
    max_seq_len: int
    num_tokens: int
    n_layer: int
    n_head: int
    n_embd: int
    causal_mask: bool = True
    full_context: bool = True
    positional_encodings: bool = True
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
                dim=cfg.n_embd, depth=cfg.n_layer, heads=cfg.n_head
            ),
            scaled_sinu_pos_emb=cfg.positional_encodings,
            use_abs_pos_emb=cfg.positional_encodings,
        )
        self.cfg = cfg
        self.no_ctx_emb = nn.Parameter(data=torch.rand(cfg.n_embd), requires_grad=True)

    def forward(
        self,
        input_ids,
        context_enc=None,
        attn_mask=None,
        only_last_logits=False,
        return_embeddings=False,
        shift_enc=True,
    ):
        bs, l = input_ids.shape
        if self.cfg.full_context:
            out = super().forward(
                x=input_ids,
                mask=attn_mask,
                prepend_embeds=context_enc,
                return_embeddings=return_embeddings,
            )
            if context_enc != None:
                # Remove the prepended encodings
                out = out[:, context_enc.shape[1] :]
        else:
            if context_enc != None:
                if shift_enc:
                    # Shift context embedding to the right and add <no_context> for the first token
                    no_ctx_emb = einx.rearrange(
                        "d -> b 1 d", self.no_ctx_emb, b=len(input_ids)
                    )
                    context_enc = torch.concatenate([no_ctx_emb, context_enc], dim=1)[
                        :, :-1
                    ]

                context_enc = einx.rearrange("b l d -> (b l) 1 d", context_enc)
                input_ids = einx.rearrange("b l -> (b l) 1 ", input_ids)

            # Run backbone
            out = super().forward(
                x=input_ids,
                prepend_embeds=context_enc,
                return_embeddings=return_embeddings,
            )
            if context_enc != None:
                out = einx.rearrange("(b l) L d -> b l L d", out, b=bs, l=l, L=2)
                out = out[:, :, -1]  # Decode from the second token

        out = out[:, [-1]] if only_last_logits else out

        return out


@dataclass
class MLPDecoderConfig:
    max_seq_len: int
    num_tokens: int
    n_embd: int
    expansion_factor: int
    tag: Optional[str] = None


# NOTE: the way this model takes a context encoding is by appending it to its context
class MLPDecoder(DecoderModel):
    def __init__(self, cfg: Optional[MLPDecoderConfig] = None, **kwargs):
        super().__init__()

        if cfg is None:
            cfg = MLPDecoderConfig(**kwargs)
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.num_tokens, embedding_dim=cfg.n_embd
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd * 2, cfg.n_embd * cfg.expansion_factor),
            nn.GELU(),
            nn.Linear(
                cfg.n_embd * cfg.expansion_factor, cfg.n_embd * cfg.expansion_factor
            ),
            nn.GELU(),
            nn.Linear(
                cfg.n_embd * cfg.expansion_factor, cfg.n_embd * cfg.expansion_factor
            ),
            nn.GELU(),
            nn.Linear(cfg.n_embd * cfg.expansion_factor, cfg.num_tokens),
            nn.GELU(),
        )

        self.no_ctx_emb = nn.Parameter(data=torch.rand(cfg.n_embd), requires_grad=True)

    def forward(
        self,
        input_ids,
        context_enc,
        shift_enc=True,
        **kwargs,
    ):
        bs, l = input_ids.shape
        
        if shift_enc:
            # Shift context embedding to the right and add <no_context> for the first token
            no_ctx_emb = einx.rearrange(
                "d -> b 1 d", self.no_ctx_emb, b=len(input_ids)
            )
            context_enc = torch.concatenate([no_ctx_emb, context_enc], dim=1)[
                :, :-1
            ]

        context_enc = einx.rearrange("b l d -> (b l) d", context_enc)
        input_tokens = einx.rearrange("b l d -> (b l) d ", self.embedding(input_ids))

        x = torch.concatenate([context_enc, input_tokens], -1)
        x = self.mlp(x)

        x = einx.rearrange("(b l) d -> b l d", x, b=bs, l=l)

        return x


@dataclass
class GRUDecoderConfig:
    num_tokens: int
    n_layer: int
    n_embd: int
    mlp_head: bool = False
    tag: Optional[str] = None


class GRUDecoder(DecoderModel):
    def __init__(self, cfg: Optional[GRUDecoderConfig] = None, **kwargs) -> None:
        if cfg is None:
            cfg = GRUDecoderConfig(**kwargs)
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=cfg.num_tokens, embedding_dim=cfg.n_embd
        )
        self.backbone = nn.GRU(
            input_size=cfg.n_embd,
            hidden_size=cfg.n_embd,
            num_layers=cfg.n_layer,
            batch_first=True,
        )
        if cfg.mlp_head:
            self.lm_head = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            )
        else:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.num_tokens)
        self.cfg = cfg

    def forward(
        self,
        input_ids,
        context_enc=None,
        return_hiddens=False,
        only_last_logits=False,
        return_embeddings=False,
    ):
        """
        context_enc : Initial hidden state. Shape (n_layer, batch, n_embd). Defaults to None.
        """
        x = self.embedding(input_ids)
        if context_enc != None:
            context_enc = context_enc.transpose(0, 1).contiguous()
        x, hiddens = self.backbone(x, context_enc)

        if return_embeddings == False:
            x = self.lm_head(x)

        if only_last_logits:
            x = x[:, [-1]]

        if return_hiddens:
            return x, hiddens.transpose(0, 1)
        else:
            return x


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
            hidden_states, residual = layer(hidden_states, residual)
        if not self.backbone.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.backbone.norm_f(
                residual.to(dtype=self.backbone.norm_f.weight.dtype)
            )
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
                is_rms_norm=isinstance(self.backbone.norm_f, RMSNorm),
            )
        if only_last_logits:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.lm_head(hidden_states)

        return lm_logits
