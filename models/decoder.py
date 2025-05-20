from abc import ABC
from dataclasses import dataclass
from typing import *
import os

import einx
import torch
import torch.nn as nn
from x_transformers.x_transformers import Decoder, Encoder, TransformerWrapper

from models.base import DecoderModel
from tasks.metalearn import MetaLearningTask


@dataclass
class TransformerDecoderConfig:
    max_seq_len: Optional[int] = None
    num_tokens: Optional[int] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    causal_mask: Optional[bool] = None
    sin_posemb: Optional[bool] = None

    soft_token_enc: Optional[bool] = None
    pretrained_id: Optional[str] = None
    enc_dim: Optional[int] = None
    tag: Optional[str] = None


# NOTE: the way this model takes a context encoding is by appending it to its context
class TransformerDecoder(TransformerWrapper, DecoderModel):
    def __init__(self, cfg: Optional[TransformerDecoderConfig] = None, **kwargs):
        if cfg is None:
            cfg = TransformerDecoderConfig(**kwargs)

        if cfg.pretrained_id != None:
            task = MetaLearningTask.load_from_checkpoint(
                os.path.join(
                    os.environ["LATENT_CONTROL_CKPT_DIR"],
                    cfg.pretrained_id,
                    "last.ckpt",
                ),
                strict=False,
            )
            # TODO: hacky af but for quick submission, will fix later
            cfg.max_seq_len = task.model.encoder.cfg.max_seq_len
            cfg.num_tokens = task.model.encoder.cfg.num_tokens
            cfg.n_layer = task.model.encoder.cfg.n_layer
            cfg.n_head =task.model.encoder.cfg.n_head
            cfg.n_embd =  task.model.encoder.cfg.n_embd
            cfg.causal_mask =task.model.encoder.cfg.causal_mask
            cfg.sin_posemb =task.model.encoder.cfg.sin_posemb

            assert task.model.decoder == None, "The model should be implicit"
            assert cfg.soft_token_enc == True

        attn_layer_cls = Decoder if cfg.causal_mask else Encoder
        super().__init__(
            num_tokens=cfg.num_tokens,
            max_seq_len=cfg.max_seq_len,
            attn_layers=attn_layer_cls(
                dim=cfg.n_embd,
                depth=cfg.n_layer,
                heads=cfg.n_head,
                cross_attend=not cfg.soft_token_enc,
            ),
            scaled_sinu_pos_emb=cfg.sin_posemb,
            use_abs_pos_emb=cfg.sin_posemb,
        )

        if cfg.pretrained_id != None:
            self.load_state_dict(task.model.encoder.state_dict(), strict=True)

        if cfg.enc_dim != None:
            self.enc_proj = nn.Linear(cfg.enc_dim, cfg.n_embd, bias=False)
        else:
            self.enc_proj = nn.Identity()

        self.cfg = cfg

    def forward(self, input_ids, context_enc, **kwargs):
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
            out = super().forward(x=input_ids, context=context_enc)

        return out
