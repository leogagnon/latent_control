from abc import ABC
from dataclasses import dataclass
from typing import *
import torch.nn as nn
import torch
from models.gpt import GPT, GPTConfig

@dataclass
class KnownEncoderConfig:
    n_embd: int
    latents_shape: List[int]

class EncoderModel(ABC, nn.Module):
    def forward(self, tokens, true_latents, **kwargs):
        pass

class KnownEncoder(EncoderModel):
    def __init__(self, cfg: Optional[KnownEncoderConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = KnownEncoderConfig(**kwargs)
        self.latent_embedding = nn.ModuleList(
            [nn.Embedding(n, cfg.n_embd) for n in cfg.latents_shape]
        )

    def forward(self, tokens, true_latents):
        out = torch.stack(
            [self.latent_embedding[i](l) for i, l in enumerate(true_latents.T)]
        ).sum(0)
        return out[:,None]


class GPTEncoder(EncoderModel, GPT):
    def __init__(self, config: Optional[GPTConfig] = None, **kwargs):
        if config is None:
            config = GPTConfig(**kwargs)
        super().__init__(config)

    def forward(
        self,
        tokens,
        true_latents,
        attn_mask=None,
    ):
        for block in self.transformer.h:
            x = block(tokens, attn_mask)
        x = self.transformer.ln_f(x)

        # Only returning the last token
        logits = self.lm_head(x[:, -1, :])
        return logits