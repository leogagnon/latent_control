import math
import inspect
from dataclasses import dataclass
from typing import List, Optional
from data.hmm import CompositionalHMMDataset

import hydra
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList
import einops

@dataclass
class ExplicitConfig:
    enc_cfg: dict
    pred_cfg: dict
    vocab_size: Optional[int] = None
    markov_pred: Optional[bool] = False
    tag: Optional[str] = None

@dataclass
class KnownEncoderConfig:
    n_embd: int
    latents_shape: Optional[List[int]] = None # None means it will be set later

class KnownEncoder(nn.Module):
    def __init__(self, cfg: Optional[KnownEncoderConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = KnownEncoderConfig(**kwargs)
        assert cfg.latents_shape != None
        self.embeds = ModuleList([nn.Embedding(n, cfg.n_embd) for n in cfg.latents_shape])
    
    def forward(self, latents):
        out = torch.stack([self.embeds[i](l) for i,l in enumerate(latents.T)]).sum(0)
        return out


class ExplicitModel(nn.Module):
    def __init__(self, cfg: Optional[ExplicitConfig] = None, **kwargs) -> None:
        super().__init__()
        if cfg is None:
            cfg = ExplicitConfig(**kwargs)
        self.cfg = cfg
        self.pred = hydra.utils.instantiate(cfg.pred_cfg)
        self.enc = hydra.utils.instantiate(cfg.enc_cfg)

        if 'MLP' in str(self.pred.__class__):
            if 'KnownEncoder' in str(self.enc.__class__):
                self.wte = nn.Embedding(cfg.vocab_size, cfg.enc_cfg['n_embd'])

    def get_wte(self):
        if hasattr(self, 'wte'):
            return self.wte
        elif 'GPT' in str(self.enc.__class__):
            return self.enc.transformer.wte
        else:
            raise NotImplemented


    def forward(self, input_ids, targets=None, only_last_logits=False, attn_mask=None, latents=None):
        if "KnownEncoder" in self.cfg.enc_cfg['_target_']:
            assert latents != None
            latents_enc = self.enc(latents)[:, None]
        else:
            latents_enc = self.enc(input_ids) 
        
        if self.cfg.markov_pred:
            if 'MLP' in self.cfg.pred_cfg['_target_']:
                # Embed the tokens with the encoder's embedding
                x = self.get_wte()(input_ids)
                # Combine with the latent
                if latents_enc.shape != x.shape:
                    latents_enc = einops.repeat(latents_enc, 'b () h -> b l h', l=x.shape[1])
                x = torch.concatenate([latents_enc, x],-1)
                logits = self.pred(x)
                return logits
            else: 
                logits = self.pred(input_ids.view(-1)[:,None], prefix=latents_enc.view(-1,512)[:,None])[:,1]
                logits = logits.view(latents_enc.shape[0], latents_enc.shape[1], logits.shape[-1])
                return logits
        else: 
            raise NotImplementedError
    
class MLP(nn.Module):
    def __init__(self, in_dim, dim, out_dim, n_hidden_layers):
        super(MLP, self).__init__()
        layers = []
        for i in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)