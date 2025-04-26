import os
from dataclasses import dataclass
from typing import Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
from einops import repeat
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from x_transformers.x_transformers import (AttentionLayers, Encoder,
                                           ScaledSinusoidalEmbedding)

import data
from data.diffusion import (ContextDiffusionDataset,
                            ContextDiffusionDatasetConfig)
from tasks.metalearn import MetaLearningTask


@dataclass
class DirectPosteriorConfig:
    pretrained_id: str
    batch_size: int
    val_split: float
    lr: float
    dataset: ContextDiffusionDatasetConfig
    n_embd: int
    n_layers: int
    n_heads: int
    seq_conditional_dim: Optional[int] = None
    cond_encoder_kwargs: Optional[dict] = None
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0


class DirectPosterior(L.LightningModule):
    """Trains a Transformer to directly predict a discrete \theta from x_{1...k}) using a cross-entropy loss, leading to p(\theta | x_{1...k})"""

    def __init__(self, cfg: Optional[DirectPosteriorConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(DirectPosteriorConfig),
                    OmegaConf.create(kwargs),
                )
            )

        base_task = MetaLearningTask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"], cfg.pretrained_id, "last.ckpt"
            ), strict=False
        )
        dataset = ContextDiffusionDataset(cfg.dataset, base_task)
        self.full_data = dataset
        self.train_data, self.val_data = random_split(
            self.full_data, [1 - cfg.val_split, cfg.val_split]
        )
        self.train_data = self.full_data

        if cfg.seq_conditional_dim is None:
            cfg.seq_conditional_dim = self.full_data.cond_dim

        self.latent_model = AttentionLayers(
            dim=cfg.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout,
            ff_dropout=cfg.ff_dropout,
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=True,
            causal=cfg.dataset.sequential,
        )

        if cfg.dataset.sequential:
            self.null_embedding = nn.Embedding(1, cfg.n_embd)
        else:
            self.null_embedding = nn.Embedding(
                len(base_task.data.latent_shape), cfg.n_embd
            )

        if cfg.cond_encoder_kwargs["vocab_size"] != None:
            self.cond_embedding = nn.Embedding(
                num_embeddings=cfg.cond_encoder_kwargs["vocab_size"],
                embedding_dim=cfg.seq_conditional_dim,
            )
            self.cond_posemb = ScaledSinusoidalEmbedding(cfg.seq_conditional_dim)

        self.cond_encoder = Encoder(
            dim=cfg.seq_conditional_dim,
            depth=cfg.cond_encoder_kwargs["n_layers"],
            heads=cfg.cond_encoder_kwargs["n_heads"],
        )

        self.cond_proj = nn.Linear(cfg.seq_conditional_dim, cfg.n_embd)

        # A different output matrix for each latent dimension
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(cfg.n_embd, latent_dim)
                for latent_dim in base_task.data.latent_shape
            ]
        )
        self.norm = nn.LayerNorm(cfg.n_embd)

        self.cfg = cfg

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def forward(self, x=None, cond_input_ids=None, cond_tokens=None, cond_mask=None):

        if self.cfg.cond_encoder_kwargs["vocab_size"] != None:
            assert cond_input_ids != None
            cond_tokens = self.cond_embedding(cond_input_ids)
            cond_tokens = cond_tokens + self.cond_posemb(cond_tokens)
        else:
            assert cond_tokens != None
            cond_tokens = cond_tokens

        cond_tokens = self.cond_encoder(cond_tokens)
        cond_tokens = self.cond_proj(cond_tokens)

        if x == None:
            if self.cfg.dataset.sequential:
                start_emb = repeat(
                    self.null_embedding.weight,
                    "1 d -> b 1 d",
                    b=cond_input_ids.shape[0],
                )
                x = start_emb
            else:
                x = repeat(
                    self.null_embedding.weight,
                    "L d -> b L d",
                    b=cond_input_ids.shape[0],
                )

        pred = self.latent_model(x=x, context=cond_tokens, context_mask=cond_mask)

        pred = self.norm(pred)
        pred = [self.out_proj[i](pred[:, i]) for i in range(pred.shape[1])]

        return pred

    def training_step(self, batch, batch_idx=None):

        if self.cfg.dataset.sequential:
            start_emb = repeat(
                self.null_embedding.weight,
                "1 d -> b 1 d",
                b=batch["cond_input_ids"].shape[0],
            )
            x = torch.concatenate([start_emb, batch["latent"]], dim=1)
        else:
            x = repeat(
                self.null_embedding.weight,
                "L d -> b L d",
                b=batch["cond_input_ids"].shape[0],
            )

        pred = self.forward(
            x=x,
            cond_input_ids=batch["cond_input_ids"],
            cond_tokens=batch["cond_tokens"],
            cond_mask=torch.logical_not(batch["cond_ignore_mask"]),
        )

        loss = sum(
            [
                nn.functional.cross_entropy(
                    pred[i].squeeze(), batch["raw_latent"][:, i]
                ).mean()
                for i in range(len(pred))
            ]
        )
        acc = sum(
            [
                (pred[i].squeeze().argmax(1) == batch["raw_latent"][:, i])
                .float()
                .mean()
                for i in range(len(pred))
            ]
        ) / len(pred)

        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
        )
        self.log(
            "train/acc",
            acc.detach().cpu().numpy().item(),
            prog_bar=True,
        )

        return loss
