import lightning as L
from tasks.metalearn import MetaLearningTask
import torch.nn as nn
from dataclasses import dataclass
import torch
import hydra
import data
from torch.utils.data import random_split, DataLoader
from einops import repeat
from data.diffusion import (
    KnownEncoderDiffusionDatasetConfig,
    KnownEncoderDiffusionDataset,
)
from x_transformers.x_transformers import AttentionLayers, ScaledSinusoidalEmbedding, Encoder

@dataclass
class DirectPosteriorConfig:
    pretrained_id: str 
    batch_size: int
    val_split: float
    lr: float
    dataset: KnownEncoderDiffusionDatasetConfig

    n_embd: int
    n_layers: int
    n_heads: int
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0
    cond_encoder_kwargs: Optional[dict]
    seq_conditional_dim: int


class DirectPosterior(L.LightningModule):
    """Trains a Transformer to directly predict a discrete \theta from x_{1...k}) using a cross-entropy loss, leading to p(\theta | x_{1...k}) """

    cfg_cls: DirectPosteriorConfig

    def __init__(self, cfg: DirectPosteriorConfig):
        super().__init__()

        self.base_task = MetaLearningTask(cfg.pretrained_id)
        for param in self.base_task.parameters():
            param.requires_grad = False

        self.latent_model = AttentionLayers(
            dim=cfg.n_embd,
            depth=cfg.n_layers,
            heads=cfg.n_heads,
            attn_dropout=cfg.attn_dropout, 
            ff_dropout=cfg.ff_dropout,  
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=True,
            causal=cfg.dataset.sequential_latent
        )

        if cfg.dataset.sequential_latent:
            self.null_embedding = nn.Embedding(
                len(self.base_task.full_data.latent_shape), cfg.n_embd
            )
        else:
            self.null_embedding = nn.Embedding(1, cfg.dataset)

        if cfg.cond_encoder_kwargs["vocab_size"] != None:
            self.cond_embedding = nn.Embedding(
                num_embeddings=cfg.cond_encoder_kwargs["vocab_size"],
                embedding_dim=cfg.seq_conditional_dim,
            )
            self.cond_posemb = ScaledSinusoidalEmbedding(cfg.seq_conditional_dim)

        self.cond_encoder = Encoder(
            dim=cfg.cond_encoder_kwargs["n_embd"],
            depth=cfg.cond_encoder_kwargs["n_layers"],
            heads=cfg.cond_encoder_kwargs["n_heads"],
        )

        self.cond_proj = nn.Linear(cfg.seq_conditional_dim, cfg.n_embd)

        # A different output matrix for each latent dimension
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(cfg.n_embd, latent_dim)
                for latent_dim in self.base_task.full_data.latent_shape
            ]
        )
        self.norm = nn.LayerNorm(cfg.n_embd)

        self.cfg = cfg

    def setup(self, stage):
        with torch.no_grad():
            dataset = KnownEncoderDiffusionDataset(self.cfg.dataset, self)
            self.full_data = dataset
            self.train_data, self.val_data = random_split(
                self.full_data, [1 - self.cfg.val_split, self.cfg.val_split]
            )

    def configure_optimizers(self):
        params = []
        for name, p in self.named_parameters():
            if not ("base_task" in name):
                params += [p]

        opt = torch.optim.AdamW(params, lr=self.cfg.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def training_step(self, batch, batch_idx=None):


        # Process the conditionning
        if self.cfg.cond_encoder_kwargs["vocab_size"] != None:
            assert batch["cond_input_ids"] != None
            cond_tokens = self.cond_embedding(batch["cond_input_ids"])
            cond_tokens = cond_tokens + self.cond_posemb(cond_tokens)
        else:
            assert batch["cond_tokens"] != None
            cond_tokens = batch["cond_tokens"]
        cond_tokens = self.cond_encoder(cond_tokens)
        cond_tokens = self.cond_proj(cond_tokens)

        if self.cfg.dataset.sequential_latent:
            start_emb = repeat(
                self.null_embedding.weight,
                "1 d -> b 1 d",
                b=batch["cond_input_ids"].shape[0],
            )
            x = torch.concatenate([start_emb, batch['latent']], dim=1)
        else:
            x = repeat(
                self.null_embedding.weight,
                "L d -> b L d",
                b=batch["cond_input_ids"].shape[0],
            )

        pred = self.latent_model(
            x=x,
            context=cond_tokens,
            context_mask=batch["cond_input_ids"]
        )
        
        pred = self.norm(pred)
        pred = [self.out_proj[i](pred[:,i]) for i in range(len(self.out_proj))]

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
