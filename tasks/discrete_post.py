import lightning as L
from tasks.metalearn import MetaLearningTask
from models.x_transformer import Encoder, ScaledSinusoidalEmbedding
import torch.nn as nn
from dataclasses import dataclass
import torch
import hydra
import data
from torch.utils.data import random_split, DataLoader
from einops import repeat

@dataclass
class DiscretePosteriorConfig:
    pretrained_id: str
    cond_dim: int
    n_embd: int
    batch_size: int
    val_split: float
    lr: float
    cond_encoder: bool
    dataset: dict


class DiscretePosterior(L.LightningModule):
    def __init__(
        self, cfg: DiscretePosteriorConfig
        
    ):
        super().__init__()

        self.base_task = MetaLearningTask(cfg.pretrained_id)
        for param in self.base_task.parameters():
            param.requires_grad = False

        self.latent_model = Encoder(
            dim=cfg.n_embd,
            depth=3,
            heads=6,
            attn_dropout=0.0,  # dropout post-attention
            ff_dropout=0.0,  # feedforward dropout
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=True,
        )
        
        self.null_embedding = nn.Embedding(len(self.base_task.full_data.latent_shape), cfg.n_embd)

        if cfg.cond_encoder:
            self.seq_conditional_encoder = Encoder(
                dim=cfg.cond_dim,
                depth=3,
                heads=6,
                attn_dropout=0.0,  # dropout post-attention
                ff_dropout=0.0,  # feedforward dropout
                rel_pos_bias=False,
                ff_glu=True,
            )
            self.seq_conditional_emb = nn.Embedding(
                num_embeddings=50,
                embedding_dim=cfg.cond_dim,
            )
            self.seq_conditional_posemb = ScaledSinusoidalEmbedding(cfg.cond_dim)

        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.n_embd)
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(cfg.n_embd, latent_shape)
                for latent_shape in self.base_task.full_data.latent_shape
            ]
        )
        self.norm = nn.LayerNorm(cfg.n_embd)

    def setup(self, stage):
        with torch.no_grad():
            dataset_cfg = hydra.utils.instantiate(self.cfg.dataset)
            if "GRU" in self.cfg.dataset["_target_"]:
                dataset_cls = data.diffusion.GRUDiffusionDataset
            elif "Mamba" in self.cfg.dataset["_target_"]:
                dataset_cls = data.diffusion.MambaDiffusionDataset
            elif "KnownEncoder" in self.cfg.dataset["_target_"]:
                dataset_cls = data.diffusion.KnownEncoderDiffusionDataset
            else:
                assert False

            dataset = dataset_cls(dataset_cfg, self)
            self.full_data = dataset
            self.train_data, self.val_data = random_split(
                self.full_data, [1 - self.cfg.val_split, self.cfg.val_split]
            )

            # NOTE: training on all the HMMs for now to make sure this is not the limiting factor
            self.train_data = self.full_data

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def training_step(self, batch, batch_idx=None):
        
        if self.cond_encoder:
            cond = self.seq_conditional_emb(batch['cond_input_ids'])
            cond = cond + self.seq_conditional_posemb(cond)
            cond = self.seq_conditional_encoder(cond)
        else:
            cond = batch['cond_tokens']
            
        cond = self.cond_proj(cond)

        init_emb = repeat(self.null_embedding.weight, "1 d -> b 1 d", b=batch['cond_input_ids'].shape[0])

        pred = self.latent_model(
            init_emb,
            context=cond,
        )
        pred = self.norm(pred)
        pred = [proj(pred) for proj in self.out_proj]

        loss = sum(
            [
                nn.functional.cross_entropy(pred[i].squeeze(), batch['raw_latent'][:, i]).mean()
                for i in range(len(pred))
            ]
        )
        acc = sum(
            [
                (pred[i].squeeze().argmax(1) == batch['raw_latent'][:, i]).float().mean()
                for i in range(len(pred))
            ]
        ) / len(pred)
        # loss = torch.mean((pred - latent)**2)

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