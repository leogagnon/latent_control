from dataclasses import dataclass

import lightning as L
import torch
import torch.nn as nn
from einops import repeat
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

from data.diffusion import GRUDiffusionDataset, GRUDiffusionDatasetConfig
from tasks.metalearn import MetaLearningTask


@dataclass
class RawFromLatentConfig:
    pretrained_id: str
    dataset: GRUDiffusionDatasetConfig
    val_split: float = 0.2
    batch_size: int = 512
    lr: float = 1e-3


class RawFromLatent(L.LightningModule):
    """Trains a model to predict a discrete theta from z_T using a cross-entropy loss, leading to p(theta | z_T)"""

    cfg_cls: RawFromLatentConfig

    def __init__(self, cfg: RawFromLatentConfig):
        super().__init__()

        dataset = GRUDiffusionDataset(
            cfg=cfg.dataset, base_task=MetaLearningTask.from_id(cfg.pretrained_id)
        )
        self.full_data = dataset
        self.train_data, self.val_data = random_split(
            self.full_data, [1 - cfg.val_split, cfg.val_split]
        )
        self.train_data = self.full_data

        # A different output matrix for each latent dimension
        self.out_proj = nn.ModuleList(
            [
                nn.Linear(torch.prod(torch.tensor(dataset.latent_shape)), latent_dim)
                for latent_dim in self.full_data.latent_shape
            ]
        )

        self.cfg = cfg

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )
        self.wandb_dict = dict({})

    def on_save_checkpoint(self, ckpt) -> None:
        ckpt["train_data"] = self.train_data
        ckpt["val_data"] = self.val_data

    def on_load_checkpoint(self, ckpt):
        self.train_data = ckpt["train_data"]
        self.val_data = ckpt["val_data"]

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

    def forward(self, z):
        z = z.flatten(start_dim=1)
        pred = [self.out_proj[i](z) for i in range(len(self.out_proj))]
        return pred

    def training_step(self, batch, batch_idx=None):
        pred = self.forward(batch["latent"])
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
