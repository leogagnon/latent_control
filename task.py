import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import *
from dataclasses import dataclass
import torch
from models.gpt import GPT, GPTConfig

from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig


@dataclass
class TaskConfig:
    data: CompositionalHMMDatasetConfig
    model: GPTConfig
    val_ratio: float
    batch_size: int
    lr: float


class MetaLearningTask(L.LightningModule):
    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = GPT(cfg.model)

    def setup(self, stage: str = None):

        self.full_data = CompositionalHMMDataset(self.cfg.data)
        train_latents = set(np.arange(len(self.full_data)))

        # Choose the validation latents, and remove them from train
        val_latents = np.random.choice(
            len(self.full_data),
            int(len(train_latents) * self.cfg.val_ratio),
            replace=False,
        )
        train_latents.difference_update(val_latents)

        train_latents = np.array(list(train_latents))
        self.train_data = Subset(self.full_data, indices=train_latents)
        self.val_data = Subset(self.full_data, indices=val_latents)

    def get_finetuning_datasets(self, latent_id: int):
        id_is_active = self.full_data.index_to_latent[:, latent_id] == True

        active_set = set(id_is_active.nonzero()[0])
        inactive_set = set((~id_is_active).nonzero()[0])
        train_set = set(self.train_data.indices)
        val_set = set(self.val_data.indices)

        finetuning_dataset = {
            "active_train": Subset(self.full_data, list(active_set & train_set)),
            "active_val": Subset(self.full_data, list(active_set & val_set)),
            "inactive_train": Subset(self.full_data, list(inactive_set & train_set)),
            "inactive_val": Subset(self.full_data, list(active_set & val_set)),
        }

        return finetuning_dataset

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg.batch_size)

    def training_step(self, batch, batch_idx=None):

        shift_idx = batch[..., :-1].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        logits, loss = self.model(idx=shift_idx, targets=shift_labels)

        pred = logits.argmax(-1)
        acc = (pred == shift_labels).float().mean()

        self.log("train/acc", acc)
        self.log("train/ce_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx=None):

        shift_idx = batch[..., :-1].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        logits, loss = self.model(idx=shift_idx, targets=shift_labels)

        pred = logits.argmax(-1)
        acc = (pred == shift_labels).float().mean()

        self.log("val/acc", acc)
        self.log("val/ce_loss", loss, prog_bar=True)

        return loss
