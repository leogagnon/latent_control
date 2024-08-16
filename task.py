import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import *
from dataclasses import dataclass
import torch
from models.gpt import GPT, GPTConfig

from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig

@dataclass
class GPT2ConfigDataclass:
    n_positions: int
    n_ctx: int
    n_embd: int
    n_layer: int
    n_head: int
    activation_function: str
    resid_pdrop: float
    embd_pdrop: float
    attn_pdrop: float
    vocab_size: Optional[int]
    n_inner: Optional[int]

@dataclass
class TaskConfig:
    data : CompositionalHMMDatasetConfig
    model : GPTConfig
    val_ratio: float
    batch_size: int
    lr: float
    
class MetaLearningTask(L.LightningModule):
    def __init__(self, cfg : TaskConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        
        self.cfg = cfg
        self.model = GPT(cfg.model)
        
    def setup(self, stage: str = None):

        self.full_data = CompositionalHMMDataset(self.cfg.data)
        train_latents = set(np.arange(len(self.full_data)))

        # Choose the validation latents, and remove them from train
        val_latents = np.random.choice(len(self.full_data), int(len(train_latents) * self.cfg.val_ratio), replace=False)
        train_latents.difference_update(val_latents)

        train_latents = np.array(list(train_latents))
        self.train_data = Subset(self.full_data, indices=train_latents)
        self.val_data = Subset(self.full_data, indices=val_latents)

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
    