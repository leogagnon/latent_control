import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import *
from dataclasses import dataclass
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

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
    model : GPT2ConfigDataclass
    val_ratio: float
    batch_size: int
    lr: float
    
class MetaLearningTask(L.LightningModule):
    def __init__(self, cfg : TaskConfig) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.model = GPT2LMHeadModel(GPT2Config(**cfg.model.__dict__))
        
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

        labels = batch.clone()
        out = self.model(input_ids=batch, labels=labels)
        out: CausalLMOutputWithCrossAttentions

        pred = torch.roll(out.logits, shifts=1, dims=1).argmax(-1)
        acc = (pred == batch).float().mean()

        self.log("train/acc", acc)
        self.log("train/ce_loss", out.loss, prog_bar=True)

        return out.loss
    
    def validation_step(self, batch, batch_idx=None):

        labels = batch.clone()
        out = self.model(input_ids=batch, labels=labels)
        out: CausalLMOutputWithCrossAttentions

        pred = torch.roll(out.logits, shifts=1, dims=1).argmax(-1)
        acc = (pred == batch).float().mean()

        self.log("train/acc", acc)
        self.log("train/ce_loss", out.loss, prog_bar=True)

        return out.loss