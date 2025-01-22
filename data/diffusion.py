import math
import random
from abc import ABC
from dataclasses import dataclass
from typing import *

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN

from lightning_modules.metalearn import MetaLearningTask
from models.diffusion import DiffusionTransformerConfig
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from models.utils import exists, right_pad_dims_to
from models.x_transformer import ScaledSinusoidalEmbedding


class DiffusionDataset(ABC, Dataset):
    """Dataset used to train a diffusion model (encoder)"""

    def __init__(
        self, cfg, task: MetaLearningTask, diffusion: DiffusionEncoderConfig
    ) -> None:
        # Should verify here that the config, pre-trained model and diffusion encoder are compatible
        pass

    def __getitems__(self, Iterable):
        """
        Args:
            indices (Iterable)

        Returns:
            dict: With keys ['latent', 'cond', 'cond_mask']
                    latent : the thing we want to sample
                    cond : the thing we want to condition on (optional)
                    cond_mask : possibly a mask for this conditionning (optional)
                    cond_input_ids: the actual sequence of the conditioning (optional)
        """
        pass


@dataclass
class KnownLatentDiffusionDatasetConfig:
    size: int
    context_length: Tuple[int]


class KnownLatentDiffusionDataset(DiffusionDataset):
    def __init__(
        self,
        cfg: KnownLatentDiffusionDatasetConfig,
        task: MetaLearningTask,
        diffusion: DiffusionEncoder,
    ) -> None:
        super().__init__(cfg, task, diffusion)
        assert "KnownEncoder" in str(task.model.encoder.__class__)
        assert "TransformerDecoder" in str(task.model.decoder.__class__)

        # Sample some environemnts
        env_indices = torch.randint(
            low=0, high=len(task.full_data), size=(cfg.size,)
        ).cpu()

        # Compute task-latent embedding from known latent encoder
        env_latents = j2t(task.full_data.index_to_latent)[env_indices].to(torch.long)
        self.env_latents = task.model.encoder(true_latents=env_latents)

        self.embedding = nn.Embedding(
            num_embeddings=task.full_data.cfg.n_obs,
            embedding_dim=diffusion.cfg.seq_conditional_dim,
        ).cuda()
        self.pos_emb = ScaledSinusoidalEmbedding(diffusion.cfg.seq_conditional_dim).cuda()

        # Generate sequences
        cond = []
        mask = []
        cond_input_ids = []
        for batch in torch.split(env_indices, 512):
            out = task.full_data.__getitems__(batch, length=cfg.context_length)

            cond_input_ids.append(out["input_ids"])
            tokens = self.embedding(out["input_ids"])
            cond.append(tokens + self.pos_emb(tokens))

            mask.append(out["ignore_mask"])
        self.cond_input_ids = torch.concatenate(cond_input_ids, dim=0)
        self.cond = torch.concatenate(cond, dim=0)
        self.mask = torch.concatenate(mask, dim=0)

        self.cfg = cfg

    def __len__(self):
        return self.cfg.size

    def __getitem__(self, idx):
        return self.__getitem__([idx])

    def __getitems__(self, indices):
        indices = torch.LongTensor(indices)

        return {
            "latent": self.env_latents[indices],
            "cond_input_ids": self.cond_input_ids[indices],
            "cond": self.cond[indices],
            "cond_mask": self.mask[indices],
        }
