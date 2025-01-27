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
                    cond_mask : possibly a mask for this conditionning (optional) [FALSE WHERE MASKED]
                    cond_input_ids: the actual sequence of the conditioning (optional)
        """
        pass


@dataclass
class KnownLatentDiffusionDatasetConfig:
    context_length: Tuple[int]


# NOTE : For now this dataset is infinite for simplicity
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

        self.task = task
        self.cfg = cfg
    
    def __len__(self):
        return len(self.task.full_data)

    def __getitem__(self, idx):
        return self.__getitems__([idx])

    def __getitems__(self, indices):
        indices = torch.LongTensor(indices)

        # Gather HMM latent and encode it with the known-latent encoder
        raw_latent = j2t(self.task.full_data.index_to_latent)[indices].to(torch.long).cuda()
        env_latents = self.task.model.encoder(true_latents=raw_latent)

        # Sample a sequence from that HMM
        hmm_sample = self.task.full_data.__getitems__(indices, length=self.cfg.context_length)
        cond_ignore_mask = hmm_sample.get('ignore_mask', torch.zeros_like(hmm_sample['input_ids'], dtype=torch.bool))

        return {
            "raw_latent": raw_latent,
            "latent": env_latents,
            "cond_input_ids": hmm_sample['input_ids'],
            "cond_ignore_mask": cond_ignore_mask,
        }
