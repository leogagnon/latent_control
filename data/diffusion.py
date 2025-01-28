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
from models.encoder import KnownEncoder, KnownEncoderConfig

@dataclass
class LatentDiffusionDatasetConfig:
    context_length: Tuple[int]
    cond_tokens_type: Optional[str] = None
    latent_type: Optional[str] = None


class LatentDiffusionDataset(Dataset):
    """Dataset used to train a diffusion model (encoder)"""

    def __init__(
        self,
        cfg: LatentDiffusionDatasetConfig,
        task: MetaLearningTask,
        diffusion: DiffusionEncoder,
    ) -> None:
        
        # Should verify here that the config, pre-trained model and diffusion encoder are compatible
        if cfg.latent_type == 'known_encoder_pretrained':
            assert "TransformerDecoder" in str(task.model.decoder.__class__)
            assert "KnownEncoder" in str(self.task.model.encoder.__class__)
            self.known_encoder = self.task.model.encoder
        elif cfg.latent_type == 'known_encoder_new':
            self.known_encoder = KnownEncoder(KnownEncoderConfig(n_embd=diffusion.cfg.n_embd, latents_shape=task.full_data.latent_shape)).cuda()
        else:
            assert cfg.latent_type == None

        if cfg.cond_tokens_type == 'pretrained':
            assert "TransformerDecoder" in str(task.model.decoder.__class__)
        else:
            assert cfg.cond_tokens_type in [None, 'pretrained']

        self.task = task
        self.cfg = cfg
        self.diffusion = diffusion

    def __len__(self):
        return len(self.task.full_data)

    def __getitem__(self, idx):
        return self.__getitems__([idx])

    def __getitems__(self, indices) -> dict:
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
        indices = torch.LongTensor(indices)

        # Gather HMM latent
        raw_latent = (
            j2t(self.task.full_data.index_to_latent)[indices].to(torch.long).cuda()
        )

        # Sample a sequence from that HMM (and the associated ignore_mask)
        hmm_sample = self.task.full_data.__getitems__(
            indices, length=self.cfg.context_length
        )
        cond_ignore_mask = hmm_sample.get(
            "ignore_mask", torch.zeros_like(hmm_sample["input_ids"], dtype=torch.bool)
        )

        out_dict = {
            "raw_latent": raw_latent,
            "cond_input_ids": hmm_sample["input_ids"],
            "cond_ignore_mask": cond_ignore_mask,
            "cond_tokens": None,
            "latent": None
        }

        if (self.cfg.latent_type == 'known_encoder_new') or (self.cfg.latent_type == 'known_encoder_pretrained'):
            env_latents = self.known_encoder(true_latents=out_dict["raw_latent"])
            out_dict["latent"] = env_latents

        if self.cfg.cond_tokens_type == 'pretrained':
            out_dict["cond_tokens"] = self.task.model.decoder(out_dict["cond_input_ids"], return_embeddings=True)

        return out_dict