import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import *

import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch2jax import j2t, t2j
from torch.nn import init
from torch.utils.data import DataLoader, StackDataset, Subset, TensorDataset
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers.activations import ACT2FN

from data.hmm import MetaHMM, MetaHMMConfig
from models.base import MetaLearner, MetaLearnerConfig
from utils import *

IGNORE_INDEX = -100


@dataclass
class MetaLearningConfig:
    data: MetaHMMConfig
    model: MetaLearnerConfig
    batch_size: int
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None
    lr: Optional[float] = 1e-3

class MetaLearningTask(L.LightningModule):

    def __init__(self, cfg: Optional[MetaLearningConfig] = None, **kwargs):
        super().__init__()

        if cfg == None:
            cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.create(MetaLearningConfig),
                    OmegaConf.create(kwargs),
                )
            )

        self.cfg = cfg

        self.register_buffer("seen_tokens", torch.tensor(0))

        # Init model and data
        self.model = MetaLearner(cfg.model)
        # If the encoder is pre-trained, use the same dataset seed
        try:
            if self.model.encoder.cfg.pretrained_id != None:
                cfg.data.seed = MetaLearningTask.load_from_checkpoint(
                    os.path.join(
                        os.environ["LATENT_CONTROL_CKPT_DIR"],
                        self.model.encoder.cfg.pretrained_id,
                        "last.ckpt",
                    ),
                    strict=False,
                ).data.cfg.seed
        except:
            pass 
        self.data = MetaHMM(cfg.data)
        self.data.to_device("cpu")

        # Build the train/validation set
        if self.cfg.val_size is not None:
            val_size = self.cfg.val_size
        elif self.cfg.val_ratio is not None:
            val_size = int(len(self.data) * self.cfg.val_ratio)
        else:
            raise Exception("Either val_size or val_ratio have to be defined")

        val_latents = np.random.choice(
            len(self.data),
            val_size,
            replace=False,
        )
        train_latents = set(range(len(self.data)))
        train_latents.difference_update(val_latents)
        train_latents = np.array(list(train_latents))

        self.register_buffer("val_latents", torch.IntTensor(val_latents))
        self.register_buffer("train_latents", torch.IntTensor(train_latents))

        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    def evaluate_pp(
        self,
        samples: Union[int, jax.Array, torch.Tensor],
        predicted_envs: Optional[np.array] = None,
        envs: Optional[Any] = None,
        assumed_envs: Optional[np.array] = None,
        seed: Optional[int] = None,
        n_steps: Optional[int] = None,
        compare_to_known: Optional[bool] = False,
    ) -> dict:
        """Computes the KL divergence between the model posterior predictive and the ground-truth

        Args:
            samples : How many sequences to compute KL (int) or the sequences themselves (jax.Array or torch.Tensor)
            predicted_envs Optional[np.array]: Environments to sample sequences from (if sampling them)
            n_steps Optional[int]: Length of the sampled sequences (if sampling them)
            assumed_envs Optional[np.array]: Environments considered possible by the oracle
            seed Optional[int]: Seeds the whole process

        Returns:
            Tuple[np.array, np.array]: Forward KL, Backward KL
        """
        if seed is None:
            seed = np.random.randint(1e10)

        # Sample HMMs, and sequences from these HMMs
        if isinstance(samples, int):
            if predicted_envs is None:
                predicted_envs = np.unique(self.val_latents.to(device="cpu"))

            envs = jr.choice(jr.PRNGKey(seed), predicted_envs, (samples,))
            Xs, Zs = jax.vmap(self.data.sample, (0, None, 0))(
                envs, n_steps, jr.split(jr.PRNGKey(seed), len(envs))
            )
        elif isinstance(samples, torch.Tensor):
            Xs = t2j(samples)
        else:
            Xs = samples

        # Compute the model's posterior predictive
        with torch.no_grad():
            model_pp = torch.softmax(
                self.model(
                    input_ids=j2t(Xs),
                    states=j2t(Zs),
                    true_envs=j2t(envs),
                    dataset=self.data
                ),
                dim=-1,
            )
            model_pp = jnp.array(model_pp.tolist())[..., : self.data.cfg.n_obs]

        # Compute the oracle's posterior predictive
        if assumed_envs is None:
            assumed_envs = jnp.arange(len(self.data))
        oracle_pp = []
        for i, X in tqdm(enumerate(Xs)):
            if compare_to_known:
                assumed_envs = envs[i][None]
            oracle_pp.append(self.data.bayesian_oracle(assumed_envs, X)["post_pred"])
        oracle_pp = jnp.stack(oracle_pp)[:, 1:, : self.data.cfg.n_obs]

        # Convert everything to torch.Tensor
        oracle_pp = j2t(oracle_pp).cpu()
        model_pp = j2t(model_pp).cpu()
        Xs = j2t(Xs).cpu()

        # Compute forward/backward KL and NLL of model and oracle
        f_kl = KLDiv(oracle_pp, model_pp)
        b_kl = KLDiv(model_pp, oracle_pp)
        model_nll = NLL(model_pp, Xs)
        oracle_nll = NLL(oracle_pp, Xs)

        return {
            "ForwardKL": f_kl,
            "BackwardKL": b_kl,
            "ModelNLL": model_nll,
            "OracleNLL": oracle_nll,
        }

    def setup(self, **kwargs):
        """Setup the data"""
        self.train_data = Subset(self.data, indices=self.train_latents)
        self.val_data = Subset(self.data, indices=self.val_latents)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
        )

    def training_step(self, batch, batch_idx=None):

        bs = batch["input_ids"].shape[0]

        # Shift tokens, labels and mask
        shift_idx = batch["input_ids"][..., :-1].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()

        # Apply pad mask
        if "ignore_mask" in batch.keys():
            shift_labels[batch["ignore_mask"][..., 1:]] = IGNORE_INDEX

        # Count the number of non-padding tokens seen
        self.seen_tokens += torch.sum(shift_labels != IGNORE_INDEX)

        logits = self.model(
            input_ids=shift_idx,
            true_envs=j2t(batch["envs"]),
            states=batch["states"][..., :-1],
            dataset=self.data
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), shift_labels.long().view(-1), ignore_index=IGNORE_INDEX
        )

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != IGNORE_INDEX].float().mean()
        )

        self.log(
            "seen_tokens",
            float(self.seen_tokens),
            add_dataloader_idx=False,
            batch_size=bs,
        )
        self.log("train/acc", acc, add_dataloader_idx=False)
        self.log(
            "train/ce_loss",
            loss,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=bs,
        )

        return loss

    def on_validation_start(self) -> None:
        self.data.val_mode = True

    def on_validation_end(self) -> None:
        self.data.val_mode = False

    def validation_step(self, batch, batch_idx=None):

        bs = batch["input_ids"].shape[0]

        # Shift tokens, labels and mask
        shift_idx = batch["input_ids"][..., :-1].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()

        if "ignore_mask" in batch.keys():
            shift_labels[batch["ignore_mask"][..., 1:]] = IGNORE_INDEX

        logits = self.model(
            input_ids=shift_idx,
            true_envs=j2t(batch["envs"]),
            states=batch["states"][..., :-1],
            dataset=self.data
        )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), shift_labels.long().view(-1), ignore_index=IGNORE_INDEX
        )

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != IGNORE_INDEX].float().mean()
        )
        self.log(
            "seen_tokens",
            float(self.seen_tokens),
            add_dataloader_idx=False,
            batch_size=bs,
        )
        self.log("val/acc", acc, add_dataloader_idx=False, batch_size=bs)
        self.log(
            "val/ce_loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=bs
        )

        return loss