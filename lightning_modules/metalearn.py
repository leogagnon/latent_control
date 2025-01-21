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
import pyreft
import pyvene as pv
import scipy.special
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
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

from data.hmm import (CompositionalHMMDataset, CompositionalHMMDatasetConfig,
                      PrecomputedDataset, SubsetIntervened)
from models.base import MetaLearner

@dataclass
class MetaLearningConfig:
    data: CompositionalHMMDatasetConfig
    model: dict
    batch_size: int
    explicit: Optional[str] = None
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None
    lr: Optional[float] = 1e-3
    n_workers: Optional[int] = None

class MetaLearningTask(L.LightningModule):

    @singledispatchmethod
    def __init__(self, cfg):
        pass

    @__init__.register(MetaLearningConfig)
    def _from_cfg(self, cfg: MetaLearningConfig) -> None:
        super().__init__()

        if cfg.n_workers is None:
            cfg.n_workers = len(os.sched_getaffinity(0))

        self.cfg = cfg
        
        self.model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        self.model: MetaLearner
        self.wandb_dict = dict({})
        self.seen_tokens = 0
        self.full_data = None

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    @__init__.register(str)
    def _from_id(self, id: str):
        # Parse the checkpoint directory
        dir = os.path.join(os.environ["SCRATCH"], "latent_control_log/checkpoints/", id)
        ckpts = []
        for f in os.listdir(dir):
            if f != "last.ckpt":
                ckpts.append(f)

        # Load the last checkpoint
        ckpt = torch.load(os.path.join(dir, ckpts[-1]), weights_only=False)
        cfg = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.create(MetaLearningConfig),
                OmegaConf.create(ckpt[MetaLearningTask.CHECKPOINT_HYPER_PARAMS_KEY]),
            )
        )
        self._from_cfg(cfg)
        self.seen_tokens = ckpt.get("seen_tokens", 0)
        self.full_data = ckpt["dataset"]
        self.full_data.to_device("cpu")  # make sure the dataset in on the CPU
        self.train_data = Subset(self.full_data, ckpt["train_latents"])
        self.val_data = Subset(self.full_data, ckpt["val_latents"])
        print(f"Loaded dataset : ({len(self.train_data)}/{len(self.val_data)})")
        self.wandb_dict.update(
            {
                "id": id,
                "ckpts_dir": dir,
                "default_ckpt": "last.ckpt",
                "ckpts_names": ckpts,
            }
        )
        self.set_to_checkpoint(-1)

    def evaluate_pp(
        self,
        samples: Union[int, jax.Array, torch.Tensor],
        predicted_envs: Optional[np.array] = None,
        assumed_envs: np.array = None,
        seed: Optional[int] = None,
        n_steps: Optional[int] = None

    ) -> dict:
        """Computes the KL divergence between the model posterior predictive and the ground-truth

        Args:
            n_samples (int): How many sequences to compute KL on
            seq_len (int): Lenght of the sequences

        Returns:
            Tuple[np.array, np.array]: Forward KL, Backward KL
        """
        assert self.full_data is not None
        if seed is None:
            seed = np.random.randint(1e10)

        data = self.full_data

        # Sample HMMs, and sequences from these HMMs
        if isinstance(samples, int):
            if predicted_envs is None:
                predicted_envs = np.unique(self.val_data.indices)

            envs = jr.choice(jr.PRNGKey(seed), predicted_envs, (samples,))
            Xs = jax.vmap(data.sample, (0, None, 0))(
                envs, n_steps, jr.split(jr.PRNGKey(seed), len(envs))
            )[0]
        elif isinstance(samples, torch.Tensor):
            Xs = t2j(samples)
        else:
            Xs = samples

        # Gather the model's posterior predictive
        Xs_torch = j2t(Xs)
        true_latents = j2t(self.full_data.index_to_latent[envs]).long().to(Xs_torch.device)
        with torch.no_grad():
            model_pp = torch.softmax(
                self.model(j2t(Xs), only_last_logits=False, true_latents=true_latents),
                dim=-1,
            )
            model_pp = jnp.array(model_pp.tolist())[..., : data.cfg.n_obs]

        torch.cuda.empty_cache()

        # Gather the ground truth posterior predictive
        if assumed_envs is None:
            assumed_envs = jnp.arange(len(data))
        oracle_pp = []
        for i, X in tqdm(enumerate(Xs)):
            if 'ExplicitModel' in str(self.model.__class__):
                #if 'KnownEncoder' in str(self.model.enc.__class__):
                assumed_envs = envs[i][None]
            oracle_pp.append(data.bayesian_oracle(assumed_envs, X)["post_pred"])
        oracle_pp = jnp.stack(oracle_pp)[:, 1:, : data.cfg.n_obs]

        f = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            oracle_pp, model_pp[..., : data.cfg.n_obs]
        ).sum(-1)
        b = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            model_pp[..., : data.cfg.n_obs], oracle_pp
        ).sum(-1)
        nll_model = jax.vmap(nll, (0, 0))(model_pp[:, :-1], Xs[:, 1:])
        nll_oracle = jax.vmap(nll, (0, 0))(oracle_pp[:, :-1], Xs[:, 1:])

        return {
            "ForwardKL": j2t(f).cpu(),
            "BackwardKL": j2t(b).cpu(),
            "ModelNLL": j2t(nll_model).cpu(),
            "OracleNLL": j2t(nll_oracle).cpu(),
        }

    def model_loglikelihood(self, batch):
        shift_idx = batch[..., :-1].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        logits = self.model(input_ids=shift_idx, only_last_logits=False)
        logsoft = torch.log_softmax(logits, dim=-1)
        loglike = logsoft[
            torch.arange(shift_labels.shape[0], device=shift_labels.device)[:, None],
            torch.arange(shift_labels.shape[1], device=shift_labels.device).repeat(
                shift_labels.shape[0], 1
            ),
            shift_labels,
        ]
        return loglike.sum(-1)

    def set_to_checkpoint(
        self, ckpt_id: Optional[int] = None, step: Optional[int] = None
    ):
        assert len(self.wandb_dict) > 0
        assert sum([ckpt_id is None, step is None]) == 1

        if step is not None:
            steps = [
                int(filename.split(".ckpt")[0].split("step=")[1])
                for filename in self.wandb_dict["ckpts_names"]
            ]
            ckpt_id = np.abs((np.array(steps) / step) - 1).argmin()

        if ckpt_id == -1:
            ckpt_f = self.wandb_dict["default_ckpt"]
        else:
            ckpt_f = self.wandb_dict["ckpts_names"][ckpt_id]

        self.load_state_dict(
            torch.load(
                os.path.join(
                    self.wandb_dict["ckpts_dir"],
                    ckpt_f,
                ),
                weights_only=False,
            )["state_dict"]
        )
        print(f"Loaded checkpoing : {ckpt_f}")

    def setup(self, **kwargs):
        """Setup the data"""

        # Ensure setup is not called twice (e.g. when the model is fine-tuned)
        if self.full_data is not None:
            return

        self.full_data = CompositionalHMMDataset(self.cfg.data)
        train_latents = set(range(len(self.full_data)))

        if self.cfg.val_size is not None:
            val_size = self.cfg.val_size
        elif self.cfg.val_ratio is not None:
            val_size = int(len(self.full_data) * self.cfg.val_ratio)
        else:
            raise Exception("Either val_size or val_ratio have to be defined")

        # Choose the validation latents, and remove them from train
        val_latents = np.random.choice(
            len(self.full_data),
            val_size,
            replace=False,
        )
        train_latents.difference_update(val_latents)
        train_latents = np.array(list(train_latents))

        self.train_data = Subset(self.full_data, indices=train_latents)
        self.val_data = Subset(self.full_data, indices=val_latents)

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["seen_tokens"] = self.seen_tokens
        checkpoint["dataset"] = self.full_data
        checkpoint["train_latents"] = self.train_data.indices
        checkpoint["val_latents"] = self.val_data.indices

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
        if "attention_mask" in batch.keys():
            attn_mask = batch["attention_mask"][..., :-1, :-1]
        else:
            attn_mask = None

        # Apply pad mask
        if "ignore_mask" in batch.keys():
            shift_labels[batch["ignore_mask"][..., 1:]] = self.full_data.PAD_ID

        # Count the number of non-padding tokens seen
        self.seen_tokens += torch.sum(shift_labels != self.full_data.PAD_ID)

        true_latents = j2t(self.full_data.index_to_latent[batch["envs"]]).long().to(shift_idx.device)

        logits = self.model(input_ids=shift_idx, true_latents=true_latents, attn_mask=attn_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), shift_labels.long().view(-1)
        )

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()
        )

        self.log("seen_tokens", float(self.seen_tokens), add_dataloader_idx=False, batch_size=bs)
        self.log("train/acc", acc, add_dataloader_idx=False)
        self.log("train/ce_loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=bs)

        return loss

    def on_validation_start(self) -> None:
        self.full_data.val_mode = True

    def on_validation_end(self) -> None:
        self.full_data.val_mode = False

    def validation_step(self, batch, batch_idx=None):

        bs = batch["input_ids"].shape[0]

        # Shift tokens, labels and mask
        shift_idx = batch["input_ids"][..., :-1].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        if "attention_mask" in batch.keys():
            attn_mask = batch["attention_mask"][..., :-1, :-1]
        else:
            attn_mask = None

        if "ignore_mask" in batch.keys():
            shift_labels[batch["ignore_mask"][..., 1:]] = self.full_data.PAD_ID

        true_latents = j2t(self.full_data.index_to_latent[batch["envs"]]).long().to(shift_idx.device)

        logits = self.model(input_ids=shift_idx, true_latents=true_latents, attn_mask=attn_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), shift_labels.long().view(-1)
        )

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()
        )
        self.log("seen_tokens", float(self.seen_tokens), add_dataloader_idx=False, batch_size=bs)
        self.log("val/acc", acc, add_dataloader_idx=False, batch_size=bs)
        self.log("val/ce_loss", loss, prog_bar=True, add_dataloader_idx=False, batch_size=bs)

        return loss
    
@jax.jit
def nll(probs, seq):
    ll = probs[jnp.arange(len(probs)), seq]
    return -jnp.log(ll)


