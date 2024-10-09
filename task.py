import os
import pickle
import traceback
import numpy as np
import scipy
import scipy.special
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import *
from dataclasses import dataclass
import torch
from models.gpt import GPT, GPTConfig, ModelConfig
from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
from transformers import PreTrainedModel, PretrainedConfig
from peft import get_peft_config, get_peft_model, LoraConfig
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
from torchmetrics.functional import kl_divergence
from copy import deepcopy
import math
from omegaconf import MISSING, SCMode
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
from torch2jax import j2t, t2j

@jax.jit
def nll(probs, seq):
    ll = probs[jnp.arange(len(probs)), seq]
    return -jnp.log(ll)


def make_hf_wrapper(model: nn.Module):
    """Wraps a PyTorch module into a HF module"""

    class HFWrapperConfig(PretrainedConfig):
        model_type = "HF_Wrapper"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class HFWrapper(PreTrainedModel):
        config_class = HFWrapperConfig

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.model = deepcopy(model)
            self.PAD_TOK = model.PAD_TOK
            self.BOS_TOK = model.BOS_TOK

        def forward(self, idx, targets=None, only_last_logits=True):
            return self.model(idx, targets, only_last_logits)

    return HFWrapper(HFWrapperConfig())


@dataclass
class TaskConfig:
    data: CompositionalHMMDatasetConfig
    model: dict
    batch_size: int
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None
    lr: Optional[float] = 1e-3
    n_workers: Optional[int] = None


class MetaLearningTask(L.LightningModule):

    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__()

        if cfg.n_workers is None:
            cfg.n_workers = len(os.sched_getaffinity(0))

        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)
        self.wandb_dict = dict({})
        self.seen_tokens = 0
        self.full_data = None

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )

    @classmethod
    def from_wandb_id(cls: "MetaLearningTask", id: str) -> "MetaLearningTask":
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
                OmegaConf.create(TaskConfig),
                OmegaConf.create(ckpt[MetaLearningTask.CHECKPOINT_HYPER_PARAMS_KEY]),
            )
        )
        task = MetaLearningTask(cfg)
        task.seen_tokens = ckpt.get("seen_tokens", 0)
        task.full_data = ckpt["dataset"]
        task.train_data = Subset(task.full_data, ckpt["train_latents"])
        task.val_data = Subset(task.full_data, ckpt["val_latents"])
        print(f"Loaded dataset : ({len(task.train_data)}/{len(task.val_data)})")
        task.wandb_dict.update({"id": id, "ckpts_dir": dir, "ckpts_names": ckpts})
        task.set_to_checkpoint(-1)

        return task

    def make_lora_task(
        self, cfg: LoraConfig, constraints: List[int]
    ) -> "MetaLearningTask":
        """Make a copy of this task where the model has LoRA adapters and data is restrained"""

        task = deepcopy(self)
        task.model = get_peft_model(
            model=make_hf_wrapper(task.model),
            peft_config=cfg,
        )

        constraint_is_active = np.ones(len(self.full_data), dtype=np.bool)
        for c in constraints:
            constraint_is_active = np.logical_and(
                constraint_is_active, self.full_data.index_to_latent[:, c[0]] == c[1]
            )

        train_set = set(self.train_data.indices)
        val_set = set(self.val_data.indices)

        active_set = set(constraint_is_active.nonzero()[0])
        val_active = list(active_set & val_set)
        val_active = val_active * math.ceil(len(self.val_data) / len(val_active))
        val_active = val_active[: len(self.val_data)]

        task.train_data = Subset(self.full_data, list(active_set & train_set))
        task.val_data = Subset(self.full_data, val_active)
        task.latent_indices = np.array(list(active_set))
        task.seen_tokens = 0

        return task

    def evaluate_pp(
        self,
        n_samples: int,
        seq_len: int,
        predicted_envs: Optional[np.array] = None,
        assumed_envs: np.array = None,
        seed: Optional[int] = None,
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

        if predicted_envs is None:
            predicted_envs = np.unique(self.val_data.indices)

        # Sample HMMs, and sequences from these HMMs
        envs = jr.choice(jr.PRNGKey(seed), predicted_envs, (n_samples,))
        Xs = jax.vmap(data.sample, (0, None, 0))(
            envs, seq_len, jr.split(jr.PRNGKey(seed), len(envs))
        )

        # Gather the model's posterior predictive
        self.model.cuda()
        with torch.no_grad():
            model_pp = torch.softmax(
                self.model.forward(
                    j2t(Xs),
                    only_last_logits=False,
                )[1],
                dim=-1,
            )
            model_pp = t2j(model_pp.cpu())
        self.model.cpu()

        # Gather the ground truth posterior predictive
        if assumed_envs is None:
            assumed_envs = jnp.arange(len(data))
        oracle_pp = []
        for X in tqdm(Xs):
            oracle_pp.append(
                jax.device_put(
                    data.posterior_predictive(assumed_envs, X[1:]), jax.devices("cpu")[0]
                )
            )
        oracle_pp = jnp.stack(oracle_pp)

        f = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            oracle_pp[:, :-1], model_pp[:, :-1, : data.cfg.n_obs]
        ).sum(-1)
        b = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            model_pp[:, :-1, : data.cfg.n_obs], oracle_pp[:, :-1]
        ).sum(-1)
        nll_model = jax.vmap(nll, (0, 0))(
            model_pp[:, :-1, : data.cfg.n_obs], Xs[:,1:]
        )
        nll_oracle = jax.vmap(nll, (0, 0))(
            oracle_pp[:, :-1, : data.cfg.n_obs], Xs[:,1:]
        )

        return {
            "ForwardKL": j2t(f),
            "BackwardKL": j2t(b),
            "ModelNLL": j2t(nll_model),
            "OracleNLL": j2t(nll_oracle),
        }

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

        self.load_state_dict(
            torch.load(
                os.path.join(
                    self.wandb_dict["ckpts_dir"],
                    self.wandb_dict["ckpts_names"][ckpt_id],
                ),
                weights_only=False,
            )["state_dict"]
        )
        print(f'Loaded checkpoing : {self.wandb_dict["ckpts_names"][ckpt_id]}')

    def setup(self, **kwargs):
        """Setup the data"""

        # Ensure setup is not called twice (e.g. when the model is fine-tuned)
        if self.full_data is not None:
            return

        self.full_data = CompositionalHMMDataset(self.cfg.data)
        train_latents = set(np.arange(len(self.full_data)))

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
            self.val_data, batch_size=self.cfg.batch_size, collate_fn=lambda x: x
        )

    def training_step(self, batch, batch_idx=None):

        seqs, attn_mask, pad_mask = batch

        # Shift tokens, labels and mask
        shift_idx = seqs[..., :-1].contiguous()
        shift_labels = seqs[..., 1:].contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask[..., :-1, :-1].contiguous()

        # Apply pad mask
        if pad_mask is not None:
            shift_labels[pad_mask[...,1:]] = self.full_data.PAD_ID
        
        # Count the number of non-padding tokens seen
        self.seen_tokens += torch.sum(shift_labels != self.full_data.PAD_ID)

        loss, logits = self.model(
            idx=shift_idx, targets=shift_labels.long(), attn_masks=attn_mask
        )

        pred = logits.argmax(-1)
        acc = (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()

        self.log("seen_tokens", float(self.seen_tokens))
        self.log("train/acc", acc)
        self.log("train/ce_loss", loss, prog_bar=True)

        return loss

    def on_validation_start(self) -> None:
        self.full_data.val_mode = True

    def on_validation_end(self) -> None:
        self.full_data.val_mode = False

    def validation_step(self, batch, batch_idx=None):

        seqs, attn_mask, pad_mask = batch
        seqs: torch.Tensor

        # Shift tokens, labels and mask
        shift_idx = seqs[..., :-1].contiguous()
        shift_labels = seqs[..., 1:].contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask[..., :-1, :-1].contiguous()

        loss, logits = self.model(
            idx=shift_idx, targets=shift_labels.long(), attn_masks=attn_mask
        )

        pred = logits.argmax(-1)
        acc = (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()
        self.log("seen_tokens", float(self.seen_tokens))
        self.log("val/acc", acc)
        self.log("val/ce_loss", loss, prog_bar=True)

        return loss
