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
from models.gpt import GPT, GPTConfig
from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
from transformers import PreTrainedModel, PretrainedConfig
from peft import get_peft_config, get_peft_model, LoraConfig, LoraModel
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
from models.mamba import MambaLMHeadModel
from models.gpt import GPT
import pyvene as pv
import pyreft
from functools import singledispatchmethod
import torch.nn.functional as F
from models.mamba import MambaLMHeadModel
import dataclasses
from transformers import BatchEncoding
import torch
from collections import OrderedDict

from pyvene import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from pyreft.interventions import LowRankRotateLayer
from transformers.activations import ACT2FN
from torch.nn import init


@dataclass
class TaskConfig:
    data: CompositionalHMMDatasetConfig
    model: dict
    batch_size: int
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None
    lr: Optional[float] = 1e-3
    n_workers: Optional[int] = None


@dataclass
class TuneConfig:
    pretrained_id: str
    method_config: dict  # LoraConfig or ReftConfig
    constraints: List[List[int]]
    batch_size: Optional[int] = None


@dataclass
class ReftConfig:
    low_rank_dimension: int
    layers: List[int]
    t_slice: Tuple[int]
    component: str


class MetaLearningTask(L.LightningModule):

    @singledispatchmethod
    def __init__(self, cfg):
        pass

    @__init__.register(TaskConfig)
    def _from_cfg(self, cfg: TaskConfig) -> None:
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
                OmegaConf.create(TaskConfig),
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
        self.wandb_dict.update({"id": id, "ckpts_dir": dir, "ckpts_names": ckpts})
        self.set_to_checkpoint(-1)

    def evaluate_pp(
        self,
        samples: Union[int, jax.Array, torch.Tensor],
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

        # Sample HMMs, and sequences from these HMMs
        if isinstance(samples, int):
            if predicted_envs is None:
                predicted_envs = np.unique(self.val_data.indices)

            envs = jr.choice(jr.PRNGKey(seed), predicted_envs, (samples,))
            Xs = jax.vmap(data.sample, (0, None, 0))(
                envs, seq_len, jr.split(jr.PRNGKey(seed), len(envs))
            )
        elif isinstance(samples, torch.Tensor):
            Xs = t2j(samples)
        else:
            Xs = samples

        # Gather the model's posterior predictive
        with torch.no_grad():
            model_pp = torch.softmax(
                self.model(j2t(Xs), only_last_logits=False),
                dim=-1,
            )
            model_pp = t2j(model_pp)

        torch.cuda.empty_cache()

        # Gather the ground truth posterior predictive
        if assumed_envs is None:
            assumed_envs = jnp.arange(len(data))
        oracle_pp = []
        for X in Xs:
            oracle_pp.append(data.posterior_predictive(assumed_envs, X[1:]))
        oracle_pp = jnp.stack(oracle_pp)

        f = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            oracle_pp[:, :-1], model_pp[:, :-1, : data.cfg.n_obs]
        ).sum(-1)
        b = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            model_pp[:, :-1, : data.cfg.n_obs], oracle_pp[:, :-1]
        ).sum(-1)
        nll_model = jax.vmap(nll, (0, 0))(model_pp[:, :-1, : data.cfg.n_obs], Xs[:, 1:])
        nll_oracle = jax.vmap(nll, (0, 0))(
            oracle_pp[:, :-1, : data.cfg.n_obs], Xs[:, 1:]
        )

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
            attn_mask = attn_mask[..., :-1, :-1]

        # Apply pad mask
        if pad_mask is not None:
            shift_labels[pad_mask[..., 1:]] = self.full_data.PAD_ID

        # Count the number of non-padding tokens seen
        self.seen_tokens += torch.sum(shift_labels != self.full_data.PAD_ID)

        logits = self.model(input_ids=shift_idx, attn_mask=attn_mask)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), shift_labels.long().view(-1)
        )

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()
        )

        self.log("seen_tokens", float(self.seen_tokens), add_dataloader_idx=False)
        self.log("train/acc", acc, add_dataloader_idx=False)
        self.log("train/ce_loss", loss, prog_bar=True, add_dataloader_idx=False)

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

        logits = self.model(input_ids=shift_idx, attn_mask=attn_mask)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), shift_labels.long().view(-1)
        )

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()
        )
        self.log("seen_tokens", float(self.seen_tokens), add_dataloader_idx=False)
        self.log("val/acc", acc, add_dataloader_idx=False)
        self.log("val/ce_loss", loss, prog_bar=True, add_dataloader_idx=False)

        return loss


class FineTuningTask(MetaLearningTask):
    def __init__(self, cfg: TuneConfig) -> None:
        super().__init__(cfg.pretrained_id)

        if isinstance(cfg.method_config, dict):
            method_config = hydra.utils.instantiate(cfg.method_config)
        else:
            method_config = cfg.method_config

        # Wrap the model
        if method_config.__class__.__name__ == "LoraConfig":
            self.model = get_peft_model(
                model=make_hf_wrapper(self.model),
                peft_config=cfg.method_config,
            )
        elif method_config.__class__.__name__ == "ReftConfig":
            # DictConfig -> dict to avoid serialization error
            if self.model.config.__class__.__name__ == "MambaConfig":
                self.model.config.ssm_cfg = dict(self.model.config.ssm_cfg)

            self.model.config = PretrainedConfig.from_dict(
                dataclasses.asdict(self.model.config)
            )
            embed_dim = (
                self.model.config.n_embd
                if isinstance(self.model, GPT)
                else self.model.config.d_model
            )
            component_prefix = (
                "transformer.h" if isinstance(self.model, GPT) else "backbone.layers"
            )

            repr_configs = [
                pv.RepresentationConfig(
                    component=f"{component_prefix}[{l}].output",
                    intervention=CustomLoreftIntervention(
                        low_rank_dimension=method_config.low_rank_dimension,
                        embed_dim=embed_dim,
                        dtype=torch.float32,
                    ).to("cuda"),
                    unit="pos",
                )
                for l in method_config.layers
            ]
            self.model = DynamicIntervenableModel(
                pv.IntervenableConfig(repr_configs),
                model=self.model,
                t_slice=method_config.t_slice,
            )
        else:
            raise NotImplementedError("Unsupported method config")

        constraint_is_active = np.ones(len(self.full_data), dtype=np.bool)
        for c in cfg.constraints:
            constraint_is_active = np.logical_and(
                constraint_is_active, self.full_data.index_to_latent[:, c[0]] == c[1]
            )

        train_set = set(self.train_data.indices)
        val_set = set(self.val_data.indices)

        active_set = set(constraint_is_active.nonzero()[0])
        inactive_set = set(np.logical_not(constraint_is_active).nonzero()[0])

        val_active = list(active_set & val_set)
        val_active = val_active * math.ceil(len(self.val_data) / len(val_active))
        val_active = val_active[: len(self.val_data)]

        val_inactive = list(inactive_set & val_set)
        val_inactive = val_inactive * math.ceil(len(self.val_data) / len(val_inactive))
        val_inactive = val_inactive[: len(self.val_data)]

        self.train_data = Subset(self.full_data, list(active_set & train_set))
        self.val_data = {
            "active": Subset(self.full_data, val_active),
            "inactive": Subset(self.full_data, val_inactive),
        }
        self.latent_indices = np.array(list(active_set))
        self.seen_tokens = 0

        self.tune_cfg = cfg

        # Possibly override batch_size
        if self.tune_cfg.batch_size is not None:
            self.cfg.batch_size = self.tune_cfg.batch_size

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["seen_tokens"] = self.seen_tokens
        checkpoint["dataset"] = self.full_data
        checkpoint["train_latents"] = self.train_data.indices
        checkpoint["val_latents"] = sum(*[d.indices for d in self.val_data.values()])

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_data["active"],
                batch_size=self.cfg.batch_size,
                collate_fn=lambda x: x,
            ),
            DataLoader(
                self.val_data["inactive"],
                batch_size=self.cfg.batch_size,
                collate_fn=lambda x: x,
            ),
        ]

    def validation_step(self, batch, batch_idx=None, dataloader_idx=None):
        val_style = list(self.val_data.keys())[dataloader_idx]

        seqs, attn_mask, pad_mask = batch
        seqs: torch.Tensor

        # Shift tokens, labels and mask
        shift_idx = seqs[..., :-1].contiguous()
        shift_labels = seqs[..., 1:].contiguous()

        logits = self.model(input_ids=shift_idx)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), shift_labels.long().view(-1)
        )

        loglike = self.model_loglikelihood(seqs)

        pred = logits.argmax(-1)
        acc = (pred == shift_labels).float().mean()
        if dataloader_idx == 0:
            self.log("seen_tokens", float(self.seen_tokens), add_dataloader_idx=False)
        self.log(f"val/{val_style}_acc", acc, add_dataloader_idx=False)
        self.log(f"val/{val_style}_loglike", loglike.mean(), add_dataloader_idx=False)
        self.log(
            f"val/{val_style}_ce_loss", loss, prog_bar=True, add_dataloader_idx=False
        )

        if val_style == "active":
            pp_dict = self.evaluate_pp(
                seqs[:64],
                self.full_data.cfg.context_length[1],
                assumed_envs=self.latent_indices,
            )
            self.log(
                "val/active_kl", pp_dict["BackwardKL"].mean(), add_dataloader_idx=False
            )

        return loss


######################################
############## UTILS #################
######################################


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

        def forward(self, input_ids, only_last_logits=False, attn_mask=None):
            return self.model(input_ids, only_last_logits, attn_mask)

    return HFWrapper(HFWrapperConfig())


# Wrapper
class DynamicIntervenableModel(pv.IntervenableModel):
    def __init__(self, config, model, t_slice, **kwargs):
        super().__init__(config, model, **kwargs)
        self.t_slice = slice(*t_slice)

    def __call__(self, input_ids, only_last_logits=False, attn_mask=None):
        assert only_last_logits == False
        assert attn_mask == None
        bs, seqlen = input_ids.shape
        logits = torch.zeros(
            size=(bs, seqlen, self.model.config.vocab_size), device=input_ids.device
        )

        for i in range(1, seqlen + 1):
            idx = list(range(i))[self.t_slice]

            logits[:, i - 1] = self.forward(
                base=BatchEncoding(
                    {"input_ids": input_ids[:, :i], "only_last_logits": True}
                ),
                unit_locations={"sources->base": (None, [[idx]])},
                return_dict=True,
            )["intervened_outputs"].squeeze()[:, : self.model.config.vocab_size]

        return logits


class CustomLoreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, orthogonal_map="matrix_exp"
        )
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        self.learned_source.weight = torch.nn.Parameter(self.rotate_layer.weight.T)

        init.constant_(
            self.learned_source.bias,
            0,
        )

        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True
        ).to(self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[
            :, :overload_w_width
        ] = overload_w
        assert (
            torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True
        )  # we must match!

        return


@jax.jit
def nll(probs, seq):
    ll = probs[jnp.arange(len(probs)), seq]
    return -jnp.log(ll)
