import os
import pickle
import traceback
import numpy as np
import scipy
import scipy.special
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import *
from dataclasses import dataclass, field
import torch
from models.gpt import GPT, GPTConfig
from data.hmm import (
    CompositionalHMMDataset,
    CompositionalHMMDatasetConfig,
    SubsetIntervened,
)
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
from pyreft.interventions import LowRankRotateLayer, NoreftIntervention
from transformers.activations import ACT2FN
from torch.nn import init
from torch.utils.data import StackDataset, TensorDataset
from torchmetrics.aggregation import MeanMetric


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
    constraints: List[List[int]]
    prefix_size: Optional[Tuple[int]] = None
    precompute_pp: Optional[bool] = False
    method_config: Optional[dict] = field(
        default_factory=dict
    )  # LoraConfig or ReftConfig or Full ({})
    batch_size: Optional[int] = None
    context_length: Optional[Tuple[int]] = None
    seed: Optional[int] = 42


@dataclass
class ReftConfig:
    reft_cls: str  # loreft, direft,
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
            oracle_pp.append(data.posterior_predictive(assumed_envs, X))
        oracle_pp = jnp.stack(oracle_pp)

        f = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            oracle_pp[..., : data.cfg.n_obs], model_pp[..., : data.cfg.n_obs]
        ).sum(-1)
        b = jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
            model_pp[..., : data.cfg.n_obs], oracle_pp[..., : data.cfg.n_obs]
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
            self.val_data, batch_size=self.cfg.batch_size, collate_fn=lambda x: x
        )

    def training_step(self, batch, batch_idx=None):

        seqs, states, attn_mask, pad_mask = batch

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

        seqs, states, attn_mask, pad_mask = batch
        seqs: torch.Tensor

        # Shift tokens, labels and mask
        shift_idx = seqs[..., :-1].contiguous()
        shift_labels = seqs[..., 1:].contiguous()
        if attn_mask is not None:
            attn_mask = attn_mask[..., :-1, :-1].contiguous()

        if pad_mask is not None:
            shift_labels[pad_mask[..., 1:]] = self.full_data.PAD_ID

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

        self.tune_cfg = cfg

        L.seed_everything(self.tune_cfg.seed)

        if self.tune_cfg.context_length is not None:
            self.full_data.cfg.context_length = self.tune_cfg.context_length

        if self.tune_cfg.batch_size is not None:
            self.cfg.batch_size = self.tune_cfg.batch_size

        # Set up the fine-tuning adapters (if not full fine-tuning)
        if cfg.method_config != {}:
            if isinstance(cfg.method_config, dict):
                method_config = hydra.utils.instantiate(cfg.method_config)
            else:
                method_config = cfg.method_config

            # Wrap the model with LoRA adapters
            if method_config.__class__.__name__ == "LoraConfig":
                self.model = get_peft_model(
                    model=make_hf_wrapper(self.model),
                    peft_config=method_config,
                )
            # Wrap the model with ReFT adapters
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
                    "transformer.h"
                    if isinstance(self.model, GPT)
                    else "backbone.layers"
                )

                reft_cls = None
                if method_config.reft_cls == "loreft":
                    reft_cls = CustomLoreftIntervention
                elif method_config.reft_cls == "direft":
                    reft_cls = CustomDireftIntervention
                elif method_config.reft_cls == "noreft":
                    reft_cls = NoreftIntervention
                elif method_config.reft_cls == "consreft":
                    reft_cls = CustomConsreftIntervention

                repr_configs = [
                    pv.RepresentationConfig(
                        component=f"{component_prefix}[{l}].{method_config.component}",
                        intervention=reft_cls(
                            low_rank_dimension=method_config.low_rank_dimension,
                            embed_dim=embed_dim,
                            dtype=torch.float32,
                            add_bias=True,
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

        # Compute when the constaint is active
        constraint_is_active = np.ones(len(self.full_data), dtype=np.bool)
        if self.tune_cfg.constraints != []:
            for c in self.tune_cfg.constraints:
                constraint_is_active = np.logical_and(
                    constraint_is_active,
                    self.full_data.index_to_latent[:, c[0]] == c[1],
                )
        active_set = set(constraint_is_active.nonzero()[0].tolist())
        inactive_set = set(np.logical_not(constraint_is_active).nonzero()[0].tolist())
        self.latent_indices = np.array(list(active_set))

        # Make the training dataset
        train_envs = list(set(self.train_data.indices) & active_set)
        if self.tune_cfg.prefix_size is None:
            self.train_data = Subset(self.full_data, train_envs)
        else:
            if self.tune_cfg.prefix_size[0] == self.tune_cfg.prefix_size[1]:
                intv_idx = self.tune_cfg.prefix_size[1]
            else:
                intv_idx = self.tune_cfg.prefix_size

            self.train_data = SubsetIntervened(
                self.full_data,
                prefix_indices=self.compute_prefix_envs(train_envs),
                suffix_indices=train_envs,
                intv_idx=intv_idx,
            )

        # Make the validation datasets
        val_set = set(self.val_data.indices)
        val_active_envs = list(active_set & val_set)
        val_active_envs = val_active_envs * math.ceil(
            len(self.val_data) / len(val_active_envs)
        )
        val_active_envs = np.array(val_active_envs[: len(self.val_data)])

        if self.tune_cfg.prefix_size is None:
            val_active = Subset(
                self.full_data,
                indices=val_active_envs,
            )
        else:
            val_active = SubsetIntervened(
                self.full_data,
                prefix_indices=self.compute_prefix_envs(val_active_envs),
                suffix_indices=val_active_envs,
                intv_idx=intv_idx,
            )

        # Possibly pre-compute bayes-optimal distribution for validation. Only when prefix_size is None.
        if self.tune_cfg.precompute_pp:
            # Sample an epoch from the dataset
            val_active_seqs = []
            val_active_states = []
            val_active_pads = []
            for batch_idx in torch.split(torch.arange(len(val_active)), 256):

                seqs, states, _, pad_masks = val_active.__getitems__(batch_idx) 
                
                val_active_seqs.append(seqs)
                val_active_states.append(states)
                val_active_pads.append(pad_masks)

            val_active_seqs = torch.concatenate(val_active_seqs, 0)
            val_active_states = torch.concatenate(val_active_states, 0)
            
            if None in val_active_pads:
                val_active = TensorDataset(
                    val_active_seqs, val_active_states, oracle_pp_ft, val_active_pads
                )
            else: 
                # This means we are dealing with mid sequence intervention
                val_active_pads = torch.concatenate(val_active_pads, 0)

                # Pre-compute
                intv_idx = val_active_pads.long().argmin(1)
                initial_states = val_active_states[
                    torch.arange(len(val_active_states)), intv_idx - 1
                ]

                print("Precomputing fine-tuning oracle...")
                oracle_pp_ft = torch.full(
                    (
                        val_active_seqs.shape[0],
                        val_active_seqs.shape[1],
                        self.full_data.cfg.n_obs,
                    ),
                    fill_value=-1.0 # For debugging purposes
                )
                for j in tqdm(range(len(val_active_seqs))):
                    seq = val_active_seqs[j]
                    oracle = j2t(
                        self.full_data.posterior_predictive(
                            jnp.array(self.latent_indices),
                            jnp.array(seq[intv_idx[j] :].tolist()),    
                            initial_states[j].item()
                        )
                    )
                    oracle_pp_ft[j, -len(oracle) :] = oracle
                print("Done")
                val_active = TensorDataset(
                    val_active_seqs, val_active_states, oracle_pp_ft, val_active_pads
                )

        # Make the inactive validation dataset
        if self.tune_cfg.constraints != []:
            val_inactive = list(inactive_set & val_set)
            val_inactive = val_inactive * math.ceil(
                len(self.val_data) / len(val_inactive)
            )
            val_inactive = val_inactive[: len(self.val_data)]
            val_inactive = Subset(self.full_data, val_inactive)

            self.val_data = {
                "active": val_active,
                "inactive": val_inactive,
            }
        else:
            self.val_data = {
                "active": val_active,
            }

        # Reinit the token count
        self.seen_tokens = 0

    def compute_prefix_envs(self, suffix_envs):
        # Compute prefix envs latents
        prefix_latents = np.array(
            self.full_data.index_to_latent[jnp.array(suffix_envs)]
        )
        # Randomly change the value of the latents in the constraints
        for constraint in self.tune_cfg.constraints:
            # Latent associated with constraint
            latent_id = constraint[0]
            # New values for this latent (can be the same)
            new_values = self.full_data.generator.integers(
                low=0,
                high=self.full_data.index_to_latent[:, latent_id].max().item() + 1,
                size=len(prefix_latents),
            )
            prefix_latents[:, latent_id] = new_values
        # Compute envs ID from latents
        prefix_envs = []
        for latents in prefix_latents:
            prefix_envs.append(
                (self.full_data.index_to_latent == latents).all(-1).argmax().item()
            )
        prefix_envs = torch.IntTensor(prefix_envs)

        return prefix_envs

    def on_save_checkpoint(self, checkpoint) -> None:
        #checkpoint["seen_tokens"] = self.seen_tokens
        #checkpoint["dataset"] = self.full_data
        #checkpoint["train_latents"] = self.train_data.indices
        pass

    def val_dataloader(self):
        if self.tune_cfg.constraints != []:
            return [
                DataLoader(
                    self.val_data["active"],
                    batch_size=self.cfg.batch_size,
                    collate_fn=None if self.tune_cfg.precompute_pp else (lambda x: x),
                ),
                DataLoader(
                    self.val_data["inactive"],
                    batch_size=self.cfg.batch_size,
                    collate_fn=lambda x: x,
                ),
            ]
        else:
            return [
                DataLoader(
                    self.val_data["active"],
                    batch_size=self.cfg.batch_size,
                    collate_fn=None if self.tune_cfg.precompute_pp else (lambda x: x),
                )
            ]

    def validation_step(self, batch, batch_idx=None, dataloader_idx=0):
        val_style = list(self.val_data.keys())[dataloader_idx]

        if (val_style == "active") & self.tune_cfg.precompute_pp:
            seqs, states, oracle_pp_ft, pad_mask = batch
        else:
            seqs, states, attn_mask, pad_mask = batch
        seqs: torch.Tensor

        # Shift tokens, labels and mask
        shift_idx = seqs[..., :-1].contiguous()
        shift_labels = seqs[..., 1:].contiguous()

        if pad_mask is not None:
            shift_labels[pad_mask[..., 1:]] = self.full_data.PAD_ID

        logits = self.model(input_ids=shift_idx)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), shift_labels.long().view(-1)
        )

        # loglike = self.model_loglikelihood(seqs)

        pred = logits.argmax(-1)
        acc = (
            (pred == shift_labels)[shift_labels != self.full_data.PAD_ID].float().mean()
        )
        if dataloader_idx == 0:
            self.log("seen_tokens", float(self.seen_tokens), add_dataloader_idx=False)
        self.log(f"val/{val_style}_acc", acc, add_dataloader_idx=False)
        # self.log(f"val/{val_style}_loglike", loglike.mean(), add_dataloader_idx=False)
        self.log(
            f"val/{val_style}_ce_loss", loss, prog_bar=True, add_dataloader_idx=False
        )

        if val_style == "active":
            if self.tune_cfg.precompute_pp:

                if pad_mask is not None:
                    probs = torch.softmax(logits, dim=-1)[
                        ..., : self.full_data.cfg.n_obs
                    ]
                    b_kl = jax.vmap(jax.scipy.special.rel_entr, (0, 0))(
                        t2j(probs[~pad_mask[:, 1:]]),
                        t2j(oracle_pp_ft[:, :-1][~pad_mask[:, 1:]]),
                    ).sum(-1)
                    b_kl = j2t(b_kl).cpu()

            else:
                b_kl = self.evaluate_pp(
                    seqs[:32],
                    self.full_data.cfg.context_length[1],
                    assumed_envs=self.latent_indices,
                )["BackwardKL"].mean()

            self.log("val/active_kl", b_kl.mean(), add_dataloader_idx=False, on_epoch=True)

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
            )["intervened_outputs"].squeeze(1)[:, : self.model.config.vocab_size]

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


class CustomDireftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    DiReFT(h) = h + R^T(Wh + b)
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
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(
                self.rotate_layer.weight.dtype
            ),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.to(base.dtype))


class CustomConsreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True
        )
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer, orthogonal_map="matrix_exp"
        )
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True
        )

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)
