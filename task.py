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
import wandb
from models.gpt import GPT, GPTConfig
from data.hmm import (
    CompositionalHMMDataset,
    CompositionalHMMDatasetConfig,
    SubsetIntervened,
    PrecomputedDataset,
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
from pyreft import ReftModel
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
import matplotlib.pyplot as plt


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
    val_size: Optional[int] = None
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
                envs, samples, jr.split(jr.PRNGKey(seed), len(envs))
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
            model_pp = jnp.array(model_pp.tolist())[..., : data.cfg.n_obs]

        torch.cuda.empty_cache()

        # Gather the ground truth posterior predictive
        if assumed_envs is None:
            assumed_envs = jnp.arange(len(data))
        oracle_pp = []
        for X in Xs:
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

        # Shift tokens, labels and mask
        shift_idx = batch["input_ids"][..., :-1].contiguous()
        shift_labels = batch["input_ids"][..., 1:].contiguous()
        if "attention_mask" in batch.keys():
            attn_mask = batch["attention_mask"][..., :-1, :-1]
        else:
            attn_mask = None

        if "ignore_mask" in batch.keys():
            shift_labels[batch["ignore_mask"][..., 1:]] = self.full_data.PAD_ID

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

        self.setup_adapters()

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

        # Compute the intervention matrix (transforms the posterior over alpha)
        intervened_latents = np.array(self.full_data.index_to_latent)
        for c in self.tune_cfg.constraints:
            intervened_latents[:, c[0]] = c[1]
        intervention_map = np.zeros((len(self.full_data), len(self.full_data)))
        for i in range(len(self.full_data)):
            dest = (
                (self.full_data.index_to_latent == intervened_latents[i])
                .all(-1)
                .argmax()
                .item()
            )
            intervention_map[dest, i] = 1.0

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

        val_size = (
            len(self.val_data)
            if self.tune_cfg.val_size is None
            else self.tune_cfg.val_size
        )

        # Make the validation datasets
        val_set = set(self.val_data.indices)
        val_active_envs = list(active_set & val_set)
        val_active_envs = val_active_envs * math.ceil(val_size / len(val_active_envs))
        val_active_envs = np.array(val_active_envs[:val_size])

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

        if self.tune_cfg.precompute_pp:
            precomputed = val_active.__getitems__(torch.arange(len(val_active)))

            if "raw_seqs" in precomputed.keys():
                # This means we are dealing with mid sequence intervention

                # Retreive <intv_idx> with the padding mask (where the intervention starts, inclusively)
                intv_idx = precomputed["ignore_mask"].long().argmin(1)

                print("Precomputing fine-tuning oracle...")
                bayes_oracle = torch.full(
                    (
                        precomputed["input_ids"].shape[0],
                        precomputed["input_ids"].shape[1] + 1,
                        self.full_data.cfg.n_obs,
                    ),
                    fill_value=-1.0,  # For debugging purposes
                )
                bayes_oracle_raw = bayes_oracle.clone()

                for j in tqdm(range(len(precomputed["input_ids"]))):

                    # Run the unconstrained bayesian oracle for the prefix to get z_post and alpha_post
                    prefix_oracle = self.full_data.bayesian_oracle(
                        jnp.arange(len(self.full_data)),
                        jnp.array(precomputed["raw_seqs"][j].tolist()),
                    )

                    # Oracle on the raw sequence
                    bayes_oracle_raw[j] = j2t(prefix_oracle["post_pred"])

                    # Compute intervened alpha_prior
                    # NOTE: here we could alse write intv_idx[j] +1 (because of the oracle BOS) -1 (because we want the prior just BEFORE <intv_idx>)
                    alpha_prior = intervention_map @ jnp.exp(
                        prefix_oracle["log_alpha_post"][intv_idx[j].item()]
                    )
                    alpha_prior = alpha_prior[self.latent_indices]

                    # Run the fine-tuned bayesian oracle, starting with previous z_post, alpha_post
                    intv_oracle = self.full_data.bayesian_oracle(
                        jnp.array(self.latent_indices),
                        jnp.array(
                            precomputed["input_ids"][j]
                            .roll(-intv_idx[j].item())
                            .tolist()
                        ),
                        initial_messages=prefix_oracle["messages"][
                            intv_idx[j].item() - 1
                        ][self.latent_indices],
                        log_alpha_prior=jnp.log(alpha_prior),
                    )

                    # Oracle on the intervened sequence
                    bayes_oracle[j, : intv_idx[j]] = bayes_oracle_raw[
                        j, : intv_idx[j].item()
                    ]
                    bayes_oracle[j, intv_idx[j] :] = j2t(
                        intv_oracle["post_pred"][: -intv_idx[j].item()]
                    )
                    precomputed.update(
                        {
                            "bayes_oracle": bayes_oracle,
                            "bayes_oracle_raw": bayes_oracle_raw,
                        }
                    )

            val_active = PrecomputedDataset(precomputed)

        # Make the inactive validation dataset
        if self.tune_cfg.constraints != []:
            val_inactive = list(inactive_set & val_set)
            val_inactive = val_inactive * math.ceil(val_size / len(val_inactive))
            val_inactive = val_inactive[:val_size]
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

    def setup_adapters(self):
        if self.tune_cfg.method_config == {}:
            # This means we are doing full fine-tuning
            return

        # Instantiate method_config
        if isinstance(self.tune_cfg.method_config, dict):
            method_config = hydra.utils.instantiate(self.tune_cfg.method_config)
        else:
            method_config = self.tune_cfg.method_config

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
                "transformer.h" if isinstance(self.model, GPT) else "backbone.layers"
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

    def on_fit_start(self) -> None:
        if wandb.run is not None:
            wandb.log({"trainable_ratio": self.trainable_parameters(string=False)})

    def compute_prefix_envs(self, suffix_envs):
        """
        Compute the index of the environments where the constraint is flipped
        """
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
        # TODO
        # checkpoint["seen_tokens"] = self.seen_tokens
        # checkpoint["dataset"] = self.full_data
        # checkpoint["train_latents"] = self.train_data.indices
        pass

    def trainable_parameters(self, string=False):

        trainable_model_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        all_model_parameters = sum(p.numel() for p in self.model.model.parameters())
        if string:
            print(
                f"model params: {all_model_parameters:,d} || trainable%: {100 * trainable_model_parameters / all_model_parameters}"
            )
        else:
            return 100 * trainable_model_parameters / all_model_parameters

    def val_dataloader(self):
        if self.tune_cfg.constraints != []:
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
        else:
            return [
                DataLoader(
                    self.val_data["active"],
                    batch_size=self.cfg.batch_size,
                    collate_fn=lambda x: x,
                )
            ]

    def validation_step(self, batch, batch_idx=None, dataloader_idx=0):
        with torch.no_grad():
            val_style = list(self.val_data.keys())[dataloader_idx]

            shift_idx = batch["input_ids"][..., :-1].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()
            if "attention_mask" in batch.keys():
                attn_mask = batch["attention_mask"][..., :-1, :-1]
            else:
                attn_mask = None

            if "ignore_mask" in batch.keys():
                shift_labels[batch["ignore_mask"][..., 1:]] = self.full_data.PAD_ID

            logits = self.model(input_ids=shift_idx, attn_mask=attn_mask)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), shift_labels.long().view(-1)
            )

            pred = logits.argmax(-1)
            acc = (
                (pred == shift_labels)[shift_labels != self.full_data.PAD_ID]
                .float()
                .mean()
            )
            if dataloader_idx == 0:
                self.log(
                    "seen_tokens", float(self.seen_tokens), add_dataloader_idx=False
                )
            self.log(f"val/{val_style}_acc", acc, add_dataloader_idx=False)
            self.log(
                f"val/{val_style}_ce_loss",
                loss,
                prog_bar=True,
                add_dataloader_idx=False,
            )

            if val_style == "active":
                if self.tune_cfg.precompute_pp:
                    # Unintervened sequence
                    shift_idx_raw = batch["raw_seqs"][..., :-1].contiguous()
                    shift_labels_raw = batch["raw_seqs"][..., 1:].contiguous()
                    shift_labels_raw[batch["ignore_mask"][..., 1:]] = (
                        self.full_data.PAD_ID
                    )
                    if self.model.__class__.__name__ == "PeftModel":
                        with self.model.disable_adapter():
                            logits_raw = self.model(input_ids=shift_idx_raw)
                    else:
                        logits_raw = self.model.model(input_ids=shift_idx_raw)
                    probs_raw = torch.softmax(logits_raw, dim=-1)[
                        ..., : self.full_data.cfg.n_obs
                    ]
                    loss_raw = F.cross_entropy(
                        logits_raw.view(-1, logits_raw.size(-1)),
                        shift_labels_raw.long().view(-1),
                    )
                    self.log(
                        "val/loss_raw",
                        loss_raw,
                        add_dataloader_idx=False,
                        on_epoch=True,
                    )
                    intv_idx = batch["ignore_mask"][0].long().argmin()
                    loss_first_raw = F.cross_entropy(
                        logits_raw[0][~batch["ignore_mask"][0, 1:]][0],
                        shift_labels_raw.long()[0][~batch["ignore_mask"][0, 1:]][0],
                    )
                    loss_first = F.cross_entropy(
                        logits[0][~batch["ignore_mask"][0, 1:]][0],
                        shift_labels.long()[0][~batch["ignore_mask"][0, 1:]][0],
                    )
                    self.log(
                        "val/loss_first_raw",
                        loss_first_raw,
                        add_dataloader_idx=False,
                        on_epoch=True,
                    )
                    self.log(
                        "val/loss_first",
                        loss_first,
                        add_dataloader_idx=False,
                        on_epoch=True,
                    )
                    b_kl_raw = j2t(
                        jax.vmap(jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0))(
                            t2j(probs_raw), t2j(batch["bayes_oracle_raw"])[:, 1:-1]
                        ).sum(-1)
                    )
                    self.log(
                        "val/active_kl_raw",
                        b_kl_raw[~batch["ignore_mask"][:, 1:]].mean().cpu().item(),
                        add_dataloader_idx=False,
                        on_epoch=True,
                    )

                    # Intervened sequence
                    if batch["ignore_mask"] is not None:
                        probs = torch.softmax(logits, dim=-1)[
                            ..., : self.full_data.cfg.n_obs
                        ]
                        b_kl = j2t(
                            jax.vmap(
                                jax.vmap(jax.scipy.special.rel_entr, (0, 0)), (0, 0)
                            )(t2j(probs), t2j(batch["bayes_oracle"])[:, 1:-1]).sum(-1)
                        )
                        self.log(
                            "val/active_kl",
                            b_kl[~batch["ignore_mask"][:, 1:]].mean().cpu().item(),
                            add_dataloader_idx=False,
                            on_epoch=True,
                        )
                    if wandb.run is not None:
                        b_kl[0][batch["ignore_mask"][0, 1:]] = b_kl_raw[0][
                            batch["ignore_mask"][0, 1:]
                        ]
                        plt.plot(b_kl[0].cpu(), label="Intervened")
                        plt.plot(b_kl_raw[0].cpu(), label="Raw")
                        plt.ylabel("Backward KL")
                        wandb.log({"val/intervened_kl": plt})

                        intv_idx = batch["ignore_mask"][0].long().argmin()

                        plt.clf()
                        plt.plot(
                            batch["bayes_oracle"][0, intv_idx].cpu(), label="Oracle"
                        )
                        plt.plot(probs[0, intv_idx - 1].cpu(), label="Probs")
                        wandb.log({"val/probs": plt})

                        plt.clf()
                        plt.plot(
                            batch["bayes_oracle_raw"][0, intv_idx].cpu(),
                            label="Oracle Raw",
                        )
                        plt.plot(probs_raw[0, intv_idx - 1].cpu(), label="Probs Raw")
                        wandb.log({"val/probs_raw": plt})

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
