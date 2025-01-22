import dataclasses
import math
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import *

import hydra
import jax
import jax.numpy as jnp
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
#import pyvene as pv
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
#from peft import get_peft_model
#from pyreft.interventions import LowRankRotateLayer, NoreftIntervention
from pyvene import (DistributedRepresentationIntervention,
                    SourcelessIntervention, TrainableIntervention)
from torch2jax import j2t, t2j
from torch.nn import init
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN

from data.hmm import PrecomputedDataset, SubsetIntervened
from lightning_modules.metalearn import MetaLearningTask
from models.decoder import TransformerDecoder


@dataclass
class FineTuningConfig:
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

class FineTuningTask(MetaLearningTask):
    def __init__(self, cfg: FineTuningConfig) -> None:
        super().__init__(cfg.pretrained_id)

        assert self.model.encoder is None, "Fine-tuning for now doesnt't support"

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
                if isinstance(self.model.decoder, TransformerDecoder)
                else self.model.config.d_model
            )
            component_prefix = (
                "transformer.h" if isinstance(self.model, TransformerDecoder) else "backbone.layers"
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
                assert False, "Fine-tuning for now doesn't support variable length sequences"
                attn_mask = None

            if "ignore_mask" in batch.keys():
                shift_labels[batch["ignore_mask"][..., 1:]] = self.full_data.PAD_ID

            logits = self.model(input_ids=shift_idx)

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
