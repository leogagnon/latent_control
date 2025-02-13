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

from tasks.metalearn import MetaLearningTask
from tasks.dsm_diffusion import DSMDiffusion
from models.encoder import KnownEncoder, KnownEncoderConfig
from transformers import PretrainedConfig, BatchEncoding
import dataclasses
import pyvene as pv
from einops import repeat
import jax.numpy as jnp
from jax.scipy.special import rel_entr
import jax

@dataclass
class LatentDiffusionDatasetConfig:
    context_length: Optional[Tuple[int]] = None
    pretrained_embedding: bool = False
    pretrained_embedding_id: Optional[str] = None


class LatentDiffusionDataset(Dataset):
    """Dataset used to train a diffusion model (encoder)"""

    def __init__(
        self,
        cfg: LatentDiffusionDatasetConfig,
        diffusion: DSMDiffusion,
    ) -> None:
        
        # Seting up the pretrained embedding
        if cfg.pretrained_embedding:
            assert cfg.context_length != None, 'Cannot use pretrained embedding if there is not sequence'
            if diffusion.model.cfg.cond_encoder_kwargs != None:
                assert (
                    diffusion.model.cfg.cond_encoder_kwargs.get("vocab_size", None) == None
                ), "If cond_encoder_kwargs.vocab size is set, cannot use cond_tokens"
            if cfg.pretrained_embedding_id == None:
                self.pretrained_embedding = diffusion.base_task.model.decoder
            else:
                self.pretrained_embedding = MetaLearningTask(
                    cfg.pretrained_embedding_id
                ).model.decoder
            self.pretrained_embedding.cuda()
            for param in self.pretrained_embedding.parameters():
                param.requires_grad = False

        self.cfg = cfg
        self.diffusion = diffusion

    def evaluate(self) -> dict:
        return None

    def __len__(self):
        return len(self.diffusion.base_task.full_data)

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
            j2t(self.diffusion.base_task.full_data.index_to_latent)[indices].to(torch.long).cuda()
        )

        out_dict = {
            "raw_latent": raw_latent,
            "cond_input_ids": None,
            "cond_ignore_mask": None,
            "cond_tokens": None,
            "latent": None,
        }

        # Possibly add a sequence from that HMM
        if self.cfg.context_length != None:
            hmm_sample = self.diffusion.base_task.full_data.__getitems__(
                indices, length=self.cfg.context_length
            )
            cond_ignore_mask = hmm_sample.get(
                "ignore_mask", torch.zeros_like(hmm_sample["input_ids"], dtype=torch.bool)
            )
            out_dict['cond_input_ids'] = hmm_sample["input_ids"]
            out_dict['cond_ignore_mask'] = cond_ignore_mask

        # Possibly embed that sequence with a pretrained embedding
        if self.cfg.pretrained_embedding:
            out_dict["cond_tokens"] = self.pretrained_embedding(
                out_dict["cond_input_ids"], return_embeddings=True
            )

        return out_dict


@dataclass
class KnownEncoderDiffusionDatasetConfig(LatentDiffusionDatasetConfig):
    new_encoder: bool = False


class KnownEncoderDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: KnownEncoderDiffusionDatasetConfig,
        diffusion: DSMDiffusion,
    ) -> None:
        super().__init__(cfg, diffusion)

        if cfg.new_encoder:
            self.known_encoder = KnownEncoder(
                KnownEncoderConfig(
                    n_embd=diffusion.model.cfg.latent_shape[1],
                    latents_shape=diffusion.base_task.full_data.latent_shape,
                )
            ).cuda()
        else:
            assert "KnownEncoder" in str(diffusion.base_task.model.decoder.__class__)
            self.known_encoder = self.diffusion.base_task.model.encoder

    def evaluate(self):
        if not self.diffusion.model.cfg.seq_conditional:
            return None
        
        out_dict = self.__getitems__(torch.randperm(len(self))[128:])
        cond = (
            out_dict["cond_input_ids"]
            if out_dict["cond_tokens"] is None
            else out_dict["cond_tokens"]
        )
        cond_mask = ~out_dict["cond_ignore_mask"]
        latent = out_dict["latent"]

        # If training on constant length sequences, evaluate deterministically
        if self.cfg.context_length[0] == self.cfg.context_length[1]:
            

            z_t = self.diffusion.sample(250, cond=cond, cond_mask=cond_mask)
            if self.diffusion.cfg.normalize_latent:
                z_t = self.diffusion.unnormalize_latent(z_t)

            decoded_task_latent = [
                torch.Tensor(
                    [
                        (latent_embds.weight @ sampled_latent.T).argmax()
                        for latent_embds in self.known_encoder.latent_embedding
                    ]
                )
                for sampled_latent in z_t
            ]
            decoded_task_latent = torch.stack(decoded_task_latent, 0)
            true_task_latent = [
                torch.Tensor(
                    [
                        (latent_embds.weight @ true_latent.T).argmax()
                        for latent_embds in self.known_encoder.latent_embedding
                    ]
                )
                for true_latent in latent
            ]
            true_task_latent = torch.stack(true_task_latent, 0)
            decoding_accuracy = (
                torch.sum(true_task_latent == decoded_task_latent, dim=-1)
                / true_task_latent.shape[-1]
            )
            decoding_accuracy = decoding_accuracy.mean(0)

            return {"val/decoding_acc": decoding_accuracy.detach().cpu().numpy().item()}

        else:
            # Sample a lot of different latents for one sequence to see the empirical posterior distribution
            cond_mask = torch.BoolTensor(
                [True] * 8 + [False] * (cond_mask.shape[1] - 8)
            )
            cond_mask = repeat(cond_mask, "l -> b l", b=250).to(device=cond.device)

            # Sample latents
            z_t = self.diffusion.sample(
                250,
                cond=repeat(cond[0], "l d -> b l d", b=250),
                cond_mask=cond_mask,
                cls_free_guidance=1.5,
            )
            if self.diffusion.cfg.normalize_latent:
                z_t = self.diffusion.unnormalize_latent(z_t)

            # Decode using known encoder
            decoded_task_latent = [
                torch.Tensor(
                    [
                        (latent_embds.weight @ sampled_latent.T).argmax()
                        for latent_embds in self.known_encoder.latent_embedding
                    ]
                )
                for sampled_latent in z_t
            ]
            decoded_task_latent = torch.stack(decoded_task_latent, 0)

            # Gather associated HMM ids
            decoded_task_id = jnp.stack(
                [
                    (
                        self.diffusion.base_task.full_data.index_to_latent
                        == t2j(decoded_task_latent[i])
                    )
                    .all(-1)
                    .argmax()
                    for i in range(len(decoded_task_latent))
                ]
            )

            # Compuate empirical distribution
            empirical_dist = jnp.bincount(
                decoded_task_id, minlength=len(self.diffusion.base_task.full_data)
            )
            empirical_dist = empirical_dist / empirical_dist.sum()

            # Compute oracle distribution
            oracle_dist = self.diffusion.base_task.full_data.bayesian_oracle(
                jnp.arange(len(self.diffusion.base_task.full_data)),
                t2j(out_dict["cond_input_ids"][0]),
            )["log_alpha_post"]
            oracle_dist = oracle_dist[((~cond_mask).int().argmax() + 1).item()]
            oracle_dist = jnp.exp(oracle_dist)
            empirical_dist = jax.device_put(empirical_dist, oracle_dist.device)

            # Forward KL(oracle, empirical) (with small epsilon)
            empirical_dist_ = empirical_dist + 1e-8
            empirical_dist_ = empirical_dist_ / empirical_dist_.sum()
            f_kl = rel_entr(oracle_dist, empirical_dist_).sum()
            
            #  Backward KL(empirical, oracle) (with small epsilon)
            oracle_ = oracle_dist + 1e-8
            oracle_ = oracle_ / oracle_.sum()
            b_kl = rel_entr(empirical_dist, oracle_).sum()
            
            return {'val/f_kl': f_kl.item(), 'val/b_kl': b_kl.item()}

    def __getitems__(self, indices) -> dict:
        out_dict = super().__getitems__(indices)

        env_latents = self.known_encoder(true_latents=out_dict["raw_latent"])
        out_dict["latent"] = env_latents

        return out_dict


@dataclass
class GRUDiffusionDatasetConfig(LatentDiffusionDatasetConfig):
    suffix_size: Tuple[int] = None


class GRUDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: GRUDiffusionDatasetConfig,
        diffusion: DSMDiffusion,
    ) -> None:
        super().__init__(cfg, diffusion)
        assert "GRU" in str(diffusion.base_task.model.decoder.__class__)
        assert (
            self.cfg.context_length[0] == self.cfg.context_length[0]
        ), "The context length should be constant. <suffix_size> is what determines the effective context length in this setting."

    def __getitems__(self, indices):
        out_dict = super().__getitems__(indices)

        _, rnn_state = self.diffusion.base_task.model.decoder(
            out_dict["cond_input_ids"], return_hiddens=True
        )
        out_dict["latent"] = rnn_state

        if self.cfg.suffix_size == None:
            return out_dict

        # Take a random suffix of the long sequence
        lens = torch.randint(
            self.cfg.suffix_size[0],
            self.cfg.suffix_size[1],
            size=(out_dict["cond_input_ids"].shape[0],),
        )
        seqs = [
            out_dict["cond_input_ids"][i, -lens[i] :]
            for i in range(out_dict["cond_input_ids"].shape[0])
        ]
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        out_dict["cond_input_ids"] = seqs

        if self.cfg.pretrained_embedding:
            tokens = [
                out_dict["cond_tokens"][i, -lens[i] :]
                for i in range(out_dict["cond_input_ids"].shape[0])
            ]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
            out_dict["cond_tokens"] = tokens

        ignore_mask = torch.arange(seqs.shape[1]).tile(len(seqs), 1) >= lens[:, None]
        out_dict["cond_ignore_mask"] = ignore_mask

        return out_dict


class MambaDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: LatentDiffusionDatasetConfig,
        diffusion: DSMDiffusion,
    ) -> None:
        super().__init__(cfg, diffusion)
        assert "Mamba" in str(diffusion.base_task.model.decoder.__class__)

        # Wrap the Mamba model with a CollectIntervention
        mamba_decoder = diffusion.base_task.model.decoder
        mamba_decoder.config.ssm_cfg = dict(mamba_decoder.config.ssm_cfg)
        mamba_decoder.config = PretrainedConfig.from_dict(
            dataclasses.asdict(mamba_decoder.config)
        )
        repr_configs = [
            pv.RepresentationConfig(
                component=f"backbone.layers[{i}].mixer.output",
                intervention=pv.CollectIntervention(
                    embed_dim=mamba_decoder.config.d_model, keep_last_dim=False
                ),
            )
            for i in range(mamba_decoder.config.n_layer)
        ]
        self.collect_model = pv.IntervenableModel(
            pv.IntervenableConfig(repr_configs), model=mamba_decoder
        ).cuda()

    def __getitems__(self, indices):
        out_dict = super().__getitems__(indices)

        activations = self.collect_model(
            base=BatchEncoding(
                {"input_ids": out_dict["cond_input_ids"], "only_last_logits": True}
            ),
            unit_locations={"sources->base": out_dict["cond_input_ids"].shape[-1] - 1},
            return_dict=True,
        )["collected_activations"]
        activations = torch.stack(list(activations.values())).transpose(0, 1).squeeze()
        out_dict["latent"] = activations

        return out_dict
