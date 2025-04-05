import dataclasses
import math
import random
from abc import ABC
from dataclasses import dataclass
from typing import *

import jax
import jax.numpy as jnp
import lightning as L
import pyvene as pv
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce, repeat
from jax.scipy.special import rel_entr
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import kl_divergence
from tqdm import tqdm
from transformers import BatchEncoding, PretrainedConfig
from transformers.activations import ACT2FN

from models.encoder import KnownEncoder, KnownEncoderConfig
from models.decoder import DecoderModel, GRUDecoder
from tasks.metalearn import MetaLearningTask
from data.hmm import CompositionalHMMDataset
from tasks.dsm_diffusion import DSMDiffusion
import os
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class LatentDiffusionDatasetConfig:
    context_length: Optional[Tuple[int]] = None
    pretrained_embedding: bool = False
    pretrained_embedding_id: Optional[str] = None


class LatentDiffusionDataset(Dataset, nn.Module):
    """Dataset used to train a diffusion model (encoder)"""

    def __init__(
        self,
        cfg: LatentDiffusionDatasetConfig,
        base_task: MetaLearningTask,
    ) -> None:
        super().__init__()

        self.base_task = base_task

        if cfg.pretrained_embedding:
            # Set up the pretrained embedding from <pretrained_embedding_id>
            assert cfg.pretrained_embedding_id != None
            task_ = MetaLearningTask.load_from_checkpoint(
                os.path.join(
                    os.environ["LATENT_CONTROL_CKPT_DIR"],
                    cfg.pretrained_embedding_id,
                    "last.ckpt",
                ),
                strict=False,
            )
            assert task_.full_data.cfg == base_task.full_data.cfg
            self.pretrained_embedding = task_.model.decoder
            del task_

            # Move to GPU and freeze
            self.pretrained_embedding.cuda()
            for param in self.pretrained_embedding.parameters():
                param.requires_grad = False

        self.cfg = cfg

    def evaluate(self, *args, **kwargs) -> dict:
        return None

    @property
    def latent_shape(self):
        return None

    @property
    def cond_dim(self):
        if self.cfg.pretrained_embedding:
            # This is a bit hacky, assuming that the decoder's cfg has a <n_embd> attribute
            return self.pretrained_embedding.cfg.n_embd
        else:
            return None

    def __len__(self):
        return len(self.base_task.full_data)

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
            j2t(self.base_task.full_data.index_to_latent)[indices].to(torch.long).cuda()
        )

        out_dict = {
            "raw_latent": raw_latent,
            "cond_input_ids": None,
            "cond_ignore_mask": None,
            "cond_tokens": None,
            "latent": None
        }

        # Possibly add a sequence from that HMM
        if self.cfg.context_length != None:
            hmm_sample = self.base_task.full_data.__getitems__(
                indices, length=self.cfg.context_length
            )
            cond_ignore_mask = hmm_sample.get(
                "ignore_mask",
                torch.zeros_like(hmm_sample["input_ids"], dtype=torch.bool).cuda(),
            )
            out_dict["cond_input_ids"] = hmm_sample["input_ids"]
            out_dict["cond_ignore_mask"] = cond_ignore_mask

        # Possibly embed that sequence with a pretrained embedding
        if self.cfg.pretrained_embedding:
            out_dict["cond_tokens"] = self.pretrained_embedding(
                out_dict["cond_input_ids"], return_embeddings=True
            ).detach()

        return out_dict


@dataclass
class KnownEncoderDiffusionDatasetConfig(LatentDiffusionDatasetConfig):
    pretrained_encoder_id: Optional[str] = None
    known_n_embd: Optional[int] = None
    sequential: Optional[bool] = None


class KnownEncoderDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: KnownEncoderDiffusionDatasetConfig,
        base_task: MetaLearningTask,
    ) -> None:
        super().__init__(cfg, base_task)
        self.cfg: KnownEncoderDiffusionDatasetConfig

        if cfg.pretrained_encoder_id != None:
            task_ = MetaLearningTask.load_from_checkpoint(
                os.path.join(
                    os.environ["LATENT_CONTROL_CKPT_DIR"],
                    cfg.pretrained_encoder_id,
                    "last.ckpt",
                ),
                strict=False,
            )
            assert "KnownEncoder" in str(task_.model.encoder.__class__)
            assert (self.cfg.known_n_embd == None) & (
                self.cfg.sequential == None
            ), "Cannot give <known_n_embd> or <sequential> if you give <pretrained_encoder_id>"
            self.known_encoder = task_.model.encoder
            self.cfg.known_n_embd = self.known_encoder.cfg.n_embd
            self.cfg.sequential = self.known_encoder.cfg.sequential
            del task_
        elif cfg.known_n_embd != None:
            self.known_encoder = KnownEncoder(
                KnownEncoderConfig(
                    n_embd=cfg.known_n_embd,
                    latents_shape=base_task.full_data.latent_shape,
                    sequential=cfg.sequential,
                )
            )
        else:
            assert False, "Either give <knonw_n_embd> or <pretrained_encoder_id>"

        # Just to make sure the dataset and the encoder have consistent config
        self.known_encoder.cuda()
        for param in self.known_encoder.parameters():
            param.requires_grad = False

    @property
    def latent_shape(self):
        return [
            (
                len(self.known_encoder.cfg.latents_shape)
                if self.known_encoder.cfg.sequential
                else 1
            ),
            self.known_encoder.cfg.n_embd,
        ]

    def __getitems__(self, indices) -> dict:
        out_dict = super().__getitems__(indices)

        env_latents = self.known_encoder(true_latents=out_dict["raw_latent"])
        out_dict["latent"] = env_latents

        return out_dict

    def evaluate(self, diffusion: DSMDiffusion, batch_size: int = 250):
        assert (
            self.known_encoder.cfg.sequential == False
        ), "This evaluation assumes a single latent, but could easily be modified"

        if diffusion.model.cfg.self_condition == True:
            return None

        out_dict = self.__getitems__(torch.randperm(len(self))[128:])
        cond_mask = ~out_dict["cond_ignore_mask"]
        latent = out_dict["latent"]

        # If training on constant length sequences, evaluate deterministically
        if self.cfg.context_length[0] == self.cfg.context_length[1]:

            z_t = diffusion.sample(
                batch_size,
                cond=out_dict["cond_tokens"],
                cond_input_ids=out_dict["cond_input_ids"],
                cond_mask=cond_mask,
            )
            if diffusion.cfg.normalize_latent:
                z_t = diffusion.unnormalize_latent(z_t)

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
            cond_mask = repeat(cond_mask, "l -> b l", b=batch_size).to(
                device=out_dict["cond_ignore_mask"].device
            )

            # Sample latents
            z_t = diffusion.sample(
                batch_size,
                cond_input_ids=(
                    repeat(out_dict["cond_input_ids"][0], "l -> b l", b=batch_size)
                    if out_dict["cond_input_ids"] != None
                    else None
                ),
                cond=(
                    repeat(out_dict["cond_tokens"][0], "l d -> b l d", b=batch_size)
                    if out_dict["cond_tokens"] != None
                    else None
                ),
                cond_mask=cond_mask,
            )
            if diffusion.cfg.normalize_latent:
                z_t = diffusion.unnormalize_latent(z_t)

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
                        self.base_task.full_data.index_to_latent
                        == t2j(decoded_task_latent[i])
                    )
                    .all(-1)
                    .argmax()
                    for i in range(len(decoded_task_latent))
                ]
            )

            # Compuate empirical distribution
            empirical_dist = jnp.bincount(
                decoded_task_id, minlength=len(self.base_task.full_data)
            )
            empirical_dist = empirical_dist / empirical_dist.sum()

            # Compute oracle distribution
            oracle_dist = self.base_task.full_data.bayesian_oracle(
                jnp.arange(len(self.base_task.full_data)),
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

            return {"val/f_kl": f_kl.item(), "val/b_kl": b_kl.item()}


@dataclass
class GRUDiffusionDatasetConfig(LatentDiffusionDatasetConfig):
    suffix_size: Optional[Tuple[int]] = None
    cond_hidden: bool = False


class GRUDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: GRUDiffusionDatasetConfig,
        base_task: MetaLearningTask,
    ) -> None:
        super().__init__(cfg, base_task)
        assert "GRU" in str(base_task.model.decoder.__class__)
        assert (
            self.cfg.context_length[0] == self.cfg.context_length[0]
        ), "The context length should be constant. <suffix_size> is what determines the effective context length in this setting."
        self.cfg: GRUDiffusionDatasetConfig

    @property
    def latent_shape(self):
        return [
            self.base_task.model.decoder.cfg.n_layer,
            self.base_task.model.decoder.cfg.n_embd,
        ]
    
    @property
    def cond_dim(self):
        if self.cfg.cond_hidden:
            return self.base_task.model.decoder.cfg.n_embd
        elif self.cfg.pretrained_embedding:
            # This is a bit hacky, assuming that the decoder's cfg has a <n_embd> attribute
            return self.pretrained_embedding.cfg.n_embd
        else:
            return None

    def __getitems__(self, indices, suffix_size=None):
        out_dict = super().__getitems__(indices)

        _, rnn_state = self.base_task.model.decoder(
            out_dict["cond_input_ids"], return_hiddens=True
        )
        out_dict["latent"] = rnn_state

        if suffix_size == None:
            if self.cfg.suffix_size == None:
                return out_dict
            
            suffix_size = self.cfg.suffix_size

        # Take a random suffix of the long sequence
        lens = torch.randint(
            suffix_size[0],
            suffix_size[1] + 1,
            size=(out_dict["cond_input_ids"].shape[0],),
            device=out_dict["cond_input_ids"].device,
        )
        seqs = [
            out_dict["cond_input_ids"][i, -lens[i] :]
            for i in range(out_dict["cond_input_ids"].shape[0])
        ]
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        out_dict["cond_input_ids"] = seqs
        

        if self.cfg.cond_hidden:
            packed_input = pack_padded_sequence(
                self.base_task.model.decoder.embedding(out_dict["cond_input_ids"]),
                lengths=lens.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            _, rnn_state_ = self.base_task.model.decoder.backbone(packed_input)
            rnn_state_ = rnn_state_.transpose(0,1)
            out_dict["cond_tokens"] = rnn_state_

            out_dict["cond_ignore_mask"] = torch.zeros(
                size=(rnn_state_.shape[0], rnn_state_.shape[1]), dtype=bool
            )
        else: 
            out_dict["cond_ignore_mask"] = (
                torch.arange(seqs.shape[1], device=seqs.device).tile(len(seqs), 1)
                >= lens[:, None]
            )

            if self.cfg.pretrained_embedding:
                tokens = [
                    out_dict["cond_tokens"][i, -lens[i] :]
                    for i in range(out_dict["cond_input_ids"].shape[0])
                ]
                tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
                out_dict["cond_tokens"] = tokens

        return out_dict
    

    def evaluate(self, diffusion: DSMDiffusion, batch_size: int = 50):
        
        # Sample some HMMs
        hmms = self.base_task.val_latents[torch.randperm(len(self.base_task.val_latents))[:batch_size]].long().cpu()
        # Sample some sequences from them
        raw_latent, cond_input_ids, cond_ignore_mask, cond_tokens, latent = self.__getitems__(hmms, suffix_size=[1,5]).values()

        # Sample multiple hidden states
        decoder = self.base_task.model.decoder.lm_head.cuda()
        n_samples = 10

        z_t = diffusion.sample(
            batch_size * n_samples,
            cond=repeat(cond_tokens, "b l d -> (b n) l d", n=n_samples) if cond_tokens != None else None,
            cond_input_ids=repeat(cond_input_ids, "b l -> (b n) l", n=n_samples).cuda(),
            cond_mask=repeat(torch.logical_not(cond_ignore_mask), "b l -> (b n) l", n=n_samples).cuda(),
            cls_free_guidance=1.0,
        )

        preds = decoder(z_t[:, -1])[:, :50]
        preds = rearrange(
            torch.nn.functional.softmax(preds, dim=-1),
            "(b n) c -> b n c",
            b=batch_size,
            n=n_samples,
        )
        pw_dist = torch.cdist(preds, preds).mean().item()

        return {"val/pw_dist": pw_dist}
