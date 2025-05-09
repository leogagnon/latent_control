import os
from dataclasses import dataclass
from typing import *

import torch
import torch.nn as nn
from torch2jax import j2t, t2j
from torch.utils.data import Dataset
from transformers.activations import ACT2FN

from tasks.metalearn import MetaLearningTask
from data.hmm import MetaHMM

@dataclass
class LatentDiffusionDatasetConfig:
    pretrained_id: str
    token_cond: bool
    context_length: Optional[Tuple[int]] = None

class LatentDiffusionDataset(Dataset, nn.Module):
    """Dataset used to train a diffusion model (encoder)"""

    def __init__(
        self,
        cfg: Optional[LatentDiffusionDatasetConfig] = None,
        **kwargs
    ) -> None:
        super().__init__()

        if cfg == None:
            cfg = LatentDiffusionDatasetConfig(**kwargs)

        self.task = MetaLearningTask.load_from_checkpoint(
            os.path.join(
                os.environ["LATENT_CONTROL_CKPT_DIR"],
                cfg.pretrained_id,
                "last.ckpt",
            ),
            strict=False,
        )
        self.task : MetaLearningTask
    
        self.task.requires_grad_(False)

        self.cfg = cfg

    @property
    def metahmm(self) -> MetaHMM:
        return self.task.data
    
    def decode(self, seqs: torch.Tensor, mask: torch.Tensor, latent: torch.Tensor):
        raise NotImplementedError

    @property
    def latent_shape(self):
        return None

    @property
    def cond_dim(self):
        if self.cfg.token_cond:
            # This is a bit hacky, assuming that the decoder's cfg has a <n_embd> attribute
            return self.task.model.encoder.hidden_dim
        else:
            return None

    def __len__(self):
        return len(self.task.data)

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
            j2t(self.task.data.index_to_latent)[indices].to(torch.long).cuda()
        )

        out_dict = {
            "raw_latent": raw_latent,
            "envs": indices,
            "cond_input_ids": None,
            "cond_ignore_mask": None,
            "cond_tokens": None,
            "latent": None,
            "cond_states": None,
        }

        # Possibly add a sequence from that HMM
        if self.cfg.context_length != None:
            hmm_sample = self.task.data.__getitems__(
                indices, length=self.cfg.context_length
            )
            cond_ignore_mask = hmm_sample.get(
                "ignore_mask",
                torch.zeros_like(hmm_sample["input_ids"], dtype=torch.bool).cuda(),
            )
            out_dict["cond_states"] = hmm_sample["states"]
            out_dict["cond_input_ids"] = hmm_sample["input_ids"]
            out_dict["cond_ignore_mask"] = cond_ignore_mask

        # Possibly embed that sequence with a pretrained embedding
        if self.cfg.token_cond:
            out_dict["cond_tokens"] = self.task.model.encoder(
                out_dict["cond_input_ids"], return_embeddings=True
            ).detach()

        return out_dict

class ContextDiffusionDatasetConfig(LatentDiffusionDatasetConfig):
    pass

class ContextDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: Optional[LatentDiffusionDatasetConfig] = None,
        **kwargs
    ) -> None:
        if cfg == None:
            cfg = LatentDiffusionDatasetConfig(**kwargs)
        super().__init__(cfg)
        assert "ContextEncoder" in str(self.task.model.encoder.__class__)

    @property
    def latent_shape(self):
        return [
            self.task.model.encoder.enc_len,
            self.task.model.encoder.out_dim,
        ]
    
    def decode(self, seqs: torch.Tensor, mask: torch.Tensor, latent: torch.Tensor):
        "p(x_t | x_<t, theta)"
        out = self.task.model.decoder(input_ids=seqs, context_enc=latent)
        out = out[torch.arange(len(out)).to(device=out.device), mask.sum(1)-1]
        
        return out

    def __getitems__(self, indices) -> dict:
        out_dict = super().__getitems__(indices)

        out_dict["latent"] = self.task.model.encoder(
            dataset=self.task.data,
            states=out_dict["cond_states"],
            true_envs=torch.IntTensor(indices).to(
                device=out_dict["cond_states"].device
            ),
        )

        return out_dict
    



@dataclass
class HiddenDiffusionDatasetConfig(LatentDiffusionDatasetConfig):
    suffix_size: Optional[Tuple[int]] = None
    diffuse_logits: bool = False


class HiddenDiffusionDataset(LatentDiffusionDataset):
    def __init__(
        self,
        cfg: Optional[HiddenDiffusionDatasetConfig] = None,
        **kwargs
    ) -> None:
        if cfg == None:
            cfg = HiddenDiffusionDatasetConfig(**kwargs)
        super().__init__(cfg)

        assert (
            self.cfg.context_length[0] == self.cfg.context_length[0]
        ), "The context length should be constant. <suffix_size> is what determines the effective context length in this setting."
        assert (
            self.task.model.decoder == None
        ), "HiddenDiffusionDataset requires an implicit model"
        self.cfg: HiddenDiffusionDatasetConfig

    @property
    def latent_shape(self):
        return [
            1,
            (
                self.task.model.encoder.out_dim
                if self.cfg.diffuse_logits
                else self.task.model.encoder.hidden_dim
            ),
        ]
    
    def decode(self, seqs: torch.Tensor, mask: torch.Tensor, latent: torch.Tensor):
        "p(x_t | x_<t, theta)"
        if self.cfg.diffuse_logits:
            out = latent
        else:
            out = self.task.model.encoder.out_proj(latent)
        out = out.squeeze(1)
        return out

    def __getitems__(self, indices, suffix_size=None):
        out_dict = super().__getitems__(indices)

        # Target latent is last
        out_dict["latent"] = self.task.model.encoder(
            out_dict["cond_input_ids"], return_embeddings=not self.cfg.diffuse_logits
        )[:, [-1]]

        if suffix_size == None:
            if self.cfg.suffix_size == None:
                return out_dict
            else:
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
        
        out_dict["cond_ignore_mask"] = (
            torch.arange(seqs.shape[1], device=seqs.device).tile(len(seqs), 1)
            >= lens[:, None]
        )

        if self.cfg.token_cond:
            # Re-run the pre-trained embedding for the suffix
            out_dict["cond_tokens"] = self.task.model.encoder(
                out_dict["cond_input_ids"], return_embeddings=True
            ).detach()

        return out_dict
