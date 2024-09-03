import os
import numpy as np
from torch.utils.data import DataLoader, Subset
import lightning as L
from typing import *
from dataclasses import dataclass
import torch
from models.gpt import GPT, GPTConfig
from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
from transformers import PreTrainedModel, PretrainedConfig
from peft import get_peft_config, get_peft_model, LoraConfig
import torch.nn as nn


@dataclass
class TaskConfig:
    data: CompositionalHMMDatasetConfig
    model: GPTConfig
    batch_size: int
    lr: float
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None


class MetaLearningTask(L.LightningModule):
    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = GPT(cfg.model)
        self.wandb_dict = dict({})

    @classmethod
    def from_wandb_id(cls: "MetaLearningTask", id: str):
        dir = os.path.join(os.environ["SCRATCH"], "latent_control_log/checkpoints/", id)
        task = MetaLearningTask.load_from_checkpoint(os.path.join(dir, "last.ckpt"))
        ckpts = []
        for f in os.listdir(dir):
            if f != "last.ckpt":
                ckpts.append(f)

        task.wandb_dict.update({"ckpts_dir": dir, "ckpts_names": ckpts})

        return task

    def set_to_checkpoint(self, ckpt_id: int):
        assert len(self.wandb_dict) > 0

        self.load_state_dict(
            torch.load(
                os.path.join(
                    self.wandb_dict["ckpts_dir"],
                    self.wandb_dict["ckpts_names"][ckpt_id],
                ),
                weights_only=False,
            )["state_dict"]
        )

    def setup(self, stage: str = None):
        """Setup the data"""

        self.full_data = CompositionalHMMDataset(self.cfg.data)
        train_latents = set(np.arange(len(self.full_data)))

        if "val_size" in self.cfg:
            val_size = self.cfg.val_size
        elif "val_ratio" in self.cfg:
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
        self.state_dict()

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["dataset"] = self.full_data
        checkpoint["train_latents"] = self.train_data.indices
        checkpoint["val_latents"] = self.val_data.indices

    def on_load_checkpoint(self, checkpoint) -> None:
        print("Loading dataset...", end="")
        self.full_data = checkpoint["dataset"]
        self.train_data = Subset(self.full_data, indices=checkpoint["train_latents"])
        self.val_data = Subset(self.full_data, indices=checkpoint["val_latents"])
        print("Done")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg.batch_size)

    def training_step(self, batch, batch_idx=None):

        shift_idx = batch[..., :-1].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        loss, logits = self.model(idx=shift_idx, targets=shift_labels)

        pred = logits.argmax(-1)
        acc = (pred == shift_labels).float().mean()

        self.log("train/acc", acc)
        self.log("train/ce_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx=None):

        shift_idx = batch[..., :-1].contiguous()
        shift_labels = batch[..., 1:].contiguous()

        loss, logits = self.model(idx=shift_idx, targets=shift_labels)

        pred = logits.argmax(-1)
        acc = (pred == shift_labels).float().mean()

        self.log("val/acc", acc)
        self.log("val/ce_loss", loss, prog_bar=True)

        return loss


def make_hf_wrapper(model: nn.Module):
    class HFWrapperConfig(PretrainedConfig):
        model_type = "HF_Wrapper"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class HFWrapper(PreTrainedModel):
        config_class = HFWrapperConfig

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.model = model

        def forward(self, idx, targets):
            return self.model(idx, targets)

    return HFWrapper(HFWrapperConfig())


@dataclass
class LoraFineTuningTaskConfig:
    checkpoint_id: str
    lora_cfg: LoraConfig
    constraint: Tuple[int, bool]
    train_size: Optional[int] = None
    val_size: Optional[int] = None


class LoraFineTuningTask(MetaLearningTask):
    def __init__(self, cfg: LoraFineTuningTaskConfig) -> None:

        # Get checkpoint
        task = MetaLearningTask.load_from_checkpoint(
            os.path.join(
                os.environ["SCRATCH"],
                "latent_control_log/checkpoints/",
                cfg.checkpoint_id,
                "last.ckpt",
            )
        )
        super().__init__(task.cfg)

        # Wrap the model
        self.model = get_peft_model(
            model=make_hf_wrapper(task.model), peft_config=cfg.lora_cfg
        )
        self.latent_id = cfg.latent_id

        self.cfg = cfg

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def setup(self, stage: str = None):

        id_is_active = self.full_data.index_to_latent[:, self.latent_id] == 0

        if self.cfg.train_size is None:
            train_set = set(self.train_data.indices)
        else:
            train_set = set(
                np.random.choice(
                    self.train_data.indices, self.cfg.train_size, replace=False
                )
            )

        if self.cfg.val_size is None:
            val_set = set(self.val_data.indices)
        else:
            val_set = set(
                np.random.choice(
                    self.val_data.indices, self.cfg.val_size, replace=False
                )
            )

        active_set = set(id_is_active.nonzero()[0])

        self.train_data = Subset(self.full_data, list(active_set & train_set))
        self.val_data = Subset(self.full_data, list(active_set & val_set))
