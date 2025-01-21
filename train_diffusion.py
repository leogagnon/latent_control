import datetime
import os
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import MISSING, DictConfig, OmegaConf

from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
from lightning_modules.metalearn import (FineTuningTask, MetaLearningTask,
                                         MetaLearningConfig, TuneConfig)
from lightning_modules.diffusion_prior import DiffusionPriorTaskConfig, DiffusionPriorTask


@dataclass
class DiffusionTrainConfig:
    seed: int
    log_dir: str
    max_steps: int
    val_check_interval: int
    logger: dict
    task: Optional[DiffusionPriorTaskConfig] = None
    accelerator: Optional[str] = MISSING
    sweep_id: Optional[str] = None
    model_checkpoint: Optional[dict] = None
    early_stopping: Optional[dict] = None


cs = ConfigStore.instance()
cs.store(name="train_config", node=DiffusionTrainConfig)
OmegaConf.register_new_resolver("eval", eval)


def main(cfg: DiffusionTrainConfig):

    # Deal with warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="Trying to infer the `batch_size` from an ambiguous collection.")
    warnings.filterwarnings("ignore", message="The `srun` command is available on your system but is not used. ")
    warnings.filterwarnings("ignore", message="Because the driver is older than the PTX compiler version, XLA is disabling parallel compilation, which may slow down compilation.")
    warnings.filterwarnings("ignore", message="The number of training batches ")
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Meta stuff
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.seed)

    # Parse config
    if cfg.logger:
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        logger = False

    callbacks = []
    if cfg.model_checkpoint:
        cfg.model_checkpoint.dirpath = os.path.join(
            cfg.log_dir, "checkpoints", logger.experiment.path.split("/")[-1]
        )
        callbacks.append(hydra.utils.instantiate(cfg.model_checkpoint))

    if OmegaConf.is_missing(cfg, "accelerator"):
        cfg.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.to_object(cfg)

    task = MetaLearningTask(cfg.task)
    
    # Give the whole TrainConfig to wandb
    if cfg.logger:
        logger.experiment.config.update(OmegaConf.to_container(OmegaConf.structured(cfg)))

    trainer = L.Trainer(
        logger=logger,
        max_steps=cfg.max_steps,
        accelerator=cfg.accelerator,
        enable_checkpointing=True if cfg.model_checkpoint else False,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=None
    )
    # Do a full validation step before training
    trainer.validate(model=task)
    trainer.fit(model=task)

if __name__ == "__main__":
    hydra_wrapper = hydra.main(version_base=None, config_name="train", config_path="configs_diffusion/")
    hydra_wrapper(main)()
