import os
from dataclasses import dataclass
from typing import Optional

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import WandbLogger
from omegaconf import MISSING, DictConfig, OmegaConf
from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
from task import MetaLearningTask, TaskConfig
import warnings


@dataclass
class ExperimentConfig:
    seed: int
    log_dir: str
    offline: bool
    epochs: int
    check_val_every_n_epoch: int
    accelerator: Optional[str]
    task : TaskConfig


cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

@hydra.main(version_base=None, config_name="train", config_path="configs/")
def main(cfg: ExperimentConfig):

    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    torch.set_float32_matmul_precision('medium')
    
    L.seed_everything(cfg.seed)
    task = MetaLearningTask(cfg.task)
    logger = WandbLogger(
        dir=cfg.log_dir,
        save_dir=cfg.log_dir,
        project="latent_control",
        offline=cfg.offline,
    )
    if OmegaConf.is_missing(cfg, "accelerator"):
        cfg.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = L.Trainer(
        logger=logger,
        max_epochs=cfg.epochs,
        enable_checkpointing=False,
        accelerator=cfg.accelerator,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch
    )
    trainer.fit(model=task)

if __name__ == "__main__":
    main()