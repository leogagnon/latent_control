import os
from dataclasses import dataclass
from typing import Any, List, Optional

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import WandbLogger
from omegaconf import MISSING, DictConfig, OmegaConf
from data.hmm import CompositionalHMMDataset, CompositionalHMMDatasetConfig
from task import MetaLearningTask, TaskConfig
import warnings
import sys
import traceback

@dataclass
class ExperimentConfig:
    seed: int
    log_dir: str
    max_steps: int
    val_check_interval: int
    logger: dict
    task : TaskConfig
    accelerator: Optional[str] = MISSING
    model_checkpoint: Optional[dict] = None

cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

@hydra.main(version_base=None, config_name="train", config_path="configs/")
def main(cfg: ExperimentConfig):

    # Deal with warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    torch.set_float32_matmul_precision('medium')

    L.seed_everything(cfg.seed)

    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False

    if cfg.model_checkpoint:
        cfg.model_checkpoint.dirpath = os.path.join(cfg.log_dir, 'checkpoints', logger.experiment.path.split('/')[-1])
        model_checkpoint = hydra.utils.instantiate(cfg.model_checkpoint)

    if OmegaConf.is_missing(cfg, "accelerator"):
        cfg.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.to_object(cfg)
    task = MetaLearningTask(cfg.task)

    # Name the checkpoint folder the wandb experiment ID
    
    trainer = L.Trainer(
        logger=logger,
        max_steps=cfg.max_steps,
        accelerator=cfg.accelerator,
        enable_checkpointing= True if cfg.model_checkpoint else False,
        callbacks=[model_checkpoint],
        val_check_interval=cfg.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=None
    )
    trainer.fit(model=task)
    
   
if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()