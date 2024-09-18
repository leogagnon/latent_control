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
import traceback
from lightning.pytorch.callbacks import EarlyStopping


@dataclass
class TrainConfig:
    seed: int
    log_dir: str
    max_steps: int
    val_check_interval: int
    logger: dict
    task: TaskConfig
    max_tokens: Optional[int] = None
    accelerator: Optional[str] = MISSING
    sweep_id: Optional[str] = "none"
    model_checkpoint: Optional[dict] = None
    early_stopping: Optional[dict] = None


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)


@hydra.main(version_base=None, config_name="train", config_path="configs_train/")
def main(cfg: TrainConfig):

    # Deal with warnings
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    torch.set_float32_matmul_precision("medium")

    L.seed_everything(cfg.seed)

    if cfg.logger:
        logger = hydra.utils.instantiate(cfg.logger)
        logger.experiment.config.update({"sweep_id": cfg.sweep_id})
    else:
        logger = False

    callbacks = []
    if cfg.model_checkpoint:
        cfg.model_checkpoint.dirpath = os.path.join(
            cfg.log_dir, "checkpoints", logger.experiment.path.split("/")[-1]
        )
        callbacks.append(hydra.utils.instantiate(cfg.model_checkpoint))

    if cfg.max_tokens:
        callbacks.append(
            EarlyStopping(
                "seen_tokens",
                stopping_threshold=cfg.max_tokens,
                mode="max",
                check_on_train_epoch_end=True,
            )
        )

    if OmegaConf.is_missing(cfg, "accelerator"):
        cfg.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.to_object(cfg)
    task = MetaLearningTask(cfg.task)

    trainer = L.Trainer(
        logger=logger,
        max_steps=cfg.max_steps,
        accelerator=cfg.accelerator,
        enable_checkpointing=True if cfg.model_checkpoint else False,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=None,
    )
    trainer.fit(model=task)


if __name__ == "__main__":
    main()
