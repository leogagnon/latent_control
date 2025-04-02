import math
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings(
    "ignore", message="Trying to infer the `batch_size` from an ambiguous collection."
)
warnings.filterwarnings(
    "ignore", message="The `srun` command is available on your system but is not used. "
)
warnings.filterwarnings(
    "ignore",
    message="Because the driver is older than the PTX compiler version, XLA is disabling parallel compilation, which may slow down compilation.",
)
warnings.filterwarnings("ignore", message="The number of training batches ")
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
from dataclasses import dataclass, fields
from typing import Any, List, Optional

import hydra
import lightning as L
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode

from tasks.dsm_diffusion import DSMDiffusion, DSMDiffusionConfig
from tasks.gfn_diffusion import GFNDiffusion, GFNDiffusionConfig
from tasks.metalearn import MetaLearningConfig, MetaLearningTask
from tasks.direct_post import DirectPosterior, DirectPosteriorConfig


@dataclass
class TaskConfig:
    dsm: Optional[DSMDiffusionConfig] = None
    gfn: Optional[GFNDiffusionConfig] = None
    direct: Optional[DirectPosteriorConfig] = None
    metalearn: Optional[MetaLearningConfig] = None

    def __post_init__(self):
        attributes = [attr.name for attr in fields(self)]
        assert (
            sum([getattr(self, attr) != None for attr in attributes]) == 1
        ), "Only one task can be given at a time "


@dataclass
class TrainConfig:
    task: TaskConfig
    seed: int
    log_dir: str
    max_steps: int
    val_check_interval: int
    logger: dict
    accelerator: Optional[str] = None
    sweep_id: Optional[str] = None
    model_checkpoint: Optional[dict] = None
    early_stopping: Optional[dict] = None
    gradient_clip_val: float = 1.0


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
OmegaConf.register_new_resolver("eval", eval)


def init_task(cfg: TrainConfig):
    if cfg.task.dsm != None:
        task = DSMDiffusion(cfg.task.dsm)
        cfg.task.dsm = task.cfg
        return task
    elif cfg.task.gfn != None:
        task = GFNDiffusion(cfg.task.gfn)
        cfg.task.gfn = task.cfg
        return task
    elif cfg.task.metalearn != None:
        task = MetaLearningTask(cfg.task.metalearn)
        cfg.task.metalearn = task.cfg
        return task
    elif cfg.task.direct != None:
        task = DirectPosterior(cfg.task.direct)
        cfg.task.direct = task.cfg
        return task
    else:
        assert False, "Config not associated a lightning module"


def main(cfg: TrainConfig):

    # Meta stuff
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.seed)

    # Parse config
    if cfg.logger:
        # Add task to logger
        tags = [k for k in cfg.task.keys() if cfg.task[k] != None]
        # Add user to logger
        if 'USER' in os.environ:
            tags += [os.environ['USER']]
        cfg.logger.tags = tags
        logger = hydra.utils.instantiate(cfg.logger)
    else:
        logger = False

    callbacks = []

    # Setup checkpoint (with wandb ID as <dirpath>)
    if cfg.model_checkpoint:
        cfg.model_checkpoint.dirpath = os.path.join(
            cfg.log_dir, "checkpoints", logger.experiment.path.split("/")[-1]
        )
        callbacks.append(hydra.utils.instantiate(cfg.model_checkpoint))

    # Set device
    if cfg.accelerator == None:
        cfg.accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Init config object
    cfg = OmegaConf.to_container(
            cfg=cfg,
            resolve=True,
            throw_on_missing=False,
            enum_to_str=False,
            structured_config_mode=SCMode.INSTANTIATE,
        )

    ########################################################################
    ################ Task specific config pre-processing ###################
    ########################################################################
    if cfg.task.metalearn != None:
        if cfg.task.metalearn.model.encoder != None:
            if "KnownEncoder" in cfg.task.metalearn.model.encoder["_target_"]:
                if cfg.task.metalearn.model.encoder['latents_shape'] == None:
                    # Add latent shape to KnownEncoder from dataset config
                    cfg.task.metalearn.model.encoder['latents_shape'] = (
                        [
                            cfg.task.metalearn.data.base_cycles,
                            cfg.task.metalearn.data.base_directions,
                            cfg.task.metalearn.data.base_speeds,
                        ]
                        + [cfg.task.metalearn.data.group_per_family]
                        * cfg.task.metalearn.data.cycle_families
                        + [
                            cfg.task.metalearn.data.family_directions,
                            cfg.task.metalearn.data.family_speeds,
                        ]
                        + [cfg.task.metalearn.data.emission_group_size]
                        * cfg.task.metalearn.data.emission_groups
                        + [cfg.task.metalearn.data.emission_shifts]
                    )

        if cfg.task.metalearn.data.start_at_n != None:
            # Adjust batch size so that there is the same number of tokens in every batch
            ratio = cfg.task.metalearn.data.context_length[1] / (
                cfg.task.metalearn.data.context_length[1]
                - cfg.task.metalearn.data.start_at_n
            )
            new_bs = math.floor(cfg.task.metalearn.batch_size * ratio)
            cfg.task.metalearn.batch_size = new_bs

    ########################################################################
    ################                End                  ###################
    ########################################################################

    # Instantiate the lightning module (task)
    task = init_task(cfg)

    # 
    
    

    # Give the whole TrainConfig to wandb
    if cfg.logger:
        logger.experiment.config.update(
            OmegaConf.to_container(OmegaConf.structured(cfg))
        )

    # Instantiate the trainer
    trainer = L.Trainer(
        logger=logger,
        max_steps=cfg.max_steps,
        accelerator=cfg.accelerator,
        enable_checkpointing=True if cfg.model_checkpoint else False,
        callbacks=callbacks,
        val_check_interval=cfg.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.gradient_clip_val,
        num_sanity_val_steps=0,
    )
    # Do a full validation step before training (instead of a sanity_val_check)
    # try:
    trainer.validate(model=task)
    # except:
    #    pass
    trainer.fit(model=task)


if __name__ == "__main__":
    hydra_wrapper = hydra.main(
        version_base=None, config_name="train", config_path="configs/"
    )
    hydra_wrapper(main)()
