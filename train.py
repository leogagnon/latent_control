import argparse
import math
import subprocess
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

# Setup a trap for SIGCONT and SIGTERM
import signal
import threading
from dataclasses import dataclass, fields
from typing import Any, List, Optional

import hydra
import lightning as L
import torch
import wandb
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import WandbLogger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode

from data.hmm import MetaHMM
from tasks.direct_post import DirectPosterior, DirectPosteriorConfig
from tasks.dsm_diffusion import DSMDiffusion, DSMDiffusionConfig
from tasks.gfn_diffusion import GFNDiffusion, GFNDiffusionConfig
from tasks.metalearn import MetaLearningConfig, MetaLearningTask

PREEMPT_DIR = "/network/scratch/l/leo.gagnon/metahmm_log/preempted_runs/"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
os.environ["LATENT_CONTROL_CKPT_DIR"] = '/network/scratch/l/leo.gagnon/metahmm_log/checkpoints'

def start_preemption_monitor(trainer, wandb_id, interval=60):
    shutdown_event = threading.Event()

    def handle_preemption_signal(signum, frame):
        print(f"Received signal {signum}, (likely preemption)")
        shutdown_event.set()

    signal.signal(signal.SIGCONT, handle_preemption_signal)
    signal.signal(signal.SIGTERM, handle_preemption_signal)

    def monitor_and_requeue(*args, **kwargs):
        import time

        while not shutdown_event.is_set():
            time.sleep(30)

        if "SLURM_JOB_ID" not in os.environ.keys():
            print("No SLURM_JOB_ID found; can't requeue.")
            return None

        run_id = os.environ.get("SLURM_JOB_ID")

        if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
            run_id = run_id + "_" + os.environ.get("SLURM_ARRAY_TASK_ID")

        with open(os.path.join(PREEMPT_DIR, run_id), "w") as f:
            f.write(wandb_id)

        print("Saved wandb ID to preempt dir")

        print(f"Requeuing job manually using: scontrol requeue {run_id}")
        try:
            subprocess.run(["scontrol", "requeue", run_id], check=True)
            print("Job requeued successfully. Will resume from last checkpoint.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to requeue job: {e}")

        trainer.should_stop = True

    t = threading.Thread(target=monitor_and_requeue, daemon=True)
    t.start()


@dataclass
class TaskConfig:
    dsm: Optional[DSMDiffusionConfig] = None
    gfn: Optional[GFNDiffusionConfig] = None
    direct: Optional[DirectPosteriorConfig] = None
    metalearn: Optional[MetaLearningConfig] = None

    def __post_init__(self):
        #attributes = [attr.name for attr in fields(self)]
        #assert (
        #    sum([getattr(self, attr) != None for attr in attributes]) == 1
        #), "Only one task can be given at a time "
        pass


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

        # Retrieve the config file of the associated meta-learn run
        run = wandb.Api().run(
            f"guillaume-lajoie/metahmm/{cfg.task.dsm.dataset['pretrained_id']}"
        )
        cfg.task.metalearn = OmegaConf.merge(
            OmegaConf.structured(MetaLearningConfig), run.config['task']['metalearn']
        )

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


def main(cfg: TrainConfig, preempt_id: Optional[str] = None):

    # Check if the run is a preemption rerun
    is_preempt_rerun = False
    if preempt_id != None:
        is_preempt_rerun = True
        wandb_id = preempt_id
        api = wandb.Api()
        entity = "guillaume-lajoie"
        project = "metahmm"
        run = api.run(f"{entity}/{project}/{preempt_id}")
        cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), run.config)
    else:
        files = [
            f
            for f in os.listdir(PREEMPT_DIR)
            if os.path.isfile(os.path.join(PREEMPT_DIR, f))
        ]
        run_id = os.environ.get("SLURM_JOB_ID")
        if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
            run_id = run_id + "_" + os.environ.get("SLURM_ARRAY_TASK_ID")
        for file in files:
            if file == run_id:
                print("Preemption rerun detected!")
                is_preempt_rerun = True
                with open(os.path.join(PREEMPT_DIR, file), "r") as f:
                    wandb_id = f.read().strip()

                os.remove(os.path.join(PREEMPT_DIR, file))

                api = wandb.Api()

                entity = "guillaume-lajoie"
                project = "metahmm"

                run = api.run(f"{entity}/{project}/{wandb_id}")
                cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), run.config)

    # Meta stuff
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.seed)

    # Parse config
    if cfg.logger:
        if is_preempt_rerun:
            cfg.logger.id = wandb_id
        else:
            # Add task to logger
            tags = [k for k in cfg.task.keys() if cfg.task[k] != None]
            # Add user to logger
            if "USER" in os.environ:
                tags += [os.environ["USER"]]
            cfg.logger.tags = tags
            # Add preempted wandb ID if it exists

        logger = hydra.utils.instantiate(cfg.logger)

        if not is_preempt_rerun:
            wandb_id = logger.experiment.path.split("/")[-1]
    else:
        logger = False

    callbacks = []

    # Setup checkpoint (with wandb ID as <dirpath>)
    if cfg.model_checkpoint:
        cfg.model_checkpoint.dirpath = os.path.join(
            cfg.log_dir, "checkpoints", wandb_id
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
        latents_shape = (
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
            assert False, "Don't know if this still works"
            # Adjust batch size so that there is the same number of tokens in every batch
            ratio = cfg.task.metalearn.data.context_length[1] / (
                cfg.task.metalearn.data.context_length[1]
                - cfg.task.metalearn.data.start_at_n
            )
            new_bs = math.floor(cfg.task.metalearn.batch_size * ratio)
            cfg.task.metalearn.batch_size = new_bs

    try:
        if cfg.task.metalearn.model.encoder["latents_shape"] == None:
            cfg.task.metalearn.model.encoder["latents_shape"] = latents_shape
    except:
        pass

    try:
        if "KnownEncoder" in cfg.task.metalearn.model.encoder["backbone"]["_target_"]:
            cfg.task.metalearn.model.encoder["backbone"][
                "latents_shape"
            ] = latents_shape
    except:
        pass

    
    

    ########################################################################
    ################                End                  ###################
    ########################################################################

    # Instantiate the lightning module (task)
    task = init_task(cfg)

    # Give the whole TrainConfig to wandb
    if cfg.logger:
        logger.experiment.config.update(
            OmegaConf.to_container(OmegaConf.structured(cfg)), allow_val_change=True
        )


    # Instantiate the trainer
    trainer = L.Trainer(
        logger=logger,
        #max_steps=10000000000000 if is_preempt_rerun else cfg.max_steps,
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


    start_preemption_monitor(trainer, wandb_id)

    trainer.fit(
        model=task,
        ckpt_path=(
            os.path.join(cfg.model_checkpoint["dirpath"], "last.ckpt")
            if is_preempt_rerun
            else None
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_id")
    args, _ = parser.parse_known_args()

    if args.wandb_id == None:
        hydra_wrapper = hydra.main(
            version_base=None, config_name="train", config_path="configs/"
        )
        hydra_wrapper(main)()
    else:
        main(cfg=None, preempt_id=args.wandb_id)
