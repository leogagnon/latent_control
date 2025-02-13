import os
from typing import Optional

import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf


class CustomCheckpointing:
    """
    Using this to load a checkpoint instead of the <load_from_checkpoint> function from Lightning.
    Because that default one is fucky and doesn't work with structured configs and I don't wanna try to understand how its 1000 lines of code work.
    """

    @classmethod
    def from_id(
        self: L.LightningModule,
        id: str,
        ckpt_id: Optional[int] = -1,
        step: Optional[int] = None,
    ):

        dir = os.path.join(os.environ["SCRATCH"], "latent_control_log/checkpoints/", id)
        ckpts = []
        for f in os.listdir(dir):
            if f != "last.ckpt":
                ckpts.append(f)

        # Load the last checkpoint
        ckpt = torch.load(os.path.join(dir, ckpts[-1]), weights_only=False)

        cfg = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.create(self.cfg_cls),
                OmegaConf.create(ckpt[self.CHECKPOINT_HYPER_PARAMS_KEY]),
            )
        )
        task = self(cfg)

        task.on_load_checkpoint(ckpt)
        task.wandb_dict.update(
            {
                "id": id,
                "ckpts_dir": dir,
                "default_ckpt": "last.ckpt",
                "ckpts_names": ckpts,
            }
        )

        assert sum([ckpt_id is None, step is None]) == 1

        if step is not None:
            steps = [
                int(filename.split(".ckpt")[0].split("step=")[1])
                for filename in task.wandb_dict["ckpts_names"]
            ]
            ckpt_id = np.abs((np.array(steps) / step) - 1).argmin()

        if ckpt_id == -1:
            ckpt_f = task.wandb_dict["default_ckpt"]
        else:
            ckpt_f = task.wandb_dict["ckpts_names"][ckpt_id]

        task.load_state_dict(
            torch.load(
                os.path.join(
                    task.wandb_dict["ckpts_dir"],
                    ckpt_f,
                ),
                weights_only=False,
            )["state_dict"]
        )
        print(f"Loaded checkpoing : {ckpt_f}")

        return task
