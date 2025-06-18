import sys

sys.path.append("/home/mila/l/leo.gagnon/latent_control")


from tasks.diffusion import DSMDiffusion
import torch
import matplotlib.pyplot as plt
from data.diffusion import LatentDiffusionDataset
from data.hmm import MetaHMM
from torch2jax import j2t, t2j
import jax.numpy as jnp
import jax
from jax.scipy.special import rel_entr
from einops import repeat, rearrange
from models.encoder import KnownEncoder
from models.decoder import TransformerDecoder
from tqdm import tqdm
from tasks.metalearn import MetaLearningTask
import torch.nn as nn
import pandas as pd
import seaborn as sns
import einx
import wandb
import pandas as pd
import seaborn as sns


runs = wandb.Api().runs(
    path="guillaume-lajoie/latent_control",
    filters={
        "$or": [
            {"config.sweep_id": "workshop-diffusion_12025-05-12-14-30-35"},
        ]
    },
)
runs = [run.id for run in runs]

N_SEQS = 50

full_df = pd.DataFrame()

for run_id in runs:
    for n_samples in [1, 5, 50]:

        task = DSMDiffusion.load_from_checkpoint(
            f"/network/scratch/l/leo.gagnon/latent_control_log/checkpoints/{run_id}/last.ckpt",
            strict=False,
        )

        # Generate some short sequences
        dataset = task.dataset.metahmm
        hmms = torch.randperm(len(dataset))[:N_SEQS]
        batch = dataset.__getitems__(hmms, length=50)

        # Compute the implicit posterior predictive
        implicit_task = MetaLearningTask.load_from_checkpoint(
            f"/network/scratch/l/leo.gagnon/latent_control_log/checkpoints/{task.dataset.task.model.encoder.cfg.pretrained_id}/last.ckpt",
            strict=False,
        ).cuda()
        implicit_pred = (
            torch.nn.functional.softmax(implicit_task.model(batch["input_ids"]), dim=-1)
            .cpu()
            .detach()
        )
        del implicit_task

        # Compute the monte-carlo estimates
        mc_out = task.evaluate_mc_estimate(
            cond_input_ids=batch["input_ids"],
            n_samples=n_samples,
            max_seqs=N_SEQS,
            full_metrics=True,
            implicit_pred=implicit_pred,
        )

        df_0 = pd.DataFrame(mc_out["jensen_div"])
        df_0["Model"] = "MC-estimate"
        df_0["Size"] = task.dataset.task.model.encoder.backbone.cfg.n_embd
        df_0["Seed"] = task.dataset.task.data.cfg.seed
        df_0["Prefix length"] = task.dataset.task.model.encoder.cfg.context_length
        df_0["Samples"] = n_samples

        df_1 = pd.DataFrame(mc_out["jensen_div_implicit"])
        df_1["Model"] = "Implicit"
        df_1["Size"] = task.dataset.task.model.encoder.backbone.cfg.n_embd
        df_1["Seed"] = task.dataset.task.data.cfg.seed

        task_df = pd.concat([df_0, df_1], ignore_index=True).melt(
            id_vars=["Size", "Seed", "Prefix length", "Model", "Samples"],
            var_name="Context length",
            value_name="JensenDiv",
        )

        full_df = pd.concat([full_df, task_df], ignore_index=True)

full_df.to_csv("data_diffusion_good1.csv")
