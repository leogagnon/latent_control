from functools import singledispatchmethod
import math
import os
import random
from dataclasses import dataclass
from typing import *

import hydra
import lightning as L
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, reduce, repeat
from torch2jax import j2t, t2j
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchmetrics.functional import kl_divergence
from transformers.activations import ACT2FN
import data
from tasks.metalearn import MetaLearningTask
from models.encoder import DiffusionEncoder, DiffusionEncoderConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import jax.numpy as jnp
import jax
from jax.scipy.special import rel_entr
from models.diffusion import DiT, DiTConfig
from collections import namedtuple
from functools import partial
from tasks.utils import CustomCheckpointing
from tqdm import tqdm

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_v"]
)


@dataclass
class DSMDiffusionConfig:
    model: DiTConfig
    dataset: dict
    pretrained_id: str
    batch_size: int
    val_split: float
    lr: float

    loss: str = "l1"
    lr_scheduler: bool = False
    sampling_timesteps: int = 250
    train_schedule: str = "cosine"
    sampling_schedule: Optional[str] = None
    diffusion_objective: str = "pred_v"
    schedule_scale: float = 1.0
    sampler: str = "ddpm"
    normalize_latent: bool = False


class DSMDiffusion(L.LightningModule, CustomCheckpointing):
    """
    Trains a diffusion model with a Denoising Score Matching (DSM, https://arxiv.org/pdf/2101.09258) loss, i.e. maximum likelihood.
    """

    def __init__(self, cfg: DSMDiffusionConfig) -> None:
        super().__init__()

        # Load pre-trained meta-learning task (and freeze it)
        self.base_task = MetaLearningTask.from_id(cfg.pretrained_id)
        self.base_task: MetaLearningTask
        for param in self.base_task.parameters():
            param.requires_grad = False

        # Init diffusion model
        self.model = DiT(cfg.model)

        assert cfg.sampler in {
            "ddim",
            "ddpm",
            "dpmpp",
        }, "sampler must be one of ddim, ddpm, dpmpp"

        assert cfg.diffusion_objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be one of pred_noise, pred_x0, pred_v"

        if cfg.train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif cfg.train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif cfg.train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif cfg.train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid noise schedule {cfg.train_schedule}")

        self.train_schedule = partial(
            time_to_alpha, alpha_schedule=alpha_schedule, scale=cfg.schedule_scale
        )

        if cfg.sampling_schedule is None:
            sampling_alpha_schedule = None
        elif cfg.sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif cfg.sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif cfg.sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif cfg.sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f"invalid sampling schedule {cfg.sampling_schedule}")

        if exists(sampling_alpha_schedule):
            self.sampling_schedule = partial(
                time_to_alpha,
                alpha_schedule=sampling_alpha_schedule,
                scale=cfg.schedule_scale,
            )
        else:
            self.sampling_schedule = self.train_schedule

        if cfg.normalize_latent:
            # Buffers for latent mean and scale values
            self.register_buffer("latent_mean", torch.tensor(0).to(torch.float32))
            self.latent_mean: torch.FloatTensor
            self.register_buffer("latent_scale", torch.tensor(1).to(torch.float32))
            self.latent_scale: torch.FloatTensor

        self.cfg = cfg

        # Important for checkpoints
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(cfg)), logger=False
        )
        self.wandb_dict = dict({})

    def on_save_checkpoint(self, ckpt) -> None:
        ckpt["train_data"] = self.train_data
        ckpt["val_data"] = self.val_data

    def on_load_checkpoint(self, ckpt):
        self.train_data = ckpt["train_data"]
        self.val_data = ckpt["val_data"]

    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1 - alpha).sqrt() * noise) / alpha.sqrt().clamp(min=1e-8)

    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1 - alpha).sqrt() * v

        return x

    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1 - alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps

    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x * (1 - alpha).sqrt()

        return v

    def normalize_latent(self, x_start):
        eps = 1e-5

        return (x_start - self.latent_mean) / (self.latent_scale).clamp(min=eps)

    def unnormalize_latent(self, x_start):
        eps = 1e-5

        return x_start * (self.latent_scale.clamp(min=eps)) + self.latent_mean

    def get_sampling_timesteps(self, batch, *, device, invert=False):
        times = torch.linspace(1.0, 0.0, self.cfg.sampling_timesteps + 1, device=device)
        if invert:
            times = times.flip(dims=(0,))
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def diffusion_model_predictions(
        self,
        z_t,
        t,
        x_self_cond=None,
        class_id=None,
        cond=None,
        cond_input_ids=None,
        cond_mask=None,  # IGNORE WHEN MASK IS FALSE
        sampling=False,
        cls_free_guidance=1.0,
    ):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output = self.model(
            z_t,
            time_cond,
            x_self_cond,
            class_id=class_id,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
        )
        if cls_free_guidance != 1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(
                    class_id, fill_value=self.model.cfg.num_classes
                )
            else:
                unc_class_id = None
            unc_model_output = self.model(
                z_t,
                time_cond,
                x_self_cond,
                class_id=unc_class_id,
                cond=None,
                cond_input_ids=None,
                cond_mask=None,
            )
            model_output = model_output * cls_free_guidance + unc_model_output * (
                1 - cls_free_guidance
            )

        pred_v = None
        if self.cfg.diffusion_objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(
                z_t, t, pred_noise, sampling=sampling
            )
        elif self.cfg.diffusion_objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(
                z_t, t, x_start, sampling=sampling
            )
            pred_v = self.predict_v_from_start_and_eps(
                z_t, t, x_start, pred_noise, sampling=sampling
            )
        elif self.cfg.diffusion_objective == "pred_v":
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f"invalid objective {self.cfg.diffusion_objective}")

        return ModelPrediction(pred_noise, x_start, pred_v)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape,
        class_id,
        cond,
        cond_input_ids,
        cond_mask,
        cls_free_guidance=1.0,
        invert=False,
        z_t=None,
    ):
        print("DDIM sampling")
        batch, device = shape[0], next(self.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device, invert=invert)
        if invert:
            assert exists(z_t)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs,
            desc="sampling loop time step",
            total=self.cfg.sampling_timesteps,
        ):
            # get predicted x0

            model_output = self.diffusion_model_predictions(
                z_t,
                time,
                class_id=class_id,
                x_self_cond=x_start,
                cond=cond,
                cond_input_ids=cond_input_ids,
                cond_mask=cond_mask,
                sampling=True,
                cls_free_guidance=cls_free_guidance,
            )
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(
                partial(right_pad_dims_to, z_t), (alpha, alpha_next)
            )

            # # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise

            if (not invert) and time_next[0] <= 0:
                z_t = x_start
                continue
            if invert and time_next[0] >= 1:
                z_t = eps
                continue

            # get noise

            z_t = x_start * alpha_next.sqrt() + eps * (1 - alpha_next).sqrt()
        return z_t

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape,
        class_id,
        cond,
        cond_input_ids,
        cond_mask,
        cls_free_guidance=1.0,
        invert=False,
        z_t=None,
    ):
        batch, device = shape[0], next(self.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs,
            desc="sampling loop time step",
            total=self.cfg.sampling_timesteps,
        ):
            # get predicted x0

            model_output = self.diffusion_model_predictions(
                z_t,
                time,
                class_id=class_id,
                x_self_cond=x_start,
                cond=cond,
                cond_input_ids=cond_input_ids,
                cond_mask=cond_mask,
                sampling=True,
                cls_free_guidance=cls_free_guidance,
            )
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(
                partial(right_pad_dims_to, z_t), (alpha, alpha_next)
            )

            alpha_now = alpha / alpha_next

            # # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise

            if time_next[0] <= 0:
                z_t = x_start
                continue

            # get noise

            noise = torch.randn_like(z_t)

            z_t = (
                1
                / alpha_now.sqrt()
                * (z_t - (1 - alpha_now) / (1 - alpha).sqrt() * eps)
                + torch.sqrt(1 - alpha_now) * noise
            )
        return z_t

    @torch.no_grad()
    def dpmpp_sample(
        self,
        shape,
        class_id,
        cond,
        cond_input_ids,
        cond_mask,
        cls_free_guidance=1.0,
        invert=False,
        z_t=None,
    ):
        batch, device = shape[0], next(self.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        old_pred_x = []
        old_hs = []

        for time, time_next in tqdm(
            time_pairs,
            desc="sampling loop time step",
            total=self.cfg.sampling_timesteps,
        ):
            # get predicted x0

            model_output = self.diffusion_model_predictions(
                z_t,
                time,
                class_id=class_id,
                x_self_cond=x_start,
                cond=cond,
                cond_input_ids=cond_input_ids,
                cond_mask=cond_mask,
                sampling=True,
                cls_free_guidance=cls_free_guidance,
            )
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(
                partial(right_pad_dims_to, z_t), (alpha, alpha_next)
            )
            sigma, sigma_next = 1 - alpha, 1 - alpha_next

            alpha_now = alpha / alpha_next

            lambda_now = (log(alpha) - log(1 - alpha)) / 2
            lambda_next = (log(alpha_next) - log(1 - alpha_next)) / 2
            h = lambda_next - lambda_now

            # calculate x0 and noise
            if time_next[0] <= 0:
                z_t = x_start
                continue

            x_start = model_output.pred_x_start

            phi_1 = torch.expm1(-h)
            if len(old_pred_x) < 2:
                denoised_x = x_start
            else:
                h = lambda_next - lambda_now
                h_0 = old_hs[-1]
                r0 = h_0 / h
                gamma = -1 / (2 * r0)
                denoised_x = (1 - gamma) * x_start + gamma * old_pred_x[-1]

            z_t = (
                sigma_next.sqrt() / sigma.sqrt()
            ) * z_t - alpha_next.sqrt() * phi_1 * denoised_x
        return z_t

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        class_id=None,
        cond=None,
        cond_input_ids=None,
        cond_mask=None,
        cls_free_guidance=1.0,
    ):

        if self.cfg.sampler == "ddim":
            sample_fn = self.ddim_sample
        elif self.cfg.sampler == "ddpm":
            sample_fn = self.ddpm_sample
        elif self.cfg.sampler == "dpmpp":
            sample_fn = self.dpmpp_sample
        else:
            raise ValueError(f"invalid sampler {self.cfg.sampler}")
        return sample_fn(
            (batch_size,) + tuple(self.model.cfg.latent_shape),
            class_id,
            cond,
            cond_input_ids,
            cond_mask,
            cls_free_guidance,
        )

    @property
    def loss_fn(self):
        if self.cfg.loss == "l1":
            return F.l1_loss
        elif self.cfg.loss == "l2":
            return F.mse_loss
        elif self.cfg.loss == "smooth_l1":
            return F.smooth_l1_loss
        else:
            raise ValueError(f"invalid loss type {self.cfg.loss}")

    def setup(self, stage):
        with torch.no_grad():
            dataset_cfg = hydra.utils.instantiate(self.cfg.dataset)
            if "GRU" in self.cfg.dataset["_target_"]:
                dataset_cls = data.diffusion.GRUDiffusionDataset
            elif "Mamba" in self.cfg.dataset["_target_"]:
                dataset_cls = data.diffusion.MambaDiffusionDataset
            elif "KnownEncoder" in self.cfg.dataset["_target_"]:
                dataset_cls = data.diffusion.KnownEncoderDiffusionDataset
            else:
                assert False

            dataset = dataset_cls(dataset_cfg, self)
            self.full_data = dataset
            self.train_data, self.val_data = random_split(
                self.full_data, [1 - self.cfg.val_split, self.cfg.val_split]
            )

            # NOTE: training on all the HMMs for now to make sure this is not the limiting factor
            self.train_data = self.full_data

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        if self.cfg.lr_scheduler:
            # this is probably fake af but we put it for good luck
            scheduler = LinearWarmupCosineAnnealingLR(
                opt, warmup_epochs=300, max_epochs=100000
            )
            return [opt], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return opt

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=lambda x: x,
            shuffle=False,
        )

    def compute_diffusion_loss(
        self, latent, class_id=None, cond=None, cond_input_ids=None, cond_ignore_mask=None
    ):
        # NOTE: Important to flip the <ignore_mask> to a <don't_ignore_mask>
        cond_mask = None
        if cond_ignore_mask != None:
            cond_mask = torch.logical_not(cond_ignore_mask)

        bs, l, d = (*latent.shape,)
        device = latent.device

        times = torch.zeros((bs,), device=device).float().uniform_(0, 1.0)
        noise = torch.randn_like(latent)

        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(latent, alpha)

        z_t = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise

        if (
            self.model.cfg.class_conditional
            and self.model.cfg.class_unconditional_prob > 0
        ):
            assert exists(class_id)
            class_unconditional_mask = self.model.class_unconditional_bernoulli.sample(
                class_id.shape
            ).bool()
            class_id[class_unconditional_mask] = self.model.cfg.num_classes

        self_cond = None

        if self.model.cfg.self_condition and (
            random.random() < self.model.cfg.train_prob_self_cond
        ):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(
                    z_t,
                    times,
                    class_id=class_id,
                    cond=cond,
                    cond_mask=cond_mask,
                )
                self_cond = model_output.pred_x_start.detach()

        # predict and take gradient step

        predictions = self.diffusion_model_predictions(
            z_t,
            times,
            x_self_cond=self_cond,
            class_id=class_id,
            cond=cond,
            cond_input_ids=cond_input_ids,
            cond_mask=cond_mask,
        )

        if self.cfg.diffusion_objective == "pred_x0":
            target = latent
            pred = predictions.pred_x_start
        elif self.cfg.diffusion_objective == "pred_noise":
            target = noise
            pred = predictions.pred_noise
        elif self.cfg.diffusion_objective == "pred_v":
            target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
            assert exists(predictions.pred_v)
            pred = predictions.pred_v

        loss = self.loss_fn(pred, target, reduction="none")
        loss = rearrange(
            [reduce(loss[i], "l d -> 1", "mean") for i in range(latent.shape[0])],
            "b 1 -> b 1",
        )

        return loss.mean()

    def training_step(self, batch, batch_idx=None):

        latent = batch["latent"]
        if self.cfg.normalize_latent:
            latent_ = rearrange(latent, "b s d -> (b s) d")
            self.latent_mean = torch.mean(latent_, dim=0)
            self.latent_scale = torch.std(latent_ - self.latent_mean, unbiased=False)
            latent = self.normalize_latent(latent)

        loss = self.compute_diffusion_loss(
            latent,
            cond=batch["cond_tokens"],
            cond_input_ids=batch["cond_input_ids"],
            cond_ignore_mask=batch["cond_ignore_mask"],
        )
        self.log(
            "train/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        bs = batch["raw_latent"].shape[0]

        latent = batch["latent"]
        if self.cfg.normalize_latent:
            latent_ = rearrange(latent, "b s d -> (b s) d")
            self.latent_mean = torch.mean(latent_, dim=0)
            self.latent_scale = torch.std(latent_ - self.latent_mean, unbiased=False)
            latent = self.normalize_latent(latent)

        loss = self.compute_diffusion_loss(
            latent,
            cond=batch["cond_tokens"],
            cond_input_ids=batch["cond_input_ids"],
            cond_ignore_mask=batch["cond_ignore_mask"],
        )
        self.log(
            "val/loss",
            loss.detach().cpu().numpy().item(),
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=latent.shape[0],
        )

    # Possibly do more specific evalutions
    # Calls the <evaluate()> function of the LatentDiffusionDataset
    def on_validation_epoch_end(self) -> None:
        eval_dict = self.full_data.evaluate()
        if eval_dict != None:
            for k in eval_dict.keys():
                self.log(k, eval_dict[k], prog_bar=False, add_dataloader_idx=False)


#######################################
################ UTILS ################
#######################################


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def beta_linear_schedule(t, clip_min=1e-9):
    return torch.exp(-1e-4 - 10 * (t**2)).clamp(min=clip_min, max=1.0)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.0)


def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha


def alpha_to_shifted_log_snr(alpha, scale=1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2 * np.log(scale).item()


def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale=scale)
    return log_snr_to_alpha(shifted_log_snr)


def log(t, eps=1e-12):
    return torch.log(t.clamp(min=eps))


def exists(x):
    return x is not None
