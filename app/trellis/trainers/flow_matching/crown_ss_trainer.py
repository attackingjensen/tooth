"""Geometry-only SS ControlNet trainer for jaw->crown generation (Stage-1).

Reuses :class:`ControlNetTrainingMixin` (loads the pretrained TRELLIS SS flow into both the
main trunk and the ControlNet branch, freezes the trunk, trains only the ControlNet, and
passes ``ori_voxel`` = jaw SS latent into the denoiser) but **replaces the DINOv2 image
cross-attention condition with a constant zero token**, so the model is conditioned purely
on the jaw geometry through the ControlNet branch.

This keeps the model architecture and pretrained weights 100% intact; only the zero-init
ControlNet branch is trained.
"""
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .flow_matching import ControlNetTrainingMixin, FlowMatchingCFGTrainer
from .mask_loss_utils import masked_mse_velocity, prepare_ss_keep_mask


class CrownGeoControlNetSSTrainer(ControlNetTrainingMixin, FlowMatchingCFGTrainer):
    """jaw SS latent -> crown SS latent via ControlNet, with a zero image condition."""

    def __init__(self, *args, cond_channels: int = 1024, num_cond_tokens: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_channels = cond_channels
        self.num_cond_tokens = num_cond_tokens

    def _zero_cond(self, batch_size: int, device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_cond_tokens, self.cond_channels, device=device, dtype=dtype)

    def training_losses(self, x_0, ori_voxel=None, **kwargs):
        assert ori_voxel is not None, "CrownGeoControlNetSSTrainer requires ori_voxel (jaw SS latent)"
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self._zero_cond(x_0.shape[0], x_0.device)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, ori_voxel=ori_voxel)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": float(mse_per_instance[time_bin == i].mean())}
        return terms, {}

    @torch.no_grad()
    def sample_crown(self, denoiser, jaw_ss: torch.Tensor, steps: int = 50, cfg_strength: float = 0.0):
        """Inference helper: jaw SS latent [B,8,16,16,16] -> crown SS latent."""
        from ...pipelines.samplers.flow_euler import FlowEulerCfgSampler
        sampler = FlowEulerCfgSampler(self.sigma_min)
        noise = torch.randn_like(jaw_ss)
        cond = self._zero_cond(jaw_ss.shape[0], jaw_ss.device)
        return sampler.sample(
            denoiser, noise=noise, cond=cond, neg_cond=cond,
            ori_voxel=jaw_ss, steps=steps, cfg_strength=cfg_strength, verbose=False,
        ).samples


class CrownGeoControlNetInpaintSSTrainer(CrownGeoControlNetSSTrainer):
    """Inpainting Stage-1 SS trainer (Path-1, formulation C).

    Target ``x_0`` is the FULL jaw∪crown SS latent; the jaw SS latent is still fed to the
    ControlNet (``ori_voxel``). The flow-matching velocity loss is split by ``keep_mask``
    (1=known jaw/background, 0=socket): the socket (generate) region carries the real
    synthesis signal, the jaw (keep) region anchors faithful reproduction of the context.
    Contact/margin coherence then emerges from grid continuity instead of a contact energy.

        loss = w_gen * MSE_v(socket) + w_keep * MSE_v(jaw)
    """

    def __init__(self, *args, w_gen: float = 1.0, w_keep: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_gen = w_gen
        self.w_keep = w_keep

    def training_losses(self, x_0, ori_voxel=None, keep_mask=None, **kwargs):
        assert ori_voxel is not None, "inpaint SS trainer requires ori_voxel (jaw SS latent)"
        assert keep_mask is not None, "inpaint SS trainer requires keep_mask"
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self._zero_cond(x_0.shape[0], x_0.device)

        pred = self.training_models['denoiser'](x_t, t * 1000, cond, ori_voxel=ori_voxel)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        keep = prepare_ss_keep_mask(keep_mask, pred)   # [B,1,D,H,W], 1=keep(jaw)
        gen = 1.0 - keep                               # 1=generate(socket)

        loss_gen = masked_mse_velocity(pred, target, gen)
        loss_keep = masked_mse_velocity(pred, target, keep)

        terms = edict()
        terms["mse_gen"] = loss_gen
        terms["mse_keep"] = loss_keep
        terms["mse"] = F.mse_loss(pred, target)        # full-volume, monitoring only
        terms["loss"] = self.w_gen * loss_gen + self.w_keep * loss_keep

        # per-time-bin monitoring on the (hard) generate region
        with torch.no_grad():
            g = gen.expand_as(pred)
            num = ((pred - target) ** 2 * g).flatten(1).sum(1)
            den = g.flatten(1).sum(1).clamp_min(1e-6)
            mse_gen_pi = (num / den).cpu().numpy()
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": float(mse_gen_pi[time_bin == i].mean())}
        return terms, {}
