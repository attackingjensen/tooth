"""idea2 · Full-FT SS trainer with jaw point-cloud cross-attention condition (Stage-1).

Contrast with ``CrownGeoControlNetSSTrainer`` (the current crown route):
- That trainer feeds a CONSTANT ZERO cross-attn token and trains only a zero-init ControlNet
  branch (trunk frozen). The DINOv2-pretrained cross-attn never learns.
- This trainer REACTIVATES cross-attention: the condition is the jaw point cloud + FDI tooth
  id, encoded to tokens by the model's ``PointCondEncoder``, and the WHOLE trunk + encoder are
  fine-tuned (full FT). The image-cond SS pretrain is loaded as init (strict=False; the encoder
  keys are simply absent and stay at init).

CFG: per-sample condition dropout (``p_uncond``) is applied INSIDE the model forward via
``cond_drop`` (zeroing the cond tokens), so the encoder runs under DDP and its grads sync.
"""
from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingCFGTrainer


class CrownPointCrossSSTrainer(FlowMatchingCFGTrainer):
    """jaw point cloud (+FDI) -> crown SS latent, full fine-tuning of the TRELLIS SS trunk."""

    def __init__(self, *args, ss_pretrain: str = None, cond_channels: int = 1024, **kwargs):
        self.ss_pretrain_path = ss_pretrain
        super().__init__(*args, **kwargs)
        self.cond_channels = cond_channels

    def _pre_init_models(self, **kwargs):
        """Load the image-cond SS pretrain into the trunk as init (no freeze — full FT)."""
        super()._pre_init_models(**kwargs)
        if self.ss_pretrain_path is None:
            return
        from safetensors.torch import load_file
        pretrain = load_file(self.ss_pretrain_path)
        denoiser = self.models['denoiser']
        incompatible = denoiser.load_state_dict(pretrain, strict=False)
        missing = list(getattr(incompatible, 'missing_keys', []))
        unexpected = list(getattr(incompatible, 'unexpected_keys', []))
        n_total = sum(p.numel() for p in denoiser.parameters())
        n_train = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        if self.is_master:
            print(f"\n[PointCrossFT] loaded SS pretrain: {self.ss_pretrain_path}")
            print(f"  pretrain tensors={len(pretrain)}, missing(init)={len(missing)} "
                  f"(point_encoder.*), unexpected={len(unexpected)}")
            print(f"  Total params: {n_total:,}, Trainable (FULL FT): {n_train:,}")

    def training_losses(self, x_0, point_cloud=None, fdi=None, **kwargs):
        assert point_cloud is not None and fdi is not None, \
            "CrownPointCrossSSTrainer requires point_cloud and fdi"
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        # Per-sample classifier-free-guidance dropout (zero the cond tokens inside forward).
        cond_drop = None
        if self.p_uncond > 0:
            cond_drop = (torch.rand(x_0.shape[0], device=x_0.device) < self.p_uncond).float()

        pred = self.training_models['denoiser'](
            x_t, t * 1000, point_cloud=point_cloud, fdi=fdi, cond_drop=cond_drop,
        )
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
    def sample_crown(self, denoiser, point_cloud, fdi, steps: int = 50, cfg_strength: float = 3.0):
        """Inference helper: jaw point cloud [B,N,6] + fdi [B] -> crown SS latent [B,8,16,16,16]."""
        from ...pipelines.samplers.flow_euler import FlowEulerCfgSampler
        sampler = FlowEulerCfgSampler(self.sigma_min)
        ctx = denoiser.encode_cond(point_cloud, fdi)        # [B,(1+)N,C]
        neg = torch.zeros_like(ctx)                          # null cond = zero tokens
        noise = torch.randn(
            point_cloud.shape[0], denoiser.in_channels, *([denoiser.resolution] * 3),
            device=point_cloud.device,
        )
        return sampler.sample(
            denoiser, noise=noise, cond=ctx, neg_cond=neg,
            steps=steps, cfg_strength=cfg_strength, verbose=False,
        ).samples

    # The base flow-matching snapshot path samples with an image-cond signature; disable it
    # (i_sample is set huge in the config, but snapshot('init') still fires once at step 0).
    def run_snapshot(self, *args, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass

    def vis_cond(self, **kwargs):
        return {}
