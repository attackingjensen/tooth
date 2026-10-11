"""Geometry-only SLat ControlNet trainer for jaw->crown generation (Stage-2).

Mirrors the Stage-1 ``CrownGeoControlNetSSTrainer`` idea but for the sparse SLat space:
reuses :class:`SparseControlNetTrainingMixin` (loads pretrained TRELLIS SLat flow into trunk
+ ControlNet, freezes trunk, trains only the zero-init ControlNet, injects ``ori_slat`` = jaw
SLat) and **replaces the image/text cross-attention condition with a constant zero token**.

The dataset (``CrownJawSLat``) yields ``cond`` = list of "" strings; we ignore the text and
feed ``torch.zeros(B, num_cond_tokens, cond_channels)`` so conditioning is purely the jaw
geometry through the ControlNet branch.
"""
from typing import *

import torch

from .sparse_flow_matching import SparseControlNetTrainingMixin, SparseFlowMatchingCFGTrainer


class CrownGeoControlNetSLatTrainer(SparseControlNetTrainingMixin, SparseFlowMatchingCFGTrainer):
    """jaw SLat -> crown SLat via ControlNet, with a zero cross-attn condition."""

    def __init__(self, *args, cond_channels: int = 1024, num_cond_tokens: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_channels = cond_channels
        self.num_cond_tokens = num_cond_tokens

    def _zero_cond(self, batch_size: int, device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_cond_tokens, self.cond_channels, device=device, dtype=dtype)

    def get_cond(self, cond, **kwargs):
        """Replace the text/image cond (a list of "") with a constant zero token batch."""
        B = len(cond) if isinstance(cond, (list, tuple)) else cond.shape[0]
        return self._zero_cond(B, self.device)

    def get_inference_cond(self, cond, **kwargs):
        B = len(cond) if isinstance(cond, (list, tuple)) else cond.shape[0]
        zero = self._zero_cond(B, self.device)
        return {"cond": zero, "neg_cond": zero, **kwargs}

    @torch.no_grad()
    def sample_crown_slat(self, denoiser, x_0, ori_slat, steps: int = 25, cfg_strength: float = 0.0):
        """Inference helper on a known coord set (e.g. an overfit sample): returns sampled SLat.

        x_0 is used only for its coords/layout (the target voxel set). At real inference the
        coords come from Stage-1 ss_dec occupancy.
        """
        from ...pipelines.samplers.flow_euler_cfg_sparse import FlowEulerCfgSampler as _S  # noqa
        return None  # overfit verification uses the trainer's own loss curve; full sampler wired later
