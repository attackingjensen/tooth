"""Full-FT jaw-cross SLat trainer (Stage-2 counterpart of CrownJawSSCrossSSTrainer).

The denoiser is ``JawSlatCrossSLatFlowModel``: TRELLIS SLat flow trunk initialized from
the official image-large pretrain (all weights trainable), with the image/text
cross-attention condition replaced by tokens encoded from the jaw SLAT features
(pooled to a dense 16^3 grid -> JawSSCondEncoder). Conditioning enters only through
``jaw_slat`` — the dataset's ``cond`` strings are unused.
"""
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .sparse_flow_matching import SparseFlowMatchingTrainer


class CrownJawSlatCrossSLatTrainer(SparseFlowMatchingTrainer):
    def __init__(self, *args, slat_pretrain=None, **kwargs):
        self.slat_pretrain_path = slat_pretrain
        super().__init__(*args, **kwargs)

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)
        if self.slat_pretrain_path:
            from safetensors.torch import load_file
            denoiser = self.models["denoiser"]
            state = load_file(self.slat_pretrain_path)
            incompatible = denoiser.load_state_dict(state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            if unexpected or any(not key.startswith("jaw_cond_encoder.") for key in missing):
                raise RuntimeError(
                    "SLat pretrain is incompatible with JawSlatCrossSLatFlowModel: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if self.is_master:
                print(f"[JawSlatCross] loaded {self.slat_pretrain_path}; "
                      f"new jaw encoder tensors={len(missing)}; full trunk trainable")

    def training_losses(self, x_0, cond=None, jaw_slat=None, **kwargs):
        assert jaw_slat is not None, "jaw SLat condition is required"
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        pred = self.training_models["denoiser"](x_t, t * 1000, jaw_slat=jaw_slat)
        target = self.get_v(x_0, noise, t)
        terms = edict(mse=F.mse_loss(pred.feats, target.feats))
        terms["loss"] = terms["mse"]

        per = np.asarray([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
            for i in range(x_0.shape[0])
        ])
        bins = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (bins == i).any():
                terms[f"bin_{i}"] = {"mse": float(per[bins == i].mean())}
        return terms, {}

    @torch.no_grad()
    def sample_crown_slat(self, denoiser, x_0, jaw_slat, steps=50):
        """Sample crown SLat on the coord layout of x_0, conditioned on jaw_slat."""
        from ...pipelines.samplers.flow_euler import FlowEulerCfgSampler
        cond = denoiser.encode_cond(jaw_slat)
        noise = x_0.replace(torch.randn_like(x_0.feats))
        return FlowEulerCfgSampler(self.sigma_min).sample(
            denoiser, noise=noise, cond=cond, neg_cond=cond,
            steps=steps, cfg_strength=0.0, verbose=False,
        ).samples

    # The generic sparse snapshot expects an image/text ``cond`` field,
    # while this trainer is conditioned only by jaw SLat features.
    def run_snapshot(self, *args, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass
