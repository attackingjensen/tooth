"""Full-FT jaw GeoKV trainer with train-only curvature supervision."""
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingTrainer
from .mask_loss_utils import prepare_ss_keep_mask


def curvature_weighted_mse(pred, target, curvature, strength, eps=1e-6):
    """Normalized weighted MSE; a zero mask is exactly ordinary MSE."""
    weight = 1.0 + float(strength) * curvature
    numerator = ((pred - target).square() * weight).sum()
    denominator = weight.sum() * pred.shape[1]
    return numerator / denominator.clamp_min(eps)


def region_mse(pred, target, mask, eps=1e-6):
    """MSE in a soft spatial region, broadcasting one mask over channels."""
    numerator = ((pred - target).square() * mask).sum()
    denominator = mask.sum() * pred.shape[1]
    return numerator / denominator.clamp_min(eps)


class CrownJawSSCrossSSTrainer(FlowMatchingTrainer):
    def __init__(self, *args, ss_pretrain=None, lambda_curv=0.0, **kwargs):
        self.ss_pretrain_path = ss_pretrain
        self.lambda_curv = float(lambda_curv)
        if self.lambda_curv < 0:
            raise ValueError(f"lambda_curv must be non-negative, got {self.lambda_curv}")
        super().__init__(*args, **kwargs)

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)
        if self.ss_pretrain_path:
            from safetensors.torch import load_file
            denoiser = self.models["denoiser"]
            state = load_file(self.ss_pretrain_path)
            incompatible = denoiser.load_state_dict(state, strict=False)
            unexpected = list(incompatible.unexpected_keys)
            missing = list(incompatible.missing_keys)
            if unexpected or any(not key.startswith("jaw_cond_encoder.") for key in missing):
                raise RuntimeError(
                    "SS pretrain is incompatible with JawSSCrossSparseStructureFlowModel: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if self.is_master:
                print(f"[JawSSCross] loaded {self.ss_pretrain_path}; "
                      f"new jaw encoder tensors={len(missing)}; full trunk trainable")

    def training_losses(self, x_0, ori_voxel=None, curv_weight=None, **kwargs):
        assert ori_voxel is not None, "jaw SS latent is required"
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        pred = self.training_models["denoiser"](x_t, t * 1000, jaw_ss=ori_voxel)
        target = self.get_v(x_0, noise, t)
        terms = edict(mse=F.mse_loss(pred, target))
        terms["loss"] = terms["mse"]

        if curv_weight is not None:
            curv = prepare_ss_keep_mask(curv_weight, pred).clamp(0, 1)
            terms["mse_curv_region"] = region_mse(pred, target, curv)
            terms["mse_non_curv"] = region_mse(pred, target, 1.0 - curv)
            terms["curv_coverage"] = (curv > 0).float().mean()
            if self.lambda_curv > 0:
                terms["mse_weighted"] = curvature_weighted_mse(
                    pred, target, curv, self.lambda_curv
                )
                terms["loss"] = terms["mse_weighted"]

        per = np.asarray([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        bins = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (bins == i).any():
                terms[f"bin_{i}"] = {"mse": float(per[bins == i].mean())}
        return terms, {}

    @torch.no_grad()
    def sample_crown(self, denoiser, jaw_ss, steps=50):
        from ...pipelines.samplers.flow_euler import FlowEulerSampler
        cond = denoiser.encode_cond(jaw_ss)
        noise = torch.randn_like(jaw_ss)
        return FlowEulerSampler(self.sigma_min).sample(
            denoiser, noise=noise, cond=cond, steps=steps, verbose=False,
        ).samples

    # The generic flow-matching snapshot expects an image/text ``cond`` field,
    # while this trainer is conditioned only by jaw SS latents.
    def run_snapshot(self, *args, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass
