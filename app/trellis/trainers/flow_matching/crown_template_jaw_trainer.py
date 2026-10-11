"""Template-image + TRELLIS jaw-feature cross-attention trainer for crown Stage 1."""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingCFGTrainer
from .mixins.image_conditioned import ImageConditionedMixin


class CrownTemplateJawCrossSSTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """Use native image cross-attention for a template and an added cross-attn for jaw SS."""

    def __init__(
        self,
        *args,
        ss_pretrain: Optional[str] = None,
        train_mode: str = "jaw_attention",
        p_uncond: float = 0.1,
        p_jaw_uncond: float = 0.0,
        **kwargs,
    ):
        self.ss_pretrain_path = ss_pretrain
        self.train_mode = train_mode
        self.p_uncond = float(p_uncond)
        if not 0.0 <= self.p_uncond <= 1.0:
            raise ValueError(f"p_uncond must be in [0, 1], got {self.p_uncond}")
        self.p_jaw_uncond = float(p_jaw_uncond)
        if not 0.0 <= self.p_jaw_uncond <= 1.0:
            raise ValueError(f"p_jaw_uncond must be in [0, 1], got {self.p_jaw_uncond}")
        super().__init__(*args, **kwargs)

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)
        denoiser = self.models["denoiser"]
        if self.ss_pretrain_path:
            from safetensors.torch import load_file

            if self.is_master:
                print(f"\n[TemplateJawCross] loading SS pretrain: {self.ss_pretrain_path}")
            pretrain = load_file(self.ss_pretrain_path)
            model_state = denoiser.state_dict()
            filtered = {
                key: value for key, value in pretrain.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            skipped = sorted(set(pretrain) - set(filtered))
            incompatible = denoiser.load_state_dict(filtered, strict=False)
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if unexpected:
                raise RuntimeError(f"Unexpected SS pretrain keys: {unexpected[:10]}")
            if self.is_master and skipped:
                print(f"  skipped {len(skipped)} incompatible pretrain keys (shape/name)")

        # Must happen after checkpoint loading: copy the pretrained image attention,
        # then zero its output projection so the new path starts as an exact no-op.
        denoiser.initialize_jaw_attention()
        self._set_trainable_parameters(denoiser)

        if self.is_master:
            n_total = sum(p.numel() for p in denoiser.parameters())
            n_train = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
            print(f"  train_mode={self.train_mode}: {n_train:,}/{n_total:,} trainable parameters")

    def _set_trainable_parameters(self, denoiser) -> None:
        valid_modes = {"jaw_attention", "jaw_and_image_cross", "full"}
        if self.train_mode not in valid_modes:
            raise ValueError(f"train_mode must be one of {sorted(valid_modes)}, got {self.train_mode!r}")
        for name, param in denoiser.named_parameters():
            if self.train_mode == "full":
                trainable = True
            elif self.train_mode == "jaw_and_image_cross":
                trainable = "jaw_" in name or (
                    ".cross_attn.to_kv." in name or ".cross_attn.to_out." in name
                )
            else:
                trainable = "jaw_" in name
            param.requires_grad = trainable

    @staticmethod
    def _raw_model(model):
        return model.module if hasattr(model, "module") else model

    def _jaw_features(self, jaw_ss: torch.Tensor, image_tokens: torch.Tensor):
        model = self._raw_model(self.training_models["denoiser"])
        return model.extract_jaw_features(jaw_ss, torch.zeros_like(image_tokens))

    def training_losses(self, x_0, cond=None, ori_voxel=None, **kwargs):
        assert cond is not None, "template image condition is required"
        assert ori_voxel is not None, "jaw SS latent (ori_voxel) is required"
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        image_tokens = self.get_cond(cond, **kwargs)
        jaw_features = self._jaw_features(ori_voxel, image_tokens)
        if self.p_uncond > 0:
            image_drop = torch.rand(x_0.shape[0], device=x_0.device) < self.p_uncond
            image_tokens = torch.where(
                image_drop[:, None, None], torch.zeros_like(image_tokens), image_tokens
            )
        if self.p_jaw_uncond > 0:
            jaw_drop = torch.rand(x_0.shape[0], device=x_0.device) < self.p_jaw_uncond
            jaw_features = [
                torch.where(drop[:, None, None], torch.zeros_like(feature), feature)
                for feature in jaw_features
                for drop in [jaw_drop]
            ]
        pred = self.training_models["denoiser"](
            x_t,
            t * 1000,
            image_tokens,
            jaw_features=jaw_features,
        )
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]
        mse_per_instance = np.asarray([
            F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).any():
                terms[f"bin_{i}"] = {"mse": float(mse_per_instance[time_bin == i].mean())}
        return terms, {}

    @torch.no_grad()
    def get_inference_cond(self, cond, ori_voxel=None, **kwargs):
        assert ori_voxel is not None, "jaw SS latent (ori_voxel) is required"
        image_tokens = self.encode_image(cond)
        jaw_features = self._jaw_features(ori_voxel, image_tokens)
        return {
            "cond": image_tokens,
            "neg_cond": torch.zeros_like(image_tokens),
            "jaw_features": jaw_features,
            **kwargs,
        }

    @torch.no_grad()
    def sample_crown(
        self,
        denoiser,
        jaw_ss: torch.Tensor,
        template_image: torch.Tensor,
        steps: int = 50,
        cfg_strength: float = 3.0,
    ) -> torch.Tensor:
        from ...pipelines.samplers.flow_euler import FlowEulerCfgSampler

        image_tokens = self.encode_image(template_image)
        model = self._raw_model(denoiser)
        jaw_features = model.extract_jaw_features(jaw_ss, torch.zeros_like(image_tokens))
        noise = torch.randn_like(jaw_ss)
        return FlowEulerCfgSampler(self.sigma_min).sample(
            denoiser,
            noise=noise,
            cond=image_tokens,
            neg_cond=torch.zeros_like(image_tokens),
            jaw_features=jaw_features,
            steps=steps,
            cfg_strength=cfg_strength,
            verbose=False,
        ).samples
