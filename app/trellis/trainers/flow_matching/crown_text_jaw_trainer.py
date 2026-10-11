"""FDI anatomical text + jaw SS trainer for TRELLIS Stage-1 crown generation."""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .flow_matching import FlowMatchingCFGTrainer
from .mask_loss_utils import prepare_ss_keep_mask
from .mixins.text_conditioned import TextConditionedMixin


class CrownTextJawCrossSSTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """Fine-tune TRELLIS text XL with jaw attention and a trainable late DiT tail.

    The text encoder and jaw feature extractor are frozen.  By default all jaw
    cross-attention modules are trainable and the final ``tail_blocks`` transformer
    blocks plus the output head are unfrozen so the model can learn crown-specific
    geometry instead of only learning to inject periodontal context.
    """

    def __init__(
        self,
        *args,
        ss_pretrain: Optional[str] = None,
        train_mode: str = "jaw_and_text_tail",
        tail_blocks: int = 12,
        p_uncond: float = 0.1,
        p_jaw_uncond: float = 0.0,
        lambda_curv: float = 0.0,
        **kwargs,
    ):
        self.ss_pretrain_path = ss_pretrain
        self.train_mode = str(train_mode)
        self.tail_blocks = int(tail_blocks)
        self.p_jaw_uncond = float(p_jaw_uncond)
        self.lambda_curv = float(lambda_curv)
        if self.tail_blocks < 0:
            raise ValueError(f"tail_blocks must be >= 0, got {self.tail_blocks}")
        if not 0.0 <= self.p_jaw_uncond <= 1.0:
            raise ValueError(f"p_jaw_uncond must be in [0, 1], got {self.p_jaw_uncond}")
        super().__init__(*args, p_uncond=p_uncond, **kwargs)

    @staticmethod
    def _raw_model(model):
        return model.module if hasattr(model, "module") else model

    def _pre_init_models(self, **kwargs):
        super()._pre_init_models(**kwargs)
        denoiser = self.models["denoiser"]
        if self.ss_pretrain_path:
            from safetensors.torch import load_file

            if self.is_master:
                print(f"\n[CrownTextJaw] loading text TRELLIS pretrain: {self.ss_pretrain_path}", flush=True)
            state = load_file(self.ss_pretrain_path)
            model_state = denoiser.state_dict()
            filtered = {
                key: value for key, value in state.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            skipped = len(state) - len(filtered)
            incompatible = denoiser.load_state_dict(filtered, strict=False)
            if self.is_master:
                print(
                    f"  loaded={len(filtered)}, shape/name skipped={skipped}, "
                    f"missing model keys={len(incompatible.missing_keys)}",
                    flush=True,
                )

        # New jaw modules start as a mathematically exact no-op after base weights
        # have been loaded; this keeps the pretrained text model as the initial point.
        if hasattr(denoiser, "initialize_jaw_attention"):
            denoiser.initialize_jaw_attention()
        self._set_trainable_parameters(denoiser)

        if self.is_master:
            total = sum(p.numel() for p in denoiser.parameters())
            trainable = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
            print(
                f"  train_mode={self.train_mode}, tail_blocks={self.tail_blocks}: "
                f"trainable={trainable:,}/{total:,}",
                flush=True,
            )

    def _set_trainable_parameters(self, denoiser) -> None:
        valid = {"jaw_attention", "jaw_and_text_tail", "full"}
        if self.train_mode not in valid:
            raise ValueError(f"train_mode must be one of {sorted(valid)}, got {self.train_mode!r}")
        for name, param in denoiser.named_parameters():
            trainable = False
            if self.train_mode == "full":
                trainable = True
            elif "jaw_" in name:
                trainable = True
            elif self.train_mode == "jaw_and_text_tail":
                if name.startswith("blocks."):
                    try:
                        block_id = int(name.split(".", 2)[1])
                    except (IndexError, ValueError):
                        block_id = -1
                    trainable = block_id >= max(0, int(denoiser.num_blocks) - self.tail_blocks)
                elif name.startswith("out_layer."):
                    trainable = True
            param.requires_grad = trainable

    def _jaw_features(self, jaw_ss: torch.Tensor, text_tokens: torch.Tensor):
        model = self._raw_model(self.training_models["denoiser"])
        return model.extract_jaw_features(jaw_ss, torch.zeros_like(text_tokens))

    def training_losses(self, x_0, cond=None, ori_voxel=None, curv_weight=None, **kwargs):
        if cond is None:
            raise ValueError("anatomical text condition is required")
        if ori_voxel is None:
            raise ValueError("jaw SS latent (ori_voxel) is required")
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        text_tokens = self.get_cond(cond, **kwargs)
        jaw_features = self._jaw_features(ori_voxel, text_tokens)
        if self.p_jaw_uncond > 0:
            drop = torch.rand(x_0.shape[0], device=x_0.device) < self.p_jaw_uncond
            jaw_features = [
                torch.where(drop[:, None, None], torch.zeros_like(feature), feature)
                for feature in jaw_features
            ]

        pred = self.training_models["denoiser"](
            x_t, t * 1000, text_tokens, jaw_features=jaw_features
        )
        target = self.get_v(x_0, noise, t)
        terms = edict(mse=F.mse_loss(pred, target))
        terms["loss"] = terms["mse"]
        if self.lambda_curv > 0.0 and curv_weight is not None:
            # w = 1 + lambda_curv * curvature: high-curvature crown surface voxels
            # get extra velocity-loss weight; lambda_curv=0 falls back to plain MSE.
            # Weighted mean computed directly (masked_mse_velocity expands the mask
            # over channels and divides by C one time too many, shrinking it by 1/C).
            w = 1.0 + self.lambda_curv * prepare_ss_keep_mask(curv_weight, pred)  # [B,1,D,H,W]
            terms["mse_curv"] = ((pred - target) ** 2 * w).sum() / (w.sum() * pred.shape[1])
            terms["loss"] = terms["mse_curv"]
        per = np.asarray([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        bins = np.digitize(t.detach().cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (bins == i).any():
                terms[f"bin_{i}"] = {"mse": float(per[bins == i].mean())}
        return terms, {}

    @torch.no_grad()
    def get_inference_cond(self, cond, ori_voxel=None, **kwargs):
        if ori_voxel is None:
            raise ValueError("jaw SS latent (ori_voxel) is required")
        text_tokens = self.encode_text(cond)
        jaw_features = self._jaw_features(ori_voxel, text_tokens)
        neg_cond = self.text_cond_model["null_cond"].repeat(text_tokens.shape[0], 1, 1)
        return {"cond": text_tokens, "neg_cond": neg_cond, "jaw_features": jaw_features, **kwargs}

    def run_snapshot(self, *args, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass
