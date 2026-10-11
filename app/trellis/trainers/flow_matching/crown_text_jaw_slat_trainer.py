# 2026-09-22 新建：Stage-2（SLat 阶段）text+jaw cross trainer
# 目的：镜像 Stage-1 的 CrownTextJawCrossSSTrainer，用于训练
#   TextTemplateJawCrossStructuredLatentFlowModel（64^3 稀疏 SLat flow）。
# 改动说明：TextConditionedMixin（CLIP 编码 cond）+ SparseFlowMatchingCFGTrainer
#   （稀疏 flow matching）；_pre_init_models 里加载 TRELLIS text-xlarge 的
#   slat_flow_txt_dit_XL 预训练主干（按名字+shape 过滤，strict=False），之后重新
#   initialize_jaw_attention 使 jaw cross-attn 以 text cross-attn 热启动且输出
#   zero-init；冻结策略 train_mode/tail_blocks 与 Stage-1 完全一致。
"""FDI anatomical text + jaw SLat trainer for TRELLIS Stage-2 crown generation."""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from .sparse_flow_matching import SparseFlowMatchingCFGTrainer
from .mixins.text_conditioned import TextConditionedMixin


class CrownTextJawCrossSLatTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """Fine-tune TRELLIS text XL SLat flow with jaw attention and a trainable late DiT tail.

    Mirrors ``CrownTextJawCrossSSTrainer`` (Stage-1) for the sparse structured-latent
    stage. The text encoder and jaw feature extractor are frozen. By default all jaw
    cross-attention modules are trainable and the final ``tail_blocks`` transformer
    blocks plus the output head are unfrozen so the model can learn crown-specific
    structure instead of only learning to inject periodontal context.
    """

    def __init__(
        self,
        *args,
        slat_pretrain: Optional[str] = None,
        train_mode: str = "jaw_and_text_tail",
        tail_blocks: int = 12,
        p_uncond: float = 0.1,
        p_jaw_uncond: float = 0.0,
        **kwargs,
    ):
        self.slat_pretrain_path = slat_pretrain
        self.train_mode = str(train_mode)
        self.tail_blocks = int(tail_blocks)
        self.p_jaw_uncond = float(p_jaw_uncond)
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
        if self.slat_pretrain_path:
            from safetensors.torch import load_file

            if self.is_master:
                print(f"\n[CrownTextJawSLat] loading text TRELLIS pretrain: {self.slat_pretrain_path}", flush=True)
            state = load_file(self.slat_pretrain_path)
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

    def _jaw_features(self, jaw_slat, text_tokens: torch.Tensor):
        model = self._raw_model(self.training_models["denoiser"])
        return model.extract_jaw_features(jaw_slat, torch.zeros_like(text_tokens))

    def training_losses(self, x_0, cond=None, jaw_slat=None, fdi=None, **kwargs):
        if cond is None:
            raise ValueError("anatomical text condition is required")
        if jaw_slat is None:
            raise ValueError("jaw SLat (jaw_slat) is required")
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        text_tokens = self.get_cond(cond, **kwargs)
        jaw_features = self._jaw_features(jaw_slat, text_tokens)
        if self.p_jaw_uncond > 0:
            drop = torch.rand(x_0.shape[0], device=x_0.device) < self.p_jaw_uncond
            if drop.any():
                dropped = []
                for feature in jaw_features:
                    batch_idx = feature.coords[:, 0].long()
                    mask = drop.to(batch_idx.device)[batch_idx].unsqueeze(-1)
                    dropped.append(feature.replace(torch.where(mask, torch.zeros_like(feature.feats), feature.feats)))
                jaw_features = dropped

        pred = self.training_models["denoiser"](
            x_t, t * 1000, text_tokens, jaw_features=jaw_features
        )
        assert pred.shape == noise.shape == x_0.shape
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
    def get_inference_cond(self, cond, jaw_slat=None, **kwargs):
        if jaw_slat is None:
            raise ValueError("jaw SLat (jaw_slat) is required")
        text_tokens = self.encode_text(cond)
        jaw_features = self._jaw_features(jaw_slat, text_tokens)
        neg_cond = self.text_cond_model["null_cond"].repeat(text_tokens.shape[0], 1, 1)
        return {"cond": text_tokens, "neg_cond": neg_cond, "jaw_features": jaw_features, **kwargs}

    def run_snapshot(self, *args, **kwargs):
        return {}

    def snapshot_dataset(self, *args, **kwargs):
        pass
