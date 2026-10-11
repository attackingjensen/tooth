# 2026-10-03：64³ 路径持续存在，三尺度交换；回归和线性速度头分别训练。
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .condition_encoder import ConditionEncoder, LocalUpdate


class ExchangeStage(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.updates = nn.ModuleList([LocalUpdate(c, channels[-1], blocks) for c in channels])
        self.maps = nn.ModuleDict()
        for src in range(3):
            for dst in range(3):
                if src == dst:
                    continue
                if src < dst:
                    layers = []
                    cin = channels[src]
                    for level in range(src + 1, dst + 1):
                        layers.extend([nn.Conv3d(cin, channels[level], 4, stride=2, padding=1), nn.SiLU()])
                        cin = channels[level]
                    self.maps[f"{src}_{dst}"] = nn.Sequential(*layers)
                else:
                    self.maps[f"{src}_{dst}"] = nn.Conv3d(channels[src], channels[dst], 1)

    def forward(self, xs, condition, embedding):
        xs = [m(x, j, t, condition["mask"], embedding) for m, x, j, t in zip(self.updates, xs, condition["jaw"], condition["template"])]
        result = []
        for dst, x in enumerate(xs):
            h = x
            for src, other in enumerate(xs):
                if src == dst:
                    continue
                mapped = self.maps[f"{src}_{dst}"](other)
                if src > dst:
                    mapped = F.interpolate(mapped, size=x.shape[2:], mode="trilinear", align_corners=False)
                h = h + mapped
            result.append(F.silu(h))
        return result


class SpatialCrownNet(nn.Module):
    def __init__(self, jaw_channels=1, template_channels=1, channels=(32, 64, 128), stages=3, blocks=2, mode="regression", checkpointing=False):
        super().__init__()
        if mode not in ("regression", "flow") or len(channels) != 3 or min(channels) < 1 or stages < 1 or blocks < 1:
            raise ValueError("invalid model configuration")
        self.mode, self.checkpointing = mode, checkpointing
        self.config = dict(jaw_channels=jaw_channels, template_channels=template_channels, channels=list(channels), stages=stages, blocks=blocks, mode=mode, checkpointing=checkpointing)
        self.condition_encoder = ConditionEncoder(jaw_channels, template_channels, channels)
        self.time_embedding = nn.Sequential(nn.Linear(32, channels[-1]), nn.SiLU(), nn.Linear(channels[-1], channels[-1]))
        self.stems = nn.ModuleList([nn.Conv3d(1, c, 3, padding=1) for c in channels])
        self.stages = nn.ModuleList([ExchangeStage(channels, blocks) for _ in range(stages)])
        self.head = nn.Conv3d(channels[0], 1, 1)

    def encode_condition(self, jaw, template=None, mask=None, fdi=None, frame=None):
        return self.condition_encoder(jaw, template, mask, fdi, frame)

    def _embedding(self, t, condition):
        freq = torch.exp(torch.arange(16, device=t.device, dtype=torch.float32) * (-math.log(10000) / 15))
        angles = t.float()[:, None] * freq[None] * 1000
        encoded = torch.cat((angles.sin(), angles.cos()), dim=1).to(condition["embedding"].dtype)
        return condition["embedding"] + self.time_embedding(encoded)

    def forward(self, state, t, condition):
        return self.head(self.forward_features(state, t, condition))

    def forward_features(self, state, t, condition):
        # 2026-10-07：暴露末端64³特征，供表面辅助头复用；原TSDF出口和权重键不变。
        b = condition["jaw"][0].shape[0]
        if state.shape != (b, 1, 64, 64, 64) or t.shape != (b,):
            raise ValueError("state must be [B,1,64,64,64], t must be [B]")
        if not torch.isfinite(t).all() or torch.any((t < 0) | (t > 1)):
            raise ValueError("t must be finite in [0,1]")
        emb = self._embedding(t, condition)
        xs = [stem(state if i == 0 else F.interpolate(state, size=(64 // (2**i),) * 3, mode="trilinear", align_corners=False)) for i, stem in enumerate(self.stems)]
        for stage in self.stages:
            if self.checkpointing and self.training:
                def run(*args, module=stage):
                    return tuple(module(list(args[:3]), condition, args[3]))
                xs = list(checkpoint(run, *xs, emb, use_reentrant=False))
            else:
                xs = stage(xs, condition, emb)
        return xs[0]

    def predict_field(self, condition):
        if self.mode != "regression":
            raise ValueError("predict_field requires regression weights")
        j = condition["jaw"][0]
        return self(j.new_zeros((j.shape[0], 1, 64, 64, 64)), j.new_zeros(j.shape[0]), condition)

    def predict_velocity(self, state, t, condition):
        if self.mode != "flow":
            raise ValueError("predict_velocity requires flow weights")
        return self(state, t, condition)


class UNetStage(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.enc = nn.ModuleList([LocalUpdate(c, channels[-1], blocks) for c in channels])
        self.down = nn.ModuleList([nn.Conv3d(channels[i], channels[i+1], 4, stride=2, padding=1) for i in range(2)])
        self.up = nn.ModuleList([nn.Conv3d(channels[i+1], channels[i], 1) for i in range(2)])
        self.dec = nn.ModuleList([LocalUpdate(c, channels[-1], blocks) for c in channels[:2]])

    def forward(self, xs, condition, embedding):
        skips, h = [], xs[0]
        for i in range(3):
            h = self.enc[i](h, condition["jaw"][i], condition["template"][i], condition["mask"], embedding)
            skips.append(h)
            if i < 2:
                h = self.down[i](h)
        for i in (1, 0):
            h = skips[i] + F.interpolate(self.up[i](h), size=skips[i].shape[2:], mode="trilinear", align_corners=False)
            h = self.dec[i](h, condition["jaw"][i], condition["template"][i], condition["mask"], embedding)
        return [h, skips[1], skips[2]]


class CrownUNet(SpatialCrownNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stages = nn.ModuleList([UNetStage(self.config["channels"], self.config["blocks"])])
        # Ordinary encoder/decoder baseline uses one down/up pass.
        self.config["stages"] = 1
