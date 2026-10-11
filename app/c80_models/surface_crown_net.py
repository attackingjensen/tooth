# 2026-10-07：共享Native主干，分别输出材料TSDF与严格表面体素logits。
import torch
from torch import nn

from .spatial_crown_net import SpatialCrownNet


class SurfaceCrownNet(SpatialCrownNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_head = nn.Conv3d(self.config['channels'][0], 1, 1)
        nn.init.normal_(self.surface_head.weight, std=0.001)
        nn.init.constant_(self.surface_head.bias, -2.8)

    def predict_outputs(self, condition):
        j = condition['jaw'][0]
        features = self.forward_features(j.new_zeros((len(j), 1, 64, 64, 64)), j.new_zeros(len(j)), condition)
        return self.head(features), self.surface_head(features)


def surface_loss(logits, target):
    # 分别归一化正/负格，避免空格数量主导；Dice逐病例计算。
    logits, target = logits.float(), target.float()
    dims = (1, 2, 3, 4)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
    positive = (bce * target).sum(dims) / target.sum(dims).clamp_min(1)
    negative = (bce * (1 - target)).sum(dims) / (1 - target).sum(dims).clamp_min(1)
    balanced_bce = ((positive + negative) * 0.5).mean()
    probability = logits.sigmoid()
    dice = (1 - (2 * (probability * target).sum(dims) + 1e-6) /
            (probability.sum(dims) + target.sum(dims) + 1e-6)).mean()
    return balanced_bce, dice
