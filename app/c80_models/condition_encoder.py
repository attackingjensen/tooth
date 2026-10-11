# 2026-10-03：jaw/template 同坐标多尺度条件，模板缺省与空区域显式区分。
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(next(g for g in (8, 4, 2, 1) if channels % g == 0), channels)


class ConditionEncoder(nn.Module):
    def __init__(self, jaw_channels, template_channels, channels):
        super().__init__()
        self.jaw_channels = jaw_channels
        self.template_channels = template_channels
        def pyramid(cin):
            return nn.ModuleList([
                nn.Sequential(nn.Conv3d(cin, channels[0], 3, padding=1), norm(channels[0]), nn.SiLU()),
                nn.Sequential(nn.Conv3d(channels[0], channels[1], 4, stride=2, padding=1), norm(channels[1]), nn.SiLU()),
                nn.Sequential(nn.Conv3d(channels[1], channels[2], 4, stride=2, padding=1), norm(channels[2]), nn.SiLU()),
            ])
        self.jaw = pyramid(jaw_channels)
        self.template = pyramid(template_channels)
        self.fdi = nn.Embedding(53, channels[2])

    def forward(self, jaw, template=None, mask=None, fdi=None, frame=None):
        if jaw.ndim != 5 or jaw.shape[1:] != (self.jaw_channels, 64, 64, 64):
            raise ValueError("jaw must be [B,Cj,64,64,64]")
        b = jaw.shape[0]
        if template is None:
            template = jaw.new_zeros((b, self.template_channels, 64, 64, 64))
            mask = jaw.new_zeros((b, 1, 1, 1, 1))
        elif template.shape != (b, self.template_channels, 64, 64, 64):
            raise ValueError("template must share jaw grid and batch")
        elif mask is None:
            mask = jaw.new_ones((b, 1, 1, 1, 1))
        mask = mask.to(jaw).reshape(b, 1, 1, 1, 1)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("template mask must be binary")
        fdi = torch.zeros(b, dtype=torch.long, device=jaw.device) if fdi is None else fdi.to(device=jaw.device, dtype=torch.long)
        valid_fdi = (fdi == 0) | (((fdi // 10 >= 1) & (fdi // 10 <= 4)) & ((fdi % 10 >= 1) & (fdi % 10 <= 8)))
        if fdi.shape != (b,) or not valid_fdi.all():
            raise ValueError("FDI must be permanent tooth code 11..48 or 0 (unknown)")
        js, ts = [], []
        j, t = jaw, template * mask
        for jl, tl in zip(self.jaw, self.template):
            j, t = jl(j), tl(t)
            js.append(j)
            ts.append(t * mask)
        return {"jaw": js, "template": ts, "mask": mask, "embedding": self.fdi(fdi), "frame": frame}


class ResidualBlock(nn.Module):
    def __init__(self, channels, embedding_channels):
        super().__init__()
        self.n1, self.n2 = norm(channels), norm(channels)
        self.c1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.film = nn.Linear(embedding_channels, channels * 2)

    def forward(self, x, embedding):
        scale, shift = self.film(embedding).chunk(2, dim=1)
        h = self.n1(x) * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        h = self.c1(F.silu(h))
        return x + self.c2(F.silu(self.n2(h)))


class LocalUpdate(nn.Module):
    def __init__(self, channels, embedding_channels, blocks=2):
        super().__init__()
        self.inject = nn.Conv3d(3 * channels + 1, channels, 1)
        self.blocks = nn.ModuleList([ResidualBlock(channels, embedding_channels) for _ in range(blocks)])

    def forward(self, x, jaw, template, mask, embedding):
        x = x + self.inject(torch.cat((x, jaw, template, mask.expand(-1, 1, *x.shape[2:])), dim=1))
        for block in self.blocks:
            x = block(x, embedding)
        return x
