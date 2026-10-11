"""idea2 · Crown SS dataset with jaw point-cloud + FDI condition (full-FT route).

Target ``x_0`` = crown SS latent (formulation A, crown-only — idea2 is being validated
standalone, not yet combined with inpainting). The condition is the jaw point cloud
(xyz+normal, in the SAME jaw-driven frame as the SS latent) plus the FDI tooth id, fed to
the model's cross-attention via ``PointCondEncoder``.

Two roots:
    roots      : existing crown SS latent dir (``crown_ss_latent`` per sample, produced by
                 ``preprocess_crown_ss.py``).  -> provides x_0
    point_root : point-cloud sidecar dir (``point_cloud`` [N,6] + ``fdi`` int, produced by
                 ``preprocess_crown_point.py``).  -> provides the condition

Batch contract (matches ``CrownPointCrossSSTrainer``):
    {"x_0": crown_ss [8,16,16,16], "point_cloud": [N,6], "fdi": int}
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_stems(roots: str, split: Optional[str], split_file: Optional[str]) -> List[str]:
    candidates = []
    if split_file:
        candidates.append(split_file if os.path.isabs(split_file) else os.path.join(roots, split_file))
    if split:
        candidates.append(os.path.join(roots, f"{split}.txt"))
    for path in candidates:
        if path and os.path.isfile(path):
            stems = [l.strip() for l in open(path) if l.strip()]
            return [os.path.splitext(s)[0] for s in stems]
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(roots)
        if f.endswith((".pt", ".npz"))
    )


class CrownJawPointSS(Dataset):
    """jaw point cloud (+FDI) -> crown SS latent pairs, for the full-FT cross-attn route."""

    def __init__(
        self,
        roots: str,
        *,
        point_root: str,
        split: Optional[str] = "train",
        split_file: Optional[str] = None,
        ext: Optional[str] = None,
        point_ext: str = ".pt",
        crown_key: str = "crown_ss_latent",
        pc_key: str = "point_cloud",
        fdi_key: str = "fdi",
        num_points: Optional[int] = None,
        normalization: Optional[dict] = None,
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.roots = roots
        self.point_root = point_root
        self.crown_key = crown_key
        self.pc_key = pc_key
        self.fdi_key = fdi_key
        self.num_points = num_points
        self.value_range = (0, 1)
        self.normalization = normalization
        if normalization is not None:
            self.mean = torch.tensor(normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(normalization["std"]).reshape(-1, 1, 1, 1)

        # Auto-detect latent ext (.pt or .npz) if not given.
        self.ext = ext

        stems = _list_stems(roots, split, split_file)
        self.point_ext = point_ext

        def _latent_path(stem):
            if self.ext is not None:
                return os.path.join(roots, stem + self.ext)
            for e in (".pt", ".npz"):
                p = os.path.join(roots, stem + e)
                if os.path.isfile(p):
                    self.ext = e
                    return p
            return os.path.join(roots, stem + ".pt")

        self.stems = [
            s for s in stems
            if os.path.isfile(_latent_path(s))
            and os.path.isfile(os.path.join(point_root, s + point_ext))
        ]
        if max_samples is not None:
            self.stems = self.stems[:max_samples]
        self.loads = [1] * len(self.stems)

        print("**********************")
        print(f"CrownJawPointSS: {len(self.stems)} samples  (split={split_file or split})")
        print(f"  - latent root: {roots}  (ext={self.ext})")
        print(f"  - point  root: {point_root}  (ext={point_ext}, num_points={num_points})")
        if len(stems) != len(self.stems):
            print(f"  - dropped {len(stems) - len(self.stems)} stems missing latent or point file")
        print("**********************")

    def __len__(self):
        return len(self.stems)

    @staticmethod
    def _load_any(path: str) -> Dict[str, Any]:
        if path.endswith(".pt"):
            return torch.load(path, map_location="cpu", weights_only=False)
        return dict(np.load(path))

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
        return t.float()

    def _sample_points(self, pc: torch.Tensor) -> torch.Tensor:
        if self.num_points is None or pc.shape[0] <= self.num_points:
            return pc
        idx = torch.randperm(pc.shape[0])[: self.num_points]
        return pc[idx]

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            lat = self._load_any(os.path.join(self.roots, stem + self.ext))
            crown = self._to_tensor(lat[self.crown_key])
            pt = self._load_any(os.path.join(self.point_root, stem + self.point_ext))
            pc = self._to_tensor(pt[self.pc_key])
            fdi = int(np.asarray(pt[self.fdi_key]).reshape(-1)[0])
        except Exception as e:  # noqa: BLE001 - skip corrupt samples at runtime
            print(f"[CrownJawPointSS] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None
        if crown.ndim == 5:  # (1,8,16,16,16) -> (8,16,16,16)
            crown = crown[0]
        if self.normalization is not None:
            crown = (crown - self.mean) / self.std
        pc = self._sample_points(pc)
        return {"x_0": crown, "point_cloud": pc, "fdi": torch.tensor(fdi, dtype=torch.long)}

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    def visualize_sample(self, sample):
        if isinstance(sample, dict):
            sample = sample.get("x_0", sample)
        if not isinstance(sample, torch.Tensor):
            return sample
        x = sample.detach().float()
        if x.ndim == 5:
            proj = x.mean(dim=(1, 2))
            proj = proj - proj.amin(dim=(1, 2), keepdim=True)
            proj = proj / proj.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            return proj.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0, 1)
        return x
