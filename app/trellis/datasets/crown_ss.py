"""Crown SS dataset for jaw->crown geometry-ControlNet (Stage-1, sparse structure).

Each preprocessed sample (a ``.pt`` produced by ``preprocess_crown_ss.py``) holds:
    crown_ss_latent : (8,16,16,16)  target SS latent (TRELLIS ``ss_enc`` mean)
    jaw_ss_latent   : (8,16,16,16)  condition SS latent (fed to the ControlNet branch)
both encoded in the SAME jaw-driven normalized frame, so they are spatially aligned on
the 16^3 latent grid (the crown fills the gap inside the surrounding-teeth geometry).

Batch contract (matches the SS ControlNet trainer):
    {"x_0": crown_ss, "ori_voxel": jaw_ss}
    plus optional "curv_weight": (16,16,16) curvature mask from ``curv_root`` sidecars
    (``preprocess_crown_curv_mask_ss.py``); samples without a sidecar get zeros.

There is intentionally no image ``cond`` key — the geometry-only trainer feeds the model a
constant zero cross-attention token, so the only conditioning is the jaw SS latent that the
ControlNet branch consumes via ``ori_voxel``.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_stems(roots: str, split: Optional[str], split_file: Optional[str]) -> List[str]:
    """Resolve the list of sample stems (filenames without extension)."""
    candidates = []
    if split_file:
        candidates.append(split_file if os.path.isabs(split_file) else os.path.join(roots, split_file))
    if split:
        candidates.append(os.path.join(roots, f"{split}.txt"))
    for path in candidates:
        if path and os.path.isfile(path):
            stems = [l.strip() for l in open(path) if l.strip()]
            return [os.path.splitext(s)[0] for s in stems]
    # Fall back to every .pt/.npz in roots.
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(roots)
        if f.endswith((".pt", ".npz"))
    )


class CrownJawSS(Dataset):
    """jaw SS latent -> crown SS latent pairs for SS ControlNet training."""

    def __init__(
        self,
        roots: str,
        *,
        split: Optional[str] = "train",
        split_file: Optional[str] = None,
        ext: str = ".pt",
        crown_key: str = "crown_ss_latent",
        jaw_key: str = "jaw_ss_latent",
        normalization: Optional[dict] = None,
        max_samples: Optional[int] = None,
        strict: bool = False,
        curv_root: Optional[str] = None,
        curv_key: str = "curv_weight_16",
        **kwargs,
    ):
        super().__init__()
        self.roots = roots
        self.ext = ext
        self.crown_key = crown_key
        self.jaw_key = jaw_key
        self.value_range = (0, 1)
        self.normalization = normalization
        self.strict = bool(strict)
        self.curv_root = curv_root
        self.curv_key = curv_key
        self.curv_missing = 0
        if normalization is not None:
            self.mean = torch.tensor(normalization["mean"]).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(normalization["std"]).reshape(-1, 1, 1, 1)

        stems = _list_stems(roots, split, split_file)
        self.stems = [s for s in stems if os.path.isfile(os.path.join(roots, s + ext))]
        if max_samples is not None:
            self.stems = self.stems[:max_samples]
        self.loads = [1] * len(self.stems)

        print("**********************")
        print(f"CrownJawSS: {len(self.stems)} samples  (split={split_file or split})")
        print(f"  - root: {roots}")
        if len(stems) != len(self.stems):
            print(f"  - dropped {len(stems) - len(self.stems)} stems with no {ext} file")
        print("**********************")

    def __len__(self):
        return len(self.stems)

    def _load(self, stem: str) -> Dict[str, Any]:
        path = os.path.join(self.roots, stem + self.ext)
        if self.ext == ".pt":
            return torch.load(path, map_location="cpu", weights_only=False)
        return dict(np.load(path))

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
        return t.float()

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            d = self._load(stem)
            crown = self._to_tensor(d[self.crown_key])
            jaw = self._to_tensor(d[self.jaw_key])
        except Exception as e:  # noqa: BLE001 - skip corrupt samples at runtime
            if self.strict:
                raise RuntimeError(f"Failed to load SS sample {stem!r}") from e
            print(f"[CrownJawSS] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None
        if crown.ndim == 5:  # (1,8,16,16,16) -> (8,16,16,16)
            crown = crown[0]
        if jaw.ndim == 5:
            jaw = jaw[0]
        expected = (8, 16, 16, 16)
        if tuple(crown.shape) != expected or tuple(jaw.shape) != expected:
            message = (
                f"{stem}: expected crown/jaw latent shape {expected}, "
                f"got {tuple(crown.shape)} and {tuple(jaw.shape)}"
            )
            if self.strict:
                raise ValueError(message)
            print(f"[CrownJawSS] skip {message}", flush=True)
            return None
        if self.normalization is not None:
            crown = (crown - self.mean) / self.std
            jaw = (jaw - self.mean) / self.std
        sample = {"x_0": crown, "ori_voxel": jaw}
        if self.curv_root is not None:
            sample["curv_weight"] = self._load_curv_weight(stem)
        return sample

    def _load_curv_weight(self, stem: str) -> torch.Tensor:
        """Curvature weight sidecar (16,16,16); missing file -> zeros (no extra weight)."""
        path = os.path.join(self.curv_root, stem + ".pt")
        if not os.path.isfile(path):
            self.curv_missing += 1
            if self.curv_missing <= 5 or self.curv_missing % 1000 == 0:
                print(f"[CrownJawSS] no curv sidecar for {stem} ({self.curv_missing} missing so far)", flush=True)
            return torch.zeros(16, 16, 16)
        d = torch.load(path, map_location="cpu", weights_only=False)
        w = self._to_tensor(d[self.curv_key])
        if w.ndim == 4 and w.shape[0] == 1:  # (1,16,16,16) -> (16,16,16)
            w = w[0]
        if tuple(w.shape) != (16, 16, 16):
            raise ValueError(f"{stem}: expected curv weight shape (16,16,16), got {tuple(w.shape)}")
        return w

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    def visualize_sample(self, sample):
        """Cheap projection of an SS latent / dict for snapshot logging (unused: i_sample high)."""
        if isinstance(sample, dict):
            sample = sample.get("x_0", sample)
        if not isinstance(sample, torch.Tensor):
            return sample
        x = sample.detach().float()
        if x.ndim == 5:  # [B,C,D,H,W] -> grayscale depth projection
            proj = x.mean(dim=(1, 2))
            proj = proj - proj.amin(dim=(1, 2), keepdim=True)
            proj = proj / proj.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            return proj.unsqueeze(1).repeat(1, 3, 1, 1).clamp(0, 1)
        return x


class CrownJawInpaintSS(CrownJawSS):
    """Inpainting variant of :class:`CrownJawSS` (Path-1, Stage-1 SS).

    Produced by ``preprocess_crown_inpaint_ss.py``. The generation target is the FULL
    jaw∪crown SS latent (``full_ss_latent``); the jaw SS latent (``jaw_ss_latent``) is the
    known context fed to the ControlNet (``ori_voxel``); ``keep_mask_16`` (1=known jaw/
    background, 0=socket) lets the trainer split the velocity loss into keep vs generate.

    Batch contract (matches ``CrownGeoControlNetInpaintSSTrainer``):
        {"x_0": full_ss, "ori_voxel": jaw_ss, "keep_mask": keep_16 (float, 1=keep)}
    """

    def __init__(
        self,
        roots: str,
        *,
        full_key: str = "full_ss_latent",
        jaw_key: str = "jaw_ss_latent",
        mask_key: str = "keep_mask_16",
        **kwargs,
    ):
        super().__init__(roots, jaw_key=jaw_key, **kwargs)
        self.full_key = full_key
        self.mask_key = mask_key

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            d = self._load(stem)
            full = self._to_tensor(d[self.full_key])
            jaw = self._to_tensor(d[self.jaw_key])
            keep = torch.as_tensor(np.asarray(d[self.mask_key])).float()
        except Exception as e:  # noqa: BLE001 - skip corrupt samples at runtime
            print(f"[CrownJawInpaintSS] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None
        if full.ndim == 5:  # (1,8,16,16,16) -> (8,16,16,16)
            full = full[0]
        if jaw.ndim == 5:
            jaw = jaw[0]
        if keep.ndim == 4 and keep.shape[0] == 1:  # (1,16,16,16) -> (16,16,16)
            keep = keep[0]
        if self.normalization is not None:
            full = (full - self.mean) / self.std
            jaw = (jaw - self.mean) / self.std
        return {"x_0": full, "ori_voxel": jaw, "keep_mask": keep}
