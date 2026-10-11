"""Crown SS pairs conditioned by a picked template render and the jaw SS latent."""
import glob
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from .crown_ss import CrownJawSS


def parse_fdi(stem: str, valid_codes) -> Optional[str]:
    valid_codes = set(str(code) for code in valid_codes)
    for token in reversed(stem.split("_")):
        if token in valid_codes:
            return token
    return None


def resolve_template_paths(
    template_root: str,
    template_picks: str = "template_picks.json",
    template_meta: str = "meta.json",
    view_index: int = 6,
) -> Dict[str, str]:
    """Resolve each picked FDI code to one rendered image path."""
    template_root = os.path.abspath(os.path.expanduser(template_root))

    def resolve(path: str) -> str:
        path = os.path.expanduser(path)
        return path if os.path.isabs(path) else os.path.join(template_root, path)

    with open(resolve(template_picks), "r", encoding="utf-8") as fp:
        picks = json.load(fp)
    meta_path = resolve(template_meta)
    with open(meta_path, "r", encoding="utf-8") as fp:
        sample_meta = json.load(fp).get("samples", {})

    paths = {}
    for fdi, picked_stem in picks.items():
        info = sample_meta.get(picked_stem)
        if info is None:
            raise ValueError(f"Picked template {picked_stem!r} is absent from {meta_path}")
        render_dir = os.path.join(
            template_root, "renders", str(fdi), f"{int(info['rank']):02d}_{picked_stem}"
        )
        matches = sorted(glob.glob(os.path.join(render_dir, f"v{int(view_index):02d}_*.png")))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one template view v{int(view_index):02d} for FDI {fdi}, got {matches}"
            )
        paths[str(fdi)] = matches[0]
    return paths


def load_image_tensor(path: str, image_size: int = 518) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"), (int(image_size), int(image_size)),
            method=Image.Resampling.LANCZOS, centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CrownTemplateJawSS(CrownJawSS):
    """Return crown target, jaw SS condition, and the selected FDI template image."""

    def __init__(
        self,
        roots: str,
        *,
        template_root: str,
        template_picks: str = "template_picks.json",
        template_meta: str = "meta.json",
        view_index: int = 6,
        image_size: int = 518,
        **kwargs,
    ):
        # Template-conditioned training must never silently drop a sample.
        kwargs.setdefault("strict", True)
        super().__init__(roots, **kwargs)
        self.template_root = os.path.abspath(os.path.expanduser(template_root))
        self.image_size = int(image_size)
        self._image_cache: Dict[str, torch.Tensor] = {}

        self.template_paths = resolve_template_paths(
            self.template_root, template_picks, template_meta, view_index
        )

        original_count = len(self.stems)
        self.sample_fdi = {}
        missing_fdi = []
        for stem in self.stems:
            fdi = parse_fdi(stem, self.template_paths)
            if fdi is None:
                missing_fdi.append(stem)
            else:
                self.sample_fdi[stem] = fdi
        if missing_fdi:
            preview = ", ".join(repr(s) for s in missing_fdi[:5])
            suffix = "..." if len(missing_fdi) > 5 else ""
            raise ValueError(
                f"Missing picked template FDI for {len(missing_fdi)}/{original_count} "
                f"samples ({preview}{suffix}); refusing to drop samples"
            )
        self.loads = [1] * len(self.stems)
        print(f"CrownTemplateJawSS: {len(self.stems)}/{original_count} samples have a picked template")
        print(f"  - template root: {self.template_root}")
        print(f"  - fixed view: v{int(view_index):02d}")

    def _load_template(self, fdi: str) -> torch.Tensor:
        cached = self._image_cache.get(fdi)
        if cached is not None:
            return cached
        tensor = load_image_tensor(self.template_paths[fdi], self.image_size)
        self._image_cache[fdi] = tensor
        return tensor

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        sample = super().__getitem__(index)
        if sample is None:
            return None
        stem = self.stems[index]
        sample["cond"] = self._load_template(self.sample_fdi[stem]).clone()
        if tuple(sample["x_0"].shape) != (8, 16, 16, 16):
            raise ValueError(f"{stem}: x_0 must have shape [8,16,16,16], got {tuple(sample['x_0'].shape)}")
        if tuple(sample["ori_voxel"].shape) != (8, 16, 16, 16):
            raise ValueError(
                f"{stem}: ori_voxel must have shape [8,16,16,16], got {tuple(sample['ori_voxel'].shape)}"
            )
        if tuple(sample["cond"].shape) != (3, self.image_size, self.image_size):
            raise ValueError(
                f"{stem}: cond must have shape [3,{self.image_size},{self.image_size}], "
                f"got {tuple(sample['cond'].shape)}"
            )
        return sample
