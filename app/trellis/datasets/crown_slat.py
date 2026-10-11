"""Crown SLat dataset for jaw->crown SLat ControlNet (Stage-2, structured latent).

Each preprocessed sample (a ``.pt`` from ``preprocess_crown_slat.py``) holds SPARSE SLat
for BOTH crown (target) and the crown-hugging clipped jaw (condition), in the SAME v2
crown-self normalized frame (so their 0..63 voxel coords are directly comparable):
    crown_coords [Nc,3] int16, crown_slat [Nc,8] fp16    (generation target x_0)
    jaw_coords   [Nj,3] int16, jaw_slat   [Nj,8] fp16    (ControlNet condition)

Batch contract (matches the SLat ControlNet trainer + ControlledSLatFlowModel):
    x_0      : SparseTensor on crown coords (normalized crown SLat)
    ori_slat : SparseTensor on the SAME crown coords (jaw SLat mapped via
               map_ori_to_edit_coords: real jaw feats where jaw voxels coincide with crown
               voxels, zero elsewhere — the sparse-add control signal)
    cond     : list of "" (geometry-only; trainer feeds a zero cross-attn token)

SLat is normalized with the official TRELLIS-image-large slat_normalization (mean/std),
because the frozen main trunk operates in normalized SLat space (see pipeline decode_slat:
slat = slat*std + mean before decoding). We store raw slat from slat_enc, normalize here.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .editing_slat_text import map_ori_to_edit_coords


# Official TRELLIS-image-large SLat normalization (8-dim, from pipeline.json).
SLAT_MEAN = [-2.1687545776367188, -0.004347046371549368, -0.13352349400520325,
             -0.08418072760105133, -0.5271206498146057, 0.7238689064979553,
             -1.1414450407028198, 1.2039363384246826]
SLAT_STD = [2.377650737762451, 2.386378288269043, 2.124418020248413, 2.1748552322387695,
            2.663944721221924, 2.371192216873169, 2.6217446327209473, 2.684523105621338]


def _list_stems(roots: str, split: Optional[str], split_file: Optional[str]) -> List[str]:
    candidates = []
    if split_file:
        candidates.append(split_file if os.path.isabs(split_file) else os.path.join(roots, split_file))
    if split:
        candidates.append(os.path.join(roots, f"{split}.txt"))
    for path in candidates:
        if path and os.path.isfile(path):
            return [os.path.splitext(l.strip())[0] for l in open(path) if l.strip()]
    return sorted(os.path.splitext(f)[0] for f in os.listdir(roots) if f.endswith(".pt"))


class CrownJawSLat(Dataset):
    """jaw SLat -> crown SLat pairs for SLat ControlNet training (Stage-2)."""

    def __init__(
        self,
        roots: str,
        *,
        split: Optional[str] = "train",
        split_file: Optional[str] = None,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.roots = roots
        self.value_range = (0, 1)
        self.normalize = normalize
        self.mean = torch.tensor(SLAT_MEAN).reshape(1, -1)
        self.std = torch.tensor(SLAT_STD).reshape(1, -1)

        stems = _list_stems(roots, split, split_file)
        self.stems = [s for s in stems if os.path.isfile(os.path.join(roots, s + ".pt"))]
        if max_samples is not None:
            self.stems = self.stems[:max_samples]
        self.loads = [1] * len(self.stems)

        print("**********************")
        print(f"CrownJawSLat: {len(self.stems)} samples (split={split_file or split}) normalize={normalize}")
        print(f"  - root: {roots}")
        if len(stems) != len(self.stems):
            print(f"  - dropped {len(stems) - len(self.stems)} stems with no .pt")
        print("**********************")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            d = torch.load(os.path.join(self.roots, stem + ".pt"), map_location="cpu", weights_only=False)
            crown_coords = np.asarray(d["crown_coords"], np.int64)
            crown_feats = np.asarray(d["crown_slat"], np.float32)
            jaw_coords = np.asarray(d["jaw_coords"], np.int64)
            jaw_feats = np.asarray(d["jaw_slat"], np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[CrownJawSLat] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None

        if self.normalize:
            m, s = self.mean.numpy(), self.std.numpy()
            crown_feats = (crown_feats - m) / s
            jaw_feats = (jaw_feats - m) / s

        # map jaw feats onto crown coords (real feats where coords coincide, zero elsewhere)
        jaw_on_crown = map_ori_to_edit_coords(jaw_coords, jaw_feats, crown_coords)

        return {
            "coords": torch.tensor(crown_coords).int(),
            "feats": torch.tensor(crown_feats).float(),
            "ori_feats": torch.tensor(jaw_on_crown).float(),
            "cond": "",
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub = [batch[i] for i in group]
            coords_list, feats_list, ori_list, layout = [], [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                ori_list.append(b["ori_feats"])
                layout.append(slice(start, start + n))
                start += n
            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            ori_feats = torch.cat(ori_list)

            x_0 = SparseTensor(coords=coords, feats=feats)
            x_0._shape = torch.Size([len(group), *sub[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)

            ori_slat = SparseTensor(coords=coords.clone(), feats=ori_feats)
            ori_slat._shape = x_0._shape
            ori_slat.register_spatial_cache("layout", layout)

            packs.append({"x_0": x_0, "ori_slat": ori_slat, "cond": [b["cond"] for b in sub]})
        return packs[0] if split_size is None else packs

    def __str__(self):
        return f"CrownJawSLat ({len(self.stems)} samples)"


class CrownJawSLatCross(CrownJawSLat):
    """Stage-2 jaw-cross variant: jaw SLAT stays on its OWN coords as cross-attn condition.

    Batch contract (matches ``CrownJawSlatCrossSLatTrainer`` + ``JawSlatCrossSLatFlowModel``):
        x_0      : SparseTensor on crown coords (normalized crown SLat, generation target)
        jaw_slat : SparseTensor on jaw coords (normalized jaw SLat, cross-attn condition)
        cond     : list of "" (unused; conditioning enters via jaw_slat cross-attention)
    """

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            d = torch.load(os.path.join(self.roots, stem + ".pt"), map_location="cpu", weights_only=False)
            crown_coords = np.asarray(d["crown_coords"], np.int64)
            crown_feats = np.asarray(d["crown_slat"], np.float32)
            jaw_coords = np.asarray(d["jaw_coords"], np.int64)
            jaw_feats = np.asarray(d["jaw_slat"], np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[CrownJawSLatCross] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None

        if self.normalize:
            m, s = self.mean.numpy(), self.std.numpy()
            crown_feats = (crown_feats - m) / s
            jaw_feats = (jaw_feats - m) / s

        return {
            "coords": torch.tensor(crown_coords).int(),
            "feats": torch.tensor(crown_feats).float(),
            "jaw_coords": torch.tensor(jaw_coords).int(),
            "jaw_feats": torch.tensor(jaw_feats).float(),
            "cond": "",
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b["coords"].shape[0] + b["jaw_coords"].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub = [batch[i] for i in group]
            coords_list, feats_list, layout = [], [], []
            jaw_coords_list, jaw_feats_list = [], []
            start = 0
            for i, b in enumerate(sub):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                layout.append(slice(start, start + n))
                start += n
                nj = b["jaw_coords"].shape[0]
                jaw_coords_list.append(torch.cat([torch.full((nj, 1), i, dtype=torch.int32), b["jaw_coords"]], dim=-1))
                jaw_feats_list.append(b["jaw_feats"])

            x_0 = SparseTensor(coords=torch.cat(coords_list), feats=torch.cat(feats_list))
            x_0._shape = torch.Size([len(group), *sub[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)

            jaw_slat = SparseTensor(coords=torch.cat(jaw_coords_list), feats=torch.cat(jaw_feats_list))
            jaw_slat._shape = torch.Size([len(group), *sub[0]["jaw_feats"].shape[1:]])

            packs.append({"x_0": x_0, "jaw_slat": jaw_slat, "cond": [b["cond"] for b in sub]})
        return packs[0] if split_size is None else packs


class CrownJawInpaintSLat(CrownJawSLat):
    """Inpainting Stage-2 SLAT (Path-1, formulation C). Produced by
    ``preprocess_crown_inpaint_slat.py``: target = FULL jaw∪crown SLAT on union coords,
    condition = jaw SLAT mapped onto the union coords, plus a per-voxel ``keep_mask``
    (1=jaw/known, 0=socket) aligned with the union coords.

    Batch contract (matches CrownGeoControlNetSLatTrainer + use_mask_loss=True):
        x_0            : SparseTensor on union coords (normalized full SLAT)
        ori_slat       : SparseTensor on the SAME union coords (jaw mapped, zero off-jaw)
        mask_keep_slat : [sum_N] float, 1=keep(jaw) — drives masked-velocity loss vs v_ori
        cond           : list of ""
    """

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            d = torch.load(os.path.join(self.roots, stem + ".pt"), map_location="cpu", weights_only=False)
            full_coords = np.asarray(d["full_coords"], np.int64)
            full_feats = np.asarray(d["full_slat"], np.float32)
            jaw_coords = np.asarray(d["jaw_coords"], np.int64)
            jaw_feats = np.asarray(d["jaw_slat"], np.float32)
            keep = np.asarray(d["keep_mask"], np.float32).reshape(-1)
        except Exception as e:  # noqa: BLE001
            print(f"[CrownJawInpaintSLat] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None

        if self.normalize:
            m, s = self.mean.numpy(), self.std.numpy()
            full_feats = (full_feats - m) / s
            jaw_feats = (jaw_feats - m) / s

        jaw_on_full = map_ori_to_edit_coords(jaw_coords, jaw_feats, full_coords)
        return {
            "coords": torch.tensor(full_coords).int(),
            "feats": torch.tensor(full_feats).float(),
            "ori_feats": torch.tensor(jaw_on_full).float(),
            "keep": torch.tensor(keep).float(),
            "cond": "",
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub = [batch[i] for i in group]
            coords_list, feats_list, ori_list, keep_list, layout = [], [], [], [], []
            start = 0
            for i, b in enumerate(sub):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                ori_list.append(b["ori_feats"])
                keep_list.append(b["keep"])
                layout.append(slice(start, start + n))
                start += n
            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            ori_feats = torch.cat(ori_list)
            keep = torch.cat(keep_list)               # [sum_N], aligned with feats order

            x_0 = SparseTensor(coords=coords, feats=feats)
            x_0._shape = torch.Size([len(group), *sub[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)

            ori_slat = SparseTensor(coords=coords.clone(), feats=ori_feats)
            ori_slat._shape = x_0._shape
            ori_slat.register_spatial_cache("layout", layout)

            packs.append({"x_0": x_0, "ori_slat": ori_slat,
                          "mask_keep_slat": keep, "cond": [b["cond"] for b in sub]})
        return packs[0] if split_size is None else packs

    def __str__(self):
        return f"CrownJawInpaintSLat ({len(self.stems)} samples)"
