# 2026-09-22 新建：Stage-2（SLat 阶段）text+jaw cross 数据集
# 目的：为 TemplateJawCrossStructuredLatentFlowModel 提供训练数据。crown SLat 是生成
#   目标 x_0；jaw SLat 保留原始稀疏 token（不做 map_ori_to_edit_coords），直接作为
#   jaw 特征提取的输入；文本条件由 stem 解析 FDI 后查固定 prompt 表得到。
# 改动说明：subclass CrownJawSLat 复用其 stems 列举/SLAT 归一化；prompt 加载与 FDI
#   解析直接复用 CrownTextJawSS 的静态方法（_validate_prompts/_parse_fdi）；collate
#   产出 x_0（crown coords 上的 SparseTensor）、jaw_slat（jaw 自身 coords 上的
#   SparseTensor）、cond 文本列表与 fdi 列表。
"""FDI template-text + jaw SLat dataset for the crown Stage-2 text route.

Each preprocessed sample (a ``.pt`` from ``preprocess_crown_slat.py``) holds SPARSE SLat
for BOTH crown (target) and the crown-hugging clipped jaw (condition), in the SAME v2
crown-self normalized frame (0..63 voxel coords).

Unlike the ControlNet route (``CrownJawSLat``), the jaw SLat keeps its own sparse
token set — it is consumed by ``extract_jaw_features`` of the text+jaw cross model,
not mapped onto the crown coords.

Batch contract (matches CrownTextJawCrossSLatTrainer):
    x_0      : SparseTensor on crown coords (normalized crown SLat)
    jaw_slat : SparseTensor on the jaw's own coords (normalized jaw SLat)
    cond     : list of fixed template prompt strings (resolved from FDI via prompt table)
    fdi      : list of FDI codes (for logging/debugging)
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .crown_slat import CrownJawSLat
from .crown_text_jaw_ss import CrownTextJawSS


class CrownTextJawSLat(CrownJawSLat):
    """text prompt + jaw SLat -> crown SLat pairs for Stage-2 text+jaw cross training."""

    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str,
        prompt_version: Optional[str] = None,
        valid_fdi: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(roots, **kwargs)
        if os.path.isabs(prompts_file):
            prompt_path = prompts_file
        elif os.path.isfile(prompts_file):
            prompt_path = prompts_file
        else:
            prompt_path = os.path.join(roots, prompts_file)
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"Template prompt file does not exist: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self.prompt_path = os.path.abspath(prompt_path)
        self.prompt_version = prompt_version
        self.prompts = CrownTextJawSS._validate_prompts(payload, prompt_version, valid_fdi)
        self.sample_fdi = {}
        missing = []
        for stem in self.stems:
            fdi = CrownTextJawSS._parse_fdi(stem, self.prompts)
            if fdi is None:
                missing.append(stem)
            else:
                self.sample_fdi[stem] = fdi
        if missing:
            preview = ", ".join(repr(s) for s in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"Could not resolve an FDI template for {len(missing)}/{len(self.stems)} "
                f"samples ({preview}{suffix})"
            )
        counts = {}
        for fdi in self.sample_fdi.values():
            counts[fdi] = counts.get(fdi, 0) + 1
        print(
            f"CrownTextJawSLat: {len(self.stems)} samples, {len(self.prompts)} prompts, "
            f"version={self.prompt_version or 'unspecified'}"
        )
        print(f"  - prompt file: {self.prompt_path}")
        print(f"  - FDI counts: {counts}")

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        stem = self.stems[index]
        try:
            d = torch.load(os.path.join(self.roots, stem + ".pt"), map_location="cpu", weights_only=False)
            crown_coords = np.asarray(d["crown_coords"], np.int64)
            crown_feats = np.asarray(d["crown_slat"], np.float32)
            jaw_coords = np.asarray(d["jaw_coords"], np.int64)
            jaw_feats = np.asarray(d["jaw_slat"], np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[CrownTextJawSLat] skip {stem}: {type(e).__name__}: {e}", flush=True)
            return None

        if self.normalize:
            m, s = self.mean.numpy(), self.std.numpy()
            crown_feats = (crown_feats - m) / s
            jaw_feats = (jaw_feats - m) / s

        fdi = self.sample_fdi[stem]
        return {
            "coords": torch.tensor(crown_coords).int(),
            "feats": torch.tensor(crown_feats).float(),
            "jaw_coords": torch.tensor(jaw_coords).int(),
            "jaw_feats": torch.tensor(jaw_feats).float(),
            "cond": self.prompts[fdi],
            "fdi": fdi,
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
            coords_list, feats_list, jaw_coords_list, jaw_feats_list = [], [], [], []
            layout, jaw_layout = [], []
            start, jaw_start = 0, 0
            for i, b in enumerate(sub):
                n = b["coords"].shape[0]
                coords_list.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
                feats_list.append(b["feats"])
                layout.append(slice(start, start + n))
                start += n
                nj = b["jaw_coords"].shape[0]
                jaw_coords_list.append(torch.cat([torch.full((nj, 1), i, dtype=torch.int32), b["jaw_coords"]], dim=-1))
                jaw_feats_list.append(b["jaw_feats"])
                jaw_layout.append(slice(jaw_start, jaw_start + nj))
                jaw_start += nj
            coords = torch.cat(coords_list)
            feats = torch.cat(feats_list)
            jaw_coords = torch.cat(jaw_coords_list)
            jaw_feats = torch.cat(jaw_feats_list)

            x_0 = SparseTensor(coords=coords, feats=feats)
            x_0._shape = torch.Size([len(group), *sub[0]["feats"].shape[1:]])
            x_0.register_spatial_cache("layout", layout)

            jaw_slat = SparseTensor(coords=jaw_coords, feats=jaw_feats)
            jaw_slat._shape = torch.Size([len(group), *sub[0]["jaw_feats"].shape[1:]])
            jaw_slat.register_spatial_cache("layout", jaw_layout)

            packs.append({
                "x_0": x_0,
                "jaw_slat": jaw_slat,
                "cond": [b["cond"] for b in sub],
                "fdi": [b["fdi"] for b in sub],
            })
        return packs[0] if split_size is None else packs

    def __str__(self):
        return f"CrownTextJawSLat ({len(self.stems)} samples, prompt_file={self.prompt_path})"
