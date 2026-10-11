"""FDI template-text + jaw SS dataset for the crown Stage-1 text route.

Each sample contains a crown target and an aligned jaw latent.  The FDI code is
parsed from the sample stem and resolved to a fixed anatomical prompt table;
the dataset never derives text from the ground-truth crown.
"""
import json
import os
from typing import Any, Dict, Optional

import torch

from .crown_ss import CrownJawSS


class CrownTextJawSS(CrownJawSS):
    """Return crown SS, jaw SS, FDI, and one fixed prompt per sample."""

    def __init__(
        self,
        roots: str,
        *,
        prompts_file: str,
        prompt_version: Optional[str] = None,
        valid_fdi: Optional[list] = None,
        **kwargs,
    ):
        kwargs.setdefault("strict", True)
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
        self.prompts = self._validate_prompts(payload, prompt_version, valid_fdi)
        self.sample_fdi = {}
        missing = []
        for stem in self.stems:
            fdi = self._parse_fdi(stem, self.prompts)
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
        self.loads = [1] * len(self.stems)
        counts = {}
        for fdi in self.sample_fdi.values():
            counts[fdi] = counts.get(fdi, 0) + 1
        print(
            f"CrownTextJawSS: {len(self.stems)} samples, {len(self.prompts)} prompts, "
            f"version={self.prompt_version or 'unspecified'}"
        )
        print(f"  - prompt file: {self.prompt_path}")
        print(f"  - FDI counts: {counts}")

    @staticmethod
    def _validate_prompts(payload: dict, expected_version: Optional[str], valid_fdi: Optional[list]) -> Dict[str, str]:
        if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), dict):
            raise ValueError("Prompt file must contain an object field named 'prompts'")
        declared_version = payload.get("schema_version")
        prompts = {}
        versions = set()
        for raw_fdi, record in payload["prompts"].items():
            fdi = str(raw_fdi)
            if not fdi.isdigit() or len(fdi) != 2:
                raise ValueError(f"Invalid FDI prompt key: {raw_fdi!r}")
            if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
                raise ValueError(f"FDI {fdi}: expected a non-empty prompt string")
            prompt = record["prompt"].strip()
            if not prompt:
                raise ValueError(f"FDI {fdi}: prompt is empty")
            record_version = record.get("version", declared_version)
            versions.add(record_version)
            prompts[fdi] = prompt
        if len(versions) > 1:
            raise ValueError(f"Prompt file mixes versions: {sorted(versions)}")
        actual_version = next(iter(versions), declared_version)
        if expected_version is not None and actual_version != expected_version:
            raise ValueError(f"Prompt version mismatch: expected {expected_version!r}, got {actual_version!r}")
        if valid_fdi is not None:
            expected = {str(x) for x in valid_fdi}
            actual = set(prompts)
            if actual != expected:
                raise ValueError(f"Prompt FDI set mismatch: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}")
        return prompts

    @staticmethod
    def _parse_fdi(stem: str, prompts: Dict[str, str]) -> Optional[str]:
        for token in reversed(stem.split("_")):
            if token in prompts:
                return token
        return None

    def __getitem__(self, index) -> Optional[Dict[str, Any]]:
        sample = super().__getitem__(index)
        if sample is None:
            return None
        stem = self.stems[index]
        fdi = self.sample_fdi[stem]
        sample["cond"] = self.prompts[fdi]
        sample["fdi"] = fdi
        return sample

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        out = {
            "x_0": torch.stack([item["x_0"] for item in batch]),
            "ori_voxel": torch.stack([item["ori_voxel"] for item in batch]),
            "cond": [item["cond"] for item in batch],
            "fdi": [item["fdi"] for item in batch],
        }
        if "curv_weight" in batch[0]:
            out["curv_weight"] = torch.stack([item["curv_weight"] for item in batch])
        return out

    def __str__(self):
        return f"{self.__class__.__name__} ({len(self.stems)} samples, prompt_file={self.prompt_path})"
