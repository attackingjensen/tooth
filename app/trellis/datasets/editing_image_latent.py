"""Shared image-loading utilities for the editing dataset adapters."""
import numpy as np
import torch
from PIL import Image


_DEFAULT_IMAGE_SIZE = 518


def _load_image_as_tensor(path: str, image_size: int = _DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """
    Load an RGBA/RGB image, tight-crop on alpha when present, resize to
    (image_size, image_size), and return a [3, H, W] float tensor in [0, 1].
    Alpha (if any) is pre-multiplied so the background is black, matching the
    convention used by ImageConditionedMixin for image-to-3D training.
    """
    image = Image.open(path)
    if image.mode == 'RGBA':
        alpha = np.array(image.getchannel(3))
        nz = alpha.nonzero()
        if nz[0].size > 0:
            bbox = [nz[1].min(), nz[0].min(), nz[1].max(), nz[0].max()]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize * 1.2
            aug_bbox = [
                int(cx - aug_hsize), int(cy - aug_hsize),
                int(cx + aug_hsize), int(cy + aug_hsize),
            ]
            image = image.crop(aug_bbox)
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        rgb = image.convert('RGB')
        rgb_t = torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
        alpha_t = torch.tensor(np.array(alpha)).float() / 255.0
        return rgb_t * alpha_t.unsqueeze(0)
    else:
        rgb = image.convert('RGB').resize((image_size, image_size), Image.Resampling.LANCZOS)
        return torch.tensor(np.array(rgb)).permute(2, 0, 1).float() / 255.0
