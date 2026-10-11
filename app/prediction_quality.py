"""2026-10-07：提示稀疏结构异常；启发式检查不代表临床质量认证。"""
import numpy as np


def assess_structure(logits, jaw_occupancy):
    pred = np.asarray(logits) > 0
    jaw = np.asarray(jaw_occupancy) > 0
    if pred.shape != (64, 64, 64) or jaw.shape != pred.shape:
        raise ValueError('Expected aligned 64^3 prediction and jaw occupancy')
    coords = np.argwhere(pred)
    count = len(coords)
    extent = ((coords.max(0)-coords.min(0)+1)/64).tolist() if count else [0., 0., 0.]
    overlap = float((pred & jaw).sum()/count) if count else 0.
    reasons = []
    # Reference validation p1=6762 occupied voxels; use these only as review flags.
    if count < 6000:
        reasons.append('生成结构稀少，需检查牙冠完整性与定框')
    if max(extent) < .65:
        reasons.append('牙冠尺寸偏离训练归一化范围')
    if overlap > .5:
        reasons.append('预测与输入牙列高度重叠，需检查位置与形状')
    return {'warning': bool(reasons), 'reasons': reasons, 'n_coords': count,
            'normalized_extent': extent, 'jaw_overlap_fraction': overlap,
            'thresholds': {'min_review_coords': 6000, 'min_review_max_extent': .65, 'max_review_jaw_overlap': .5},
            'scope': 'Heuristic review flags; not proof of physical penetration or clinical acceptability'}
