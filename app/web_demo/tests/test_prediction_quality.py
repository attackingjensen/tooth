"""2026-10-07：结构异常提示与合法接触的区分。"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prediction_quality import assess_structure


def test_small_prediction_inside_jaw_is_flagged():
    pred = np.full((64, 64, 64), -1.)
    pred[20:30, 20:30, 20:30] = 1
    report = assess_structure(pred, pred > 0)
    assert report['warning'] and len(report['reasons']) == 3
    assert report['jaw_overlap_fraction'] == 1


def test_partial_contact_does_not_flag_plausible_structure():
    pred = np.full((64, 64, 64), -1.)
    pred[4:60, 25:40, 25:40] = 1
    jaw = np.zeros_like(pred)
    jaw[4:20, 25:40, 25:40] = 1
    assert not assess_structure(pred, jaw)['warning']


def test_empty_prediction_produces_finite_diagnostics():
    report = assess_structure(np.full((64, 64, 64), -1.), np.zeros((64, 64, 64)))
    assert report['warning'] and report['n_coords'] == 0
    assert report['jaw_overlap_fraction'] == 0
