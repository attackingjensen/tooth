"""2026-10-07：用封闭合成网格核对距离符号与风险提示；不模拟真实开放扫描临床判定。"""
import sys
from pathlib import Path
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from surface_clearance import assess_clearance


def test_inside_surface_warns():
    jaw=trimesh.creation.box(extents=[6,6,6])
    crown=trimesh.creation.icosphere(radius=1)
    result=assess_clearance(crown,jaw)
    assert result['warning'] and result['behind_0p5mm_fraction']==1.


def test_separated_surface_has_no_negative_side_risk():
    jaw=trimesh.creation.box(extents=[6,6,6])
    crown=trimesh.creation.icosphere(radius=1);crown.apply_translation([8,0,0])
    result=assess_clearance(crown,jaw)
    assert not result['warning'] and result['behind_0p5mm_fraction']==0.
