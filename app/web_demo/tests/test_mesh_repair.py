from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mesh_repair import repair_small_holes


def test_repairs_small_open_boundary():
    box = trimesh.creation.box()
    damaged = trimesh.Trimesh(box.vertices, box.faces[:-1], process=True)
    assert not damaged.is_watertight
    repaired, meta = repair_small_holes(damaged)
    assert repaired.is_watertight
    assert meta['boundary_edges_before'] > 0
    assert meta['boundary_edges_after'] == 0
    assert meta['status'] == 'repaired'


def test_reports_closed_topology_hole_without_solidifying_it():
    torus = trimesh.creation.torus(major_radius=2.0, minor_radius=0.5)
    repaired, meta = repair_small_holes(torus)
    assert repaired.is_watertight
    assert meta['boundary_edges_before'] == 0
    assert meta['genus'] == 1
    assert meta['status'] == 'major_topology_hole'
    assert np.allclose(repaired.bounds, torus.bounds)
