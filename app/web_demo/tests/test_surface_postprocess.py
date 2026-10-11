"""2026-10-07：检查补洞几何保护、开放边界回退与OBJ顶点法线。"""
from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import surface_postprocess as surface


def test_large_open_surface_is_not_solidified(monkeypatch):
    grid = trimesh.creation.icosphere(subdivisions=3, radius=.3)
    mesh = trimesh.Trimesh(grid.vertices, grid.faces[grid.triangles_center[:, 2] < 0], process=True)
    monkeypatch.setattr(surface, 'close_small_tunnels', lambda _: (_ for _ in ()).throw(AssertionError('Open input solidified')))
    result, meta = surface.postprocess(mesh)
    assert meta['status'] == 'open_boundaries_remaining'
    assert not result.is_watertight
    assert not meta['hole_repair_accepted']
    assert surface.within_limits(meta)


def test_excessive_candidate_displacement_is_rejected(monkeypatch):
    torus = trimesh.creation.torus(major_radius=.2, minor_radius=.08, major_sections=24, minor_sections=12)
    monkeypatch.setattr(surface, 'close_small_tunnels', lambda _: trimesh.creation.icosphere(radius=.48))
    result, meta = surface.postprocess(torus)
    assert not meta['hole_repair_accepted']
    assert meta['rejected_candidate']['extent_change_max'] > .02
    assert surface.topology(result)['handles'] == 1
    assert surface.within_limits(meta)


def test_obj_export_includes_normals_without_geometry_changes(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.3)
    path = tmp_path / 'crown.obj'
    mesh.export(path, include_normals=True)
    assert '\nvn ' in path.read_text()
    reloaded = trimesh.load(path, process=False, maintain_order=True)
    assert np.array_equal(mesh.faces, reloaded.faces)
    assert np.allclose(mesh.vertices, reloaded.vertices, atol=1e-8, rtol=0)
