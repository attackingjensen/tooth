"""Conservative crown mesh cleanup and small-boundary repair."""
from __future__ import annotations

import numpy as np
import trimesh


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return int(np.count_nonzero(counts == 1))


def repair_small_holes(mesh: trimesh.Trimesh, max_boundary_edges: int = 64):
    """Repair small open boundaries without sealing large anatomical regions.

    MeshFix limits repairs by boundary edge count. Closed handles/tunnels have no
    boundary and are reported in metadata instead of being replaced by a flat cap.
    """
    work = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.faces).copy(),
        process=True,
    )
    input_faces = len(work.faces)
    removed_components = 0

    if work.body_count > 1:
        components = sorted(work.split(only_watertight=False), key=lambda part: len(part.faces), reverse=True)
        largest = len(components[0].faces)
        tiny_limit = max(64, int(largest * 0.001))
        if all(len(part.faces) <= tiny_limit for part in components[1:]):
            removed_components = len(components) - 1
            work = components[0]

    boundary_before = _boundary_edge_count(work)
    if boundary_before:
        from pymeshfix import _meshfix

        fixer = _meshfix.PyTMesh()
        fixer.load_array(np.asarray(work.vertices), np.asarray(work.faces, dtype=np.int32))
        fixer.fill_small_boundaries(nbe=max_boundary_edges, refine=True)
        vertices, faces = fixer.return_arrays()
        work = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    boundary_after = _boundary_edge_count(work)
    watertight = bool(work.is_watertight)
    genus = None
    if watertight and work.body_count == 1 and work.is_winding_consistent:
        genus = max(0, int(round((2 - work.euler_number) / 2)))

    if boundary_after:
        status = 'open_boundaries_remaining'
    elif genus:
        status = 'major_topology_hole'
    elif removed_components or boundary_before:
        status = 'repaired'
    else:
        status = 'clean'

    metadata = {
        'status': status,
        'input_faces': input_faces,
        'output_faces': len(work.faces),
        'removed_tiny_components': removed_components,
        'boundary_edges_before': boundary_before,
        'boundary_edges_after': boundary_after,
        'max_repaired_boundary_edges': max_boundary_edges,
        'watertight': watertight,
        'euler_number': int(work.euler_number),
        'genus': genus,
    }
    return work, metadata
