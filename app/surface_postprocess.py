"""2026-10-07：交付版局部补洞、Taubin平滑及几何变化保护，无需GT。"""
import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes
from mesh_repair import repair_small_holes


def topology(mesh):
    counts = np.bincount(mesh.edges_unique_inverse)
    bodies = int(mesh.body_count)
    return {'boundary_edges': int((counts == 1).sum()),
            'components': bodies, 'watertight': bool(mesh.is_watertight),
            'handles': int(round((2*bodies-mesh.euler_number)/2)) if mesh.is_watertight else None}


def scene_for(mesh):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, np.int32))))
    return scene


def geometry_change(raw, result, seed):
    import open3d as o3d
    distances = []
    for target, source in [(raw, result), (result, raw)]:
        points = trimesh.sample.sample_surface(source, 100000, seed=seed)[0]
        distances.append(scene_for(target).compute_distance(
            o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=2).numpy())
    distances = np.concatenate(distances)
    return {'extent_change_max': float(np.max(np.abs(result.extents / np.maximum(raw.extents, 1e-12)-1))),
            'surface_shift_p95': float(np.quantile(distances, .95)),
            'surface_shift_mean': float(distances.mean())}


def within_limits(change):
    return change['extent_change_max'] <= .02 and change['surface_shift_p95'] <= .005


def close_small_tunnels(mesh):
    import open3d as o3d
    pitch, pad = 1/384, 4
    voxels = mesh.voxelized(pitch=pitch)
    occupied = np.pad(voxels.matrix, pad)
    origin = voxels.transform[:3, 3] - pad*pitch
    x, y, z = np.meshgrid(*([np.arange(-2, 3)]*3), indexing='ij')
    ball = x*x+y*y+z*z <= 4
    expanded = ndimage.binary_dilation(occupied, structure=ball)
    closed = ndimage.binary_erosion(ndimage.binary_fill_holes(expanded), structure=ball)
    closed |= occupied
    field = ndimage.gaussian_filter(closed.astype(np.float32), sigma=.55)
    vertices, faces, _, _ = marching_cubes(field, .5, spacing=(pitch,)*3, allow_degenerate=False)
    result = trimesh.Trimesh(vertices+origin, faces, process=False)
    result.fix_normals(multibody=True)
    nearest = scene_for(mesh).compute_closest_points(
        o3d.core.Tensor(np.asarray(result.vertices, np.float32)), nthreads=2)
    target = nearest['points'].numpy()
    ids = nearest['primitive_ids'].numpy().astype(np.int64)
    selected = ((np.linalg.norm(target-result.vertices, axis=1) < 2*pitch) &
                (np.einsum('ij,ij->i', result.vertex_normals, mesh.face_normals[ids]) > .5))
    projected = np.asarray(result.vertices).copy()
    projected[selected] = .9*target[selected]+.1*projected[selected]
    result.vertices = projected
    return trimesh.smoothing.filter_taubin(result, lamb=.5, nu=.53, iterations=80)


def postprocess(mesh, seed=1234):
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError('Invalid input crown mesh')
    seed = int(seed) % (2**32)
    before = topology(mesh)
    raw, boundary_meta = repair_small_holes(mesh, max_boundary_edges=32)
    repaired = False
    rejected = None
    if raw.is_watertight and (topology(raw)['handles'] or 0) > 0:
        try:
            candidate = close_small_tunnels(raw)
            change = geometry_change(mesh, candidate, seed)
            after = topology(candidate)
            repaired = bool(after['watertight'] and after['components'] <= before['components'] and
                            (after['handles'] or 0) < (topology(raw)['handles'] or 0) and within_limits(change))
            if not repaired:
                rejected = {'topology': after, **change, 'reason': 'topology or geometry gate'}
        except Exception as exc:
            rejected = {'reason': repr(exc)}
    result = candidate if repaired else trimesh.smoothing.filter_taubin(raw.copy(), lamb=.5, nu=.53, iterations=80)
    change = geometry_change(mesh, result, seed)
    passes = 80
    # Guard the smoothing fallback as well as the remeshing candidate.
    if not within_limits(change):
        repaired = False
        result = trimesh.smoothing.filter_taubin(mesh.copy(), lamb=.5, nu=.53, iterations=20)
        passes = 20
        change = geometry_change(mesh, result, seed)
        if not within_limits(change):
            result, passes = mesh.copy(), 0
            change = geometry_change(mesh, result, seed)
    after = topology(result)
    if after['boundary_edges']:
        status = 'open_boundaries_remaining'
    elif after['handles']:
        status = 'major_topology_hole'
    elif repaired or before['boundary_edges'] > after['boundary_edges']:
        status = 'repaired'
    else:
        status = 'clean'
    return result, {'status': status, 'topology_before': before, 'topology_after': after,
        'hole_repair_accepted': repaired, 'boundary_repair': boundary_meta, 'rejected_candidate': rejected,
        'taubin': {'lambda': .5, 'nu': .53, 'passes': passes}, 'pitch': 1/384,
        'closing_radius_voxels': 2, 'gaussian_sigma_voxels': .55, **change}
