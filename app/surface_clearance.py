"""2026-10-07：最终牙冠与完整输入扫描的局部法向距离风险提示；开放扫描不当作实体碰撞。"""
import numpy as np
import trimesh
import open3d as o3d


def assess_clearance(crown, jaw, seed=1234):
    points,_=trimesh.sample.sample_surface(crown,20000,seed=seed)
    scene=o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(jaw.vertices,np.float32)),
        o3d.core.Tensor(np.asarray(jaw.faces,np.int32))))
    hit=scene.compute_closest_points(o3d.core.Tensor(np.asarray(points,np.float32)),nthreads=2)
    delta=points-hit['points'].numpy()
    signed=np.einsum('ij,ij->i',delta,hit['primitive_normals'].numpy())
    risk=float((signed<-.5).mean())
    return {'warning':risk>.05,'samples':20000,'behind_0p2mm_fraction':float((signed<-.2).mean()),
        'behind_0p5mm_fraction':risk,'review_threshold':.05,
        'signed_quantiles_mm':np.quantile(signed,[0,.05,.5,.95,1]).tolist(),
        'reasons':['牙冠表面存在明显穿入扫描背侧的风险，请检查基牙、邻牙与对颌'] if risk>.05 else [],
        'scope':'Nearest oriented triangle distances on open scans; heuristic risk only, not certified solid collision or fit quality'}
