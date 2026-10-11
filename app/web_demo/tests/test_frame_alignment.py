"""2026-10-07：验证定向裁剪、双阶段共用变换和不允许镜像，防止预览与出口分歧。"""
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frame_alignment import to_model,to_world,parameters,estimate
from geometry import prepare


def test_rotated_crop_and_inverse_export():
    r=trimesh.transformations.rotation_matrix(.63,[1,2,3])[:3,:3]
    frame={'version':2,'center_world':[2.,-1.,3.],'cube_side_world':6.,
           'rotation_model_to_world':r.tolist(),'gt_used':False}
    mesh=trimesh.creation.box(extents=[4,8,6])
    mesh.vertices=to_world(mesh.vertices,dict(frame,cube_side_world=1.))
    crop,saved=prepare(mesh,frame['center_world'],'cube',6,aligned_frame=frame)
    normalized=to_model(crop.vertices,saved)
    assert np.max(np.abs(normalized))<=.500001
    assert np.isclose(saved['cube_side_world'],saved['estimated_crown_longest_edge_mm']/.7)
    assert np.allclose(normalized.max(0)-normalized.min(0),[4/6,1,1])
    assert np.allclose(to_world(normalized,saved),crop.vertices)
    # 模拟一阶段及二阶段顶点，两者用相同的反变换。
    for vertices in [np.array([[.1,.2,.3],[-.4,.2,.1]]),normalized]:
        assert np.allclose(to_model(to_world(vertices,saved),saved),vertices)


@pytest.mark.parametrize('rotation',[np.diag([-1,1,1]),np.ones((3,3)),np.full((3,3),np.nan)])
def test_reject_invalid_or_reflected_rotation(rotation):
    with pytest.raises(ValueError):parameters({'center_world':[0,0,0],'cube_side_world':10,'rotation_model_to_world':rotation})


def test_rigid_frame_estimate_equivariance():
    mesh=trimesh.creation.icosphere(subdivisions=3,radius=5)
    anchor=np.array([0.,0.,5.]);normal=np.array([0.,0.,1.])
    neighbors=[[-6,0,5],[6,0,5]]
    f=estimate(mesh,anchor,normal,neighbors=neighbors)
    r=trimesh.transformations.rotation_matrix(.7,[1,2,3])[:3,:3];shift=np.array([9.,2.,-6.])
    moved=mesh.copy();moved.vertices=moved.vertices@r.T+shift
    g=estimate(moved,anchor@r.T+shift,normal@r.T,neighbors=np.array(neighbors)@r.T+shift)
    assert np.isclose(f['cube_side_world'],g['cube_side_world'])
    assert np.allclose(np.array(f['center_world'])@r.T+shift,g['center_world'])
    assert np.isclose(abs(np.array(g['rotation_model_to_world'])[:,1]@(r@np.array(f['rotation_model_to_world'])[:,1])),1.)
    assert not f['gt_used'] and len(f['neighbor_points_world'])==2


def test_full07_frame_matches_physical_crown_extent():
    mesh=trimesh.creation.icosphere(subdivisions=3,radius=5)
    f=estimate(mesh,[0,0,5],[0,0,1],diameter=9)
    assert f['fill']==.7
    assert np.isclose(f['cube_side_world'],9/.7)
    _,saved=prepare(mesh,f['center_world'],'cube',f['cube_side_world'],aligned_frame=f)
    assert saved['fill']==.7
    assert np.isclose(saved['estimated_crown_longest_edge_mm'],9)
    crown=np.array([[-4.5,0,0],[4.5,0,0]])@np.array(saved['rotation_model_to_world']).T+saved['center_world']
    assert np.isclose(np.ptp(to_model(crown,saved)[:,0]),.7)
    with pytest.raises(ValueError,match='0.9'):
        prepare(mesh,f['center_world'],'cube',10,aligned_frame=dict(f,fill=.9))


def test_v3_explicit_frame_has_no_heuristic_shift_or_rotation():
    from frame_alignment import v3_frame
    f=v3_frame([2,3,4],9)
    assert f['center_world']==[2,3,4]
    assert np.array_equal(f['rotation_model_to_world'],np.eye(3))
    assert np.isclose(f['cube_side_world'],9/.7)
    assert f['center_offset_from_anchor_mm']==0
    assert f['crop_recipe']=='v3_identity_explicit_frame'
