# 2026-10-07：网页扫描加载与定向框三角面裁剪；和两阶段推理共用frame v2，不读取GT。
import numpy as np
import trimesh
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frame_alignment import parameters, to_model

def load_mesh(path):
    mesh=trimesh.load(path,force='mesh',process=True)
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('文件中没有可用的三角网格')
    if len(mesh.faces)>2_000_000:raise ValueError('网格超过 200 万面，请先简化')
    if not np.isfinite(mesh.vertices).all() or mesh.extents.max()<=1e-8:
        raise ValueError('网格坐标无效')
    return mesh

def clip(mesh,origin,normal):
    t=mesh.triangles;d=(t-origin)@normal;inside=d>=-1e-9
    kept=[t[inside.all(axis=1)]];pieces=[]
    for k in np.flatnonzero(inside.any(axis=1)&~inside.all(axis=1)):
        poly=[]
        for i in range(3):
            j=(i+1)%3
            if inside[k,i]:poly.append(t[k,i])
            if inside[k,i]!=inside[k,j]:
                poly.append(t[k,i]+(t[k,j]-t[k,i])*d[k,i]/(d[k,i]-d[k,j]))
        for i in range(1,len(poly)-1):pieces.append([poly[0],poly[i],poly[i+1]])
    if pieces:kept.append(np.asarray(pieces))
    tri=np.concatenate(kept)
    if not len(tri):raise ValueError('裁剪区域为空，请重新点选')
    return trimesh.Trimesh(vertices=tri.reshape(-1,3),faces=np.arange(tri.size//3).reshape(-1,3),process=True)

def prepare(mesh,center,mode,length,anchor_normal=None,aligned_frame=None):
    center=np.asarray(center,dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():raise ValueError('请先在模型上点选基牙')
    pad=mesh.extents.max()*.05
    if ((center<mesh.bounds[0]-pad)|(center>mesh.bounds[1]+pad)).any():raise ValueError('所选位置不在模型范围内')
    # 只由 jaw PCA 确定牙列长轴；不能替代逐牙分割，裁剪结果需要用户预览。
    v=mesh.vertices;sample=v[::max(1,len(v)//30000)]
    _,_,vh=np.linalg.svd(sample-sample.mean(0),full_matrices=False)
    axis=vh[0];cropped=mesh.copy()
    if mode=='cube':
        if not np.isfinite(length) or not 0<length<=mesh.extents.max()*4:raise ValueError('裁剪边长无效')
        rotation=np.eye(3) if aligned_frame is None else parameters(aligned_frame)[2]
        for axis_id in range(3):
            direction=rotation[:,axis_id]
            cropped=clip(cropped,center-direction*length/2,direction)
            cropped=clip(cropped,center+direction*length/2,-direction)
    elif mode=='local':
        if not np.isfinite(length) or not 0<length<=mesh.extents.max()*4:raise ValueError('保留范围无效')
        cropped=clip(cropped,center-axis*length/2,axis)
        cropped=clip(cropped,center+axis*length/2,-axis)
    elif mode!='full':raise ValueError('裁剪模式无效')
    side=float(length) if mode=='cube' else float(2*np.abs(cropped.vertices-center).max()/.95)
    normal=None
    if anchor_normal is not None:
        normal=np.asarray(anchor_normal,dtype=float)
        if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-6:
            raise ValueError('点选表面法向无效，请重新点选')
        normal=(normal/np.linalg.norm(normal)).tolist()
    frame={'center_world':center.tolist(),'cube_side_world':side,'anchor_normal_world':normal,'gt_used':False,
                    'mode':mode,'length':length,'longitudinal_axis':axis.tolist(),
                    'crop_faces':len(cropped.faces),'input_faces':len(mesh.faces),
                    'normalization':'(v_world-center_world)/cube_side_world',
                    'localization':'User-selected point on jaw; no automatic abutment detection',
                    'crop_rule':('Oriented cube using shared inference frame' if mode=='cube' else 'Longitudinal clipping or full jaw; inspect boundaries in preview')}
    if aligned_frame is not None:
        if mode!='cube':raise ValueError('局部对齐请使用定向立方体模式')
        frame.update(aligned_frame,center_world=center.tolist(),cube_side_world=side,mode=mode,length=length,
                     estimated_crown_longest_edge_mm=side*.9,
                     crop_faces=len(cropped.faces),input_faces=len(mesh.faces))
    if np.max(np.abs(to_model(cropped.vertices,frame)))>.50001:
        raise ValueError('裁剪结果越出归一化框，请重新预览')
    return cropped,frame
