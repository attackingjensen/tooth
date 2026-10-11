# 2026-09-23：在项目专用 Open3D 环境执行与 baseline 相同的 64³ 体素化，避免环境段错误。
import sys
import numpy as np
import open3d as o3d
payload=np.load(sys.argv[1]);m=o3d.geometry.TriangleMesh()
m.vertices=o3d.utility.Vector3dVector(np.ascontiguousarray(payload['vertices'],np.float64))
m.triangles=o3d.utility.Vector3iVector(np.ascontiguousarray(payload['faces'],np.int32))
vg=o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(m,voxel_size=1./64,min_bound=(-.5,)*3,max_bound=(.5,)*3)
vox=vg.get_voxels();occ=np.zeros((64,64,64),np.float32)
if vox:
 idx=np.array([v.grid_index for v in vox],np.int64).reshape(-1,3)
 idx=np.unique(idx[((idx>=0)&(idx<64)).all(axis=1)],axis=0)
 occ[idx[:,0],idx[:,1],idx[:,2]]=1
np.save(sys.argv[2],occ)
print('Open3D occupancy:',int(occ.sum()),flush=True)
