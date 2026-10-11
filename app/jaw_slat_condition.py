# 2026-10-09：离线生产jaw法线40视图DINO编码；与原交付桥接完全相同，不读取训练数据。
import math
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
import utils3d
from trellis.modules import sparse as sp

def sphere_hammersley(i, n, off=(0.5, 0.5)):
    def rad(b, k):
        f = 1.0; r = 0.0
        while k > 0: f /= b; r += f*(k % b); k //= b
        return r
    return 2*math.pi*(i+off[0])/n, math.asin(max(-1, min(1, 1-2*((rad(2, i)+off[1]) % 1.0))))


def cam(yaw, pitch, r=2.0):
    x = r*math.cos(pitch)*math.cos(yaw); y = r*math.cos(pitch)*math.sin(yaw); z = r*math.sin(pitch)
    eye = torch.tensor([x, y, z], dtype=torch.float32, device="cuda")
    extr = utils3d.torch.extrinsics_look_at(eye, torch.zeros(3, device="cuda"), torch.tensor([0, 0, 1.0], device="cuda"))
    fov = torch.tensor(40/180*math.pi, device="cuda")
    return extr, utils3d.torch.intrinsics_from_fov_xy(fov, fov)


def render_normal(glctx, verts, faces, nv, res):
    v = torch.tensor(verts, dtype=torch.float32, device="cuda"); f = torch.tensor(faces, dtype=torch.int32, device="cuda")
    vn = torch.tensor(np.asarray(trimesh.Trimesh(vertices=verts, faces=faces, process=False).vertex_normals, np.float32), device="cuda")
    out = []
    for i in range(nv):
        yaw, pitch = sphere_hammersley(i, nv); extr, intr = cam(yaw, pitch)
        fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]; near, far = 0.1, 10.0
        proj = torch.zeros(4, 4, device="cuda")
        proj[0,0]=2*fx; proj[1,1]=2*fy; proj[0,2]=2*cx-1; proj[1,2]=-2*cy+1; proj[2,2]=far/(far-near); proj[2,3]=near*far/(near-far); proj[3,2]=1.0
        vh = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1); vclip = (vh@extr.T)@proj.T
        rast, _ = dr.rasterize(glctx, vclip.unsqueeze(0), f, (res, res)); mask = (rast[..., -1:] > 0).float()
        nrm, _ = dr.interpolate(vn.unsqueeze(0), rast, f); nrm = F.normalize(nrm, dim=-1)
        img = dr.antialias((nrm+1)/2*mask, rast, vclip.unsqueeze(0), f)
        out.append((img[0].permute(2, 0, 1).clamp(0, 1), extr, intr))
    return out


@torch.no_grad()
def encode_jaw_on_coords(glctx, dino, enc, mimg, simg, jv, jf, idx, nv, res):
    N = idx.shape[0]
    pos = torch.tensor((idx+0.5)/64-0.5, dtype=torch.float32, device="cuda")
    fs = torch.zeros(N, 1024, device="cuda"); fc = torch.zeros(N, 1, device="cuda")
    for img, extr, intr in render_normal(glctx, jv, jf, nv, res):
        x = F.interpolate(img.unsqueeze(0), (518, 518), mode="bilinear", align_corners=False); x = (x-mimg)/simg
        o = dino(x, is_training=True); pt = o["x_prenorm"][:, dino.num_register_tokens+1:]
        pt = F.layer_norm(pt, pt.shape[-1:]); side = int(pt.shape[1]**0.5); pt = pt.permute(0, 2, 1).reshape(1, 1024, side, side)
        uv = utils3d.torch.project_cv(pos, extr[:3], intr)[0]; grid = (uv*2-1).view(1, N, 1, 2)
        samp = F.grid_sample(pt, grid, mode="bilinear", align_corners=False).squeeze(-1).squeeze(0).permute(1, 0)
        inb = ((uv[:, 0] >= 0)&(uv[:, 0] <= 1)&(uv[:, 1] >= 0)&(uv[:, 1] <= 1)).float().unsqueeze(1)
        fs += samp*inb; fc += inb
    feats = fs/fc.clamp(min=1)
    coords = torch.cat([torch.zeros(N, 1, dtype=torch.int32), torch.tensor(idx, dtype=torch.int32)], 1).cuda()
    return enc(sp.SparseTensor(feats=feats.float(), coords=coords), sample_posterior=False).feats.cpu().numpy()

