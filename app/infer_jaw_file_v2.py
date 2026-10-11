# 2026-10-09：C80表面support级联TRELLIS raw55k，显式FDI、XYZ/ZYX适配与统一frame。
import argparse, json, hashlib, time, subprocess, os, sys
from pathlib import Path
import numpy as np
import torch
import trimesh
import jaw_slat_condition as base
from c80_runtime import load_c80, predict_support, valid_fdi
from trellis.pipelines.samplers.flow_euler import FlowEulerCfgSampler
from surface_postprocess import postprocess
from prediction_quality import assess_structure
from frame_alignment import to_model, to_world, parameters
from surface_clearance import assess_clearance
from trellis import models
from trellis.models import from_pretrained
from trellis.modules import sparse as sp
from trellis.datasets.crown_slat import SLAT_MEAN, SLAT_STD
from trellis.datasets.editing_slat_text import map_ori_to_edit_coords

ROOT = Path(__file__).resolve().parent
MIN_PRED_VOXELS = 1000

def build(config, checkpoint):
    spec=json.loads(config.read_text())['models']['denoiser']
    model=getattr(models,spec['name'])(**spec['args']).cuda()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    result=model.load_state_dict(state,strict=True)
    print('Loaded', checkpoint, result, flush=True)
    return model.eval()

def export(mesh,out,name,frame):
    mesh.export(out/(name+'_normalized.ply'))
    world=mesh.copy();world.vertices=to_world(mesh.vertices,frame)
    for ext in ['stl','glb','obj']:
        options = {'include_normals': True} if ext == 'obj' else {}
        world.export(out/(name+'_world.'+ext), **options)
    return world

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jaw',type=Path,required=True)
    p.add_argument('--frame',type=Path,required=True)
    p.add_argument('--original_jaw',type=Path,help='完整原始扫描，用于最终风险诊断与叠加')
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--fdi',type=int,required=True,help='恒牙牙位11–18/21–28/31–38/41–48')
    p.add_argument('--slat_steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=1234)
    args=p.parse_args();valid_fdi(args.fdi)
    if not 1 <= args.slat_steps <= 200: p.error('--slat_steps must be 1..200')
    out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    started=time.time();torch.manual_seed(args.seed);np.random.seed(args.seed)
    frame=json.loads(args.frame.read_text());parameters(frame)
    jaw=trimesh.load(args.jaw,force='mesh',process=True)
    jvn=to_model(jaw.vertices,frame);jf=np.asarray(jaw.faces)
    assert np.isfinite(jvn).all() and np.max(np.abs(jvn))<=.50001
    ckpts=ROOT.parent/'models/TRELLIS-image-large/ckpts'
    ssck=ROOT.parent/'models/c80/c80_epoch80.pt'
    slck=ROOT/'runs/crown_slat_full/ckpts/denoiser_step0055000.pt'
    meta={'input':str(args.jaw.resolve()),'input_sha256':hashlib.sha256(args.jaw.read_bytes()).hexdigest(),'frame':frame,'gt_used':bool(frame.get('gt_used',False)),'seed':args.seed,'stage1_architecture':'SurfaceCrownNet C80','fdi':args.fdi,'stage1_deterministic':True,'surface_threshold':0.,'c80_input_layout':'ZYX/DHW','trellis_coords_layout':'XYZ','material_tsdf_head_used':False,'slat_steps':args.slat_steps,'c80_checkpoint':str(ssck),'c80_checkpoint_sha256':hashlib.sha256(ssck.read_bytes()).hexdigest(),'slat_checkpoint':str(slck),'num_views':40,'render_res':512,'cfg_strength':0.,'taubin':{'lambda':.5,'nu':.53,'iterations':80},'condition_bridge':'normalize native jaw features then exact coordinate overlap; zero elsewhere','note':'C80 epoch80 armC surface logits + SLAT55raw; shared input frame; GT support not used.'}
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    sampler=FlowEulerCfgSampler(sigma_min=1e-5)
    zero=torch.zeros(1,1,1024,device='cuda')
    print('Stage 1: voxelize jaw and predict deterministic C80 surface support',flush=True)
    np.savez(out/'voxel_input.npz',vertices=jvn,faces=jf)
    subprocess.run([os.environ.get('CROWN_VOXEL_PYTHON',sys.executable),
                    str(ROOT/'voxelize_jaw_file.py'),str(out/'voxel_input.npz'),str(out/'jaw_occupancy.npy')],check=True)
    occupancy_xyz=np.load(out/'jaw_occupancy.npy')
    jaw_idx=np.argwhere(occupancy_xyz>0).astype(np.int64)
    c80=load_c80(ssck)
    logits,idx=predict_support(c80,occupancy_xyz,args.fdi)
    np.savez_compressed(out/'stage1.npz',jaw_occupancy=occupancy_xyz,pred_logits=logits,pred_coords=idx,
                        layout='XYZ',architecture='C80 SurfaceCrownNet',fdi=args.fdi)
    meta.update(jaw_active_voxels=int(occupancy_xyz.sum()),pred_active_voxels=len(idx))
    meta['prediction_quality']=assess_structure(logits,occupancy_xyz)
    print('Stage 1 voxels:',meta['jaw_active_voxels'],'->',len(idx),flush=True)
    if len(idx) < MIN_PRED_VOXELS:
        status='empty_stage1' if not len(idx) else 'invalid_stage1'
        error=(f'Stage 1 predicted only {len(idx)} occupied voxels; '
               f'minimum plausible crown is {MIN_PRED_VOXELS}')
        meta.update(status=status,elapsed_seconds=time.time()-started,error=error,
                    min_pred_voxels=MIN_PRED_VOXELS)
        (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
        raise RuntimeError(meta['error'])
    from skimage.measure import marching_cubes
    v,f,_,_=marching_cubes(np.pad((logits>=0).astype(np.float32),1),level=.5)
    coarse=trimesh.Trimesh(vertices=(v-1+.5)/64-.5,faces=f,process=False)
    export(coarse,out,'stage1_coarse',frame)
    del c80;torch.cuda.empty_cache()
    print('Bridge: native jaw normals -> DINO -> SLat encoder -> exact predicted-coordinate overlap',flush=True)
    slenc=from_pretrained(str(ckpts/'slat_enc_swin8_B_64l8_fp16')).cuda().eval()
    dino=torch.hub.load(str(ROOT.parent/'models/dinov2/hub'),'dinov2_vitl14_reg',source='local',pretrained=False)
    dino.load_state_dict(torch.load(ROOT.parent/'models/dinov2/dinov2_vitl14_reg4_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
    dino=dino.eval().cuda()
    ctx=base.dr.RasterizeCudaContext(device='cuda')
    mi=torch.tensor([.485,.456,.406],device='cuda').view(3,1,1);si=torch.tensor([.229,.224,.225],device='cuda').view(3,1,1)
    jftr=base.encode_jaw_on_coords(ctx,dino,slenc,mi,si,jvn,jf,jaw_idx,40,512)
    np.savez_compressed(out/'native_jaw_slat.npz',coords=jaw_idx,features=jftr)
    mapped,overlap=map_ori_to_edit_coords(jaw_idx,
        (jftr-np.asarray(SLAT_MEAN,np.float32)[None])/np.asarray(SLAT_STD,np.float32)[None],
        idx,return_occ=True)
    meta['jaw_overlap_voxels']=int(overlap.sum())
    np.save(out/'normalized_jaw_features_on_pred_coords.npy',mapped)
    del dino,slenc,ctx;torch.cuda.empty_cache()
    print('Stage 2: SLat sampling and mesh decoding',flush=True)
    slm=build(ROOT/'configs/crown/slat_geo_controlnet_crown_full.json',slck)
    sldec=from_pretrained(str(ckpts/'slat_dec_mesh_swin8_B_64l8m256c_fp16')).cuda().eval()
    mean=torch.tensor(SLAT_MEAN,device='cuda').view(1,-1);std=torch.tensor(SLAT_STD,device='cuda').view(1,-1)
    coords=torch.cat([torch.zeros(len(idx),1,dtype=torch.int32),torch.tensor(idx,dtype=torch.int32)],1).cuda()
    ori=sp.SparseTensor(feats=torch.tensor(mapped,device='cuda'),coords=coords.clone())
    noise=sp.SparseTensor(feats=torch.randn(len(idx),8,device='cuda',generator=torch.Generator(device='cuda').manual_seed(args.seed+1)),coords=coords.clone())
    slat=sampler.sample(slm,noise=noise,cond=zero,neg_cond=zero,ori_slat=ori,steps=args.slat_steps,cfg_strength=0.,verbose=True).samples
    torch.save({'coords':slat.coords.cpu(),'normalized_features':slat.feats.cpu()},out/'pred_slat.pt')
    decoded=sldec(slat.replace(slat.feats*std+mean))[0]
    mesh=trimesh.Trimesh(vertices=decoded.vertices.cpu().numpy(),faces=decoded.faces.cpu().numpy(),process=False)
    assert len(mesh.faces)>0 and np.isfinite(mesh.vertices).all()
    raw=export(mesh,out,'crown_raw',frame)
    smooth=trimesh.smoothing.filter_taubin(mesh.copy(),lamb=.5,nu=.53,iterations=80)
    export(smooth,out,'crown_smooth_unrepaired',frame)
    print('Postprocess: conservative hole repair and guarded Taubin smoothing',flush=True)
    repaired,repair_meta=postprocess(mesh,seed=args.seed)
    sm=export(repaired,out,'crown_smooth',frame)
    full_jaw=trimesh.load(args.original_jaw,force='mesh',process=True) if args.original_jaw else jaw
    meta['surface_clearance']=assess_clearance(sm,full_jaw,seed=args.seed)
    meta['surface_clearance']['input_scope']='complete original scan' if args.original_jaw else 'cropped jaw only; omitted neighbors not assessed'
    if args.original_jaw:
        meta['original_input_sha256']=hashlib.sha256(args.original_jaw.read_bytes()).hexdigest()
    meta['noise_protocol']={'stage1':'deterministic C80; no SS diffusion noise','slat_cuda_seed':args.seed+1,'model_initialization_does_not_consume_sampling_rng':True}
    jaw.visual.vertex_colors=[180,190,205,255];sm.visual.vertex_colors=[245,150,50,255]
    full_jaw.visual.vertex_colors=[180,190,205,255]
    scene=trimesh.Scene();scene.add_geometry(full_jaw,node_name='input_jaw');scene.add_geometry(sm,node_name='generated_crown')
    scene.export(out/'jaw_and_crown_world.glb')
    meta.update(status='complete',elapsed_seconds=time.time()-started,raw_vertices=len(mesh.vertices),raw_faces=len(mesh.faces),raw_bounds_world=raw.bounds.tolist(),smooth_bounds_world=sm.bounds.tolist(),smooth_displacement_mean_normalized=float(np.linalg.norm(smooth.vertices-mesh.vertices,axis=1).mean()),mesh_repair=repair_meta)
    (out/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
