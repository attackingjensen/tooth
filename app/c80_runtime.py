# 2026-10-09：C80严格权重加载及XYZ↔ZYX适配；不使用TSDF作为TRELLIS support。
from pathlib import Path
import numpy as np
import torch
from c80_models import SurfaceCrownNet

DEFAULT_CHECKPOINT=Path(__file__).resolve().parents[1]/'models/c80/c80_v3_step12320.pt'

def valid_fdi(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not (1 <= value//10 <= 4 and 1 <= value%10 <= 8):
        raise ValueError('请指定恒牙FDI牙位：11–18、21–28、31–38或41–48')
    return int(value)

def load_c80(checkpoint=DEFAULT_CHECKPOINT, device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('step') != 12320 or saved.get('architecture') != 'c80':
        raise ValueError('C80 checkpoint must be v3 step12320')
    model=SurfaceCrownNet(**saved['config']).float().to(device).eval()
    model.load_state_dict(saved['model'],strict=True)
    return model

@torch.inference_mode()
def predict_support(model, occupancy_xyz, fdi):
    fdi=valid_fdi(fdi); a=np.asarray(occupancy_xyz,np.float32)
    if a.shape != (64,64,64) or not np.isfinite(a).all() or not np.isin(a,[0,1]).all() or a.sum()==0:
        raise ValueError('C80 jaw must be nonempty binary 64^3 occupancy in XYZ order')
    device=next(model.parameters()).device
    jaw=torch.from_numpy(np.ascontiguousarray(a.transpose(2,1,0)))[None,None].to(device)
    condition=model.encode_condition(jaw,fdi=torch.tensor([fdi],device=device))
    _,surface=model.predict_outputs(condition)
    logits_dhw=surface[0,0].float().cpu().numpy()
    if not np.isfinite(logits_dhw).all():raise ValueError('Nonfinite C80 prediction')
    coords_xyz=np.argwhere(logits_dhw>=0)[:,[2,1,0]].astype(np.int64)
    return np.ascontiguousarray(logits_dhw.transpose(2,1,0)), coords_xyz
