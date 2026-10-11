"""2026-10-07：命令行无GT定框入口，与网页使用相同邻牙辅助估计和定向裁剪。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'web_demo'))
from geometry import load_mesh,prepare
from frame_alignment import estimate,v3_frame


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jaw',type=Path,required=True)
    p.add_argument('--anchor',type=float,nargs=3)
    p.add_argument('--center',type=float,nargs=3,help='v3预计牙冠bbox中心，世界坐标')
    p.add_argument('--normal',type=float,nargs=3)
    p.add_argument('--neighbor',action='append',type=float,nargs=3)
    p.add_argument('--diameter',type=float)
    p.add_argument('--basis',choices=['v3_aligned','occlusal','pca_middle'],default='v3_aligned')
    p.add_argument('--out_dir',type=Path,required=True)
    args=p.parse_args()
    mesh=load_mesh(args.jaw)
    if args.basis=='v3_aligned':
        if args.center is None or args.diameter is None:p.error('v3需要--center和--diameter，输入须已按训练方向对齐')
        frame=v3_frame(args.center,args.diameter)
    else:
        if args.anchor is None:p.error('启发式估框需要--anchor')
        frame=estimate(mesh,args.anchor,args.normal,args.neighbor,args.diameter,args.basis)
    crop,frame=prepare(mesh,frame['center_world'],'cube',frame['cube_side_world'],frame['anchor_normal_world'],frame)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    crop.export(args.out_dir/'prepared.stl');(args.out_dir/'frame.json').write_text(json.dumps(frame,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(frame,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
