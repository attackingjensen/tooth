# 2026-10-07：认证、邻牙辅助定框、哈希候选、预览确认、多GPU队列和几何风险提示。
import hashlib,hmac,json,logging,os,queue,re,secrets,signal,subprocess,sys,threading,time
from pathlib import Path
from datetime import timedelta
import numpy as np
from flask import Flask,request,session,jsonify,render_template,send_file,abort
from werkzeug.exceptions import HTTPException
from geometry import load_mesh,prepare
from frame_alignment import estimate,parameters
from c80_runtime import valid_fdi

BASE=Path(__file__).resolve().parent
PIPELINE=BASE.parent
PYTHON=Path(os.environ.get('CROWN_PYTHON',sys.executable))
DATA=Path(os.environ.get('CROWN_WEB_DATA',str(BASE/'private'))).resolve()
DATA.mkdir(parents=True,exist_ok=True);DATA.chmod(0o700)
credentials=DATA/'access.json'
if not credentials.exists():
    fd=os.open(credentials,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump({'access_code':secrets.token_urlsafe(18),'session_secret':secrets.token_hex(32)},f)
AUTH=json.loads(credentials.read_text())
app=Flask(__name__)
app.config.update(SECRET_KEY=AUTH['session_secret'],MAX_CONTENT_LENGTH=100*1024*1024,
 SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict',SESSION_COOKIE_SECURE=os.environ.get('CROWN_HTTPS')=='1',PERMANENT_SESSION_LIFETIME=timedelta(hours=12))
JOBS=DATA/'jobs';JOBS.mkdir(exist_ok=True)
LOCK=threading.RLock();Q=queue.Queue(maxsize=8);ACTIVE={};RESERVED_GPUS=set()
GPU_SPEC=os.environ.get('CROWN_GPUS',os.environ.get('CROWN_GPU','auto')).strip().lower()
MIN_GPU_FREE_MB=int(os.environ.get('CROWN_MIN_GPU_FREE_MB','35000'))
MAX_WORKERS=int(os.environ.get('CROWN_MAX_WORKERS','2'))
if MAX_WORKERS < 1 or MAX_WORKERS > 8:raise RuntimeError('CROWN_MAX_WORKERS must be between 1 and 8')
if GPU_SPEC!='auto' and not re.fullmatch(r'\d+(,\d+)*',GPU_SPEC):raise RuntimeError('CROWN_GPUS must be auto or comma-separated GPU indices')
EXPORTS={'crown_smooth_world.stl','crown_smooth_unrepaired_world.stl','crown_raw_world.stl','crown_smooth_world.glb','crown_smooth_world.obj','crown_raw_world.obj','jaw_and_crown_world.glb','stage1_coarse_world.stl'}
LOGIN_ATTEMPTS={}

def saved_candidate(source):
    # 仅精确匹配已知扫描哈希；不能按文件名或病例相似性迁用校正框。
    folder=PIPELINE.parent/'production_cases'
    catalog=folder/'catalog.json'
    if not catalog.exists():return None
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    for item in json.loads(catalog.read_text()).get('cases',[]):
        if digest not in {item.get('sha256'),item.get('stl_sha256')}:
            continue
        candidate=folder/'candidates'/item['case']/'frame.json'
        if candidate.is_file():
            return {'case':item['case'],'frame':json.loads(candidate.read_text()),
                    'scope':'已知用户基牙的探索性复测框，仍须预览和检查贯穿孔、边缘贴合'}
    return None

def write(jid,d):
    p=JOBS/jid/'state.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)

def state(jid):
    if not re.fullmatch('[a-f0-9]{32}',jid):abort(404)
    p=JOBS/jid/'state.json'
    if not p.exists():abort(404)
    d=json.loads(p.read_text())
    if d['owner']!=session.get('owner'):abort(404)
    return d

def public(d):
    return {k:v for k,v in d.items() if k not in {'owner'}}

@app.before_request
def access():
    if 'csrf' not in session:session['csrf']=secrets.token_hex(24)
    if request.path.startswith('/api/') and request.path!='/api/login' and not session.get('owner'):
        return jsonify(error='请先输入访问码'),401
    if request.method in ('POST','DELETE','PUT'):
        if not hmac.compare_digest(request.headers.get('X-CSRF-Token',''),session['csrf']):
            return jsonify(error='页面已过期，请刷新后重试'),403

@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options']='nosniff';response.headers['X-Frame-Options']='DENY'
    response.headers['Referrer-Policy']='same-origin'
    if request.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
    return response

@app.errorhandler(Exception)
def error(exc):
    if isinstance(exc,HTTPException):return jsonify(error='文件过大（上限 100MB）' if exc.code==413 else exc.description),exc.code
    if isinstance(exc,(ValueError,TypeError,KeyError)):return jsonify(error=str(exc)),400
    app.logger.exception('request failed');return jsonify(error='处理失败，请检查文件或联系管理员'),500

@app.get('/')
def index():return render_template('index.html',csrf=session['csrf'],logged=bool(session.get('owner')))

@app.post('/api/login')
def login():
    ip=request.remote_addr;now=time.time();attempts=[x for x in LOGIN_ATTEMPTS.get(ip,[]) if now-x<300]
    if len(attempts)>=10:return jsonify(error='尝试次数过多，请五分钟后重试'),429
    if not hmac.compare_digest(str((request.get_json() or {}).get('code','')),AUTH['access_code']):
        LOGIN_ATTEMPTS[ip]=attempts+[now];return jsonify(error='访问码不正确'),401
    LOGIN_ATTEMPTS.pop(ip,None);session['owner']=session.get('owner',secrets.token_hex(16));session.permanent=True
    return jsonify(ok=True)

@app.post('/api/logout')
def logout():session.clear();return jsonify(ok=True)

@app.get('/api/jobs')
def jobs():
    found=[]
    for p in JOBS.glob('*/state.json'):
        d=json.loads(p.read_text())
        if d['owner']==session['owner']:found.append(public(d))
    return jsonify(sorted(found,key=lambda d:d['created'],reverse=True)[:30])

@app.post('/api/upload')
def upload():
    f=request.files.get('file')
    if not f or Path(f.filename).suffix.lower() not in {'.stl','.glb'}:raise ValueError('请上传 STL 或 GLB 文件')
    owned=[p for p in JOBS.glob('*/state.json') if json.loads(p.read_text())['owner']==session['owner']]
    if len(owned)>=30:return jsonify(error='本会话已达到 30 个文件，请联系管理员清理'),429
    jid=secrets.token_hex(16);p=JOBS/jid;p.mkdir()
    source=p/('source'+Path(f.filename).suffix.lower());f.save(source)
    try:mesh=load_mesh(source)
    except Exception:
        import shutil;shutil.rmtree(p);raise
    mesh.export(p/'input.stl')
    d={'id':jid,'owner':session['owner'],'name':Path(f.filename).name[:120],'created':time.time(),
       'status':'uploaded','bounds':mesh.bounds.tolist(),'faces':len(mesh.faces),'default_length':10.0}
    candidate=saved_candidate(source)
    if candidate:d['saved_candidate']=candidate
    write(jid,d);return jsonify(public(d))

@app.post('/api/jobs/<jid>/prepare')
def preparation(jid):
    with LOCK:
        d=state(jid)
        if d['status'] in {'queued','running'}:return jsonify(error='任务正在运行'),409
        body=request.get_json() or {};mesh=load_mesh(JOBS/jid/'input.stl')
        aligned=body.get('frame')
        if aligned is not None:
            if aligned.get('version')!=2 or aligned.get('gt_used') is not False:
                raise ValueError('定框版本无效，请重新估计并预览')
            parameters(aligned)
        crop,frame=prepare(mesh,body.get('center'),body.get('mode','cube'),float(body.get('length',0)),body.get('anchor_normal'),aligned)
        crop.export(JOBS/jid/'prepared.stl');(JOBS/jid/'frame.json').write_text(json.dumps(frame,indent=2))
        d.update(status='prepared',center=frame['center_world'],frame=frame,message='请检查裁剪边界后开始生成')
        d.pop('downloads',None);d.pop('error',None);write(jid,d)
    return jsonify(public(d))

@app.post('/api/jobs/<jid>/estimate-frame')
def estimate_frame(jid):
    with LOCK:
        d=state(jid)
        if d['status'] in {'queued','running'}:return jsonify(error='任务正在运行'),409
        body=request.get_json() or {}
        mesh=load_mesh(JOBS/jid/'input.stl')
        frame=estimate(mesh,body.get('anchor'),body.get('anchor_normal'),body.get('neighbors'),
                       body.get('diameter'),body.get('basis','occlusal'))
        # 估计不覆盖已保存裁剪，必须随后显式预览生成新revision。
    return jsonify(frame=frame)

@app.post('/api/jobs/<jid>/run')
def run(jid):
    with LOCK:
        d=state(jid)
        if d['status']!='prepared':return jsonify(error='请先完成裁剪预览'),409
        if (d.get('frame') or {}).get('version')!=2:return jsonify(error='旧定框需要重新估计和预览'),409
        fdi=valid_fdi((request.get_json() or {}).get('fdi'))
        if Q.full():return jsonify(error='队列已满，请稍后重试'),429
        d.update(fdi=fdi,architecture='C80 + TRELLIS raw55k',status='queued',message='已排队，等待推理资源',revision=secrets.token_hex(6),queued_at=time.time());write(jid,d);Q.put_nowait(jid)
    return jsonify(public(d)),202

@app.post('/api/jobs/<jid>/cancel')
def cancel(jid):
    with LOCK:
        d=state(jid)
        if d['status'] not in {'queued','running'}:return jsonify(error='当前任务不在运行或排队中'),409
        d.update(status='cancelled',message='任务已取消',finished_at=time.time());write(jid,d)
        proc=ACTIVE.get(jid)
    if proc is not None and proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    return jsonify(public(d))

@app.get('/api/jobs/<jid>')
def get_job(jid):return jsonify(public(state(jid)))

@app.get('/api/jobs/<jid>/files/<name>')
def file(jid,name):
    d=state(jid)
    if name in {'input.stl','prepared.stl'}:p=JOBS/jid/name
    elif name in EXPORTS and d['status']=='succeeded':p=JOBS/jid/('result_'+d['revision'])/name
    else:abort(404)
    if not p.is_file():abort(404)
    return send_file(p,as_attachment=request.args.get('download')=='1',download_name=name,conditional=True)

def gpu_candidates():
    try:
        cmd=['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits']
        out=subprocess.check_output(cmd,timeout=5,text=True)
        allowed=None if GPU_SPEC=='auto' else set(GPU_SPEC.split(','))
        found=[]
        for line in out.splitlines():
            index,free,util=(part.strip() for part in line.split(','))
            if allowed is not None and index not in allowed:continue
            found.append({'index':index,'free_mb':int(free),'utilization':int(util)})
        return found
    except Exception:
        logging.exception('failed to query GPUs')
        return []

def acquire_gpu():
    with LOCK:
        choices=[g for g in gpu_candidates() if g['free_mb']>=MIN_GPU_FREE_MB and g['index'] not in RESERVED_GPUS]
        if not choices:return None
        chosen=min(choices,key=lambda g:(g['utilization'],-g['free_mb']))
        RESERVED_GPUS.add(chosen['index'])
        return chosen

def release_gpu(index):
    with LOCK:RESERVED_GPUS.discard(index)

def update_job(jid,**kwargs):
    with LOCK:
        p=JOBS/jid/'state.json';d=json.loads(p.read_text());d.update(kwargs);write(jid,d)

def worker():
    while True:
        jid=Q.get();p=JOBS/jid
        gpu=None
        try:
            if json.loads((p/'state.json').read_text()).get('status')!='queued':continue
            wait_start=time.time()
            while gpu is None:
                if json.loads((p/'state.json').read_text()).get('status')=='cancelled':break
                if time.time()-wait_start>1800:raise RuntimeError('推理资源繁忙，请稍后重试')
                gpu=acquire_gpu()
                if gpu is not None:break
                update_job(jid,message=f'已排队，等待任意 GPU 剩余显存达到 {MIN_GPU_FREE_MB//1000} GB')
                time.sleep(5)
            if json.loads((p/'state.json').read_text()).get('status')=='cancelled':continue
            d=json.loads((p/'state.json').read_text());result=p/('result_'+d['revision']);log=p/('run_'+d['revision']+'.log')
            gpu_index=gpu['index']
            update_job(jid,status='running',message=f'正在 GPU {gpu_index} 编码 jaw',started_at=time.time(),gpu=int(gpu_index),gpu_free_mb_at_start=gpu['free_mb'])
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu_index,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',ATTN_BACKEND='flash_attn')
            cmd=[str(PYTHON),'-u',str(PIPELINE/'infer_jaw_file_v2.py'),'--jaw',str(p/'prepared.stl'),'--original_jaw',str(p/'input.stl'),'--frame',str(p/'frame.json'),'--out_dir',str(result),'--fdi',str(valid_fdi(d.get('fdi')))]
            with log.open('w') as handle:
                proc=subprocess.Popen(cmd,cwd=PIPELINE,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                with LOCK:ACTIVE[jid]=proc
                start=time.time()
                try:
                    while proc.poll() is None:
                        if json.loads((p/'state.json').read_text()).get('status')=='cancelled':break
                        if time.time()-start>1200:raise TimeoutError('生成超时，请联系管理员')
                        tail=log.read_text(errors='replace')[-12000:]
                        msg=f'GPU {gpu_index}：正在生成牙冠结构'
                        if 'Bridge:' in tail:msg=f'GPU {gpu_index}：正在提取牙列细节'
                        if 'Stage 2:' in tail:msg=f'GPU {gpu_index}：正在生成精细牙冠'
                        if 'Postprocess:' in tail:msg=f'GPU {gpu_index}：正在平滑与修复网格'
                        update_job(jid,message=msg);time.sleep(2)
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    with LOCK:ACTIVE.pop(jid,None)
            if json.loads((p/'state.json').read_text()).get('status')=='cancelled':continue
            meta=json.loads((result/'run_metadata.json').read_text()) if (result/'run_metadata.json').exists() else {}
            if meta.get('status') in {'empty_stage1','invalid_stage1'}:
                count=meta.get('pred_active_voxels',0)
                update_job(jid,status='empty',message=f'一阶段仅生成 {count} 个有效体素，结果不可信。请调整 3D 裁剪框后重试。',quality_warning=True,finished_at=time.time())
            elif proc.returncode==0 and meta.get('status')=='complete' and (result/'crown_smooth_world.stl').exists():
                repair=meta.get('mesh_repair') or {}
                repair_status=repair.get('status')
                structure=meta.get('prediction_quality') or {}
                clearance=meta.get('surface_clearance') or {}
                warning=bool(structure.get('warning')) or bool(clearance.get('warning')) or repair_status in {'major_topology_hole','open_boundaries_remaining'}
                if clearance.get('warning'):message='生成完成，但几何需检查：'+'；'.join(clearance['reasons'])
                elif structure.get('warning'):message='生成完成，但结构需检查：'+'；'.join(structure['reasons'])
                elif warning:message='生成完成，但仍有孔洞，需要检查网格'
                elif repair_status=='repaired':message='生成完成，已自动修复小型孔洞'
                else:message='生成完成，请检查牙冠位置与形状'
                update_job(jid,status='succeeded',message=message,quality_warning=warning,prediction_quality=structure,surface_clearance=clearance,mesh_repair=repair,downloads=sorted(EXPORTS),elapsed_seconds=meta['elapsed_seconds'],finished_at=time.time())
            else:raise RuntimeError('生成失败，请联系管理员检查任务日志')
        except Exception as e:
            current=json.loads((p/'state.json').read_text()).get('status')
            if current!='cancelled':
                logging.exception('job %s failed',jid);update_job(jid,status='failed',message=str(e),finished_at=time.time())
        finally:
            if gpu is not None:release_gpu(gpu['index'])
            Q.task_done()

def start_worker():
    queued=[]
    for p in JOBS.glob('*/state.json'):
        d=json.loads(p.read_text())
        if d['status']=='running':
            d.update(status='failed',message='服务重启，任务已中断，请重新预览后提交');write(d['id'],d)
        elif d['status']=='queued':queued.append(d)
    for d in sorted(queued,key=lambda item:item.get('queued_at',item['created'])):Q.put_nowait(d['id'])
    for i in range(MAX_WORKERS):threading.Thread(target=worker,daemon=True,name=f'gpu-worker-{i}').start()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8090);args=parser.parse_args()
    # 单进程持锁，避免两个服务实例竞争同一任务目录/GPU 队列。
    import fcntl
    lock_file=(DATA/'server.lock').open('w');fcntl.flock(lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (DATA/'server.pid').write_text(str(os.getpid()))
    start_worker()
    from waitress import serve
    serve(app,host=args.host,port=args.port,threads=6,max_request_body_size=100*1024*1024)
