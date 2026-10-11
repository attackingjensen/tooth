# 2026-09-23：验证访问隔离、CSRF、任务状态、裁剪几何及错误上传。
import io,json,os,sys,tempfile
from pathlib import Path
os.environ['CROWN_WEB_DATA']=tempfile.mkdtemp(prefix='crown-web-test-')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as server
import trimesh,numpy as np
import hashlib

def client():
    c=server.app.test_client();c.get('/')
    with c.session_transaction() as s:csrf=s['csrf']
    headers={'X-CSRF-Token':csrf}
    assert c.post('/api/login',json={'code':server.AUTH['access_code']},headers=headers).status_code==200
    return c,headers

def upload(c,h):
    mesh=trimesh.creation.box(extents=(4,8,6));data=mesh.export(file_type='stl')
    r=c.post('/api/upload',data={'file':(io.BytesIO(data),'jaw.stl')},headers=h)
    assert r.status_code==200,r.json
    return r.json['id']

def test_auth_csrf_owner_and_file_whitelist():
    c=server.app.test_client();assert c.get('/api/jobs').status_code==401
    a,h=client();assert a.post('/api/upload').status_code==403
    jid=upload(a,h);b,bh=client()
    assert b.get('/api/jobs/'+jid).status_code==404
    assert b.get('/api/jobs/'+jid+'/files/input.stl').status_code==404
    assert a.get('/api/jobs/'+jid+'/files/state.json').status_code==404
    assert a.get('/api/jobs/'+jid+'/files/input.stl').status_code==200

def test_preview_queue_and_busy_guard():
    c,h=client();jid=upload(c,h);url='/api/jobs/'+jid
    assert c.post(url+'/run',headers=h).status_code==409
    r=c.post(url+'/prepare',json={'center':[0,0,0],'mode':'local','length':4},headers=h)
    assert r.status_code==200,r.json
    crop=trimesh.load(server.JOBS/jid/'prepared.stl',force='mesh')
    # 长轴为 Y，裁剪不改变高度或宽度。
    assert np.allclose(crop.extents,[4,4,6])
    # 旧模式仍可用于几何预览，但提交必须重新使用v2定向框。
    assert c.post(url+'/run',headers=h).status_code==409
    frame={'version':2,'gt_used':False,'center_world':[0,0,0],'cube_side_world':6,
           'rotation_model_to_world':np.eye(3).tolist()}
    assert c.post(url+'/prepare',json={'center':[0,0,0],'mode':'cube','length':6,'frame':frame},headers=h).status_code==200
    for invalid in [None, 0, 19, 50, 16.5, True, '16']:
        assert c.post(url+'/run',json={'fdi':invalid},headers=h).status_code==400
    assert c.post(url+'/run',json={'fdi':16},headers=h).status_code==202
    assert c.post(url+'/run',headers=h).status_code==409
    assert c.post(url+'/prepare',json={'center':[0,0,0]},headers=h).status_code==409

def test_cancel_queued_job_and_owner_guard():
    c,h=client();jid=upload(c,h);url='/api/jobs/'+jid
    frame={'version':2,'gt_used':False,'center_world':[0,0,0],'cube_side_world':6,
           'rotation_model_to_world':np.eye(3).tolist()}
    assert c.post(url+'/prepare',json={'center':[0,0,0],'mode':'cube','length':6,'frame':frame},headers=h).status_code==200
    assert c.post(url+'/run',json={'fdi':16},headers=h).status_code==202
    r=c.post(url+'/cancel',headers=h)
    assert r.status_code==200 and r.json['status']=='cancelled'
    assert c.post(url+'/cancel',headers=h).status_code==409
    other,oh=client();assert other.post(url+'/cancel',headers=oh).status_code==404

def test_gpu_scheduler_uses_available_lowest_load(monkeypatch):
    server.RESERVED_GPUS.clear()
    monkeypatch.setattr(server,'gpu_candidates',lambda:[
        {'index':'0','free_mb':90000,'utilization':99},
        {'index':'2','free_mb':34000,'utilization':0},
        {'index':'7','free_mb':80000,'utilization':40},
    ])
    first=server.acquire_gpu();second=server.acquire_gpu()
    assert first['index']=='7'
    assert second['index']=='0'
    server.release_gpu('7');server.release_gpu('0')

def test_bad_upload_and_center():
    c,h=client()
    assert c.post('/api/upload',data={'file':(io.BytesIO(b'x'),'bad.exe')},headers=h).status_code==400
    jid=upload(c,h)
    assert c.post('/api/jobs/'+jid+'/prepare',json={'center':[999,0,0],'mode':'local','length':10},headers=h).status_code==400

def test_download_requires_success_and_owned_task():
    c,h=client();jid=upload(c,h)
    p=server.JOBS/jid;d=json.loads((p/'state.json').read_text());d.update(status='succeeded',revision='fixture')
    result=p/'result_fixture';result.mkdir();content=trimesh.creation.box().export(file_type='stl');(result/'crown_smooth_world.stl').write_bytes(content);server.write(jid,d)
    assert c.get('/api/jobs/'+jid+'/files/crown_smooth_world.stl?download=1').data==content
    d['status']='empty';server.write(jid,d)
    assert c.get('/api/jobs/'+jid+'/files/crown_smooth_world.stl').status_code==404

def test_cube_matches_explicit_frame_without_padding():
    from geometry import prepare
    mesh=trimesh.creation.box(extents=(10,12,14))
    # 保留穿过立方体的表面；用一个较小盒子横跨裁剪边界。
    mesh=trimesh.creation.box(extents=(4,8,6))
    crop,frame=prepare(mesh,[0,0,0],'cube',6)
    assert frame['cube_side_world']==6
    assert np.allclose(crop.extents,[4,6,6])
    assert np.max(np.abs((crop.vertices-np.array(frame['center_world']))/frame['cube_side_world']))<=.500001

def test_saved_candidate_requires_exact_scan_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'PIPELINE',tmp_path/'app')
    folder=tmp_path/'production_cases';(folder/'candidates/case1').mkdir(parents=True)
    source=tmp_path/'case1.glb';source.write_bytes(b'original scan')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    (folder/'catalog.json').write_text(json.dumps({'cases':[{'case':'case1','sha256':digest}]}))
    (folder/'candidates/case1/frame.json').write_text(json.dumps({'version':2,'gt_used':False}))
    assert server.saved_candidate(source)['case']=='case1'
    source.write_bytes(b'different scan with same filename')
    assert server.saved_candidate(source) is None
