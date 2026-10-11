"""Verify release bytes; optionally strictly load the C80 checkpoint."""
import argparse, hashlib, json, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent
manifest=json.loads((root/'release_manifest.json').read_text())
for e in manifest['files']:
    path=root/e['path']
    if not path.is_file():raise SystemExit('Missing file: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if path.stat().st_size!=e['size'] or h.hexdigest()!=e['sha256']:raise SystemExit('Checksum mismatch: '+str(path))
print('RELEASE_HASHES_PASS',len(manifest['files']),flush=True)
if args.cpu_only:
    sys.path.insert(0,str(root/'app'))
    from c80_runtime import load_c80
    load_c80(device='cpu')
    print('C80_STRICT_LOAD_PASS',flush=True)
