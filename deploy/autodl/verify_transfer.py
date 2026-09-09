"""Offline transfer-integrity check, no downloads or checkpoint writes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import socket

ROOT = Path(__file__).resolve().parents[2]

def verify(root, weights):
    root, weights = Path(root), Path(weights)
    spec = importlib.util.spec_from_file_location('fetch_alignment_weights', ROOT/'deploy/idc/fetch_alignment_weights.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = json.loads((weights/'transfer_plan.json').read_text())
    if plan['revision'] != module.REVISION or len(plan['files']) != 16:
        raise ValueError('Wrong pinned weight plan')
    for row in plan['files']:
        if not module.digest(weights/row['name'], row):
            raise ValueError('Weight checksum mismatch:'+row['name'])
        print('Verified weight '+row['name'], flush=True)
    audit = json.loads((root/'sequential_decode_verified.json').read_text())
    if not audit['complete'] or len(audit['records']) != 2652:
        raise ValueError('Incomplete decode certificate')
    for row in audit['records']:
        digest = hashlib.sha256()
        with Path(row['path']).open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != row['sha256']:
            raise ValueError('Transferred video checksum mismatch:'+row['path'])
    counts = {}
    for split, expected in [('train',796),('val',88)]:
        entries = json.loads((root/split/'piper_video_paths.json').read_text())
        if len(entries) != expected:
            raise ValueError('Unexpected split size')
        counts[split] = len(entries)
        for entry in entries:
            ep = Path(entry['video_dir'] if isinstance(entry,dict) else entry)
            for filename in ['provenance.json','task_paths.json','task_paths_eval.json']:
                json.loads((ep/filename).read_text())
    if shutil.disk_usage(root).free < 110*1024**3:
        raise ValueError('Insufficient final-checkpoint reserve')
    result = dict(verified=True, hostname=socket.gethostname(), weights=16, videos=2652,
                  splits=counts, free_bytes=shutil.disk_usage(root).free)
    print(json.dumps(result),flush=True)
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--weights',required=True)
    args=parser.parse_args()
    verify(args.root,args.weights)
