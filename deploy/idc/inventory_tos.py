"""Read one TOS directory level at a time; stage metadata and a small pilot on IDC.

Uses configured eip-kit authentication only, serial operations, stops on failures.
Does not recursively download task roots (annotation videos are out of scope).
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

BASE = Path('/pfs/user/data/host_piper_paper')
ROOTS = {
    'bottles': ('瓶子立起-松灵臂-纯遥操-主从臂', 'Stand the bottles upright on the table.'),
    'pens': ('笔筒装笔-松灵臂-纯遥操-主从臂', 'Put the pens into the pen holder.'),
    'basket': ('物件入筐-松灵臂-纯遥操-主从臂', 'Put the objects into the basket.'),
    'lemon': ('把柠檬放到盘子里-纯遥操-松灵臂', 'Put the lemon onto the plate.'),
}


def call(*args):
    p = subprocess.run(['eip-kit', *args], text=True, capture_output=True)
    if p.returncode:
        # Do not emit raw authentication output or retry access-denied errors.
        if 'auth_failed' in p.stdout+p.stderr or 'Missing token' in p.stdout+p.stderr:
            subprocess.run(['eip-kit', 'auth', 'check'], check=True, capture_output=True)
            p = subprocess.run(['eip-kit', *args], text=True, capture_output=True)
        if p.returncode:
            raise RuntimeError(f'eip-kit failed ({p.returncode}) during {args[0:2]}; stopped, no retry')
    return p.stdout


def listing(url):
    out = call('tos', 'ls', url)
    folders = [s.strip() for s in out.splitlines() if s.strip().startswith(url) and s.strip().endswith('/')]
    # Human-readable sizes are estimates, not exact object checksums.
    sizes = re.findall(r'\d{4}-\d\d-\d\dT\S+\s+(\d+(?:\.\d+)?)(B|KB|MB|GB)\s', out)
    size = sum(float(n) * {'B':1,'KB':1024,'MB':1024**2,'GB':1024**3}[u] for n,u in sizes)
    return folders, size


def copy(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    call('tos', 'cp', url, str(dest))


def main(pilot):
    if subprocess.check_output(['hostname'], text=True).strip() != 'dev-instance-shenrongtian':
        raise RuntimeError('IDC hostname mismatch')
    call('auth', 'check')
    sources = []
    for task, (root_name, instruction) in ROOTS.items():
        task_root = 'tos://magiclab/datasight/dataset/'+root_name+'/'
        task_dirs, _ = listing(task_root)
        for task_dir in task_dirs:
            sessions, _ = listing(task_dir)
            for session in sessions:
                source_id = task+'_'+task_dir.rstrip('/').split('/')[-1]+'_'+session.rstrip('/').split('/')[-1]
                local = BASE/'raw'/source_id
                for filename in ('info.json','episodes.jsonl','tasks.jsonl'):
                    copy(session+'meta/'+filename, local/'meta'/filename)
                info = json.loads((local/'meta/info.json').read_text())
                camera_keys = [k for k,v in info['features'].items() if v.get('dtype') == 'video']
                size = 0
                for chunk in range(info['total_chunks']):
                    for camera in camera_keys:
                        _, s = listing(session+f'videos/chunk-{chunk:03d}/'+camera+'/')
                        size += s
                record = dict(id=source_id, task=task, instruction=instruction, tos=session,
                    root=str(local), total_episodes=info['total_episodes'], total_frames=info['total_frames'],
                    fps=info['fps'], cameras=camera_keys, estimated_video_bytes=round(size),
                    features={k:v.get('shape') for k,v in info['features'].items() if v.get('dtype') != 'video'})
                sources.append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in ('features','cameras')}, ensure_ascii=False), flush=True)
                if pilot:
                    first = json.loads((local/'meta/episodes.jsonl').read_text().splitlines()[0])
                    idx = first['episode_index']
                    fmt = dict(episode_index=idx, episode_chunk=idx//info['chunks_size'])
                    rel = info['data_path'].format(**fmt)
                    copy(session+rel, local/rel)
                    for camera in camera_keys:
                        rel = info['video_path'].format(**fmt, video_key=camera)
                        copy(session+rel, local/rel)
    manifest = dict(seed=42, val_fraction=0.1, conventions=dict(confirmed=False,
        position_unit=None, angle_unit=None, rpy=None, end_frame=None,
        target_source=None, command_semantics_confirmed=False), sources=sources)
    path = BASE/'inventory.json'
    with path.open('x') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(sources=len(sources), episodes=sum(s['total_episodes'] for s in sources),
        frames=sum(s['total_frames'] for s in sources), estimated_video_bytes=sum(s['estimated_video_bytes'] for s in sources)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    main(parser.parse_args().pilot)
