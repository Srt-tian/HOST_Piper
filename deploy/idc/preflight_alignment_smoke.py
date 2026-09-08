"""No-GPU artifact gate for the Piper EIP alignment smoke. Never downloads files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from fetch_alignment_weights import REVISION, digest


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check(root, weights, output):
    root, weights, output = map(lambda p: Path(p).resolve(), (root, weights, output))
    data = Path('/pfs/user/data/host_piper_paper').resolve()
    errors = []
    if any(not p.is_relative_to(data) for p in (root, weights, output)):
        raise ValueError('All artifacts must remain in the isolated IDC HOST data root')
    if output.exists():
        errors.append('Output already exists; a new smoke directory is required')
    ready = json.loads((root/'preparation_summary.json').read_text())
    if ready.get('seek_safe') is not True or ready.get('visual_only') is not True:
        errors.append('Completed seek-safe visual dataset required')
    videos = [json.loads(line) for line in (root/'verified_videos.jsonl').read_text().splitlines() if line]
    destinations = set()
    for row in videos:
        dest = Path(row['destination']).resolve()
        if not dest.is_relative_to(root) or dest in destinations:
            raise ValueError('Duplicate or escaped video certificate')
        destinations.add(dest)
        if row.get('seek_exact_match') is not True or not dest.is_file() or sha256(dest) != row['cache_file_sha256']:
            errors.append(f'Invalid derived video certificate: {dest.name}')
    if len(videos) != ready['videos'] or len(videos) != 3*ready['episodes']:
        errors.append('Unexpected video certificate coverage')
    for split in ('train','val'):
        paths = json.loads((root/split/'piper_video_paths.json').read_text())
        if len(paths) != ready[split]:
            errors.append(f'Episode list count mismatch: {split}')
        for entry in paths:
            ep = Path(entry['video_dir'] if isinstance(entry, dict) else entry).resolve()
            if not ep.is_relative_to(root):
                raise ValueError('Episode escaped the certified pilot')
            for camera in ('cam_front','cam_left','cam_right'):
                if (ep/f'{camera}.mp4').resolve() not in destinations:
                    errors.append(f'Uncertified episode video: {ep.name}/{camera}')
    plan = json.loads((weights/'transfer_plan.json').read_text())
    if plan['revision'] != REVISION or plan['output'] != str(weights):
        raise ValueError('Unexpected pinned weight plan')
    marker = weights/'verified_artifacts.json'
    missing = [row['name'] for row in plan['files'] if not (weights/row['name']).is_file()]
    if missing:
        errors.append('Incomplete pretrained model: '+', '.join(missing))
    if not marker.is_file():
        errors.append('Missing final weight verification marker')
    else:
        m = json.loads(marker.read_text())
        if m.get('verified') is not True or m.get('revision') != REVISION:
            errors.append('Invalid final weight verification marker')
    if not missing:
        for row in plan['files']:
            if not digest(weights/row['name'], row):
                errors.append('Weight integrity failure: '+row['name'])
    free = shutil.disk_usage(data).free
    if free < 250*1024**3:
        errors.append('Reserve at least250GiB for full smoke checkpoint, replay and temporary files')
    return dict(ready=not errors, errors=errors, root=str(root), weights=str(weights),
        output=str(output), certified_videos=len(videos), free_bytes=free,
        inputs='visual-only; action units/norm stats/DTW progress labels not required',
        production_ready=False, weights_complete=not missing)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','weights','output'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    result = check(args.root, args.weights, args.output)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['ready'] else 2)
