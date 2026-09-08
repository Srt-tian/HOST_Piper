"""Build a small split-preserving, verified-video lemon alignment pilot on IDC.

No action conversion, labels, GPU work or network. Sequential cache creation only.
Existing raw/full datasets stay immutable; a failed new output is kept for diagnosis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_seek_safe import transcode
from convert_lerobot import CAMERAS, read_json, write_json


def select(episodes, anchor, limit):
    if limit < 2 or limit > len(episodes):
        raise ValueError('Need at least two episodes and sufficient split-local data')
    by_path = {ep['source_path']: ep for ep in episodes}
    if anchor not in by_path:
        raise ValueError('Reviewed anchor is not in the selected task/split')
    others = sorted((ep for ep in episodes if ep['source_path'] != anchor),
                    key=lambda ep: hashlib.sha256(ep['source_path'].encode()).hexdigest())
    return [by_path[anchor]]+others[:limit-1]


def prepare(root, review_path, output, train_count=12, val_count=4, gop=16):
    root, output = Path(root).resolve(), Path(output)
    data_root = Path('/pfs/user/data/host_piper_paper').resolve()
    if not root.is_relative_to(data_root) or not output.is_absolute() or not output.resolve().is_relative_to(data_root):
        raise ValueError('Pilot must stay under the IDC project data root')
    if output.exists():
        raise ValueError('Use a new pilot directory')
    review = read_json(review_path)
    selected, anchors = [], {}
    for split, count in [('train', train_count), ('val', val_count)]:
        group = [g for g in review['groups'] if (g['task'], g['split']) == ('lemon', split)]
        if len(group) != 1 or group[0]['reviewed'] is not True:
            raise ValueError('Missing reviewed lemon anchor')
        pool = []
        for path in read_json(root/split/'piper_video_paths.json'):
            prov = read_json(Path(path)/'provenance.json')
            if prov['task'] == 'lemon':
                if prov['split'] != split or prov['frames'] < 96:
                    raise ValueError('Unexpected split or too-short episode; review manually')
                pool.append(dict(prov, source_path=path))
        members = select(pool, group[0]['anchor'], count)
        for ep in members:
            ep['output'] = str(output/'episodes'/split/ep['source_id']/f"episode_{ep['episode_index']:06d}")
        anchors[split] = members[0]['output']
        selected.extend(members)
    # Conservative bound before writing; source frames retained at native resolution.
    estimated_bound = sum(ep['frames'] for ep in selected)*3*640*480*3
    if shutil.disk_usage(data_root).free < estimated_bound+10*1024**3:
        raise ValueError('Insufficient space for bounded pilot')
    write_json(output/'pilot_plan.json', dict(task='lemon', gop=gop, episodes=selected,
        raw_rgb_bytes_upper_estimate=estimated_bound, source_review=str(review_path), training_ready=False))
    verified = 0
    with (output/'verified_videos.jsonl').open('x') as ledger:
        for ep in selected:
            dest = Path(ep['output'])
            dest.mkdir(parents=True, exist_ok=True)
            for cam in CAMERAS:
                original = Path(ep['source_path'])/f'{cam}.mp4'
                result = transcode(original.resolve(), dest/f'{cam}.mp4', gop)
                if result['frames'] != ep['frames']:
                    raise ValueError('Verified video does not match trajectory metadata')
                result.update(episode=str(dest), camera=cam, source_episode=ep['source_path'])
                ledger.write(json.dumps(result)+'\n')
                ledger.flush()
                verified += 1
                print(json.dumps(dict(verified_videos=verified, total_videos=len(selected)*3,
                                     bytes=result['cache_bytes'], seconds=result['elapsed_seconds'])), flush=True)
            # Avoid claiming that the original raw video paths are certified outputs.
            provenance = dict(ep, videos={cam: str(dest/f'{cam}.mp4') for cam in CAMERAS},
                              seek_safe=True, visual_only=True)
            write_json(dest/'provenance.json', provenance)
            with (dest/'instruction.txt').open('x') as handle:
                handle.write(ep['instruction']+'\n')
            write_json(dest/'task_paths.json', {'same': [peer['output'] for peer in selected
                if peer['split'] == ep['split'] and peer['output'] != ep['output']]})
            write_json(dest/'task_paths_eval.json', dict(same=[anchors[ep['split']]], canonical_anchor_reviewed=True))
    for split in ('train', 'val'):
        write_json(output/split/'piper_video_paths.json', [ep['output'] for ep in selected if ep['split'] == split])
    write_json(output/'cam_mapping/piper_cam_mapping.json', {str(Path(ep['output']).parent): list(CAMERAS) for ep in selected})
    write_json(output/'anchor_review.json', dict(pilot_only=True, groups=[dict(task='lemon', split=s,
        reviewed=True, anchor=anchor, source_review=str(review_path)) for s, anchor in anchors.items()]))
    write_json(output/'manifest.json', read_json(root/'manifest.json'))
    summary = dict(episodes=len(selected), train=train_count, val=val_count, videos=verified,
        frames=sum(ep['frames'] for ep in selected), seek_safe=True, visual_only=True,
        progress_generated=False, training_ready=False, gop=gop)
    write_json(output/'preparation_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'review', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--train-count', type=int, default=12)
    p.add_argument('--val-count', type=int, default=4)
    p.add_argument('--gop', type=int, choices=(1, 16), default=16)
    args = p.parse_args()
    prepare(args.root, args.review, args.output, args.train_count, args.val_count, args.gop)
