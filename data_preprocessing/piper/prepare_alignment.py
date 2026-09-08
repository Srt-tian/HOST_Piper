"""Prepare a visual-only HOST dataset without assuming EEF/action semantics.

Uses the policy converter's deterministic split. Raw videos are symlinked, never
rewritten. Anchor candidates require visual review; no progress labels are made.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_lerobot import CAMERAS, assign_splits, read_json, write_json


def prepare(manifest, output):
    output = Path(output)
    if not output.is_absolute() or output.exists():
        raise ValueError('Output must be a new absolute directory')
    episodes, seen = [], set()
    for source in manifest['sources']:
        source_id = source['id']
        if source_id in seen or not source_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Invalid/duplicate source id')
        seen.add(source_id)
        root = Path(source['root']).resolve()
        info = read_json(root/'meta/info.json')
        if info['fps'] != 30 or info['codebase_version'] != 'v2.1':
            raise ValueError('Expected native 30Hz LeRobot v2.1')
        bad = root/'meta/is_bad_data.jsonl'
        if bad.exists() and bad.read_text().strip():
            raise ValueError(f'Review quality marks first: {bad}')
        if not source['task'] or not source['instruction']:
            raise ValueError('Missing semantic task/instruction')
        for line in (root/'meta/episodes.jsonl').read_text().splitlines():
            if not line:
                continue
            meta = json.loads(line)
            idx = meta['episode_index']
            fmt = dict(episode_index=idx, episode_chunk=idx//info['chunks_size'])
            videos = {cam: str(root/info['video_path'].format(**fmt,
                      video_key=f'observation.images.{cam}')) for cam in CAMERAS}
            if not all(Path(video).is_file() for video in videos.values()):
                raise ValueError(f'Missing video: {source_id}/{idx}')
            episodes.append(dict(source_id=source_id, episode_index=idx,
                task=source['task'], instruction=source['instruction'],
                frames=meta['length'], videos=videos, visual_only=True))
    assign_splits(episodes, manifest.get('val_fraction', 0.1), manifest.get('seed', 42))
    groups = {}
    for ep in episodes:
        ep['output'] = str(output/'episodes'/ep['split']/ep['source_id']/f"episode_{ep['episode_index']:06d}")
        groups.setdefault((ep['split'], ep['task']), []).append(ep)
    anchor_plan = []
    for (split, task), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda ep: (ep['frames'], ep['source_id'], ep['episode_index']))
        median = ordered[len(ordered)//2]['frames']
        candidates = sorted(group, key=lambda ep: (abs(ep['frames']-median), ep['output']))[:5]
        anchor_plan.append(dict(split=split, task=task, reviewed=False, anchor=None,
            candidates=[dict(path=ep['output'], frames=ep['frames']) for ep in candidates],
            review='Check successful completion, full visibility and compatible manipulation order; length is NOT a quality label.'))
        for ep in group:
            dest = Path(ep['output'])
            write_json(dest/'provenance.json', ep)
            with (dest/'instruction.txt').open('x') as handle:
                handle.write(ep['instruction']+'\n')
            for cam, video in ep['videos'].items():
                (dest/f'{cam}.mp4').symlink_to(video)
            write_json(dest/'task_paths.json', {'same': [peer['output'] for peer in group if peer is not ep]})
    for split in ('train', 'val'):
        write_json(output/split/'piper_video_paths.json', [ep['output'] for ep in episodes if ep['split'] == split])
    write_json(output/'cam_mapping/piper_cam_mapping.json', {
        str(Path(ep['output']).parent): list(CAMERAS) for ep in episodes})
    write_json(output/'anchor_review.json', {'groups': anchor_plan})
    write_json(output/'manifest.json', manifest)
    summary = dict(episodes=len(episodes), frames=sum(ep['frames'] for ep in episodes),
        splits={s: sum(ep['split'] == s for ep in episodes) for s in ('train', 'val')},
        groups=len(groups), visual_only=True, progress_generated=False,
        source_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    write_json(output/'preparation_summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(prepare(read_json(args.manifest), args.output)))
