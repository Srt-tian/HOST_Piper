"""Install reviewed canonical references for alignment evaluation, without labels."""
import argparse
import json
from pathlib import Path


def prepare(root, review):
    root = Path(root).resolve()
    groups = {}
    for split in ('train', 'val'):
        for path in json.loads((root/split/'piper_video_paths.json').read_text()):
            ep = Path(path)
            if not ep.resolve().is_relative_to(root):
                raise ValueError('Episode outside visual dataset')
            prov = json.loads((ep/'provenance.json').read_text())
            groups.setdefault((split, prov['task']), []).append(ep)
    selected = {}
    for entry in review['groups']:
        key = entry['split'], entry['task']
        if entry.get('reviewed') is not True or not entry.get('anchor'):
            raise ValueError('All canonical anchors require explicit visual review')
        if key in selected or key not in groups or Path(entry['anchor']) not in groups[key]:
            raise ValueError('Duplicate, unknown or cross-split/task anchor')
        selected[key] = entry['anchor']
    if set(selected) != set(groups):
        raise ValueError('Missing canonical anchor group')
    pending = {}
    for key, episodes in groups.items():
        for ep in episodes:
            dest = ep/'task_paths_eval.json'
            if dest.exists():
                raise ValueError('Never overwrite existing evaluation references')
            pending[dest] = dict(same=[selected[key]], canonical_anchor_reviewed=True)
    for dest, contents in pending.items():
        with dest.open('x') as handle:
            json.dump(contents, handle)
    return dict(episodes=len(pending), canonical_groups=len(selected), progress_generated=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--review', required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, json.loads(Path(args.review).read_text()))))
