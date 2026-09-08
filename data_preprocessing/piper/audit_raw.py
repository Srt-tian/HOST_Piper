"""Read-only schema/numeric audit after download; does not infer command semantics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import av


def audit(manifest):
    results = []
    for source in manifest['sources']:
        root = Path(source['root'])
        info = json.loads((root/'meta/info.json').read_text())
        episodes = [json.loads(l) for l in (root/'meta/episodes.jsonl').read_text().splitlines() if l]
        report = dict(source=source['id'], episodes=len(episodes), frames=0, videos=0,
                      action_equals_qpos=True, max_timestamp_gap=0, video_length_mismatches=[])
        bad_file = root/'meta/is_bad_data.jsonl'
        report['quality_mark_records'] = len(bad_file.read_text().splitlines()) if bad_file.exists() else None
        for ep in episodes:
            idx = ep['episode_index']
            fmt = dict(episode_index=idx, episode_chunk=idx//info['chunks_size'])
            path = root/info['data_path'].format(**fmt)
            table = pq.read_table(path)
            if len(table) != ep['length']:
                raise ValueError(f'Parquet length mismatch: {path}')
            arrays = {}
            for key, width in [('state_end',12),('action_end',12),('observation.qpos',14),('action',14),('real_action',14)]:
                a = np.asarray(table[key].to_pylist(), dtype=float)
                if a.shape != (len(table),width) or not np.isfinite(a).all():
                    raise ValueError(f'Bad numeric field: {path}: {key}')
                arrays[key] = a
            report['action_equals_qpos'] &= bool(np.array_equal(arrays['action'], arrays['observation.qpos']))
            times = np.asarray(table['timestamp'].to_pylist(), dtype=float).reshape(-1)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f'Bad timestamps: {path}')
            report['max_timestamp_gap'] = max(report['max_timestamp_gap'], float(np.diff(times).max()))
            report['frames'] += len(table)
            for key, feature in info['features'].items():
                if feature.get('dtype') != 'video':
                    continue
                video = root/info['video_path'].format(**fmt, video_key=key)
                with av.open(str(video)) as c:
                    # Fast container count; full decode remains a conversion-time check.
                    frames = c.streams.video[0].frames
                    first = next(c.decode(video=0), None)
                if frames != len(table) or first is None:
                    report['video_length_mismatches'].append(dict(video=str(video), frames=frames, rows=len(table)))
                report['videos'] += 1
        results.append(report)
        print(json.dumps(report), flush=True)
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    args = p.parse_args()
    results = audit(json.loads(Path(args.manifest).read_text()))
    if any(r['video_length_mismatches'] for r in results):
        raise SystemExit(2)
