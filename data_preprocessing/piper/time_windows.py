"""Measure gap-free action-window availability without editing/interpolating data.

This is a candidate-window primitive/audit, NOT yet wired into HOST's sampler.
Static-frame removal and variable action spacing need a separate integration test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def valid_starts(timestamps, frames, fps=30, max_gap_periods=1.5):
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if frames < 2 or fps <= 0 or max_gap_periods <= 0:
        raise ValueError('Positive fps/gap threshold and at least two frames required')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must be finite and strictly increasing')
    if len(times) < frames:
        return np.array([], dtype=np.int64)
    bad = np.diff(times) > max_gap_periods/fps+1e-7
    prefix = np.concatenate(([0], np.cumsum(bad)))
    starts = np.arange(len(times)-frames+1)
    return starts[(prefix[starts+frames-1]-prefix[starts]) == 0]


def audit(manifest, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new audit path')
    tasks = {}
    for source in manifest['sources']:
        root = Path(source['root'])
        info = json.loads((root/'meta/info.json').read_text())
        task = tasks.setdefault(source['task'], {})
        for line in (root/'meta/episodes.jsonl').read_text().splitlines():
            if not line:
                continue
            ep = json.loads(line)
            fmt = dict(episode_index=ep['episode_index'], episode_chunk=ep['episode_index']//info['chunks_size'])
            times = pq.read_table(root/info['data_path'].format(**fmt), columns=['timestamp'])['timestamp'].to_pylist()
            #31 covers30 adjacent intervals;61 covers60. These are audit spans, not
            #a claim that the released sampler always requests exactly these indices.
            for frames in (31, 61, 121):
                row = task.setdefault(str(frames), dict(episodes=0, possible_windows=0, valid_windows=0, episodes_with_valid_windows=0))
                count = len(valid_starts(times, frames, info['fps']))
                row['episodes'] += 1
                row['possible_windows'] += max(0, len(times)-frames+1)
                row['valid_windows'] += count
                row['episodes_with_valid_windows'] += int(count > 0)
    for task in tasks.values():
        for row in task.values():
            row['retained_fraction'] = row['valid_windows']/max(1,row['possible_windows'])
    result = dict(tasks=tasks, threshold='No adjacent timestamp gap >1.5/fps',
                  sampler_integrated=False, raw_data_modified=False)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    audit(json.loads(Path(args.manifest).read_text()), args.output)
