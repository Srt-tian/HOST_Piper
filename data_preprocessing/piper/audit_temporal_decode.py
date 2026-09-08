"""Read-only timing audit and bounded Decord seek-vs-sequential pilot on IDC."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def timing(times, fps=30):
    times = np.asarray(times, dtype=float).reshape(-1)
    if len(times) < 2 or not np.isfinite(times).all():
        raise ValueError('Invalid timestamp sequence')
    delta = np.diff(times)
    if np.any(delta <= 0):
        raise ValueError('Non-increasing timestamps')
    return dict(frames=len(times), duration=float(times[-1]-times[0]),
                max_gap=float(delta.max()), gaps_over_1p5_period=int((delta > 1.5/fps).sum()),
                estimated_missing_periods=int(np.maximum(np.rint(delta*fps)-1, 0).sum()),
                index_clock_drift_seconds=float(times[-1]-times[0]-(len(times)-1)/fps))


def compare_seek(video):
    import decord
    reader = decord.VideoReader(str(video), width=224, height=224, num_threads=1)
    count = len(reader)
    indices = sorted(set(np.linspace(0, count-1, min(count, 17), dtype=int).tolist()))
    selected = set(indices)
    sequential = {}
    for i in range(count):
        frame = reader.next()
        if i in selected:
            sequential[i] = frame.asnumpy().copy()
    del reader
    comparisons = []
    # Fresh reader for each access pattern; tests both monotonic and backwards seeking.
    for order in (indices, list(reversed(indices))):
        reader = decord.VideoReader(str(video), width=224, height=224, num_threads=1)
        batch = reader.get_batch(order).asnumpy()
        for index, frame in zip(order, batch):
            error = np.abs(frame.astype(np.int16)-sequential[index].astype(np.int16))
            comparisons.append(dict(index=index, mae=float(error.mean()), max_error=int(error.max())))
        del reader
    return dict(video=str(video), frames=count, sampled_indices=indices,
                exact_match=all(c['max_error'] == 0 for c in comparisons),
                max_mae=max(c['mae'] for c in comparisons), comparisons=comparisons)


def audit(manifest, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new report path')
    results = dict(timing=[], decode=[], scope='All trajectory timestamps; first episode / all cameras per source only for decode')
    for source in manifest['sources']:
        root = Path(source['root'])
        info = json.loads((root/'meta/info.json').read_text())
        episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line]
        for ep in episodes:
            fmt = dict(episode_index=ep['episode_index'], episode_chunk=ep['episode_index']//info['chunks_size'])
            path = root/info['data_path'].format(**fmt)
            table = pq.read_table(path, columns=['timestamp'])
            result = dict(source=source['id'], task=source['task'], episode_index=ep['episode_index'],
                          **timing(table['timestamp'].to_pylist(), info['fps']))
            results['timing'].append(result)
        fmt = dict(episode_index=episodes[0]['episode_index'], episode_chunk=episodes[0]['episode_index']//info['chunks_size'])
        for cam in source['cameras']:
            video = root/info['video_path'].format(**fmt, video_key=cam)
            result = dict(source=source['id'], camera=cam, **compare_seek(video))
            results['decode'].append(result)
            print(json.dumps({k:v for k,v in result.items() if k != 'comparisons'}), flush=True)
    results['summary'] = dict(episodes=len(results['timing']),
        episodes_with_gaps=sum(r['gaps_over_1p5_period'] > 0 for r in results['timing']),
        max_index_clock_drift_seconds=max(abs(r['index_clock_drift_seconds']) for r in results['timing']),
        pilot_videos=len(results['decode']), seek_mismatch_videos=sum(not r['exact_match'] for r in results['decode']))
    with output.open('x') as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    print(json.dumps(results['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    audit(json.loads(Path(args.manifest).read_text()), args.output)
