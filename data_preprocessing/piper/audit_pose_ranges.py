"""Read-only numeric evidence for EEF conventions; never auto-confirms semantics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WIDTHS = {'state_end': 12, 'action_end': 12, 'observation.qpos': 14, 'real_action': 14}


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected a nonempty finite matrix')
    return dict(min=a.min(0).tolist(), p01=np.quantile(a, .01, axis=0).tolist(),
                median=np.median(a, axis=0).tolist(), p99=np.quantile(a, .99, axis=0).tolist(),
                max=a.max(0).tolist())


def pose_summary(a):
    result = {'per_dimension': stats(a)}
    for arm, start in [('left', 0), ('right', 6)]:
        xyz, rpy = a[:, start:start+3], a[:, start+3:start+6]
        result[arm] = dict(xyz_min=float(xyz.min()), xyz_max=float(xyz.max()),
            position_norm_p99=float(np.quantile(np.linalg.norm(xyz, axis=1), .99)),
            rpy_min=float(rpy.min()), rpy_max=float(rpy.max()),
            angle_abs_p99=float(np.quantile(np.abs(rpy), .99)),
            fraction_angles_above_pi=float((np.abs(rpy) > np.pi+1e-5).mean()))
    return result


def audit(manifest_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh output path')
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    report = dict(scope='All numeric rows, no FK or collector-code verification',
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        conclusions=dict(working_position_unit='m', working_angle_unit='rad',
            unit_choice_origin='User-authorized provisional assumption, inspect statistics',
            convention_verified=False, rpy_order_verified=False, tcp_verified=False,
            command_semantics_verified=False), sources=[])
    for source in manifest['sources']:
        root = Path(source['root'])
        info = json.loads((root/'meta/info.json').read_text())
        episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line]
        fields = {key: [] for key in WIDTHS}
        deltas = {'state_end': [], 'action_end': []}
        identities, records = [], []
        for episode in episodes:
            index = episode['episode_index']
            path = root/info['data_path'].format(episode_index=index, episode_chunk=index//info['chunks_size'])
            table = pq.read_table(path, columns=list(WIDTHS)+['action', 'timestamp'])
            n = len(table)
            if n != episode['length']:
                raise ValueError(f'Frame count mismatch: {path}')
            arrays = {}
            for key, width in WIDTHS.items():
                a = np.asarray(table[key].to_pylist(), dtype=np.float64)
                if a.shape != (n, width) or not np.isfinite(a).all():
                    raise ValueError(f'Invalid numeric field: {path}: {key}')
                arrays[key] = a
                fields[key].append(a)
                if key in deltas and n > 1:
                    deltas[key].append(np.abs(np.diff(a, axis=0)))
            times = np.asarray(table['timestamp'].to_pylist(), dtype=np.float64).reshape(-1)
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError(f'Invalid timestamp: {path}')
            action = np.asarray(table['action'].to_pylist(), dtype=np.float64)
            identities.append(bool(np.array_equal(action, arrays['observation.qpos'])))
            records.append(dict(episode=index, frames=n, parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        merged = {key: np.concatenate(value) for key, value in fields.items()}
        diff = merged['action_end']-merged['state_end']
        row = dict(source=source['id'], task=source['task'], episodes=len(episodes),
            frames=len(diff), poses={key: pose_summary(merged[key]) for key in deltas},
            joints={key: stats(merged[key]) for key in ('observation.qpos', 'real_action')},
            absolute_row_delta={key: stats(np.concatenate(value)) for key, value in deltas.items() if value},
            action_exactly_equals_qpos=all(identities),
            action_end_minus_state_end_raw_mae=np.abs(diff).mean(0).tolist(),
            action_end_exactly_equals_state_end_fraction=float(np.all(diff == 0, axis=1).mean()),
            note='RPY subtraction/deltas are raw component diagnostics, NOT rotation distance; no lag or FK inference.',
            records=records)
        report['sources'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k in ('source','task','episodes','frames','poses',
            'action_exactly_equals_qpos','action_end_exactly_equals_state_end_fraction')}, ensure_ascii=False), flush=True)
    report['total_frames'] = sum(s['frames'] for s in report['sources'])
    report['total_episodes'] = sum(s['episodes'] for s in report['sources'])
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), total_frames=report['total_frames'],
        total_episodes=report['total_episodes'], convention_verified=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.manifest, args.output)
