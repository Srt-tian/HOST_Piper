"""Strict 3-camera coupling adapter for official alignment evaluation records.

No model calls, no linear-index fallback. Writes only into a new artifact directory;
never modifies raw episodes. Requires explicit reviewed common anchors and model ID.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = ('cam_front', 'cam_left', 'cam_right')


def anchor_frames(paths, episode, frames, context_frames):
    if context_frames not in (1, 2):
        raise ValueError('Explicit 1 or 2 temporal context frames required')
    size = len(CAMERAS)*context_frames
    if not paths or len(paths) % size:
        raise ValueError('Frame path count is inconsistent with three-camera group size')
    result = []
    # Resolve episode directory only, NOT its camera symlinks (which point to raw data).
    episode = Path(episode).resolve()
    for start in range(0, len(paths), size):
        times = []
        for offset in range(context_frames):
            view_times = []
            for v, cam in enumerate(CAMERAS):
                path = paths[start+offset*3+v]
                if not path.startswith('video://'):
                    raise ValueError('Piper adapter requires explicit video:// frame paths')
                parts = path[8:].split('::')
                if len(parts) not in (2, 3):
                    raise ValueError('Malformed frame path')
                video, index = Path(parts[0]), int(parts[1])
                if video.parent.resolve() != episode or video.name != cam+'.mp4':
                    raise ValueError('Wrong episode/camera order in alignment record')
                if not 0 <= index < frames:
                    raise ValueError('Out-of-range video frame')
                view_times.append(index)
            if len(set(view_times)) != 1:
                raise ValueError('Cameras disagree on anchor timestamp/frame index')
            times.append(view_times[0])
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Reversed temporal context')
        result.append(times[-1])
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError('Anchor frames must be strictly increasing')
    return np.asarray(result, dtype=int)


def align_record(record, main_frames, ref_frames, context_frames):
    loss = record.get('loss')
    if loss is None or not np.isfinite(loss) or loss > 0.15:
        raise ValueError('Missing/nonfinite/high alignment loss')
    main = anchor_frames(record['frame_paths'], record['main_video_path'], main_frames, context_frames)
    ref = anchor_frames(record['ref_frame_paths'], record['ref_video_path'], ref_frames, context_frames)
    indices = np.asarray(record['forward_argmax_indices'])
    if indices.shape != main.shape or indices.dtype.kind not in 'iu':
        raise ValueError('Alignment length/type must exactly match anchor groups; no truncation')
    if np.any(indices < 0) or np.any(indices >= len(ref)):
        raise ValueError('Out-of-range correspondence')
    if np.ptp(indices) <= 0.8*len(ref):
        raise ValueError('Insufficient reference coverage')
    if np.any(np.diff(indices) < 0) or np.any(np.diff(indices) > 7/24*len(ref)):
        raise ValueError('Backward alignment or excessive jump')
    values = (ref[indices]+1)/ref_frames
    # Retain repeated correspondence points, preserving observed pauses. The official
    # converter drops those points before interpolation; that difference is explicit.
    dense = np.interp(np.arange(main_frames), main, values)
    return dict(ref_path=record['ref_video_path'],
        aligned_progress={str(i): float(p) for i, p in enumerate(dense)},
        coupling=dict(method='official_forward_argmax_indices', context_frames=context_frames,
            cameras=list(CAMERAS), loss=float(loss), matched_points=len(main),
            main_support=[int(main[0]), int(main[-1])],
            reference_coverage=float(np.ptp(indices)/len(ref)),
            repeated_matches='preserve plateau samples', outside_support='hold endpoint'))


def export(root, records, anchors, output, model_id, context_frames):
    root, output = Path(root).resolve(), Path(output)
    if not output.is_absolute() or output.exists() or not model_id.strip():
        raise ValueError('New absolute output directory and alignment model ID required')
    paths = {}
    for split in ('train', 'val'):
        for path in json.loads((root/split/'piper_video_paths.json').read_text()):
            ep = Path(path)
            if not ep.resolve().is_relative_to(root):
                raise ValueError('Episode outside dataset root')
            provenance = json.loads((ep/'provenance.json').read_text())
            paths[path] = dict(split=split, task=provenance['task'], frames=provenance['frames'])
    canonical = {}
    for group in anchors['groups']:
        key = group['split'], group['task']
        anchor = group['anchor']
        if group['reviewed'] is not True or anchor not in paths:
            raise ValueError('Canonical anchor requires completed visual review')
        if (paths[anchor]['split'], paths[anchor]['task']) != key or key in canonical:
            raise ValueError('Cross-task/split or duplicate canonical anchor')
        canonical[key] = anchor
    pending = {}
    for record in records:
        main, ref = record['main_video_path'], record['ref_video_path']
        if main not in paths or ref not in paths or main in pending:
            raise ValueError('Unknown or duplicate episode in records')
        m, r = paths[main], paths[ref]
        key = m['split'], m['task']
        if (r['split'], r['task']) != key or canonical.get(key) != ref:
            raise ValueError('Record does not use the same-task/split canonical reference')
        label = align_record(record, m['frames'], r['frames'], context_frames)
        label['coupling'].update(alignment_model_id=model_id,
            source_record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest())
        pending[main] = label
    if not pending:
        raise ValueError('No model-derived records supplied')
    # All validation happens before any writes. Incomplete coverage remains explicit;
    # this directory is an artifact, not a training-ready dataset.
    output.mkdir(parents=True)
    for main, label in pending.items():
        dest = output/Path(main).relative_to(root)/'info_dtw.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('x') as handle:
            json.dump(label, handle, allow_nan=False)
    summary = dict(generated=len(pending), expected=len(paths), missing=sorted(set(paths)-set(pending)),
                   training_ready=False, alignment_model_id=model_id)
    with (output/'summary.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'records', 'anchors', 'output', 'model-id'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--context-frames', required=True, type=int, choices=(1, 2))
    args = p.parse_args()
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines() if line]
    print(json.dumps(export(args.root, records, json.loads(Path(args.anchors).read_text()),
                           args.output, args.model_id, args.context_frames)))
