"""Fail closed on missing semantics, real progress, text caches or split-local peers."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def check(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['conventions'].get('confirmed') is not True:
        raise ValueError('Data conventions have not been confirmed')
    all_splits = {s: set(json.loads((root/s/'piper_video_paths.json').read_text())) for s in ('train','val')}
    if all_splits['train'] & all_splits['val']:
        raise ValueError('Train/val episode leakage')
    progress_anchors = {}
    for split, paths in all_splits.items():
        for path in paths:
            ep = Path(path)
            provenance = json.loads((ep/'provenance.json').read_text())
            n = len(json.loads((ep/(ep.name+'.json')).read_text())['data'])
            if any((ep/name).exists() for name in ('bad_info.json','bad_info_dtw.json','bad_action.json')):
                raise ValueError(f'Quality-filtered episode in manifest: {ep}')
            progress_data = json.loads((ep/'info_dtw.json').read_text())
            progress = progress_data['aligned_progress']
            anchor = progress_data.get('ref_path')
            if not anchor or anchor not in paths:
                raise ValueError(f'Unknown or cross-split progress reference: {ep}')
            group = (split, provenance['task'])
            if progress_anchors.setdefault(group, anchor) != anchor:
                raise ValueError(f'Inconsistent progress coordinate references for {group}')
            values = np.array([progress[str(i)] for i in range(n)], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError(f'Invalid progress values: {ep}')
            if (np.diff(values) < -1e-7).any() or np.ptp(values) < 0.05:
                raise ValueError(f'Non-monotonic or degenerate progress: {ep}')
            peers = json.loads((ep/'task_paths.json').read_text())['same']
            if not peers or path in peers or not set(peers) <= paths:
                raise ValueError(f'Invalid/cross-split peers: {ep}')
            for peer in peers:
                other = json.loads((Path(peer)/'provenance.json').read_text())
                if other['task'] != provenance['task']:
                    raise ValueError(f'Cross-task reference: {ep} -> {peer}')
            text = torch.load(ep/'instruction.pt', map_location='cpu', weights_only=True)
            context, mask = text['context'], text['mask']
            if context.ndim != 2 or context.shape[-1] != 4096 or mask.shape != context.shape[:1]:
                raise ValueError(f'Invalid T5 embedding shape: {ep}')
            if not torch.isfinite(context).all() or not mask.any() or context.abs().sum() == 0:
                raise ValueError(f'Empty/nonfinite text embedding: {ep}')
    return {s: len(p) for s,p in all_splits.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    print(json.dumps(check(p.parse_args().root)))
