"""Explicitly adapt ONLY camera/position embeddings of a HOST checkpoint on CPU.

Map target cameras to known source camera indices; null initializes a new camera.
If source camera identities are unknown use all null, not an invented semantic mapping.
Never alters action/proprio layers or overwrites the source checkpoint.
"""
import argparse
import json
from pathlib import Path
import torch


def adapt_visual_state(state, camera_map, seed=42):
    out = dict(state)
    camera, pos = state['camera_emb'], state['pos_embed']
    if camera.ndim != 2 or pos.ndim != 3 or pos.shape[0] != 1:
        raise ValueError('Unexpected camera embedding shape')
    nsrc = camera.shape[0]
    if not camera_map or pos.shape[1] % nsrc:
        raise ValueError('Invalid number of cameras/patch tokens')
    patches = pos.shape[1] // nsrc
    if any(i is not None and (type(i) is not int or not 0 <= i < nsrc) for i in camera_map):
        raise ValueError('camera_map entries must be valid source indices or null')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new_cam = torch.empty(len(camera_map), camera.shape[1], dtype=camera.dtype)
        new_pos = torch.empty(1, len(camera_map)*patches, pos.shape[2], dtype=pos.dtype)
        torch.nn.init.trunc_normal_(new_cam, std=0.02)
        torch.nn.init.trunc_normal_(new_pos, std=0.02)
    for target, source in enumerate(camera_map):
        if source is not None:
            new_cam[target] = camera[source]
            new_pos[:, target*patches:(target+1)*patches] = pos[:, source*patches:(source+1)*patches]
    out['camera_emb'], out['pos_embed'] = new_cam, new_pos
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--camera-map', required=True, help='JSON, e.g. [null,null,null]')
    args = p.parse_args()
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if target.exists() or source == target:
        raise ValueError('Output must be a new file distinct from input')
    payload = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    mapping = json.loads(args.camera_map)
    payload['visual_encoder'] = adapt_visual_state(payload['visual_encoder'], mapping)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        torch.save(payload, f)
    with Path(str(target)+'.adaptation.json').open('x') as f:
        json.dump(dict(source=str(source), camera_map=mapping,
            modified_keys=['visual_encoder.camera_emb','visual_encoder.pos_embed']), f, indent=2)
