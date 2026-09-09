"""CPU-only validation: raw sequential reads and memory-optimization equivalence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'alignment'))


def audit(root):
    import av
    root = Path(root)
    target = root/'sequential_decode_verified.json'
    if target.exists():
        raise ValueError('Audit already exists; do not overwrite')
    jobs = []
    for split in ('train','val'):
        entries = json.loads((root/split/'piper_video_paths.json').read_text())
        for entry in entries:
            ep = Path(entry['video_dir'] if isinstance(entry, dict) else entry)
            frames = json.loads((ep/'provenance.json').read_text())['frames']
            for camera in ('cam_front','cam_left','cam_right'):
                jobs.append((ep/f'{camera}.mp4', frames))
    def check(item):
        path, expected = item
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2), b''):
                digest.update(chunk)
        with av.open(str(path)) as container:
            container.streams.video[0].thread_count = 1
            count = sum(1 for frame in container.decode(video=0))
        if count != expected:
            raise ValueError(f'Decoded frame count mismatch:{path}:{count}!={expected}')
        return dict(path=str(path), frames=count, sha256=digest.hexdigest())
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(check, jobs):
            records.append(record)
            if len(records)%64 == 0:
                print(f'Sequential decode verified {len(records)}/{len(jobs)} videos', flush=True)
    if len(records) != 2652:
        raise ValueError('Expected2652 full-task videos')
    result = dict(complete=True, videos=len(records), episodes=884,
                  decode='PyAV sequential, no seek', records=records)
    with target.open('x') as f:
        json.dump(result,f)
    print('Full sequential decode audit complete',flush=True)


def equivalence(pilot, weights, output):
    import numpy as np
    import torch
    from piper_low_memory import sequential_load_imgs
    from piper_profile import strict_load_imgs
    from piper_smoke import configure, batch, make_model, loss_for
    pilot = Path(pilot)
    rows = [json.loads(x) for x in (pilot/'verified_videos.jsonl').read_text().splitlines() if x]
    pixels = 0
    for row in rows:
        ids = row['sampled_indices']
        raw_paths = [f'video://{row["source"]}::{i}::224x224' for i in ids]
        cached_paths = [f'video://{row["destination"]}::{i}::224x224' for i in ids]
        os.environ.pop('HOST_VIDEO_DECODE_MODE',None)
        expected = strict_load_imgs(None, cached_paths)
        actual = sequential_load_imgs(raw_paths)
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            pixels += 1
    print(f'Exact resized RGB matches:{pixels} across{len(rows)} videos',flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)
    cfg = configure(pilot,4)
    data = batch(pilot,weights,cfg,0)
    model = make_model(weights,True)
    os.environ.pop('HOST_ALIGNMENT_LOGITS_TO_KEEP',None)
    full_loss = loss_for(model,data,0)
    full_loss.backward()
    grads = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    value = full_loss.detach().clone()
    model.zero_grad(set_to_none=True)
    del full_loss
    os.environ['HOST_ALIGNMENT_LOGITS_TO_KEEP']='1'
    small_loss = loss_for(model,data,0)
    small_loss.backward()
    torch.testing.assert_close(small_loss.detach(),value,rtol=1e-6,atol=1e-7)
    checked = 0
    for name,param in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(param.grad,grads[name],rtol=1e-5,atol=1e-6)
            checked += 1
        elif param.grad is not None:
            raise ValueError('Unexpected gradient in logits-limited model:'+name)
    result = dict(passed=True, random_tiny_cpu_only=True, full_loss=float(value),
                  one_token_loss=float(small_loss.detach()), gradient_tensors_checked=checked,
                  exact_pixel_samples=pixels, videos=len(rows), pretrained8b_gpu_validated=False)
    with Path(output).open('x') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['audit','equivalence'],required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--weights')
    p.add_argument('--output')
    a=p.parse_args()
    if a.mode=='audit':
        audit(a.root)
    else:
        equivalence(a.root,a.weights,a.output)
