"""Opt-in HOST Piper strict loading and ZeRO-2 model-only checkpoint safety."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch

MODULES = ('mot', 'proprio_encoder', 'progress_encoder', 'progress_decoder', 'visual_encoder')


def adapt_camera(state, target_count, seed=42):
    out = dict(state)
    cam, pos = out['camera_emb'], out['pos_embed']
    if cam.shape[0] == target_count:
        return out
    if cam.shape[0] != 2 or target_count != 3 or pos.shape != (1, 512, 4096):
        raise ValueError('Only explicit official 2->3 camera initialization is supported')
    # Official camera identities are unknown: never invent a semantic correspondence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        out['camera_emb'] = torch.empty((3, cam.shape[1]), dtype=cam.dtype)
        out['pos_embed'] = torch.empty((1, 768, 4096), dtype=pos.dtype)
        for key in ('camera_emb', 'pos_embed'):
            torch.nn.init.trunc_normal_(out[key], std=0.02)
    return out


def load_strict_host(model, path, optimizer=None):
    if optimizer is not None:
        raise ValueError('Weights-only restart does not restore an optimizer')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not set(MODULES) <= set(payload):
        raise ValueError(f'Missing HOST modules: {set(MODULES) - set(payload)}')
    for name in MODULES:
        module = getattr(model, name)
        state = payload[name]
        if name == 'visual_encoder':
            state = adapt_camera(state, module.num_cameras)
        module.load_state_dict(state, strict=True)
    print('HOST_STRICT_LOAD_OK: all five modules; only explicit camera/position adaptation', flush=True)
    return {'step': payload.get('step'), 'strict_loaded': True}


def _weight_payload(model, step):
    payload = {'step': int(step), 'torch_dtype': 'torch.bfloat16'}
    for name in MODULES:
        payload[name] = {}
        for key, value in getattr(model, name).state_dict().items():
            tensor = value.detach().to('cpu').contiguous()
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Nonfinite checkpoint tensor {name}.{key}')
            payload[name][key] = tensor
    return payload


def save_model_only(trainer):
    accel = trainer.accelerator
    plugin = accel.state.deepspeed_plugin
    if plugin is None or plugin.deepspeed_config['zero_optimization']['stage'] != 2:
        raise ValueError('This saver requires replicated ZeRO-2 model weights; not ZeRO-3')
    accel.wait_for_everyone()
    path = Path(trainer.weights_dir) / f'step_{trainer.global_step:06d}.pt'
    if accel.is_main_process and not path.exists():
        model = accel.unwrap_model(trainer.model)
        payload = _weight_payload(model, trainer.global_step)
        nbytes = sum(v.numel()*v.element_size() for n in MODULES for v in payload[n].values())
        existing = sorted(path.parent.glob('step_*.pt'))
        # If necessary prune only older verified run-owned files, retaining the latest.
        while shutil.disk_usage(path.parent).free < nbytes + 8*2**30 and len(existing) > 1:
            old = existing.pop(0)
            certificate = old.with_suffix('.verified.json')
            if not certificate.is_file():
                raise RuntimeError(f'Refuse to prune uncertified checkpoint {old}')
            old.unlink()
            certificate.unlink()
        if shutil.disk_usage(path.parent).free < nbytes + 8*2**30:
            raise RuntimeError('Insufficient safe disk budget; newest checkpoint preserved')
        temp = path.with_suffix('.partial')
        with temp.open('xb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        restored = torch.load(temp, map_location='cpu', weights_only=True, mmap=True)
        for name in MODULES:
            assert set(restored[name]) == set(payload[name])
            for key, value in payload[name].items():
                if not torch.equal(restored[name][key], value):
                    raise RuntimeError(f'Checkpoint round-trip mismatch: {name}.{key}')
        del restored, payload
        digest = hashlib.sha256()
        with temp.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                digest.update(block)
        os.replace(temp, path)
        path.with_suffix('.verified.json').write_text(json.dumps({
            'step': trainer.global_step, 'sha256': digest.hexdigest(),
            'bytes': path.stat().st_size, 'model_only': True, 'verified': True})+'\n')
        existing = sorted(path.parent.glob('step_*.pt'))
        for old in existing[:-int(trainer.cfg.get('keep_model_checkpoints', 2))]:
            certificate = old.with_suffix('.verified.json')
            if not certificate.is_file():
                raise RuntimeError(f'Refuse to prune uncertified checkpoint {old}')
            old.unlink()
            certificate.unlink()
        print(f'MODEL_ONLY_CHECKPOINT_OK {path}', flush=True)
    accel.wait_for_everyone()
    return {'weights_path': str(path), 'state_path': None}


def check_train_health(trainer, loss):
    peak = torch.cuda.max_memory_reserved() / 2**30
    norm = trainer.model.get_global_grad_norm()
    norm = float(norm) if norm is not None else float('nan')
    if not (0 < norm < float('inf')):
        raise FloatingPointError(f'Invalid DeepSpeed gradient norm: {norm}')
    if peak > float(trainer.cfg.get('max_reserved_gib', 74)):
        raise RuntimeError(f'Peak memory {peak:.2f}GiB exceeds safety gate')
    if shutil.disk_usage(trainer.output_dir).free < 8*2**30:
        raise RuntimeError('Disk reserve below8GiB')
    record = dict(step=trainer.global_step, rank=trainer.accelerator.process_index,
                  loss=float(loss.detach()), grad_norm=norm, peak_reserved_gib=peak,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  current_allocated_gib=torch.cuda.memory_allocated()/2**30,
                  elapsed_s=time.perf_counter()-trainer.run_start_time)
    with (Path(trainer.output_dir)/f'health_rank{record["rank"]}.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
