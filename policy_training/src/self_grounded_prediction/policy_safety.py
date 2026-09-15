"""Opt-in HOST Piper strict loading and ZeRO-2 model-only checkpoint safety."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import math
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
    reasons = []
    if not (0 < norm < float('inf')):
        reasons.append(f'Invalid DeepSpeed gradient norm: {norm}')
    if peak > float(trainer.cfg.get('max_reserved_gib', 74)):
        reasons.append(f'Peak memory {peak:.2f}GiB exceeds safety gate')
    if shutil.disk_usage(trainer.output_dir).free < 8*2**30:
        reasons.append('Disk reserve below8GiB')
    record = dict(step=trainer.global_step, rank=trainer.accelerator.process_index,
                  loss=float(loss.detach()), grad_norm=norm, peak_reserved_gib=peak,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  current_allocated_gib=torch.cuda.memory_allocated()/2**30,
                  current_reserved_gib=torch.cuda.memory_reserved()/2**30,
                  errors=reasons,
                  elapsed_s=time.perf_counter()-trainer.run_start_time)
    with (Path(trainer.output_dir)/f'health_rank{record["rank"]}.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    if reasons:
        raise RuntimeError('; '.join(reasons))


def summarize_window(records):
    """Sample-weighted metric sums; ignore diagnostic NaN sentinels, not loss NaNs."""
    if not records:
        raise ValueError('Empty metric window')
    totals = {}
    for record in records:
        size = record['batch_size']
        for key, value in record['metrics'].items():
            if value is None or not math.isfinite(value):
                continue
            sums = totals.setdefault(key, [0.0, 0.0])
            sums[0] += value * size
            sums[1] += size
    return totals


def record_microbatch(trainer, sample, loss, loss_dict):
    # Scalars only: never retain a loss tensor/autograd graph between microsteps.
    metrics = {'loss_total': float(loss.detach()), **{k: float(v) for k,v in loss_dict.items()}}
    metrics = {k: v if math.isfinite(v) else None for k,v in metrics.items()}
    trainer._diagnostic_microstep = getattr(trainer, '_diagnostic_microstep', 0) + 1
    record = dict(update=trainer.global_step+1, microstep=trainer._diagnostic_microstep,
                  rank=trainer.accelerator.process_index, batch_size=int(sample['action'].shape[0]),
                  metrics=metrics, agent_episodes=sample.get('agent_episode_dir'),
                  reference_episodes=sample.get('task_episode_dir'),
                  task_video_dropped=sample.get('task_video_dropped'),
                  video_shape=list(sample['video'].shape),
                  task_video_shape=list(sample['task_video'].shape),
                  elapsed_s=time.perf_counter()-trainer.run_start_time)
    for name in ('action', 'proprio', 'progress_gt'):
        value = sample.get(name)
        if isinstance(value, torch.Tensor):
            value = value.detach().float()
            record[name] = dict(shape=list(value.shape), min=float(value.min()),
                                max=float(value.max()), finite=bool(torch.isfinite(value).all()))
    with (Path(trainer.output_dir)/f'micro_rank{record["rank"]}.jsonl').open('a') as f:
        f.write(json.dumps(record, allow_nan=False)+'\n')
    if not hasattr(trainer, '_metric_window'):
        trainer._metric_window = []
    trainer._metric_window.append({'batch_size': record['batch_size'], 'metrics': metrics})


def finish_metric_window(trainer, device):
    totals = summarize_window(trainer._metric_window)
    # Every rank/model exposes the same loss_dict keys; diagnostic NaNs contribute0count.
    keys = sorted(trainer._metric_window[0]['metrics'])
    tensor = torch.tensor([totals.get(k, [0.,0.]) for k in keys], device=device, dtype=torch.float64)
    tensor = trainer.accelerator.reduce(tensor, reduction='sum').cpu()
    means = {k: float(row[0]/row[1]) if row[1] else None for k,row in zip(keys,tensor)}
    record = dict(step=trainer.global_step, metrics=means,
                  samples=int(tensor[keys.index('loss_total'),1]),
                  grad_norm=float(trainer.model.get_global_grad_norm()),
                  elapsed_s=time.perf_counter()-trainer.run_start_time)
    if trainer.accelerator.is_main_process:
        with (Path(trainer.output_dir)/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        print('UPDATE_METRICS '+json.dumps(record, allow_nan=False), flush=True)
    trainer._metric_window.clear()
    torch.cuda.reset_peak_memory_stats()
    return means['loss_total']
