"""Model-only BF16 checkpoint saving with verified bounded retention."""
import hashlib
import json
from pathlib import Path
import re
import shutil

OWNERSHIP = 'host-piper-weights-v1'

def checkpoint_due(step, total_steps=3000, interval=100):
    if step <= 0 or interval <= 0:
        return False
    return step == 20 or step % interval == 0 or step == total_steps

def model_schema(module):
    schema = {}
    def visit(layer, prefix=''):
        for name, parameter in layer.named_parameters(recurse=False):
            shape = getattr(parameter, 'ds_shape', parameter.shape)
            schema[prefix+name] = (tuple(shape), parameter.dtype)
        for name, buffer in layer.named_buffers(recurse=False):
            if name not in layer._non_persistent_buffers_set:
                schema[prefix+name] = (tuple(buffer.shape), buffer.dtype)
        for name, child in layer.named_children():
            visit(child, prefix+name+'.')
    visit(module)
    return schema

def validate_weights(path, schema):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if set(state) != set(schema):
        raise ValueError('Incomplete or unexpected model-only checkpoint keys')
    floating_bytes = 0
    for name, (shape, dtype) in schema.items():
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError('Checkpoint shape/dtype mismatch:'+name)
        if tensor.is_floating_point():
            for chunk in tensor.reshape(-1).split(1024*1024):
                if not torch.isfinite(chunk).all():
                    raise ValueError('Nonfinite checkpoint weights:'+name)
            floating_bytes += tensor.numel()*tensor.element_size()
    del state
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2), b''):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=Path(path).stat().st_size,
                model_tensors=len(schema), floating_tensor_bytes=floating_bytes)

def prune_owned(root, keep=3):
    """Remove only old, complete checkpoints produced by this saver in this run."""
    root = Path(root).resolve()
    if keep < 1:
        raise ValueError('Keep at least one complete checkpoint')
    candidates = []
    for path in root.iterdir():
        if not re.fullmatch(r'checkpoint-[0-9]{6}', path.name) or path.is_symlink() or not path.is_dir():
            continue
        marker = path/'manifest.json'
        if not marker.is_file() or marker.is_symlink():
            continue
        info = json.loads(marker.read_text())
        if info.get('owner') != OWNERSHIP or info.get('complete') is not True:
            continue
        if info.get('step') != int(path.name.split('-')[-1]):
            raise ValueError('Checkpoint step mismatch')
        entries = list(path.iterdir())
        if {p.name for p in entries} != {'pytorch_model.bin','manifest.json'}:
            raise ValueError('Refuse to delete checkpoint containing unknown files')
        if any(p.is_symlink() or not p.is_file() for p in entries):
            raise ValueError('Refuse to delete unexpected checkpoint file type')
        candidates.append(path)
    candidates.sort(key=lambda p:p.name)
    removed = []
    for path in candidates[:-keep]:
        for name in ('pytorch_model.bin','manifest.json'):
            (path/name).unlink()
        path.rmdir()
        removed.append(path.name)
    return removed

def save_weights(engine, output, step, final=False, keep=3):
    """All ranks participate; rank0 verifies before atomic publication and pruning."""
    import torch
    import torch.distributed as dist
    root = Path(output)/'weights'
    partial = root/f'checkpoint-{step:06d}.incomplete'
    destination = root/f'checkpoint-{step:06d}'
    rank = dist.get_rank()
    status = [None]
    schema = model_schema(engine.module)
    if rank == 0:
        try:
            root.mkdir(exist_ok=True)
            if destination.exists() or partial.exists():
                raise ValueError('Do not overwrite checkpoint or partial save')
            if shutil.disk_usage(root).free < 32*1024**3:
                raise ValueError('Need32GiB free BEFORE saving; never delete the previous good checkpoint first')
            partial.mkdir()
            status[0] = dict(ok=True)
        except Exception as exc:
            status[0] = dict(ok=False,error=str(exc))
    dist.broadcast_object_list(status,src=0)
    if not status[0]['ok']:
        raise ValueError(status[0]['error'])
    torch.cuda.empty_cache()
    if not engine.save_16bit_model(str(partial), save_filename='pytorch_model.bin'):
        raise ValueError('DeepSpeed refused BF16 consolidation')
    dist.barrier()
    if rank == 0:
        try:
            report = validate_weights(partial/'pytorch_model.bin',schema)
            report.update(owner=OWNERSHIP, complete=True, step=step, final=final,
                          weights_only=True, optimizer_saved=False, scheduler_saved=False,
                          strict_schema_verified=True)
            with (partial/'manifest.json').open('x') as handle:
                json.dump(report,handle,indent=2)
            partial.rename(destination)
            removed = prune_owned(root,keep)
            print(json.dumps(dict(event='weights_checkpoint_saved',path=str(destination),
                                  step=step,bytes=report['bytes'],removed=removed)),flush=True)
            status[0] = dict(ok=True,path=str(destination),removed=removed)
        except Exception as exc:
            status[0] = dict(ok=False,error=str(exc))
    dist.broadcast_object_list(status,src=0)
    if not status[0]['ok']:
        raise ValueError(status[0]['error'])
    torch.cuda.empty_cache()
    return status[0]
