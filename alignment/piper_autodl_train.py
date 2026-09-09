"""Four-task AutoDL alignment training: low-memory, only one FINAL checkpoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time

from piper_low_memory import apply_low_memory_config, check_paths


def ds_config(cfg):
    base = json.loads((Path(__file__).resolve().parent/'scripts/ds_config_zero3.json').read_text())
    base.update(copy.deepcopy(dict(cfg.DS_CONFIG)))
    base['gradient_accumulation_steps'] = 4
    base['zero_optimization']['offload_optimizer'] = dict(device='cpu', pin_memory=True)
    base['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] = False
    return base


def run(args):
    if os.environ.get('HOST_AUTODL_TRAIN_APPROVED') != '1' or int(os.environ.get('WORLD_SIZE','0')) != 4:
        raise ValueError('Approved four-GPU AutoDL launch required')
    import numpy as np
    import torch
    import deepspeed
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from piper_smoke import make_model, loss_for, to_device
    from datasets import AlignmentDataset, AlignmentCollator, worker_init_fn

    root, weights, output = Path(args.root), Path(args.weights), Path(args.output)
    rank = int(os.environ['RANK'])
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    random.seed(42+rank)
    np.random.seed(42+rank)
    torch.manual_seed(42)
    cfg = apply_low_memory_config(root)
    os.environ['HOST_ALIGNMENT_MODEL_PATH'] = str(weights)
    deepspeed.init_distributed()
    if rank == 0:
        check_paths(root, weights, output)
        output.mkdir(exist_ok=False)
    torch.distributed.barrier()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    dataset = AlignmentDataset(mode='train', processor=processor,
        video_paths_json=str(root/'train/piper_video_paths.json'))
    if len(dataset) != 796 or not dataset.weights:
        raise ValueError('Expected796 four-task train episodes and official weighted sampler')
    generator = torch.Generator().manual_seed(42+rank)
    sampler = WeightedRandomSampler(torch.DoubleTensor(dataset.weights), len(dataset.weights),
                                    replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2,
        prefetch_factor=1, pin_memory=False, persistent_workers=True, drop_last=True,
        collate_fn=AlignmentCollator(processor=processor, mode='train'),
        worker_init_fn=worker_init_fn)
    model = make_model(str(weights), False)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config(cfg))
    started = time.monotonic()
    report = dict(rank=rank, approved=True, data_root=str(root), episodes=796,
                  validation_episodes=88, pretrained_revision='2c4565515e0f265c6511776e7193b22c0968ddc7',
                  effective_ds=ds_config(cfg), final_only_checkpoint=True, optimizer_steps=0,
                  micro_steps=0, production_quality_validated=False, progress_generated=False)
    (output/f'config_rank{rank}.json').write_text(json.dumps(report, indent=2))
    stream = (output/f'metrics_rank{rank}.jsonl').open('x')
    try:
        done = False
        while not done:
            for data in loader:
                data = to_device(data, torch.device('cuda', device))
                before = engine.global_steps
                loss = loss_for(engine, data, before)
                engine.backward(loss)
                engine.step()
                report['micro_steps'] += 1
                report['optimizer_steps'] = engine.global_steps
                if engine.global_steps > before:
                    torch.cuda.synchronize()
                    grad = float(engine.get_global_grad_norm())
                    if not np.isfinite(grad) or grad <= 0:
                        raise ValueError('Nonfinite/zero gradient norm')
                    free, total = torch.cuda.mem_get_info()
                    record = dict(rank=rank, step=engine.global_steps,
                        micro_step=report['micro_steps'], loss=float(loss.detach()),
                        grad_norm=grad, lr=engine.get_lr(),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        device_used_bytes=total-free, seconds=time.monotonic()-started)
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                    stream.flush()
                    print(json.dumps(record, allow_nan=False), flush=True)
                    # Fail closed after20 real updates if the reserved-memory margin is unsafe.
                    if engine.global_steps == 20:
                        peak = torch.tensor(record['peak_reserved_bytes'], device=torch.device('cuda', device), dtype=torch.float64)
                        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
                        if peak.item() > 65*1024**3:
                            raise ValueError('20-update memory gate exceeded65GiB reserved; no checkpoint saved')
                        if rank == 0:
                            (output/'memory_gate_passed.json').write_text(json.dumps(
                                dict(optimizer_steps=20, max_reserved_bytes=peak.item(), checkpoint_saved=False)))
                del loss
                if engine.global_steps >= 3000:
                    done = True
                    break
        import shutil
        if shutil.disk_usage(output).free < 110*1024**3:
            raise ValueError('Insufficient110GiB final checkpoint reserve')
        engine.save_checkpoint(str(output/'checkpoint'), tag='final',
                               client_state=dict(step=3000, final_only=True, data_root=str(root)))
        torch.distributed.barrier()
        report.update(complete=True, seconds=time.monotonic()-started, final_checkpoint='checkpoint/final')
        (output/f'complete_rank{rank}.json').write_text(json.dumps(report, indent=2))
        if rank == 0:
            (output/'train_complete.json').write_text(json.dumps(dict(complete=True, step=3000, production_quality_validated=False)))
    except BaseException as exc:
        report.update(complete=False, error_type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-started)
        (output/f'failure_rank{rank}.json').write_text(json.dumps(report, indent=2))
        raise
    finally:
        stream.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root','weights','output'):
        parser.add_argument('--'+key, required=True)
    run(parser.parse_args())
