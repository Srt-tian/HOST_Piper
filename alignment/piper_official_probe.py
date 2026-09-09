"""Run the released training loop/settings on certified Piper video, bounded at
10 actual optimizer updates. No production alignment-quality claim.
prepare: CPU sampler/processor check. train/reload: separately approved EIP only.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time

UPDATES = 10


def write_json(path, record):
    with Path(path).open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)


def require_approval():
    if os.environ.get('HOST_OFFICIAL_PROBE_APPROVED') != '1':
        raise ValueError('Explicit official-probe approval required')
    if int(os.environ.get('WORLD_SIZE', '0')) != 4:
        raise ValueError('Exactly four ranks required')


def configure(root):
    from piper_profile import apply_profile, enable_strict_loading
    cfg = apply_profile(root, context_frames=1)
    enable_strict_loading()
    # Preserve released config.py + single-node train_scripts/run_ds.sh values.
    assert cfg.TRAIN.NUM_FRAMES == 24 and cfg.TRAIN.MAX_BATCH_FRAMES == 96
    assert cfg.TRAIN.NUM_ALIGN_FRAMES == 24
    assert cfg.TRAIN.CHUNK_PROBS == [.1, .2, .7]
    assert cfg.DS_CONFIG.train_micro_batch_size_per_gpu == 4
    cfg.DS_CONFIG.scheduler.params.total_num_steps = 3000
    cfg.TRAIN.MAX_ITERS = 3000
    return cfg


def effective_ds(cfg):
    path = Path(__file__).resolve().parent/'scripts/ds_config_zero3.json'
    ds = json.loads(path.read_text())
    # Matches train.py: config.py owns optimizer/scheduler and runtime owns accumulation.
    ds.update(copy.deepcopy(dict(cfg.DS_CONFIG)))
    ds['gradient_accumulation_steps'] = 4  # official launcher:4 /1 node
    return ds


def batch_record(data):
    q = data['qwen_input']
    counts = (q['input_ids'] == 151664).sum(-1).tolist()
    if counts != [24]*8:
        raise ValueError('Official batch requires8 Qwen rows x24 anchor CLS tokens')
    pairs = []
    for main, ref in zip(data['name'], data['ref_name']):
        a = json.loads((Path(main)/'provenance.json').read_text())
        b = json.loads((Path(ref)/'provenance.json').read_text())
        if main == ref or a['split'] != 'train' or b['split'] != 'train' or a['task'] != b['task']:
            raise ValueError('Invalid self/cross-task/cross-split training pair')
        pairs.append(dict(main=main, reference=ref, task=a['task'], split=a['split']))
    gids = q['group_ids'].tolist()
    cids = q['chunk_ids'].tolist()
    return dict(pairs=pairs, input_ids_shape=list(q['input_ids'].shape),
                cls_per_row=counts, group_ids=gids, chunk_ids=cids,
                unique_main_episodes=len(set(p['main'] for p in pairs)),
                chosen_steps=data['chosen_steps'].tolist(),
                ref_chosen_steps=data['ref_chosen_steps'].tolist(),
                tensor_shapes={k:list(v.shape) for k,v in q.items() if hasattr(v,'shape')})


def prepare(args):
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from datasets import AlignmentDataset, AlignmentCollator
    cfg = configure(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.weights, local_files_only=True, trust_remote_code=False)
    dataset = AlignmentDataset(mode='train', processor=processor,
                               video_paths_json=str(Path(args.root)/'train/piper_video_paths.json'))
    if not dataset.weights or len(dataset.weights) != len(dataset):
        raise ValueError('Expected official weighted-sampling path')
    torch.set_num_threads(4)
    for rank in range(4):
        torch.manual_seed(42+rank)
        random.seed(42+rank)
        np.random.seed(42+rank)
        generator = torch.Generator().manual_seed(42+rank)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.DoubleTensor(dataset.weights), len(dataset.weights),
            replacement=True, generator=generator)
        indices = list(sampler)[:4]
        if len(indices) != 4:
            raise ValueError('Insufficient local training samples for official batch4')
        samples = [dataset[index] for index in indices]
        data = AlignmentCollator(processor=processor, mode='train')(samples)
        record = batch_record(data)
        record.update(rank=rank, sampler='WeightedRandomSampler(replacement=True)',
                      dataset_entries=len(dataset), indices=indices,
                      gpu_used=False, model_forward=False)
        write_json(output/f'prepare_rank{rank}.json', record)
        print(json.dumps(record), flush=True)
    write_json(output/'effective_ds.json', effective_ds(cfg))
    write_json(output/'prepare_complete.json',
               dict(complete=True, ranks=4, gpu_used=False, production_ready=False))


class ProbeComplete(Exception):
    pass


def gpu(args):
    require_approval()
    import numpy as np
    import torch
    import deepspeed
    from piper_smoke import make_model
    cfg = configure(args.root)
    rank = int(os.environ['RANK'])
    output = Path(args.output)
    if args.mode == 'reload' and not (output/'train_complete.json').is_file():
        raise ValueError('Completed training required for fresh-process reload')
    marker = json.loads((Path(args.weights)/'verified_artifacts.json').read_text())
    if marker.get('verified') is not True or marker.get('revision') != '2c4565515e0f265c6511776e7193b22c0968ddc7':
        raise ValueError('Pinned verified8B weights required')
    os.environ['HOST_ALIGNMENT_MODEL_PATH'] = args.weights
    torch.manual_seed(42)
    random.seed(42+rank)
    np.random.seed(42+rank)
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    state = dict(mode=args.mode, rank=rank, complete=False, micro_steps=0,
                 optimizer_steps=0, effective_ds=effective_ds(cfg),
                 production_quality_validated=False, progress_generated=False)

    def memory():
        torch.cuda.synchronize()
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    device_total_bytes=torch.cuda.get_device_properties(
                        int(os.environ['LOCAL_RANK'])).total_memory)

    try:
        if args.mode == 'reload':
            deepspeed.init_distributed()
            base = make_model(args.weights, False)
            engine, _, _, _ = deepspeed.initialize(
                model=base, model_parameters=base.parameters(), config=effective_ds(cfg))
            path, client = engine.load_checkpoint(
                str(output/'checkpoint'), tag='probe_step10', load_module_strict=True,
                load_optimizer_states=True, load_lr_scheduler_states=True)
            if not path or client.get('step') != UPDATES or engine.global_steps != UPDATES:
                raise ValueError('Full model/optimizer/scheduler reload failed')
            state.update(complete=True, optimizer_steps=engine.global_steps,
                         checkpoint_reload_passed=True, replay_loss_checked=False)
        else:
            from absl import app
            import train as official_train
            original_init = deepspeed.initialize

            def tracked_init(*positional, **keywords):
                if keywords['config'] != effective_ds(cfg):
                    raise ValueError('Actual DeepSpeed config differs from reviewed official config')
                result = original_init(*positional, **keywords)
                engine = result[0]
                backward, step = engine.backward, engine.step
                micro_started = [time.monotonic()]

                def before_forward(module, inputs):
                    data = inputs[0]
                    info = batch_record(data)
                    write_json(output/f'input_rank{rank}_micro{state["micro_steps"]}.json', info)
                    micro_started[0] = time.monotonic()

                engine.register_forward_pre_hook(before_forward)

                def tracked_backward(loss, *a, **kw):
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise ValueError('Nonfinite/nonscalar loss')
                    state['last_loss'] = float(loss.detach())
                    return backward(loss, *a, **kw)

                def tracked_step(*a, **kw):
                    previous = engine.global_steps
                    result = step(*a, **kw)
                    state['micro_steps'] += 1
                    state['optimizer_steps'] = engine.global_steps
                    grad = engine.get_global_grad_norm()
                    if engine.global_steps > previous and (
                            grad is None or not np.isfinite(float(grad)) or float(grad) <= 0):
                        raise ValueError('Nonfinite/zero optimizer-boundary gradient norm')
                    record = dict(rank=rank, micro_step=state['micro_steps'],
                                  optimizer_step=engine.global_steps,
                                  loss=state['last_loss'], lr=engine.get_lr(),
                                  grad_norm=None if grad is None else float(grad), **memory(),
                                  seconds=time.monotonic()-micro_started[0])
                    write_json(output/f'metrics_rank{rank}_micro{state["micro_steps"]}.json', record)
                    print(json.dumps(record), flush=True)
                    if engine.global_steps >= UPDATES:
                        # Stop by real engine updates, not the loop's accumulation-counter label.
                        engine.save_checkpoint(str(output/'checkpoint'), tag='probe_step10',
                                               client_state={'step': UPDATES})
                        state['complete'] = True
                        raise ProbeComplete()
                    return result

                engine.backward = tracked_backward
                engine.step = tracked_step
                return result

            deepspeed.initialize = tracked_init
            argv = ['train.py', '--alsologtostderr',
                    '--logdir='+str(output/'official_logs'),
                    '--video_paths='+str(Path(args.root)/'train/piper_video_paths.json'),
                    '--network=Qwen3-VL-Embedding-8B',
                    '--gradient_accumulation_steps=4',
                    '--ds_config='+str(Path(__file__).resolve().parent/'scripts/ds_config_zero3.json'),
                    '--max_iters=3000', '--save_interval=500']

            def bounded_main(unused):
                try:
                    official_train.train(unused)
                except ProbeComplete:
                    pass
                if not state['complete'] or state['optimizer_steps'] != UPDATES:
                    raise ValueError('Official loop ended before10 real optimizer updates')

            # absl app.run exits; keep final report/cleanup within this function.
            try:
                app.run(bounded_main, argv=argv)
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
        state.update(memory(), seconds=time.monotonic()-started)
        write_json(output/f'{args.mode}_rank{rank}.json', state)
        torch.distributed.barrier()
        if rank == 0:
            write_json(output/f'{args.mode}_complete.json', dict(complete=True, production_ready=False))
    except BaseException as exc:
        state.update(error_type=type(exc).__name__, seconds=time.monotonic()-started)
        write_json(output/f'{args.mode}_failure_rank{rank}.json', state)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare','train','reload'), required=True)
    for key in ('root','weights','output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    else:
        gpu(args)
