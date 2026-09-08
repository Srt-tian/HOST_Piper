"""Bounded real-video alignment smoke, never a source of production progress GT.

cpu-tiny: random reduced Qwen, CPU only, real HOST forward/loss/backward/checkpoint.
train/reload: approved four-GPU ZeRO-3 smoke with the pinned pretrained 8B model.
Reload must run in a fresh torchrun process; ZeRO-3 cannot safely reload its own
partitioned engine in-place. No production progress artifacts are exported.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from piper_profile import apply_profile, enable_strict_loading


def configure(root, anchors):
    root = Path(root)
    ready = json.loads((root/'preparation_summary.json').read_text())
    if ready.get('seek_safe') is not True or ready.get('visual_only') is not True:
        raise ValueError('Completed seek-safe visual preparation required')
    cfg = apply_profile(root, context_frames=1)
    enable_strict_loading()
    cfg.TRAIN.NUM_FRAMES = cfg.EVAL.NUM_FRAMES = anchors
    cfg.TRAIN.MAX_BATCH_FRAMES = anchors
    cfg.TRAIN.NUM_ALIGN_FRAMES = 6 if anchors == 4 else 24
    cfg.TRAIN.CHUNK_PROBS = cfg.EVAL.CHUNK_PROBS = [1., 0., 0.]
    cfg.AUGMENTATION.BRIGHTNESS = cfg.AUGMENTATION.CONTRAST = False
    cfg.EVAL.REF_CACHE_MAXSIZE = 48
    cfg.EVAL.REF_CACHE_MAIN_ONLY = False
    return cfg


def batch(root, weights, cfg, rank):
    from transformers import AutoProcessor
    from datasets import AlignmentDataset, AlignmentCollator
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'data_preprocessing/piper'))
    from build_progress import anchor_frames
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    data = AlignmentDataset(mode='eval', video_paths_json=str(Path(root)/'train/piper_video_paths.json'),
                            processor=processor)
    sample = data[1+rank % (len(data)-1)]
    if sample['name'] == sample['ref_name']:
        raise ValueError('Smoke requires distinct main/reference videos')
    for name, paths, steps in [('name','frame_paths','chosen_steps'),
                                ('ref_name','ref_frame_paths','ref_chosen_steps')]:
        provenance = json.loads((Path(sample[name])/'provenance.json').read_text())
        np.testing.assert_array_equal(anchor_frames(sample[paths], sample[name], provenance['frames'], 1),
                                      sample[steps].numpy())
    result = AlignmentCollator(processor=processor, mode='eval')([sample])
    inputs = result['qwen_input']
    torch.testing.assert_close((inputs['input_ids'] == cfg.SPECIAL_TOKENS.CLS_TOKEN_ID).sum(-1),
                               inputs['num_mains']+inputs['num_refs'])
    return result


def make_model(weights, tiny):
    from algos.alignment import Alignment
    if not tiny:
        os.environ['HOST_ALIGNMENT_MODEL_PATH'] = weights
        return Alignment()
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from models import BaseModel, LinearEmbedder
    from monkey_patch_forward import replace_qwen3_with_mixed_modality_forward
    config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    text = config.text_config
    text.hidden_size, text.intermediate_size = 128, 256
    text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 3, 4, 2
    text.head_dim = 32
    text.rope_scaling = dict(rope_type='default', mrope_section=[4,6,6], mrope_interleaved=True)
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size = 3, 64, 128
    vision.num_heads, vision.out_hidden_size = 4, 128
    vision.deepstack_visual_indexes = [0,1,2]
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    replace_qwen3_with_mixed_modality_forward()
    wrapper = BaseModel.__new__(BaseModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.base_model = Qwen3VLForConditionalGeneration(config).float()
    wrapper.base_model.gradient_checkpointing_enable()
    wrapper.num_steps = 4
    return Alignment(model={'cnn': wrapper, 'emb': LinearEmbedder(128)})


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    return x


def loss_for(model, data, step, training=True):
    # HOST's forward mutates qwen_input selection; keep the replay batch immutable.
    data = copy.deepcopy(data)
    steps = torch.cat([data['chosen_steps'], data['ref_chosen_steps']], dim=0)
    lengths = torch.cat([data['seq_lens'], data['ref_seq_lens']], dim=0)
    embeddings = model(data, steps, lengths, training=training)
    module = getattr(model, 'module', model)
    loss, _ = module.compute_loss(embeddings, steps, lengths, step, True,
        frame_labels=data.get('frame_labels'), seq_labels=data.get('seq_labels'), metadata=data)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError('Nonfinite/non-scalar HOST alignment loss')
    return loss


def ds_config():
    return dict(bf16={'enabled': True}, train_micro_batch_size_per_gpu=1,
        gradient_accumulation_steps=1, gradient_clipping=3.,
        zero_optimization=dict(stage=3, overlap_comm=True, contiguous_gradients=True,
            reduce_bucket_size=200000000, allgather_bucket_size=200000000,
            stage3_gather_16bit_weights_on_model_save=False),
        optimizer=dict(type='AdamW', params=dict(lr=1e-5, betas=[.9,.999], eps=1e-8, weight_decay=1e-5)),
        steps_per_print=1, wall_clock_breakdown=False)


def run(args):
    tiny = args.mode == 'cpu-tiny'
    if not tiny:
        if os.environ.get('HOST_SMOKE_APPROVED') != '1' or int(os.environ.get('WORLD_SIZE', '0')) != 4:
            raise ValueError('Resolved approval and four-rank torchrun required')
        verified = json.loads((Path(args.weights)/'verified_artifacts.json').read_text())
        if verified.get('verified') is not True or verified.get('revision') != '2c4565515e0f265c6511776e7193b22c0968ddc7':
            raise ValueError('Pinned complete 8B artifacts required')
    rank = int(os.environ.get('RANK', '0')) if not tiny else 0
    output = Path(args.output)
    if args.mode == 'reload':
        if not (output/'train_complete.json').is_file():
            raise ValueError('Completed train phase required before fresh-process reload')
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(42+rank)
    np.random.seed(42+rank)
    torch.manual_seed(42)
    cfg = configure(args.root, args.anchors)
    data = batch(args.root, args.weights, cfg, rank)
    started = time.monotonic()
    if tiny:
        model = make_model(args.weights, True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    else:
        import deepspeed
        deepspeed.init_distributed()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        data = to_device(data, device)
        base = make_model(args.weights, False)
        model, optimizer, _, _ = deepspeed.initialize(model=base, model_parameters=base.parameters(), config=ds_config())
    results = []
    if args.mode != 'reload':
        for step in range(args.steps):
            loss = loss_for(model, data, step)
            if tiny:
                loss.backward()
                norms = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
                grad_norm = torch.linalg.vector_norm(torch.stack(norms))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                model.backward(loss)
                model.step()
                grad_norm = torch.as_tensor(model.get_global_grad_norm())
            if not torch.isfinite(grad_norm) or grad_norm <= 0:
                raise ValueError('Nonfinite/zero gradient norm')
            results.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(grad_norm)))
            print(json.dumps(dict(rank=rank, mode=args.mode, **results[-1])), flush=True)
        model.eval()
        with torch.no_grad():
            replay_loss = float(loss_for(model, data, args.steps, training=False))
        if tiny:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), output/'cpu_random_test.pt')
            restored = torch.load(output/'cpu_random_test.pt', map_location='cpu', weights_only=True)
            fresh = make_model(args.weights, True)
            fresh.load_state_dict(restored['model'], strict=True)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-5, weight_decay=1e-5)
            fresh_optimizer.load_state_dict(restored['optimizer'])
            fresh.eval()
            with torch.no_grad():
                torch.testing.assert_close(loss_for(fresh, data, args.steps, training=False), torch.tensor(replay_loss), rtol=1e-5, atol=1e-6)
        else:
            model.save_checkpoint(str(output/'checkpoint'), tag='smoke', client_state={'steps':args.steps})
            torch.save(dict(data=data, replay_loss=replay_loss), output/f'replay_rank{rank}.pt')
            torch.distributed.barrier()
    else:
        path, state = model.load_checkpoint(str(output/'checkpoint'), tag='smoke', load_module_strict=True)
        if not path or state.get('steps') != args.steps:
            raise ValueError('Full training checkpoint reload failed')
        saved = torch.load(output/f'replay_rank{rank}.pt', map_location=device, weights_only=False)
        model.eval()
        with torch.no_grad():
            replay_loss = float(loss_for(model, saved['data'], args.steps, training=False))
        np.testing.assert_allclose(replay_loss, saved['replay_loss'], rtol=1e-3, atol=1e-4)
    report = dict(mode=args.mode, rank=rank, steps=results, seconds=time.monotonic()-started,
        anchors=args.anchors, replay_loss=replay_loss, pretrained_8b=not tiny,
        production_alignment_quality_validated=False, progress_generated=False,
        checkpoint_reload_passed=tiny or args.mode=='reload')
    if not tiny:
        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with (output/f'{args.mode}_rank{rank}.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    if not tiny:
        torch.distributed.barrier()
    if rank == 0:
        with (output/('reload_complete.json' if args.mode=='reload' else 'train_complete.json')).open('x') as handle:
            json.dump(dict(mode=args.mode, complete=True, production_ready=False), handle)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['cpu-tiny','train','reload'], required=True)
    for name in ['root', 'weights', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--anchors', choices=[4,24], type=int, default=4)
    p.add_argument('--steps', choices=[2], type=int, default=2)
    run(p.parse_args())
