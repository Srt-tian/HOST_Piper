"""Offline real-video Qwen processor check; optional RANDOM tiny-model interface test.

Never exports embeddings/progress labels and never runs an optimizer. Tiny mode is
an architecture compatibility test, NOT the pretrained8B model or a memory estimate.
"""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from piper_profile import apply_profile, enable_strict_loading


def smoke(root, weights, output, tiny_forward=False):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new report path')
    ready = json.loads((Path(root)/'preparation_summary.json').read_text())
    if ready.get('seek_safe') is not True or ready.get('visual_only') is not True:
        raise ValueError('This smoke requires the completed seek-safe visual pilot')
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = apply_profile(root, context_frames=1)
    enable_strict_loading()
    # Explicit reduced structural test, not the paper's24/96-anchor training recipe.
    cfg.TRAIN.NUM_FRAMES = cfg.EVAL.NUM_FRAMES = 4
    cfg.TRAIN.MAX_BATCH_FRAMES = 4
    cfg.TRAIN.NUM_ALIGN_FRAMES = 6
    cfg.EVAL.CHUNK_PROBS = [1.0, 0.0, 0.0]
    from datasets import AlignmentDataset, AlignmentCollator
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    if processor.tokenizer.convert_tokens_to_ids('<|file_sep|>') != cfg.SPECIAL_TOKENS.CLS_TOKEN_ID:
        raise ValueError('CLS token ID mismatch with public tokenizer')
    if processor.tokenizer.convert_tokens_to_ids('<|fim_pad|>') != cfg.SPECIAL_TOKENS.ALIGN_END_TOKEN_ID:
        raise ValueError('Alignment separator token ID mismatch')
    dataset = AlignmentDataset(mode='eval', video_paths_json=str(Path(root)/'train/piper_video_paths.json'), processor=processor)
    # Index0 is the canonical anchor. Use index1 to test distinct main/reference videos.
    sample = dataset[1]
    if sample['name'] == sample['ref_name']:
        raise ValueError('Expected distinct main/reference sample')
    started = time.monotonic()
    collated = AlignmentCollator(processor=processor, mode='eval')([sample])
    inputs = collated['qwen_input']
    ids = inputs['input_ids']
    cls_counts = (ids == cfg.SPECIAL_TOKENS.CLS_TOKEN_ID).sum(-1)
    expected = inputs['num_mains']+inputs['num_refs']
    torch.testing.assert_close(cls_counts, expected)
    if ids.shape[0] != 2 or cls_counts.tolist() != [4, 4]:
        raise ValueError('Packed Main/Ref or anchor count mismatch')
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite processor tensor: {key}')
    grid = inputs['video_grid_thw']
    expected_video_tokens = int((grid.prod(-1)//4).sum())
    config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    actual_video_tokens = int((ids == config.video_token_id).sum())
    if expected_video_tokens != actual_video_tokens:
        raise ValueError('Visual placeholder token/grid mismatch')
    report = dict(processor_passed=True, main=sample['name'], reference=sample['ref_name'],
        tensor_shapes={k:list(v.shape) for k,v in inputs.items() if torch.is_tensor(v)},
        cls_counts=cls_counts.tolist(), video_tokens=actual_video_tokens,
        processor_seconds=time.monotonic()-started, tiny_forward_requested=tiny_forward,
        pretrained_model_forward=False, gpu_test=False, progress_generated=False)
    if tiny_forward:
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
        from monkey_patch_forward import replace_qwen3_with_mixed_modality_forward
        from models import BaseModel
        replace_qwen3_with_mixed_modality_forward()
        backbone = Qwen3VLForConditionalGeneration(config).float().eval()
        # Exercise the actual HOST BaseModel.forward/CLS extraction without invoking
        # its8B from_pretrained constructor. No random tensor is saved as an artifact.
        wrapper = BaseModel.__new__(BaseModel)
        torch.nn.Module.__init__(wrapper)
        wrapper.base_model, wrapper.num_steps = backbone, cfg.DATA.NUM_STEPS
        started = time.monotonic()
        with torch.no_grad():
            features = wrapper({'qwen_input': inputs})
        if features.shape != (8,128) or not torch.isfinite(features).all() or (features.abs().sum(-1) == 0).any():
            raise ValueError(f'Invalid tiny-model features: {features.shape}')
        report['tiny_forward'] = dict(passed=True, shape=list(features.shape), seconds=time.monotonic()-started,
                                      initialization='RANDOM, interface test only, not trained GT')
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','weights','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--tiny-forward', action='store_true')
    args = p.parse_args()
    smoke(args.root, args.weights, args.output, args.tiny_forward)
