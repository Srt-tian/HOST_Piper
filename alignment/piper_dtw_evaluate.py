"""Bounded held-out HOST DTW evaluation alongside training, no optimizer/GT mutation."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'data_preprocessing/piper'))

def run(args):
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from config import apply_eval_overrides
    from piper_profile import apply_profile,enable_strict_loading
    from piper_smoke import make_model,to_device
    from piper_dtw_eval_helpers import select_pairs,install_last_hidden_only,quality
    from datasets import AlignmentDataset,AlignmentCollator
    from algos.algorithm import RefEmbeddingCache
    from build_progress import anchor_frames
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    output=Path(args.output)
    if output.exists(): raise ValueError('Use a new evaluation output')
    if shutil.disk_usage(output.parent).free < 40*1024**3:
        raise ValueError('Need40GiB free to preserve training checkpoint rotation headroom')
    if args.device=='cuda':
        torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info()
        if free < 30*1024**3: raise ValueError('Insufficient shared-GPU headroom; training untouched')
        torch.cuda.set_per_process_memory_fraction(22*1024**3/total,0)
        torch.cuda.reset_peak_memory_stats()
    apply_eval_overrides()
    cfg=apply_profile(args.root,1)
    enable_strict_loading()
    cfg.EVAL.REF_CACHE_MAIN_ONLY=False
    cfg.EVAL.REF_CACHE_MAXSIZE=4
    os.environ['HOST_ALIGNMENT_MODEL_PATH']=args.weights
    os.environ['HOST_ALIGNMENT_LOGITS_TO_KEEP']='1'
    os.environ['HOST_VIDEO_DECODE_MODE']='sequential'
    splits=['train','val'] if args.split=='both' else [args.split]
    selection=[]
    for split in splits:
        selection.extend(dict(row,split=split) for row in select_pairs(args.root,args.per_task,split))
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard selection')
    selection=selection[args.shard_index::args.num_shards]
    if not selection: raise ValueError('Empty selection')
    if args.limit: selection=selection[:args.limit]
    output.mkdir()
    # Protect the exact snapshot from rotation without duplicating17.5GB.
    candidates=sorted(Path(args.checkpoints).glob('checkpoint-[0-9]*'))
    checkpoint=next(p for p in reversed(candidates) if p.is_dir()
        and not p.name.endswith('.incomplete') and (p/'manifest.json').is_file())
    if args.checkpoint_step is not None:
        checkpoint=Path(args.checkpoints)/f'checkpoint-{args.checkpoint_step:06d}'
    manifest=json.loads((checkpoint/'manifest.json').read_text())
    if manifest.get('complete') is not True or manifest.get('weights_only') is not True:
        raise ValueError('Complete model-only checkpoint required')
    pinned=output/'pinned_weights.bin'
    os.link(checkpoint/'pytorch_model.bin',pinned)
    digest=hashlib.sha256()
    with pinned.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''): digest.update(chunk)
    if digest.hexdigest()!=manifest['sha256']: raise ValueError('Checkpoint SHA256 mismatch')
    (output/'selection.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        manifest=manifest,selection=selection,selection_rule='within-task duration quantiles, no self pairs',
        inference_gpu_cap_gib=22,training_modified=False),indent=2))
    print(json.dumps(dict(stage='loading',checkpoint=checkpoint.name,pairs=len(selection))),flush=True)
    processor=AutoProcessor.from_pretrained(args.weights,local_files_only=True,trust_remote_code=False)
    datasets={split:AlignmentDataset(mode='eval',processor=processor,
        video_paths_json=str(Path(args.root)/split/'piper_video_paths.json')) for split in splits}
    collator=AlignmentCollator(processor=processor,mode='eval')
    model=make_model(args.weights,False)
    state=torch.load(pinned,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True)
    del state
    model=model.to(device=args.device,dtype=torch.bfloat16).eval()
    model.cnn.base_model.gradient_checkpointing_disable()
    handles=install_last_hidden_only(model)
    from piper_serial_inference import install_serial_cnn
    install_serial_cnn(model)
    started=time.monotonic()
    reports=[]
    with torch.no_grad(),(output/'records.jsonl').open('x') as stream:
        for i,selected in enumerate(selection):
            model._ref_cache=RefEmbeddingCache(maxsize=4)
            sample=datasets[selected['split']][selected['index']]
            if sample['name']!=selected['path'] or sample['ref_name']!=selected['reference']:
                raise ValueError('Unexpected sampled reference')
            data=to_device(collator([sample]),torch.device(args.device))
            steps=torch.cat([data['chosen_steps'],data['ref_chosen_steps']],dim=0)
            lengths=torch.cat([data['seq_lens'],data['ref_seq_lens']],dim=0)
            print(json.dumps(dict(stage='forward',pair=i,task=selected['task'],
                input_shape=list(data['qwen_input']['input_ids'].shape))),flush=True)
            embeddings=model(data,steps,lengths,training=False)
            loss,details=model.compute_loss(embeddings,steps,lengths,manifest['step'],False,
                frame_labels=data.get('frame_labels'),seq_labels=data.get('seq_labels'),metadata=data)
            metadata=embeddings['merged_metadata']
            if len(metadata)!=1: raise ValueError('Expected one merged pair')
            meta=metadata[0]
            pair_loss=details['per_sample_loss'].float().mean().item()
            record=dict(step=manifest['step'],task=selected['task'],split=selected['split'],loss=pair_loss,
                main_video_path=meta['main_name'],ref_video_path=meta['ref_name'],
                frame_paths=meta['frame_paths'],ref_frame_paths=meta['ref_frame_paths'])
            for key in ['forward_argmax_indices','backward_argmax_indices',
                        'forward_top5_probs','forward_top5_indices']:
                record[key]=details[key][0].float().cpu().tolist() if 'probs' in key else details[key][0].cpu().tolist()
            record['diagnostics']=quality(record,selected['frames'],selected['ref_frames'])
            record['main_frames']=selected['frames']; record['ref_frames']=selected['ref_frames']
            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            reports.append(dict(pair=i,task=selected['task'],**record['diagnostics']))
            print(json.dumps(dict(stage='pair_complete',**reports[-1]),allow_nan=False),flush=True)
            del data,embeddings,details,loss,steps,lengths
            model._ref_cache=RefEmbeddingCache(maxsize=4)
            gc.collect()
            if args.device=='cuda': torch.cuda.empty_cache()
    summary=dict(complete=True,checkpoint_step=manifest['step'],pairs=reports,
        export_passed=sum(r['export_gate_pass'] for r in reports),total=len(reports),
        semantic_quality_reviewed=False,training_ready=False,seconds=time.monotonic()-started)
    if args.device=='cuda':
        summary.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','weights','checkpoints','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--per-task',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--split',choices=['train','val','both'],default='val')
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--checkpoint-step',type=int)
    args=parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        output=Path(args.output)
        if output.is_dir():
            (output/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc))))
        raise
