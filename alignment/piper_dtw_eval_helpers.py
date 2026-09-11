"""Deterministic held-out selection and inference-only last-hidden-state capture."""
import json
from pathlib import Path

def select_pairs(root, per_task=2, split='val'):
    root=Path(root)
    groups={}
    entries=json.loads((root/split/'piper_video_paths.json').read_text())
    for index,path in enumerate(entries):
        ep=Path(path)
        prov=json.loads((ep/'provenance.json').read_text())
        refs=json.loads((ep/'task_paths_eval.json').read_text())
        if refs.get('canonical_anchor_reviewed') is not True or len(refs['same'])!=1:
            raise ValueError('Explicit reviewed canonical reference required')
        ref=Path(refs['same'][0])
        rp=json.loads((ref/'provenance.json').read_text())
        if (prov['task'],prov['split'])!=(rp['task'],rp['split']) or prov['split']!=split:
            raise ValueError('Cross-task/split reference rejected')
        if ep==ref: continue
        groups.setdefault(prov['task'],[]).append(dict(index=index,path=str(ep),
            reference=str(ref),frames=prov['frames'],ref_frames=rp['frames'],task=prov['task']))
    if set(groups)!={'basket','bottles','lemon','pens'}:
        raise ValueError('Expected four tasks')
    selected=[]
    for task,rows in sorted(groups.items()):
        rows.sort(key=lambda r:(r['frames'],r['path']))
        indices=range(len(rows)) if per_task==0 else sorted(set(
            min(len(rows)-1,int(len(rows)*(i+.5)/per_task)) for i in range(per_task)))
        selected.extend(rows[i] for i in indices)
    return selected

def install_last_hidden_only(model):
    """Equivalent final hidden state without retaining every decoder layer; eval only."""
    import torch
    holder={}
    causal=model.cnn.base_model
    def pre(module,args,kwargs):
        if torch.is_grad_enabled():
            raise ValueError('This memory adapter is inference-only')
        kwargs=dict(kwargs,output_hidden_states=False,use_cache=False)
        return args,kwargs
    def capture(module,args,output):
        holder['last']=output.last_hidden_state
    def finish(module,args,output):
        output.hidden_states=(holder.pop('last'),)
        return output
    handles=[causal.register_forward_pre_hook(pre,with_kwargs=True),
             causal.model.register_forward_hook(capture),
             causal.register_forward_hook(finish)]
    return handles

def quality(record,main_frames,ref_frames):
    import numpy as np
    from build_progress import align_record,anchor_frames
    main=anchor_frames(record['frame_paths'],record['main_video_path'],main_frames,1)
    ref=anchor_frames(record['ref_frame_paths'],record['ref_video_path'],ref_frames,1)
    matched=np.asarray(record['forward_argmax_indices'],dtype=int)
    changes=np.diff(matched)
    linear=np.rint(main/max(1,main_frames-1)*(ref_frames-1)).astype(int)
    mapped=ref[matched]
    longest=current=1
    for diff in changes:
        current=current+1 if diff==0 else 1
        longest=max(longest,current)
    result=dict(loss=record['loss'],anchors=len(main),
        backward_count=int((changes<0).sum()),coverage=float(np.ptp(matched)/len(ref)),
        max_jump=int(changes.max(initial=0)),longest_plateau_anchors=longest,
        mean_abs_deviation_from_time_fraction=float(np.mean(np.abs(mapped-linear))/ref_frames))
    try:
        align_record(record,main_frames,ref_frames,1)
        result.update(export_gate_pass=True,rejection=None)
    except ValueError as exc:
        result.update(export_gate_pass=False,rejection=str(exc))
    return result
