"""Bounded real-weight row-batching benchmark; no production outputs altered."""
import argparse,copy,gc,json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'alignment'))

def run(a):
 import torch,numpy as np,random
 from transformers import AutoProcessor
 from config import apply_eval_overrides
 from piper_profile import apply_profile,enable_strict_loading
 from piper_smoke import make_model,to_device
 from piper_dtw_eval_helpers import select_pairs,install_last_hidden_only
 from piper_serial_inference import install_serial_cnn
 from algos.algorithm import RefEmbeddingCache
 from datasets import AlignmentDataset,AlignmentCollator
 torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
 out=Path(a.output)
 if out.exists():raise ValueError('New output required')
 free,total=torch.cuda.mem_get_info()
 if free<48*1024**3:raise ValueError('Need48GiB free for isolated benchmark')
 torch.cuda.set_per_process_memory_fraction(42*1024**3/total)
 apply_eval_overrides();cfg=apply_profile(a.root,1);enable_strict_loading()
 cfg.EVAL.REF_CACHE_MAIN_ONLY=False;cfg.EVAL.REF_CACHE_MAXSIZE=4
 os.environ['HOST_ALIGNMENT_MODEL_PATH']=a.weights
 os.environ['HOST_ALIGNMENT_LOGITS_TO_KEEP']='1'
 os.environ['HOST_VIDEO_DECODE_MODE']='sequential'
 proc=AutoProcessor.from_pretrained(a.weights,local_files_only=True)
 ds=AlignmentDataset(mode='eval',processor=proc,video_paths_json=str(Path(a.root)/'val/piper_video_paths.json'))
 selected=select_pairs(a.root,2)[0]
 data=to_device(AlignmentCollator(processor=proc,mode='eval')([ds[selected['index']]]),torch.device('cuda'))
 model=make_model(a.weights,False)
 state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
 model.load_state_dict(state,strict=True);del state
 model=model.to('cuda',dtype=torch.bfloat16).eval()
 model.cnn.base_model.gradient_checkpointing_disable();install_last_hidden_only(model)
 original=model.cnn.forward
 steps=torch.cat([data['chosen_steps'],data['ref_chosen_steps']],0)
 lengths=torch.cat([data['seq_lens'],data['ref_seq_lens']],0)
 rows=[];base=None;base_idx=None
 with torch.no_grad():
  for n in (1,2,4,8):
   model.cnn.forward=original;install_serial_cnn(model,n)
   model._ref_cache=RefEmbeddingCache(maxsize=4)
   gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   torch.cuda.synchronize();t=time.monotonic()
   try:
    e=model(copy.deepcopy(data),steps,lengths,training=False)
    loss,d=model.compute_loss(e,steps,lengths,1500,False,frame_labels=data.get('frame_labels'),seq_labels=data.get('seq_labels'),metadata=data)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-t
    feats=e['embs'].detach().float().cpu()
    idx=d['forward_argmax_indices'][0].cpu()
    if base is None:base=feats.clone();base_idx=idx.clone()
    delta=float((feats-base).abs().max())
    close=bool(torch.allclose(feats,base,rtol=.01,atol=.01))
    row=dict(rows_per_forward=n,seconds=elapsed,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
      embedding_max_abs=delta,embeddings_close=close,argmax_equal=bool(torch.equal(idx,base_idx)))
    rows.append(row);print(json.dumps(row),flush=True)
    del e,d,loss,feats,idx
   except torch.cuda.OutOfMemoryError as ex:
    rows.append(dict(rows_per_forward=n,oom_at_42gib_cap=True));print(json.dumps(rows[-1]),flush=True)
    break
 result=dict(checkpoint=a.checkpoint,results=rows,shared_gpu_benchmark=True,production_modified=False)
 with out.open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('root','weights','checkpoint','output'):p.add_argument('--'+name,required=True)
 run(p.parse_args())
