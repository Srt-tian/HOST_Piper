"""Four-GPU RANDOM tiny-model test, not a trained alignment checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'alignment'))

def run(output):
    import torch
    import torch.distributed as dist
    import deepspeed
    from piper_weight_checkpoints import save_weights
    rank=int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    deepspeed.init_distributed()
    root=Path(output)
    if rank==0: root.mkdir(exist_ok=False)
    dist.barrier()
    def make():
        return torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,4))
    config=dict(train_micro_batch_size_per_gpu=1,gradient_accumulation_steps=1,
        bf16=dict(enabled=True),
        zero_optimization=dict(stage=3,offload_optimizer=dict(device='cpu',pin_memory=True),
                               stage3_gather_16bit_weights_on_model_save=True),
        optimizer=dict(type='AdamW',params=dict(lr=1e-3)))
    model=make()
    engine,_,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=config)
    x=torch.ones((1,8),device='cuda',dtype=torch.bfloat16)
    loss=engine(x).float().square().mean()
    engine.backward(loss)
    engine.step()
    # Exercise actual4-rank consolidation, strict reload and rolling pruning.
    for step in [20,100,200,300]:
        saved=save_weights(engine,root,step,keep=3)
    if rank==0:
        weights=torch.load(Path(saved['path'])/'pytorch_model.bin',map_location='cpu',weights_only=True)
        restored=make().bfloat16()
        restored.load_state_dict(weights,strict=True)
    for name,param in engine.module.named_parameters():
        with deepspeed.zero.GatheredParameters([param],modifier_rank=None):
            if rank==0:
                torch.testing.assert_close(param.detach().cpu(),weights[name],rtol=0,atol=0)
    # Check consolidation leaves the live ZeRO3 engine able to train.
    loss=engine(x).float().square().mean()
    engine.backward(loss)
    engine.step()
    assert torch.isfinite(loss)
    if rank==0:
        names=sorted(p.name for p in (root/'weights').iterdir())
        assert names==['checkpoint-000100','checkpoint-000200','checkpoint-000300']
        result=dict(passed=True,random_tiny_only=True,strict_reload=True,
                    live_tensor_equality=True,train_after_save=True,retained=names)
        (root/'verification.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    run(p.parse_args().output)
