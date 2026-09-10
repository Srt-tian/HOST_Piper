import json
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'alignment'))
from piper_weight_checkpoints import checkpoint_due, model_schema, validate_weights, prune_owned, OWNERSHIP
from piper_autodl_train import ds_config

def test_save_schedule():
    assert all(checkpoint_due(s) for s in [20,100,200,2900,3000])
    assert not any(checkpoint_due(s) for s in [0,1,19,21,99])
    assert checkpoint_due(3000,interval=700)

def test_only_model_gather_enabled():
    cfg=type('Config',(),{'DS_CONFIG':{'train_micro_batch_size_per_gpu':4}})()
    config=ds_config(cfg,True)
    assert config['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] is True
    assert config['gradient_accumulation_steps']==4

def test_weight_roundtrip(tmp_path):
    original=torch.nn.Linear(8,4).bfloat16()
    target=tmp_path/'pytorch_model.bin'
    torch.save(original.state_dict(),target)
    report=validate_weights(target,model_schema(original))
    assert report['model_tensors']==2 and report['bytes']>0
    other=torch.nn.Linear(8,4).bfloat16()
    other.load_state_dict(torch.load(target,weights_only=True),strict=True)
    for a,b in zip(original.parameters(),other.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('kind',['extra','missing','nan','dtype'])
def test_invalid_weights_rejected(tmp_path,kind):
    model=torch.nn.Linear(8,4).bfloat16()
    state=model.state_dict()
    if kind=='extra': state['optimizer']={}
    elif kind=='missing': del state['bias']
    elif kind=='nan': state['weight'][0,0]=float('nan')
    else: state['weight']=state['weight'].float()
    path=tmp_path/'bad.bin'
    torch.save(state,path)
    with pytest.raises(ValueError):
        validate_weights(path,model_schema(model))

def checkpoint(root,step):
    path=root/f'checkpoint-{step:06d}'
    path.mkdir()
    (path/'pytorch_model.bin').write_bytes(b'tiny-test-only')
    (path/'manifest.json').write_text(json.dumps(dict(owner=OWNERSHIP,complete=True,step=step)))
    return path

def test_retention_keeps_latest_three_and_partial(tmp_path):
    for step in [20,100,200,300]: checkpoint(tmp_path,step)
    partial=tmp_path/'checkpoint-000400.incomplete'
    partial.mkdir()
    assert prune_owned(tmp_path,3)==['checkpoint-000020']
    assert partial.is_dir() and (tmp_path/'checkpoint-000100').is_dir()

def test_retention_rejects_unknown_files_without_deleting(tmp_path):
    older=checkpoint(tmp_path,20)
    (older/'user_notes').write_text('preserve')
    checkpoint(tmp_path,100)
    with pytest.raises(ValueError,match='unknown'):
        prune_owned(tmp_path,1)
    assert (older/'pytorch_model.bin').is_file()

def test_retention_ignores_unowned_and_symlinks(tmp_path):
    unowned=checkpoint(tmp_path,20)
    (unowned/'manifest.json').write_text(json.dumps(dict(owner='someone-else',complete=True,step=20)))
    checkpoint(tmp_path,100)
    (tmp_path/'checkpoint-000001').symlink_to(unowned,target_is_directory=True)
    assert prune_owned(tmp_path,1)==[]
    assert (unowned/'pytorch_model.bin').is_file()

def test_conservative_restart_keeps_effective_batch():
    cfg=type('Config',(),{'DS_CONFIG':{'train_micro_batch_size_per_gpu':2}})()
    config=ds_config(cfg,True,8)
    assert config['train_micro_batch_size_per_gpu']*config['gradient_accumulation_steps']*4==64
