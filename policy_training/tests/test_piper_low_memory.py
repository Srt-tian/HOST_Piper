import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'alignment'))
from piper_low_memory import check_paths
from piper_autodl_train import ds_config

def test_offload_preserves_batch_schedule():
    cfg=type('Config',(),{'DS_CONFIG':{
        'train_micro_batch_size_per_gpu':4,
        'scheduler':{'type':'WarmupCosineLR','params':{'total_num_steps':3000}}
    }})()
    actual=ds_config(cfg)
    assert actual['train_micro_batch_size_per_gpu']==4
    assert actual['gradient_accumulation_steps']==4
    assert actual['zero_optimization']['offload_optimizer']['device']=='cpu'
    assert actual['zero_optimization']['stage']==3
    assert actual['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] is False
    actual['scheduler']['params']['total_num_steps']=20
    assert cfg.DS_CONFIG['scheduler']['params']['total_num_steps']==3000

def test_output_not_overwritten(tmp_path):
    with pytest.raises(ValueError,match='new output'):
        check_paths(tmp_path,tmp_path,tmp_path)

def test_space_gate(tmp_path,monkeypatch):
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda _:type('Usage',(),{'free':109*1024**3})())
    with pytest.raises(ValueError,match='Reserve110GiB'):
        check_paths(tmp_path,tmp_path,tmp_path/'new')

def test_final_only_call_site():
    import ast
    tree=ast.parse((ROOT/'alignment/piper_autodl_train.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=='save_checkpoint']
    assert len(calls)==1
    assert any(k.arg=='tag' and isinstance(k.value,ast.Constant) and k.value.value=='final'
               for k in calls[0].keywords)
