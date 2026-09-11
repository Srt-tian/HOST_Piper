from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'alignment'))
from piper_dtw_eval_helpers import install_last_hidden_only

class Output:
    def __init__(self,last):
        self.last_hidden_state=last
        self.hidden_states=None

class Base(torch.nn.Module):
    def forward(self,x,output_hidden_states=True,use_cache=True):
        result=Output(x*2)
        if output_hidden_states: result.hidden_states=(x,x*2)
        return result

class Causal(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=Base()
    def forward(self,x,**kwargs):
        return self.model(x,**kwargs)

def test_inference_hidden_capture_exact():
    causal=Causal()
    model=type('Model',(),{'cnn':type('CNN',(),{'base_model':causal})()})()
    x=torch.ones(3)
    with torch.no_grad():
        expected=causal(x).hidden_states[-1]
        handles=install_last_hidden_only(model)
        actual=causal(x)
        assert len(actual.hidden_states)==1
        torch.testing.assert_close(expected,actual.hidden_states[-1],rtol=0,atol=0)
    for handle in handles: handle.remove()

def test_inference_adapter_rejects_grad_enabled():
    causal=Causal()
    model=type('Model',(),{'cnn':type('CNN',(),{'base_model':causal})()})()
    handles=install_last_hidden_only(model)
    with pytest.raises(ValueError,match='inference-only'):
        causal(torch.ones(3))
    for handle in handles: handle.remove()
