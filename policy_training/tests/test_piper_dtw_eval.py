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

from piper_serial_inference import split_qwen_rows, install_serial_cnn
from types import SimpleNamespace

def serial_inputs():
    return dict(input_ids=torch.tensor([[9,9,8,0],[9,9,9,9]]),
        attention_mask=torch.ones(2,4), num_mains=torch.tensor([1,1]),
        num_refs=torch.tensor([1,1]), cls_token_id=torch.tensor(8),
        pixel_values=None,image_grid_thw=None,
        pixel_values_videos=torch.arange(24).reshape(24,1),
        video_grid_thw=torch.tensor([[1,2,4],[1,4,4]]))

def serial_config():
    return SimpleNamespace(image_token_id=7,video_token_id=9,
        vision_config=SimpleNamespace(spatial_merge_size=2))

def test_serial_visual_slices():
    rows=split_qwen_rows(serial_inputs(),serial_config())
    assert len(rows)==2
    assert rows[0]['pixel_values_videos'].flatten().tolist()==list(range(8))
    assert rows[1]['pixel_values_videos'].flatten().tolist()==list(range(8,24))
    assert rows[1]['video_grid_thw'].tolist()==[[1,4,4]]

def test_serial_rejects_token_mismatch():
    data=serial_inputs(); data['input_ids'][1,0]=0
    with pytest.raises(ValueError,match='do not match'):
        split_qwen_rows(data,serial_config())

def test_serial_preserves_main_then_ref_order():
    class CNN:
        base_model=SimpleNamespace(config=serial_config())
        def forward(self,data):
            first=float(data['qwen_input']['pixel_values_videos'][0,0])
            return torch.tensor([[first],[first+100]])
    model=SimpleNamespace(cnn=CNN())
    install_serial_cnn(model)
    with torch.no_grad():
        got=model.cnn.forward({'qwen_input':serial_inputs()})
    assert got.flatten().tolist()==[0,8,100,108]
    with pytest.raises(ValueError,match='inference-only'):
        model.cnn.forward({'qwen_input':serial_inputs()})
