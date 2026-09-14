from types import SimpleNamespace
from pathlib import Path
import torch
import pytest
from self_grounded_prediction.policy_safety import MODULES, adapt_camera, load_strict_host, save_model_only


def tiny():
    modules = {name: torch.nn.Linear(2, 2) for name in MODULES}
    modules['visual_encoder'] = torch.nn.Module()
    ve = modules['visual_encoder']
    ve.num_cameras = 3
    ve.camera_emb = torch.nn.Parameter(torch.zeros(3, 2))
    ve.pos_embed = torch.nn.Parameter(torch.zeros(1, 6, 4))
    return SimpleNamespace(**modules)


def test_strict_reject_missing_key(tmp_path):
    model = tiny()
    payload = {n: getattr(model, n).state_dict() for n in MODULES}
    del payload['mot']['bias']
    path = tmp_path/'bad.pt'
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_strict_host(model, path)


def test_camera_only_seeded():
    state = {'camera_emb': torch.ones(2,1920), 'pos_embed': torch.ones(1,512,4096), 'other':torch.ones(1)}
    a, b = adapt_camera(state,3), adapt_camera(state,3)
    assert a['camera_emb'].shape == (3,1920)
    assert a['pos_embed'].shape == (1,768,4096)
    assert torch.equal(a['camera_emb'], b['camera_emb'])
    assert a['other'] is state['other']
    assert state['camera_emb'].shape == (2,1920)


def test_native30_model_inputs_guard(monkeypatch):
    from self_grounded_prediction.models.wan22.self_grounded_predictor import SelfGroundedPredictor
    model = SimpleNamespace(device='cpu', torch_dtype=torch.float32,
        video_expert=SimpleNamespace(action_conditioned=False), proprio_encoder=None,
        visual_encoder=None, _encode_video_latents=lambda video,tiled:torch.zeros(1,48,2,2,2))
    sample = dict(video=torch.zeros(1,3,5,32,32), action=torch.zeros(1,30,22),
                  context=torch.ones(1,8,4096),context_mask=torch.ones(1,8,dtype=torch.bool))
    monkeypatch.delenv('HOST_POLICY_NATIVE_30HZ', raising=False)
    with pytest.raises(ValueError, match='divisible'):
        SelfGroundedPredictor.build_inputs(model,sample)
    monkeypatch.setenv('HOST_POLICY_NATIVE_30HZ','1')
    result = SelfGroundedPredictor.build_inputs(model,sample)
    assert result['action'].shape == (1,30,22)
    assert result['input_latents'].shape[2] == 2
    model.video_expert.action_conditioned=True
    with pytest.raises(ValueError, match='divisible'):
        SelfGroundedPredictor.build_inputs(model,sample)


def test_model_only_roundtrip_retention(tmp_path):
    model = tiny()
    accel = SimpleNamespace(is_main_process=True, wait_for_everyone=lambda:None,
        unwrap_model=lambda m:m, state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(
            deepspeed_config={'zero_optimization':{'stage':2}})))
    trainer = SimpleNamespace(model=model, accelerator=accel, weights_dir=str(tmp_path),
                              cfg={'keep_model_checkpoints':2}, global_step=0)
    for step in (1,2,3):
        trainer.global_step = step
        result = save_model_only(trainer)
        assert result['state_path'] is None
    assert sorted(p.name for p in tmp_path.glob('*.pt')) == ['step_000002.pt','step_000003.pt']
    restored = tiny()
    load_strict_host(restored, tmp_path/'step_000003.pt')
    for n in MODULES:
        for k,v in getattr(model,n).state_dict().items():
            assert torch.equal(v, getattr(restored,n).state_dict()[k])
    accel.state.deepspeed_plugin.deepspeed_config['zero_optimization']['stage']=3
    with pytest.raises(ValueError, match='ZeRO-2'):
        save_model_only(trainer)
