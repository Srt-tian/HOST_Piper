import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('official_probe', ROOT/'alignment/piper_official_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_separate_approval(monkeypatch):
    monkeypatch.setenv('HOST_SMOKE_APPROVED', '1')
    monkeypatch.delenv('HOST_OFFICIAL_PROBE_APPROVED', raising=False)
    with pytest.raises(ValueError, match='approval'):
        probe.require_approval()


def test_four_rank_gate(monkeypatch):
    monkeypatch.setenv('HOST_OFFICIAL_PROBE_APPROVED', '1')
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(ValueError, match='four ranks'):
        probe.require_approval()


def test_original_launcher_accumulation_and_schedule():
    cfg = type('Config', (), {'DS_CONFIG': {
        'train_micro_batch_size_per_gpu':4,
        'scheduler':{'type':'WarmupCosineLR','params':{'total_num_steps':3000}}
    }})()
    ds = probe.effective_ds(cfg)
    assert ds['gradient_accumulation_steps'] == 4
    assert ds['train_micro_batch_size_per_gpu'] == 4
    assert ds['zero_optimization']['stage'] == 3
    assert ds['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] is True
    assert ds['scheduler']['params']['total_num_steps'] == 3000
    ds['scheduler']['params']['total_num_steps'] = 10
    assert cfg.DS_CONFIG['scheduler']['params']['total_num_steps'] == 3000


def test_no_overwrite(tmp_path):
    path = tmp_path/'result.json'
    probe.write_json(path, {'complete':True})
    with pytest.raises(FileExistsError):
        probe.write_json(path, {'complete':False})
    assert json.loads(path.read_text())['complete']
