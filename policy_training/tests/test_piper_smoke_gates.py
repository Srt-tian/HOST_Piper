import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pose_ranges_are_diagnostics_not_automatic_conventions():
    audit = load('pose_audit_test', 'data_preprocessing/piper/audit_pose_ranges.py')
    values = np.zeros((10,12))
    values[:,0] = .5
    values[:,3] = np.pi
    result = audit.pose_summary(values)
    assert result['left']['position_norm_p99'] == .5
    assert result['left']['fraction_angles_above_pi'] == 0
    assert result['per_dimension']['median'][3] == np.pi
    with pytest.raises(ValueError, match='finite'):
        audit.stats([[float('nan')]])
    with pytest.raises(ValueError, match='nonempty'):
        audit.stats(np.empty((0,12)))


@pytest.mark.parametrize('mode', ['train','reload'])
def test_gpu_smoke_requires_explicit_approval(monkeypatch, mode):
    monkeypatch.syspath_prepend(str(ROOT/'alignment'))
    monkeypatch.delenv('HOST_SMOKE_APPROVED', raising=False)
    monkeypatch.setenv('WORLD_SIZE', '4')
    module = load('piper_smoke_gate', 'alignment/piper_smoke.py')
    with pytest.raises(ValueError, match='approval'):
        module.run(argparse.Namespace(mode=mode))


def test_gpu_smoke_rejects_wrong_world_size(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'alignment'))
    monkeypatch.setenv('HOST_SMOKE_APPROVED', '1')
    monkeypatch.setenv('WORLD_SIZE', '1')
    module = load('piper_smoke_world', 'alignment/piper_smoke.py')
    with pytest.raises(ValueError, match='four-rank'):
        module.run(argparse.Namespace(mode='train'))
    cfg = module.ds_config()
    assert cfg['zero_optimization']['stage'] == 3
    assert cfg['train_micro_batch_size_per_gpu'] == 1
    assert cfg['gradient_accumulation_steps'] == 1
    assert cfg['bf16']['enabled'] is True


def test_smoke_preflight_rejects_external_output(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'deploy/idc'))
    module = load('piper_smoke_preflight', 'deploy/idc/preflight_alignment_smoke.py')
    with pytest.raises(ValueError, match='isolated IDC'):
        module.check('/tmp/other', '/tmp/weights', '/tmp/output')
