import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('piper_future', Path(__file__).resolve().parents[1]/'src/self_grounded_prediction/datasets/custom/piper_future.py')
future = importlib.util.module_from_spec(spec)
spec.loader.exec_module(future)


def test_exact_future_state_not_current():
    state = np.arange(100*20).reshape(100, 20)
    ids = future.future_indices(10, 30, 100)
    np.testing.assert_array_equal(state[ids], state[11:41])
    assert ids[0] == 11 and ids[-1] == 40


def test_last_possible_window():
    times = np.arange(31)/30
    assert future.valid_future_starts(times, np.arange(31), 30).tolist() == [0]
    assert future.future_indices(0, 30, 31).tolist() == list(range(1,31))
    with pytest.raises(ValueError): future.future_indices(1, 30, 31)


def test_gaps_exclude_crossing_only():
    times = np.arange(70)/30
    times[35:] += 1/30
    got = future.valid_future_starts(times, np.arange(70), 30)
    expected = [s for s in range(40) if np.all(np.diff(times[s:s+31]) <= .05+1e-7)]
    assert got.tolist() == expected
    assert 4 in got and 5 not in got and 35 in got


def test_no_static_compression_or_bad_time():
    with pytest.raises(ValueError, match='source rows'):
        future.valid_future_starts([0,.033,.066], [0,2,3], 1)
    with pytest.raises(ValueError, match='timestamps'):
        future.valid_future_starts([0,0,.066], [0,1,2], 1)
    assert future.valid_future_starts([0,.033], [0,1], 30).size == 0


def test_actual_video_sampler_uses_safe_start_and_endpoint():
    import sys
    from types import SimpleNamespace
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from self_grounded_prediction.datasets.custom.mydatasets import CustomDataset
    sampler = SimpleNamespace(action_frames=30, action_video_freq_ratio=8, use_augmentation=False)
    sampler._get_target_size = lambda name: (32,24)
    sampler._downsample_segments_to_vae_indices = CustomDataset._downsample_segments_to_vae_indices
    seen = []
    def load(paths, target_size):
        seen.append(list(paths))
        return [Image.new('RGB', target_size) for _ in paths]
    sampler._load_frame = load
    paths = [[f'{cam}/{i}' for i in range(70)] for cam in range(3)]
    got = CustomDataset.random_frames_to_tensor(sampler, paths, 31,
        action_frames_list=[30,30], dataset_name='piper', eligible_starts=[35])
    assert got[1] == 35 and got[2] == [35,65]
    assert all(p[0].endswith('/35') and p[-1].endswith('/65') for p in seen)
    assert future.future_indices(got[1],30,70).tolist() == list(range(36,66))


def test_profile_applies_same_future_contract_to_val():
    import yaml
    path = Path(__file__).resolve().parents[1]/'configs/task/piper_3cam_future_state.yaml'
    cfg = yaml.safe_load(path.read_text())
    assert cfg['data']['train']['future_executed_state']
    assert cfg['data']['val']['future_executed_state']
    assert cfg['data']['train']['remove_static_frames'] is False
