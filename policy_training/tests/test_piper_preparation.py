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


windows = load('piper_time_windows', 'data_preprocessing/piper/time_windows.py')
pilot = load('piper_video_pilot', 'data_preprocessing/piper/prepare_seek_safe_pilot.py')
fetch = load('piper_artifacts', 'deploy/idc/fetch_alignment_weights.py')


def test_gap_windows_boundary():
    # Gap between indices3/4 invalidates starts2/3 for three-frame windows, not0/1/4.
    times = np.arange(8, dtype=float)/30
    times[4:] += 1/30
    assert windows.valid_starts(times, 3).tolist() == [0,1,4,5]
    assert windows.valid_starts(times, 20).size == 0
    assert windows.valid_starts([0,1/30,2/30], 3).tolist() == [0]
    with pytest.raises(ValueError, match='strictly increasing'):
        windows.valid_starts([0,0,1], 2)


def test_gap_windows_match_exhaustive():
    rng = np.random.default_rng(123)
    times = np.cumsum(rng.choice([1/30,2/30,3/30], size=100, p=[.9,.05,.05]))
    for size in (2,7,31,61):
        expected = [s for s in range(len(times)-size+1) if np.all(np.diff(times[s:s+size]) <= .05+1e-7)]
        assert windows.valid_starts(times, size).tolist() == expected


def test_pilot_selection_keeps_anchor_and_is_order_independent():
    pool = [dict(source_path=f'/source/ep_{i}') for i in range(20)]
    a = pilot.select(pool, '/source/ep_8', 4)
    b = pilot.select(list(reversed(pool)), '/source/ep_8', 4)
    assert a == b and a[0]['source_path'] == '/source/ep_8'
    assert len({ep['source_path'] for ep in a}) == 4
    with pytest.raises(ValueError, match='anchor'):
        pilot.select(pool, '/unknown', 4)


def test_artifact_git_and_lfs_integrity(tmp_path):
    import hashlib
    artifact = tmp_path/'config.json'
    contents = b'{"test":true}'
    artifact.write_bytes(contents)
    row = dict(size=len(contents), blob_id=hashlib.sha1(f'blob {len(contents)}\0'.encode()+contents).hexdigest())
    assert fetch.digest(artifact, row)
    row['sha256'] = hashlib.sha256(contents).hexdigest()
    assert fetch.digest(artifact, row)
    artifact.write_bytes(b'bad')
    assert not fetch.digest(artifact, row)


@pytest.mark.parametrize('status,content_range', [(200,None), (206,'bytes 0-99/100')])
def test_resume_rejects_wrong_range_without_modifying_partial(tmp_path, monkeypatch, status, content_range):
    import json
    row = dict(name='model-test.safetensors', size=40*1024**2, sha256='0'*64)
    plan = tmp_path/'transfer_plan.json'
    plan.write_text(json.dumps(dict(revision=fetch.REVISION, output=str(tmp_path),
        pilot={'status':206}, files=[row], total_bytes=row['size'])))
    partial = tmp_path/'model-test.safetensors.partial'
    partial.write_bytes(b'preserve-this-prefix')
    monkeypatch.setattr(fetch, 'OUTPUT', tmp_path)
    monkeypatch.setattr(fetch, 'PLAN', plan)
    monkeypatch.setattr(fetch, 'validate_host', lambda: None)
    class Response:
        def __enter__(self):
            self.status = status
            self.headers = {'Content-Range':content_range}
            return self
        def __exit__(self, *args):
            return False
        def read(self, size):
            raise AssertionError('Must reject before reading/appending')
    monkeypatch.setattr(fetch, 'request', lambda *args: Response())
    with pytest.raises(ValueError, match='Resume response'):
        fetch.download(resume=True)
    assert partial.read_bytes() == b'preserve-this-prefix'
