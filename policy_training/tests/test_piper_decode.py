import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('piper_decode', ROOT/'src/self_grounded_prediction/datasets/custom/piper_decode.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


@pytest.fixture
def clip(tmp_path):
    import av
    path = tmp_path/'test.mkv'
    expected = [np.full((24, 32, 3), i*30, dtype=np.uint8) for i in range(5)]
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('ffv1', rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'bgr0'
        for array in expected:
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    return path, expected


def test_out_of_order_duplicates_exact(clip):
    path, expected = clip
    ids = [4, 0, 2, 2, 1]
    frames = decoder.load_frames_strict([f'video://{path}::{i}' for i in ids], (32, 24))
    for i, frame in zip(ids, frames): np.testing.assert_array_equal(frame, expected[i])
    assert frames[2] is not frames[3]


def test_out_of_range_fails_without_placeholder(clip):
    path, _ = clip
    with pytest.raises(ValueError, match='out-of-range'):
        decoder.load_frames_strict([f'video://{path}::5'], (32, 24))


@pytest.mark.parametrize('path', ['video://x::-1', 'video://x', 'video://::0'])
def test_malformed_path_rejected(path):
    with pytest.raises(ValueError): decoder.load_frames_strict([path], (32, 24))


def test_missing_video_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        decoder.load_frames_strict([f'video://{tmp_path}/absent.mp4::0'], (32, 24))
