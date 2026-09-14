"""Native-row Piper supervision: observation t -> executed states t+1..t+H."""
import numpy as np


def valid_future_starts(timestamps, source_indices, horizon, fps=30, max_gap_periods=1.5):
    times = np.asarray(timestamps, dtype=np.float64)
    indices = np.asarray(source_indices)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Finite strictly increasing timestamps required')
    if indices.shape != times.shape or indices.dtype.kind not in 'iu' or not np.array_equal(indices, np.arange(len(times))):
        raise ValueError('Original contiguous source rows required; no static filtering')
    if type(horizon) is not int or horizon < 1 or not np.isfinite(fps) or fps <= 0 or max_gap_periods <= 0:
        raise ValueError('Positive horizon/fps/gap threshold required')
    bad = np.diff(times) > max_gap_periods / fps + 1e-7
    prefix = np.concatenate(([0], np.cumsum(bad)))
    starts = np.arange(max(0, len(times) - horizon), dtype=np.int64)
    return starts[prefix[starts + horizon] == prefix[starts]]


def future_indices(start, horizon, total):
    if any(type(x) is not int for x in (start, horizon, total)) or start < 0 or horizon < 1 or start + horizon >= total:
        raise ValueError('Future action window outside episode')
    return np.arange(start + 1, start + horizon + 1, dtype=np.int64)
