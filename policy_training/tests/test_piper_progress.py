import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


progress = load('piper_progress', 'data_preprocessing/piper/build_progress.py')
temporal = load('piper_temporal', 'data_preprocessing/piper/audit_temporal_decode.py')
eval_pairs = load('piper_eval_pairs', 'data_preprocessing/piper/prepare_eval_pairs.py')


def record(context=1, main='/main', ref='/ref'):
    anchors = np.linspace(0, 99, 24, dtype=int)
    def paths(ep):
        return [f'video://{ep}/{cam}.mp4::{max(0, int(t)-10*(context-1-c))}::224x224'
                for t in anchors for c in range(context) for cam in progress.CAMERAS]
    return dict(loss=0.01, main_video_path=main, ref_video_path=ref,
                frame_paths=paths(main), ref_frame_paths=paths(ref),
                forward_argmax_indices=list(range(24)))


@pytest.mark.parametrize('context', [1, 2])
def test_three_camera_grouping(context):
    r = record(context)
    dense = progress.align_record(r, 100, 100, context)
    np.testing.assert_allclose(list(dense['aligned_progress'].values()), (np.arange(100)+1)/100)
    assert dense['coupling']['matched_points'] == 24


def test_dtw_not_main_index_and_pause_is_preserved():
    r = record()
    r['forward_argmax_indices'][8:11] = [8, 8, 8]
    dense = progress.align_record(r, 100, 100, 1)['aligned_progress']
    # Different main frames mapped to one reference frame form a real pause.
    assert dense['35'] == dense['40']
    assert dense['40'] != pytest.approx(41/100)


@pytest.mark.parametrize('mutation, match', [
    (lambda r: r['frame_paths'].pop(), 'group size'),
    (lambda r: r['forward_argmax_indices'].pop(), 'length/type'),
    (lambda r: r.update(loss=float('nan')), 'alignment loss'),
    (lambda r: r['forward_argmax_indices'].__setitem__(12, -1), 'Out-of-range'),
    (lambda r: r['forward_argmax_indices'].__setitem__(12, 24), 'Out-of-range'),
    (lambda r: r['forward_argmax_indices'].__setitem__(12, 3), 'Backward'),
    (lambda r: r['frame_paths'].__setitem__(1, r['frame_paths'][0]), 'camera order'),
    (lambda r: r.update(forward_argmax_indices=[0]*24), 'coverage'),
])
def test_invalid_records_fail_closed(mutation, match):
    r = record()
    mutation(r)
    with pytest.raises(ValueError, match=match):
        progress.align_record(r, 100, 100, 1)


def test_timing_preserves_real_gaps():
    report = temporal.timing([0, 1/30, 2/30, 4/30])
    assert report['gaps_over_1p5_period'] == 1
    assert report['estimated_missing_periods'] == 1
    assert report['index_clock_drift_seconds'] == pytest.approx(1/30)
    with pytest.raises(ValueError, match='Non-increasing'):
        temporal.timing([0, 0])


@pytest.mark.parametrize('use_alias', [False, True])
def test_export_requires_review_and_records_provenance(tmp_path, use_alias):
    root = tmp_path/'visual'
    if use_alias:
        root.mkdir()
        alias = tmp_path/'idc_alias'
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    episodes = [root/'episodes/train/source'/f'ep_{i}' for i in range(2)]
    for ep in episodes:
        ep.mkdir(parents=True)
        (ep/'provenance.json').write_text(json.dumps(dict(task='bottle', frames=100)))
    for split, paths in [('train', [str(ep) for ep in episodes]), ('val', [])]:
        (root/split).mkdir()
        (root/split/'piper_video_paths.json').write_text(json.dumps(paths))
    anchors = {'groups': [dict(task='bottle', split='train', reviewed=False, anchor=str(episodes[1]))]}
    r = record(main=str(episodes[0]), ref=str(episodes[1]))
    dest = tmp_path/'labels'
    with pytest.raises(ValueError, match='visual review'):
        progress.export(root, [r], anchors, dest, 'synthetic-test-only', 1)
    assert not dest.exists()
    anchors['groups'][0]['reviewed'] = True
    result = progress.export(root, [r], anchors, dest, 'synthetic-test-only', 1)
    assert result['generated'] == 1 and not result['training_ready']
    label = json.loads((dest/episodes[0].relative_to(root)/'info_dtw.json').read_text())
    assert label['coupling']['alignment_model_id'] == 'synthetic-test-only'
    assert len(label['coupling']['source_record_sha256']) == 64
    assert not (episodes[0]/'info_dtw.json').exists()


@pytest.mark.parametrize('cross_split', [False, True])
def test_explicit_eval_references(tmp_path, cross_split):
    groups = []
    for split in ('train', 'val'):
        paths = []
        for i in range(2):
            ep = tmp_path/'episodes'/split/f'ep_{i}'
            ep.mkdir(parents=True)
            (ep/'provenance.json').write_text(json.dumps(dict(task='bottle', split=split)))
            paths.append(str(ep))
        (tmp_path/split).mkdir()
        (tmp_path/split/'piper_video_paths.json').write_text(json.dumps(paths))
        groups.append(dict(split=split, task='bottle', reviewed=True, anchor=paths[0]))
    if cross_split:
        groups[1]['anchor'] = groups[0]['anchor']
        with pytest.raises(ValueError, match='cross-split/task'):
            eval_pairs.prepare(tmp_path, {'groups': groups})
        assert not list(tmp_path.rglob('task_paths_eval.json'))
    else:
        result = eval_pairs.prepare(tmp_path, {'groups': groups})
        assert result['episodes'] == 4 and not result['progress_generated']
        for file in tmp_path.rglob('task_paths_eval.json'):
            pairs = json.loads(file.read_text())
            assert pairs['canonical_anchor_reviewed'] and len(pairs['same']) == 1
