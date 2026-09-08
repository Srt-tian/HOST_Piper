import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


converter = module('piper_converter', 'data_preprocessing/piper/convert_lerobot.py')
camera = module('piper_camera', 'policy_training/scripts/piper_camera_checkpoint.py')
preflight = module('piper_preflight', 'policy_training/scripts/piper_preflight.py')


def test_target_semantics_and_units():
    table = pa.table({'state_end': [[100,200,300,0,0,90]*2]*3,
                      'action_end': [[400,500,600,0,0,45]*2]*3,
                      'observation.qpos': [[0]*6+[0.02]+[0]*6+[0.03]]*3,
                      'real_action': [[0]*6+[0.04]+[0]*6+[0.05]]*3,
                      'timestamp': [0,1/30,2/30]})
    rows, _ = converter.make_rows(table, {'position_unit':'mm','angle_unit':'deg'})
    assert rows[0]['follow_left_position'] == [0.1,0.2,0.3]
    assert rows[0]['master_left_position'] == [0.4,0.5,0.6]
    assert rows[0]['follow_left_rotation'][2] == pytest.approx(np.pi/2)
    assert rows[0]['follow_right_gripper'] == 0.03
    assert rows[0]['master_right_gripper'] == 0.05
    assert rows[0]['source_frame_index'] == 0  # no implicit shift


def test_unconfirmed_semantics_fail():
    with pytest.raises(ValueError, match='Confirm units'):
        converter.validate_manifest({'conventions': {'confirmed': False}})


def test_split_is_deterministic_and_task_local():
    eps = [dict(task=task, source_id=f'{task}_session', episode_index=i)
           for task in ('bottle','pen') for i in range(10)]
    converter.assign_splits(eps, 0.2, 42)
    reverse = [dict(e) for e in reversed(eps)]
    converter.assign_splits(reverse, 0.2, 42)
    assert {(e['task'],e['episode_index']):e['split'] for e in eps} == {
        (e['task'],e['episode_index']):e['split'] for e in reverse}
    for task in ('bottle','pen'):
        assert sum(e['split']=='val' for e in eps if e['task']==task) == 2


def test_stats_exclude_validation_outliers():
    def row(x):
        return {k: ([x]*3 if 'gripper' not in k else x)
                for k in converter.keys('follow')+converter.keys('master')}
    norm = converter.normalization([
        {'split':'train','rows':[row(0),row(1)]},
        {'split':'val','rows':[row(999)]}], 'executed_state')
    assert norm['action_keys'] == converter.keys('follow')
    assert norm['norm_min_delta']['follow_left_position']['delta'] == [1,1,1]
    assert norm['norm_min_delta']['follow_left_position_relative']['min'] == [-1,-1,-1]


def test_camera_adapter_only_changes_two_keys():
    original = {'camera_emb':torch.arange(8.).reshape(2,4),
                'pos_embed':torch.arange(24.).reshape(1,6,4), 'backbone.w':torch.ones(1)}
    out = camera.adapt_visual_state(original, [1,None,0])
    assert out['camera_emb'].shape == (3,4)
    assert out['pos_embed'].shape == (1,9,4)
    assert torch.equal(out['camera_emb'][0], original['camera_emb'][1])
    assert torch.equal(out['pos_embed'][:,6:], original['pos_embed'][:,:3])
    assert out['backbone.w'] is original['backbone.w']
    assert original['camera_emb'].shape == (2,4)
    with pytest.raises(ValueError):
        camera.adapt_visual_state(original, [2])


def test_upstream_vertical_concat_and_rotation():
    from self_grounded_prediction.datasets.custom.mydatasets import CustomDataset
    views = [[np.full((224,224,3), v, dtype=np.uint8)] for v in (0,127,255)]
    out = CustomDataset._frames_to_video_tensor(None, views)
    assert out.shape == (3,1,672,224)
    assert torch.all(out[:,:,0:224] == -1)
    assert torch.all(out[:,:,448:] == 1)
    raw, _, _ = CustomDataset._assemble_raw_actions(
        [{'follow_left_rotation':[0,0,np.pi/2]}], ['follow_left_rotation'], True)
    np.testing.assert_allclose(raw[0], [0,1,0,-1,0,0], atol=1e-6)


@pytest.mark.parametrize('task', ['piper_3cam_checkpoint_compat','piper_3cam_paper_objective'])
def test_hydra_profile(task, monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    for k in ('HOST_PREPARED_ROOT','HOST_DINO_REPO','HOST_DINO_WEIGHTS',
              'HOST_SIGLIP_WEIGHTS','HOST_INIT_CHECKPOINT','HOST_RUN_DIR'):
        monkeypatch.setenv(k, '/pfs/user/data/host_piper_paper/test_placeholder')
    with initialize_config_dir(config_dir=str(ROOT/'policy_training/configs'), version_base=None):
        cfg = compose(config_name='train', overrides=[f'task={task}'])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved['model']['visual_encoder']['num_cameras'] == 3
    assert resolved['data']['train']['processor']['action_output_dim'] == 22
    assert resolved['data']['train']['dataset_fps']['piper'] == 30
    assert resolved['model']['mot_checkpoint_mixed_attn']
    assert '/open_data/' not in json.dumps(resolved)
    if task.endswith('paper_objective'):
        assert resolved['model']['action_prediction_type'] == 'velocity'
        assert not resolved['data']['train']['use_relative_action']


def test_preflight_rejects_missing_progress(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'conventions':{'confirmed':True}}))
    ep = tmp_path/'episodes/episode_0'
    ep.mkdir(parents=True)
    (ep/'episode_0.json').write_text(json.dumps({'data':[{}]*5}))
    (ep/'provenance.json').write_text(json.dumps({'task':'bottle'}))
    for split, paths in [('train',[str(ep)]),('val',[])]:
        (tmp_path/split).mkdir()
        (tmp_path/split/'piper_video_paths.json').write_text(json.dumps(paths))
    with pytest.raises(FileNotFoundError, match='info_dtw.json'):
        preflight.check(tmp_path)


def test_conversion_loads_in_upstream(tmp_path):
    import av
    import pyarrow.parquet as pq
    from self_grounded_prediction.datasets.custom.mydatasets import CustomDataset
    raw = tmp_path/'raw'
    (raw/'meta').mkdir(parents=True)
    info = dict(codebase_version='v2.1', fps=30, chunks_size=1000,
                data_path='data/episode_{episode_index:06d}.parquet',
                video_path='videos/{video_key}/episode_{episode_index:06d}.mp4')
    (raw/'meta/info.json').write_text(json.dumps(info))
    (raw/'meta/episodes.jsonl').write_text('\n'.join(
        json.dumps(dict(episode_index=i,length=3)) for i in range(4)))
    (raw/'data').mkdir()
    for i in range(4):
        table = pa.table({'state_end':[[0,0,0,0,0,0]*2]*3,
            'action_end':[[0.1,0,0,0,0,0]*2]*3,
            'observation.qpos':[[0]*14]*3, 'real_action':[[0.1]*14]*3,
            'timestamp':[0,1/30,2/30]})
        pq.write_table(table, raw/f'data/episode_{i:06d}.parquet')
        for cam in converter.CAMERAS:
            video = raw/f'videos/observation.images.{cam}/episode_{i:06d}.mp4'
            video.parent.mkdir(parents=True,exist_ok=True)
            with av.open(str(video),'w') as c:
                stream = c.add_stream('mpeg4',rate=30)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                for _ in range(3):
                    frame = av.VideoFrame.from_ndarray(np.zeros((32,32,3),dtype=np.uint8),format='rgb24')
                    for packet in stream.encode(frame):
                        c.mux(packet)
                for packet in stream.encode():
                    c.mux(packet)
    manifest = dict(conventions=dict(confirmed=True,position_unit='m',angle_unit='rad',
        rpy='Rz(yaw)Ry(pitch)Rx(roll)',end_frame='synthetic_tcp',target_source='command',
        command_semantics_confirmed=True), sources=[dict(id='synthetic',task='test',
        instruction='Synthetic test only',root=str(raw))])
    out = tmp_path/'converted'
    episodes = converter.convert(manifest,out)
    ds = object.__new__(CustomDataset)
    ds.use_6d_rotation = ds.use_relative_action = True
    ds._joint_action_mapping_cache = {'piper':json.loads(
        (out/'joint_action_mapping/piper_joint_action_mapping.json').read_text())}
    data = ds._load_episode_raw({'dataset':'piper','video_dir':episodes[0]['output']})
    assert data is not None
    assert data['raw_actions'].shape == data['raw_joints'].shape == (3,20)
    assert data['raw_actions'][0,0] == pytest.approx(0.1)
    assert data['raw_joints'][0,0] == 0
    assert not list(out.rglob('info_dtw.json'))  # never fabricate progress
    assert not list(out.rglob('instruction.pt'))
    for ep in episodes:
        peers = json.loads((Path(ep['output'])/'task_paths.json').read_text())['same']
        assert all(f"/{ep['split']}/" in peer for peer in peers)
