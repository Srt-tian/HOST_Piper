import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from self_grounded_prediction.policy_safety import summarize_window, record_microbatch, finish_metric_window, check_train_health


def test_sample_weighted_micro_metrics():
    totals = summarize_window([
        {'batch_size':1,'metrics':{'loss_total':4.,'has_task_video':None}},
        {'batch_size':3,'metrics':{'loss_total':8.,'has_task_video':1.}},
    ])
    assert totals['loss_total'] == [28.,4.]
    assert totals['has_task_video'] == [3.,3.]
    with pytest.raises(ValueError):
        summarize_window([])


def test_micro_records_release_graph_and_average(tmp_path, monkeypatch):
    trainer = SimpleNamespace(global_step=0, output_dir=str(tmp_path), run_start_time=0.,
        accelerator=SimpleNamespace(process_index=0, is_main_process=True, reduce=lambda x,reduction:x),
        model=SimpleNamespace(get_global_grad_norm=lambda:10.))
    sample = dict(action=torch.ones(1,30,22), proprio=torch.ones(1,1,20),
        progress_gt=torch.tensor([[0.,1.]]), video=torch.zeros(1,3,5,16,16),
        task_video=torch.zeros(1,3,21,16,16), agent_episode_dir=['episode0'],
        task_episode_dir=['ref0'], task_video_dropped=[False])
    for x in (2.,6.):
        record_microbatch(trainer,sample,torch.tensor(x,requires_grad=True),
                          {'loss_action':x,'optional':float('nan')})
    assert isinstance(trainer._metric_window[0]['metrics']['loss_total'], float)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    trainer.global_step=1
    assert finish_metric_window(trainer,'cpu') == 4.
    assert trainer._metric_window == []
    data=json.loads((tmp_path/'metrics.jsonl').read_text())
    assert data['samples']==2 and data['metrics']['optional'] is None
    lines=(tmp_path/'micro_rank0.jsonl').read_text().splitlines()
    assert len(lines)==2 and json.loads(lines[0])['agent_episodes']==['episode0']


def test_failure_health_written_before_raise(tmp_path,monkeypatch):
    for method,value in [('max_memory_reserved',65*2**30),('max_memory_allocated',40*2**30),
                         ('memory_allocated',30*2**30),('memory_reserved',45*2**30)]:
        monkeypatch.setattr(torch.cuda,method,lambda value=value:value)
    trainer=SimpleNamespace(global_step=3,output_dir=str(tmp_path),run_start_time=0.,
        cfg={'max_reserved_gib':64},model=SimpleNamespace(get_global_grad_norm=lambda:20.),
        accelerator=SimpleNamespace(process_index=3))
    with pytest.raises(RuntimeError,match='Peak memory'):
        check_train_health(trainer,torch.tensor(1.))
    record=json.loads((tmp_path/'health_rank3.jsonl').read_text())
    assert record['peak_allocated_gib']==40. and record['errors']


def test_cpu_offload_config_preserves_training_inputs(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/'deploy/autodl/policy_zero2_cpu.json').read_text())
    assert config['zero_optimization']['stage']==2
    assert config['zero_optimization']['offload_optimizer']['device']=='cpu'
    assert config['zero_force_ds_cpu_optimizer']
    for key in ('HOST_PREPARED_ROOT','HOST_DINO_REPO','HOST_DINO_WEIGHTS',
                'HOST_SIGLIP_WEIGHTS','HOST_INIT_CHECKPOINT','HOST_RUN_DIR'):
        monkeypatch.setenv(key,'/test')
    with initialize_config_dir(config_dir=str(root/'policy_training/configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=piper_autodl_cpu'])
    cfg=OmegaConf.to_container(cfg,resolve=True)
    assert cfg['require_cpu_optimizer_offload'] and cfg['max_reserved_gib']==64
    assert cfg['batch_size']==1 and cfg['gradient_accumulation_steps']==4
    assert cfg['save_at_steps']==[1,5]
    assert cfg['data']['train']['max_action_len']==30
    assert cfg['data']['train']['future_executed_state']
    assert cfg['model']['visual_encoder']['num_cameras']==3
