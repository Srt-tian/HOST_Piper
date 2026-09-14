import importlib.util,json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('candidates',Path(__file__).resolve().parents[2]/'data_preprocessing/piper/assemble_policy_candidates.py')
candidates=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidates)


def fixture(tmp):
    converted=tmp/'converted';visual=tmp/'visual';labels=tmp/'labels';raw=tmp/'raw';raw.mkdir()
    def write(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
    paths=[]
    for i in range(3):
        rel=Path('episodes/train/source')/f'ep_{i}'
        ep=converted/rel;vp=visual/rel;ep.mkdir(parents=True);vp.mkdir(parents=True)
        videos={}
        for cam in ('cam_front','cam_left','cam_right'):
            src=raw/f'{i}_{cam}.mp4';src.write_bytes(b'unit-test-only-not-a-real-video')
            (ep/(cam+'.mp4')).symlink_to(src);videos[cam]=str(src)
        p=dict(source_id='source',episode_index=i,split='train',task='test',videos=videos)
        write(ep/'provenance.json',p);write(vp/'provenance.json',dict(p,frames=2))
        write(ep/(ep.name+'.json'),dict(data=[{},{}]));(ep/'instruction.txt').write_text('Synthetic fixture')
        paths.append(str(ep))
    for split in ('train','val'):write(converted/split/'piper_video_paths.json',paths if split=='train' else [])
    write(converted/'joint_action_mapping/piper_joint_action_mapping.json',{});write(converted/'manifest.json',{})
    anchor=visual/'episodes/train/source/ep_0'
    write(labels/'installed_anchor_provenance.json',dict(groups=[dict(split='train',task='test',reviewed=True,anchor=str(anchor))]))
    write(labels/'summary.json',dict(generated=1,alignment_model_id='synthetic-test-only'))
    write(labels/'episodes/train/source/ep_1/info_dtw.json',dict(ref_path=str(anchor),aligned_progress={'0':.5,'1':1.},coupling=dict(alignment_model_id='synthetic-test-only')))
    return converted,visual,labels


def test_candidate_mapping_and_no_rejected_peer(tmp_path):
    roots=fixture(tmp_path);out=tmp_path/'output'
    result=candidates.assemble(*roots,out)
    assert result['episodes']==2 and result['model_derived_pairs']==1 and result['canonical_axes']==1
    assert result['training_ready'] is False and result['text_embeddings_generated'] is False
    ep=out/'episodes/train/source/ep_1';anchor=out/'episodes/train/source/ep_0'
    assert json.loads((ep/'info_dtw.json').read_text())['ref_path']==str(anchor)
    assert json.loads((ep/'task_paths.json').read_text())['same']==[str(anchor)]
    axis=json.loads((anchor/'info_dtw.json').read_text())
    assert axis['coupling']['is_model_derived_pair'] is False
    assert not (roots[0]/'episodes/train/source/ep_1/info_dtw.json').exists()
    assert not (out/'episodes/train/source/ep_2').exists()


def test_wrong_reference_rejected_before_writes(tmp_path):
    roots=fixture(tmp_path);file=roots[2]/'episodes/train/source/ep_1/info_dtw.json'
    label=json.loads(file.read_text());label['ref_path']=str(roots[1]/'episodes/train/source/ep_2');file.write_text(json.dumps(label))
    out=tmp_path/'output'
    with pytest.raises(ValueError,match='coordinate'):candidates.assemble(*roots,out)
    assert not out.exists()


def test_progress_rejects_missing_rows():
    with pytest.raises(ValueError,match='every original'):
        candidates.validated_progress(dict(aligned_progress={'0':0.5}),2)
