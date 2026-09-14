"""Build a NEW candidate policy view; explicit canonical axes, no synthetic peer DTW."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def validated_progress(label, frames):
    values=label['aligned_progress']
    if set(values)!=set(map(str,range(frames))):
        raise ValueError('Progress keys must cover every original source row')
    array=np.array([values[str(i)] for i in range(frames)],dtype=float)
    if not np.isfinite(array).all() or np.any(array<0) or np.any(array>1) or np.any(np.diff(array)<-1e-7):
        raise ValueError('Invalid/nonmonotonic progress')
    if np.ptp(array)<.05:raise ValueError('Degenerate progress')


def canonical_axis(path,frames,model_id):
    if frames<2:raise ValueError('Canonical axis requires at least2 frames')
    return dict(ref_path=str(path),aligned_progress={str(i):(i+1)/frames for i in range(frames)},
        coupling=dict(method='canonical_reference_coordinate_axis',alignment_model_id=model_id,
            is_model_derived_pair=False,explanation='Reference timeline defines coordinate axis; never fallback for non-reference episodes'))


def assemble(converted,visual,labels,output):
    converted,visual,labels=map(lambda p:Path(p).resolve(),(converted,visual,labels))
    output=Path(output)
    if not output.is_absolute() or output.exists():raise ValueError('New absolute output required')
    summary=read(labels/'summary.json');model_id=summary['alignment_model_id']
    anchors=read(labels/'installed_anchor_provenance.json')['groups']
    canonical={}
    for a in anchors:
        key=(a['split'],a['task'])
        if a['reviewed'] is not True or key in canonical:raise ValueError('Unreviewed/duplicate canonical axis')
        canonical[key]=Path(a['anchor']).resolve()
    pending={};paths_by_group={};seen=set();paired=0
    for split in ('train','val'):
        for raw_path in read(converted/split/'piper_video_paths.json'):
            ep=Path(raw_path).resolve()
            if ep in seen:raise ValueError('Duplicate or cross-split source episode')
            seen.add(ep)
            relative=ep.relative_to(converted);v=visual/relative
            p=read(ep/'provenance.json');vp=read(v/'provenance.json')
            if any(p[k]!=vp[k] for k in ('source_id','episode_index','task','split')) or p['split']!=split:
                raise ValueError('Visual/policy identity mismatch')
            for cam in ('cam_front','cam_left','cam_right'):
                if Path(p['videos'][cam]).resolve()!=Path(vp['videos'][cam]).resolve():
                    raise ValueError('Visual/policy camera source mismatch')
                if not (ep/(cam+'.mp4')).is_file():raise ValueError('Missing camera file')
            n=len(read(ep/(ep.name+'.json'))['data'])
            if n!=vp['frames']:raise ValueError('Visual/policy row-count mismatch')
            group=(split,p['task']);anchor=canonical[group]
            dest=output/relative
            source_label=labels/relative/'info_dtw.json'
            if v==anchor:
                if source_label.exists():raise ValueError('Unexpected model-derived canonical self record')
                label=canonical_axis(anchor,n,model_id)
            elif source_label.is_file():
                label=read(source_label);paired+=1
                if Path(label['ref_path']).resolve()!=anchor:raise ValueError('Wrong canonical progress coordinate')
                if label['coupling']['alignment_model_id']!=model_id:raise ValueError('Mixed alignment checkpoints')
                label['coupling']['source_label_sha256']=hashlib.sha256(source_label.read_bytes()).hexdigest()
            else:continue
            validated_progress(label,n)
            label['coupling']['source_visual_reference']=label['ref_path']
            label['ref_path']=str(output/anchor.relative_to(visual))
            paths_by_group.setdefault(group,[]).append(str(dest))
            pending[dest]=(ep,p,label,n)
    if paired!=summary['generated']:raise ValueError('Candidate label/source coverage mismatch')
    for group,paths in paths_by_group.items():
        target=str(output/canonical[group].relative_to(visual))
        if target not in paths or len(paths)<2:raise ValueError('Missing reference/independent peers')
    # All semantic/path/count checks above precede writes; no existing dataset is edited.
    output.mkdir(parents=True)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,allow_nan=False)
    for dest,(ep,p,label,n) in pending.items():
        dest.mkdir(parents=True)
        os.link(ep/(ep.name+'.json'),dest/(dest.name+'.json'))
        os.link(ep/'instruction.txt',dest/'instruction.txt')
        for cam in ('cam_front','cam_left','cam_right'):
            (dest/(cam+'.mp4')).symlink_to((ep/(cam+'.mp4')).resolve())
        provenance=copy.deepcopy(p);provenance.update(output=str(dest),source_converted_episode=str(ep))
        write(dest/'provenance.json',provenance)
        write(dest/'info_dtw.json',label)
        write(dest/'task_paths.json',dict(same=[x for x in paths_by_group[(p['split'],p['task'])] if x!=str(dest)]))
    for split in ('train','val'):
        write(output/split/'piper_video_paths.json',[str(d) for d,(_,p,_,_) in pending.items() if p['split']==split])
    write(output/'cam_mapping/piper_cam_mapping.json',{str(d.parent):['cam_front','cam_left','cam_right'] for d in pending})
    write(output/'joint_action_mapping/piper_joint_action_mapping.json',read(converted/'joint_action_mapping/piper_joint_action_mapping.json'))
    write(output/'manifest.json',read(converted/'manifest.json'))
    result=dict(episodes=len(pending),model_derived_pairs=paired,canonical_axes=len(canonical),
        frames=sum(x[3] for x in pending.values()),groups={'/'.join(k):len(v) for k,v in paths_by_group.items()},
        training_ready=False,semantic_quality_reviewed=False,text_embeddings_generated=False,
        normalization='Original conversion train-only extrema retained; includes numerically rejected train episodes',
        source_converted=str(converted),source_labels=str(labels),alignment_model_id=model_id)
    write(output/'candidate_summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('converted','visual','labels','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(assemble(a.converted,a.visual,a.labels,a.output)),flush=True)
