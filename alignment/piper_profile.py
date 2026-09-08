"""Explicit, visual-only Piper alignment profile and strict CPU data smoke.

Import apply_profile BEFORE constructing upstream datasets/models. Does not start
training, download weights, or produce progress GT. Safe for the isolated container.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np


def apply_profile(root, context_frames=1):
    from config import CONFIG
    root = Path(root).resolve()
    if context_frames not in (1, 2) or not (root/'cam_mapping/piper_cam_mapping.json').is_file():
        raise ValueError('Prepared Piper camera mapping and explicit context size required')
    CONFIG.DATA.CAM_MAPPING_DIR = str(root/'cam_mapping')
    CONFIG.DATA.USE_CAM_MAPPING = True
    CONFIG.DATA.NUM_STEPS = 3*context_frames
    CONFIG.DATA.TASK_PATHS_TRANSFORMS = {}
    CONFIG.DATA.DATASET_WEIGHTS = {'piper': 1.0}
    CONFIG.JOINTS.USE_JOINTS = False
    CONFIG.JOINTS.JOINT_ACTION_MAPPING_DIR = ''
    # Left/right camera identities are fixed; do not silently mirror embodiment.
    CONFIG.AUGMENTATION.RANDOM_FLIP = False
    CONFIG.LOGDIR = '/pfs/user/data/host_piper_paper/logs/alignment'
    return CONFIG


def strict_load_imgs(self, paths):
    """Same batched Decord -> PIL resize path; fail instead of substituting black RGB."""
    import decord
    from PIL import Image
    grouped, result = {}, [None]*len(paths)
    for position, path in enumerate(paths):
        if not isinstance(path, str) or not path.startswith('video://'):
            raise ValueError('Piper strict loader requires explicit video frame paths')
        parts = path[8:].rsplit('::', 2)
        if len(parts) != 3:
            raise ValueError('Piper strict loader requires explicit resize dimensions')
        video, index, dimensions = parts
        width, height = map(int, dimensions.split('x'))
        grouped.setdefault(video, []).append((position, int(index), width, height))
    for video, group in grouped.items():
        reader = decord.VideoReader(video, num_threads=1)
        indices = [entry[1] for entry in group]
        if any(index < 0 or index >= len(reader) for index in indices):
            raise ValueError('Out-of-range frame request')
        frames = reader.get_batch(indices).asnumpy()
        for frame, (position, _, width, height) in zip(frames, group):
            result[position] = Image.fromarray(frame).convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    if any(frame is None for frame in result):
        raise ValueError('Missing decoded frame')
    return result


def enable_strict_loading():
    import datasets
    original_load = datasets.AlignmentDataset._load_video_data_from_json
    def checked_load(self, index):
        info = self.video_paths[index]
        ep = Path(info['video_dir'] if isinstance(info, dict) else info)
        filename = 'task_paths_eval.json' if self.mode == 'eval' else 'task_paths.json'
        pairs = json.loads((ep/filename).read_text())  # Missing eval pairs MUST NOT self-align.
        if not pairs.get('same'):
            raise ValueError('Empty explicit reference list')
        main = json.loads((ep/'provenance.json').read_text())
        for peer in pairs['same']:
            other = json.loads((Path(peer)/'provenance.json').read_text())
            if (main['task'], main['split']) != (other['task'], other['split']):
                raise ValueError('Cross-task/split reference')
            if self.mode == 'train' and Path(peer) == ep:
                raise ValueError('Training self-reference is not permitted')
        if self.mode == 'eval' and (pairs.get('canonical_anchor_reviewed') is not True or len(pairs['same']) != 1):
            raise ValueError('Evaluation requires one reviewed canonical anchor')
        result = original_load(self, index)
        if result is None or result['ref_video_path'] not in pairs['same']:
            raise ValueError('Dataset silently substituted a reference or skipped a sample')
        return result
    if not getattr(original_load, '_piper_strict', False):
        checked_load._piper_strict = True
        datasets.AlignmentDataset._load_video_data_from_json = checked_load
    # Scoped to a process that explicitly opted into this Piper profile.
    datasets.AlignmentDataset.__getitem__ = datasets.AlignmentDataset._get_item_impl
    datasets.AlignmentCollator.load_imgs = strict_load_imgs


def smoke(root, output, context_frames=1, dataset_mode='train'):
    cfg = apply_profile(root, context_frames)
    enable_strict_loading()
    from datasets import AlignmentDataset
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'data_preprocessing/piper'))
    from build_progress import anchor_frames
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output report')
    random.seed(42)
    np.random.seed(42)
    records = []
    for split in ('train', 'val'):
        dataset = AlignmentDataset(mode=dataset_mode, video_paths_json=str(Path(root)/split/'piper_video_paths.json'))
        checked = set()
        for index, ep in enumerate(dataset.video_paths):
            path = ep['video_dir'] if isinstance(ep, dict) else ep
            source = Path(path).parent.name
            if source in checked:
                continue
            sample = dataset[index]
            main_prov = json.loads((Path(sample['name'])/'provenance.json').read_text())
            ref_prov = json.loads((Path(sample['ref_name'])/'provenance.json').read_text())
            if (dataset_mode == 'train' and sample['name'] == sample['ref_name']) or main_prov['task'] != ref_prov['task'] or ref_prov['split'] != split:
                raise ValueError('Self/cross-task/cross-split pairing')
            if sample['main_joint_per_step'] is not None or sample['ref_joint_per_step'] is not None:
                raise ValueError('Unexpected robot-state conditioning')
            main = anchor_frames(sample['frame_paths'], sample['name'], main_prov['frames'], context_frames)
            ref = anchor_frames(sample['ref_frame_paths'], sample['ref_name'], ref_prov['frames'], context_frames)
            np.testing.assert_array_equal(main, sample['chosen_steps'].numpy())
            np.testing.assert_array_equal(ref, sample['ref_chosen_steps'].numpy())
            records.append(dict(split=split, source=source, task=main_prov['task'], main=sample['name'],
                reference=sample['ref_name'], anchor_count=len(main), paths_per_anchor=cfg.DATA.NUM_STEPS))
            checked.add(source)
    report = dict(samples=records, sample_count=len(records), context_frames=context_frames, dataset_mode=dataset_mode,
        robot_joint_conditioning=False, black_frame_fallback=False, random_episode_retry=False,
        scope='CPU dataset/path-layout check only; no model, collator processor, GPU or DTW evaluation')
    # Until reviewed anchors are installed, evaluation must reject missing files,
    # not quietly self-align and accidentally create index-like progress labels.
    first = Path(records[0]['main'])
    if not (first/'task_paths_eval.json').exists():
        dataset = AlignmentDataset(mode='eval', video_paths_json=str(Path(root)/'train/piper_video_paths.json'))
        try:
            dataset[0]
        except FileNotFoundError as exc:
            if 'task_paths_eval.json' not in str(exc):
                raise
            report['missing_eval_anchor_rejected'] = True
        else:
            raise ValueError('Missing evaluation anchor was silently accepted')
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--context-frames', type=int, choices=(1, 2), default=1)
    p.add_argument('--dataset-mode', choices=('train', 'eval'), default='train')
    args = p.parse_args()
    print(json.dumps(smoke(args.root, args.output, args.context_frames, args.dataset_mode)))
